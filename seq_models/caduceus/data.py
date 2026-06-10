""" Dataset and data loaders for the STR length/variation task with Caduceus-Ph.

Requires a TSV file with columns:
- ID: Locus identifier (used to pair reverse-complement samples downstream)
- chrom: Chromosome name (e.g., 'chr1')
- str_start: Start index of the STR (0-based, half-open)
- str_end: End index of the STR (first base after the STR)
- motif: Repeat unit sequence (e.g., 'AC'). May be blank; the tiling unit length
  is derived from the coordinates and ref_copy_number, not from this string.
- rev_comp: Boolean indicating if sequence should be reverse complemented
- split: One of 'train', 'val', or 'test' indicating data split
- One column per regression target (defaults: 'mode_copy_number' for length,
  'heterozygosity' for variation; see `targets` below).

As well as a reference genome FASTA file to extract sequences (.fa and .fai,
see data/STR_data/reference_genome/download.sh).

Unlike the HyenaDNA soft-prompt pipeline this module is built for full
fine-tuning of Caduceus-Ph:
- The STR boundary is synthesized by tiling the reference repeat unit to
  `n_str_bp` bases on each side rather than copying real STR bases, so the
  emitted length never reveals the true tract length (no leakage, no validity
  check needed).
- Learnable tokens can be placed at the sequence start, in the STR gap, and at
  the sequence end. Their ids start just past the tokenizer vocab; the model
  must resize its embedding matrix to match (handled in a later step).
- Targets are multi-objective: the output dict uses simplified, overridable keys
  (e.g. 'length', 'variation').

Function create_data_module() creates a PyTorch Lightning DataModule based on the
arguments in a config file used by the training script. See its docstring for
required and optional config keys.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

from pyfaidx import Fasta

from seq_models.caduceus.model import (
	CADUCEUS_CHECKPOINT,
	DEFAULT_TARGETS,
	get_caduceus_vocab_size,
)


_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(seq):
	"""Return the reverse complement of an (uppercase) DNA string."""
	return seq.translate(_COMPLEMENT)[::-1]


def tile_motif(unit, length, anchor="left"):
	"""Tile a motif unit to a fixed length.

	Repeats `unit` until it is at least `length` characters, then takes the
	first `length` chars (anchor='left', for the start side) or the last
	`length` chars (anchor='right', for the end side).

	Example: tile_motif('AT', 6, 'left') -> 'ATATAT';
	         tile_motif('TA', 6, 'right') -> 'TATATA'.
	"""
	if length <= 0:
		return ""
	reps = -(-length // len(unit))  # ceil division
	tiled = unit * reps
	return tiled[:length] if anchor == "left" else tiled[-length:]


def create_data_module(config):
	""" Create STRLengthDataModule from config.

	Args:
		config (dict): Configuration dictionary.

	Required config keys:
		data_path (str): Path to the data file.
		ref_path (str): Path to the reference genome FASTA file.
		batch_size (int): Batch size for data loaders.
		n_flanking_bp (int): Number of flanking base pairs to include on each
			side of the STR.
		n_str_bp (int): Number of synthesized (motif-tiled) base pairs to include
			on each side of the STR gap.

	Optional config keys:
		n_prefix_prompt_tokens (int): Learnable tokens at sequence start.
			Default 0.
		n_str_prompt_tokens (int): Learnable tokens in the STR gap. Default 0.
		n_suffix_prompt_tokens (int): Learnable tokens at sequence end.
			Default 0.
		targets (dict): Mapping of output_name -> source column. Default
			DEFAULT_TARGETS ({'length': 'mode_copy_number',
			'variation': 'heterozygosity'}). Provide a subset to train on a
			single objective.
		dataloader_num_workers (int): Number of workers for data loaders.
			Default 4.
		dev_subset_n (int): If set, cap every split to its first n rows for fast
			dev iteration. Default None (use all rows).
	"""
	return STRLengthDataModule(
		data_path=config["data_path"],
		ref_path=config["ref_path"],
		batch_size=config["batch_size"],
		n_flanking_bp=config["n_flanking_bp"],
		n_str_bp=config["n_str_bp"],
		n_prefix_prompt_tokens=config.get("n_prefix_prompt_tokens", 0),
		n_str_prompt_tokens=config.get("n_str_prompt_tokens", 0),
		n_suffix_prompt_tokens=config.get("n_suffix_prompt_tokens", 0),
		targets=config.get("targets", DEFAULT_TARGETS),
		num_workers=config.get("dataloader_num_workers", 4),
		dev_subset_n=config.get("dev_subset_n", None),
	)


class STRLengthDataset(Dataset):
	""" Dataset for STR length/variation prediction with Caduceus-Ph.

	Attributes:
		str_df (pd.DataFrame): DataFrame with STR information.
		ref_path (str): Path to the reference genome FASTA file.
		tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
		n_flanking_bp (int): Flanking base pairs on each side of the STR.
		n_str_bp (int): Synthesized motif-tiled base pairs on each side.
		prefix_prompt_ids / str_prompt_ids / suffix_prompt_ids (torch.Tensor):
			Learnable token ids at start / STR gap / end.
		targets (dict): Mapping of output_name -> source column.
	"""

	def __init__(
		self,
		str_df,
		ref_path,
		tokenizer,
		n_flanking_bp,
		n_str_bp,
		motif_len,
		prompt_start_id,
		n_prefix_prompt_tokens,
		n_str_prompt_tokens,
		n_suffix_prompt_tokens,
		targets,
	):
		"""Initialize dataset.

		Args:
			str_df (pd.DataFrame): DataFrame with STR information.
			ref_path (str): Path to the reference genome FASTA file.
			tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
			n_flanking_bp (int): Flanking base pairs on each side of the STR.
			n_str_bp (int): Synthesized motif-tiled base pairs on each side.
			motif_len (int): Repeat-unit length shared by all loci in the file;
				sets how many reference bases form the tiled unit.
			prompt_start_id (int): First id used for learnable tokens (just past
				the tokenizer vocab).
			n_prefix_prompt_tokens (int): Learnable tokens at sequence start.
			n_str_prompt_tokens (int): Learnable tokens in the STR gap.
			n_suffix_prompt_tokens (int): Learnable tokens at sequence end.
			targets (dict): Mapping of output_name -> source column.
		"""

		self.str_df = str_df.reset_index(drop=True)
		self.ref_path = ref_path
		self.tokenizer = tokenizer
		self.n_flanking_bp = n_flanking_bp
		self.n_str_bp = n_str_bp
		self.motif_len = motif_len
		self.targets = dict(targets)

		# Do NOT initialize Fasta here.
		# It breaks pickling for num_workers > 0.
		self.ref_genome = None

		# Create learnable token ids: a contiguous block starting just past the
		# tokenizer vocab, split across the three structural positions.
		n_total = (
			n_prefix_prompt_tokens
			+ n_str_prompt_tokens
			+ n_suffix_prompt_tokens
		)
		all_prompt_ids = torch.arange(
			prompt_start_id,
			prompt_start_id + n_total,
		)

		self.prefix_prompt_ids = all_prompt_ids[:n_prefix_prompt_tokens]
		self.str_prompt_ids = all_prompt_ids[
			n_prefix_prompt_tokens:n_prefix_prompt_tokens + n_str_prompt_tokens
		]
		self.suffix_prompt_ids = all_prompt_ids[
			n_prefix_prompt_tokens + n_str_prompt_tokens:
		]

	def __len__(self):
		return len(self.str_df)

	def __getitem__(self, idx):
		"""Get item at index idx.

		Args:
			idx (int): Index of item to get.

		Returns:
			dict: Keys 'input_ids', one entry per configured target (e.g.
				'length', 'variation'), 'id', and 'rev_comp'.
		"""

		# Lazy Load Fasta
		if self.ref_genome is None:
			# sequence_always_upper=True is safer for tokenizers
			self.ref_genome = Fasta(self.ref_path, sequence_always_upper=True)

		row = self.str_df.iloc[idx]
		chrom = row["chrom"]
		start = row["str_start"]
		end = row["str_end"]
		# Repeat-unit length is a per-file constant (derived once in the
		# DataModule from coordinates + ref_copy_number), not the 'motif' string.
		motif_len = self.motif_len

		# Flanks (real reference sequence) up to, but not into, the STR.
		left_flank = self.ref_genome[chrom][
			start - self.n_flanking_bp : start
		].seq
		right_flank = self.ref_genome[chrom][
			end : end + self.n_flanking_bp
		].seq

		# Synthesize the STR boundary by tiling the motif unit (read from the
		# reference so its phase is correct) to n_str_bp on each side.
		start_unit = self.ref_genome[chrom][start : start + motif_len].seq
		end_unit = self.ref_genome[chrom][end - motif_len : end].seq
		start_fill = tile_motif(start_unit, self.n_str_bp, anchor="left")
		end_fill = tile_motif(end_unit, self.n_str_bp, anchor="right")

		pre_seq = left_flank + start_fill
		post_seq = end_fill + right_flank

		# Flip if reverse complement (swap sides, keep learnable tokens in their
		# structural positions).
		if row["rev_comp"]:
			final_pre_seq = reverse_complement(post_seq)
			final_post_seq = reverse_complement(pre_seq)
		else:
			final_pre_seq = pre_seq
			final_post_seq = post_seq

		# Tokenize each DNA piece separately so learnable tokens can be inserted.
		start_tokens = self.tokenizer(
			final_pre_seq,
			add_special_tokens=False,
			return_tensors='pt'
		).input_ids.squeeze(0)

		end_tokens = self.tokenizer(
			final_post_seq,
			add_special_tokens=False,
			return_tensors='pt'
		).input_ids.squeeze(0)

		input_ids = torch.cat(
			[
				self.prefix_prompt_ids,
				start_tokens,
				self.str_prompt_ids,
				end_tokens,
				self.suffix_prompt_ids,
			],
			dim=0
		)

		item = {
			"input_ids": input_ids,
			# id + rev_comp let downstream code pair the two orientations for
			# post-hoc conjoining (averaging) at test/interpretation time.
			"id": str(row["ID"]),
			"rev_comp": torch.tensor(bool(row["rev_comp"])),
		}
		for out_name, col in self.targets.items():
			item[out_name] = torch.tensor(row[col], dtype=torch.float32)

		return item


class STRLengthDataModule(pl.LightningDataModule):
	""" DataModule for STR length/variation prediction with Caduceus-Ph. """

	def __init__(
		self,
		data_path,
		ref_path,
		batch_size,
		n_flanking_bp,
		n_str_bp,
		n_prefix_prompt_tokens=0,
		n_str_prompt_tokens=0,
		n_suffix_prompt_tokens=0,
		targets=None,
		num_workers=4,
		dev_subset_n=None,
	):
		super().__init__()
		if targets is None:
			targets = DEFAULT_TARGETS
		self.save_hyperparameters()

		self.tokenizer = None
		# First id for learnable tokens; set in setup() from the model vocab.
		self.prompt_start_id = None

	def setup(self, stage=None):

		# Load tokenizer (single fixed Caduceus-Ph checkpoint). The repo ships a
		# custom CharacterTokenizer, so trust_remote_code is required.
		self.tokenizer = AutoTokenizer.from_pretrained(
			CADUCEUS_CHECKPOINT,
			trust_remote_code=True,
		)

		# Learnable token ids live just past the backbone's (padded) vocabulary,
		# so they match the model's embedding rows. The tokenizer has 12 real
		# tokens but the embedding is padded to 16; the model splits its embedding
		# at this same id.
		self.prompt_start_id = get_caduceus_vocab_size()

		# Load data and split.
		full_df = pd.read_csv(
			self.hparams.data_path,
			sep="\t"
		)

		# Every locus in a file shares one repeat-unit length (each file is built
		# for a single STR motif length, 1 or 2). Derive that single length from
		# the coordinates and reference copy number -- not the 'motif' string,
		# which is blank for a few source loci -- since for every locus
		# str_end - str_start == ref_copy_number * motif_len.
		unit_lens = (
			(full_df["str_end"] - full_df["str_start"])
			/ full_df["ref_copy_number"]
		).round().astype(int)

		# Cross-check once: the derived length must be uniform across the file and
		# equal the length of every 'motif' string that is present.
		motif_str = full_df["motif"].astype(str).str.strip()
		present = full_df["motif"].notna() & (motif_str != "")
		if unit_lens.nunique() != 1 or (
			motif_str[present].str.len() != unit_lens[present]
		).any():
			raise ValueError(
				"Inconsistent repeat-unit length: derived "
				f"{sorted(unit_lens.unique())} from coordinates/ref_copy_number, "
				f"present motif lengths "
				f"{sorted(motif_str[present].str.len().unique())}."
			)
		motif_len = int(unit_lens.iloc[0])
		print(
			f"Using motif_len={motif_len} (derived from coordinates and "
			f"ref_copy_number; consistent with {int(present.sum())} labeled "
			f"motif(s), {int((~present).sum())} blank)."
		)

		train_df = full_df[full_df["split"] == "train"].reset_index(drop=True)
		val_df = full_df[full_df["split"] == "val"].reset_index(drop=True)
		test_df = full_df[full_df["split"] == "test"].reset_index(drop=True)

		# Dev option: cap every split for fast iteration.
		if self.hparams.dev_subset_n is not None:
			n = self.hparams.dev_subset_n
			train_df = train_df.head(n).reset_index(drop=True)
			val_df = val_df.head(n).reset_index(drop=True)
			test_df = test_df.head(n).reset_index(drop=True)

		dataset_args = {
			"ref_path": self.hparams.ref_path,
			"tokenizer": self.tokenizer,
			"n_flanking_bp": self.hparams.n_flanking_bp,
			"n_str_bp": self.hparams.n_str_bp,
			"motif_len": motif_len,
			"prompt_start_id": self.prompt_start_id,
			"n_prefix_prompt_tokens": self.hparams.n_prefix_prompt_tokens,
			"n_str_prompt_tokens": self.hparams.n_str_prompt_tokens,
			"n_suffix_prompt_tokens": self.hparams.n_suffix_prompt_tokens,
			"targets": self.hparams.targets,
		}

		self.train_dataset = STRLengthDataset(train_df, **dataset_args)
		self.val_dataset = STRLengthDataset(val_df, **dataset_args)
		self.test_dataset = STRLengthDataset(test_df, **dataset_args)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.hparams.batch_size,
			shuffle=True,
			num_workers=self.hparams.num_workers,
			persistent_workers=self.hparams.num_workers > 0,
			pin_memory=True,
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			persistent_workers=self.hparams.num_workers > 0,
			pin_memory=True,
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			persistent_workers=self.hparams.num_workers > 0,
			pin_memory=True,
		)
