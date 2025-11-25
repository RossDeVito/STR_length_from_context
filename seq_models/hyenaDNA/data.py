""" Dataset and data loaders for STR length task with soft prompting. 

Requires a TSV file with columns:
- chrom: Chromosome name (e.g., 'chr1')
- str_start: Start index of the STR
- str_end: End index of the STR (first base after the STR)
- copy_number: Copy number of the STR in reference genome
- rev_comp: Boolean indicating if sequence is reverse complement
- split: One of 'train', 'val', or 'test' indicating data split

As well as reference genome FASTA file to extract sequences. (.fa and .fai,
see data/STR_data/reference_genome/download.sh)

Function create_data_module() creates a PyTorch Lightning DataModule
based on the arguments in a config file used by the training script.
See create_data_module() docstring for required and optional config keys.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

from pyfaidx import Fasta

from seq_models.hyenaDNA.hyenaDNA_info import PROMPT_START_ID


def create_data_module(config):
	""" Create STRLengthDataModule from config.

	Args:
		config (dict): Configuration dictionary.

	Required config keys:
		data_path (str): Path to the data file.
		ref_path (str): Path to the reference genome FASTA file.
		hyenaDNA_checkpoint (str): Checkpoint name or path for tokenizer.
		batch_size (int): Batch size for data loaders.
		n_flanking_bp (int): Number of flanking base pairs to include
			on each side of the STR.
		n_str_bp (int): Number of base pairs of the STR to include
			on each side of the STR.
		n_prompt_tokens (int): Number of soft prompt tokens to use. Will
			be placed in the middle of the STR.
	
	Optional config keys:
		dataloader_num_workers (int): Number of workers for data loaders.
			Default: 4.
	"""

	return STRLengthDataModule(
		data_path=config["data_path"],
		ref_path=config["ref_path"],
		tokenizer_checkpoint=config["hyenaDNA_checkpoint"],
		batch_size=config["batch_size"],
		n_flanking_bp=config["n_flanking_bp"],
		n_str_bp=config["n_str_bp"],
		n_prompt_tokens=config["n_prompt_tokens"],
		num_workers=config.get("dataloader_num_workers", 4),
	)


class STRLengthDataset(Dataset):
	""" Dataset for STR length prediction with soft prompting. 
	
	Attributes:
		str_df (pd.DataFrame): DataFrame with STR information.
		ref_path (str): Path to the reference genome FASTA file.
		tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
		n_flanking_bp (int): Number of flanking base pairs to include on
			each side of the STR.
		n_str_bp (int): Number of base pairs of the STR to include on
			each side of the STR.
		n_prompt_tokens (int): Number of soft prompt tokens to use. Will
			be placed in the middle of the STR.
		prompt_token_ids (torch.Tensor): Tensor of prompt token IDs.
	"""

	def __init__(
		self,
		str_df,
		ref_path,
		tokenizer,
		n_flanking_bp,
		n_str_bp,
		n_prompt_tokens,
	):
		"""Initialize dataset.
		
		Args:
			str_df (pd.DataFrame): DataFrame with STR information.
			ref_path (str): Path to the reference genome FASTA file.
			tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
			n_flanking_bp (int): Number of flanking base pairs to include on
				each side of the STR.
			n_str_bp (int): Number of base pairs of the STR to include on
				each side of the STR.
			n_prompt_tokens (int): Number of soft prompt tokens to use. Will
				be placed in the middle of the STR.
		"""
		
		self.str_df = str_df.reset_index(drop=True)
		self.ref_path = ref_path
		self.tokenizer = tokenizer
		self.n_flanking_bp = n_flanking_bp
		self.n_str_bp = n_str_bp
		self.n_prompt_tokens = n_prompt_tokens

		# Do NOT initialize Fasta here. 
		# It breaks pickling for num_workers > 0.
		self.ref_genome = None

		# Create prompt token IDs
		self.prompt_token_ids = torch.arange(
			PROMPT_START_ID,
			PROMPT_START_ID + n_prompt_tokens
		)

		self.validity_check()

	def __len__(self):
		return len(self.str_df)
	
	def validity_check(self):
		"""Throws warning is 2 * n_str_bp is greater than min STR length."""
		min_str_length = (self.str_df["str_end"] - self.str_df["str_start"]).min()

		# Raise error if n_str_bp is longer than smallest STR
		if self.n_str_bp > min_str_length:
			raise ValueError(
				f"n_str_bp ({self.n_str_bp}) is greater than "
				f"minimum STR length ({min_str_length}). "
				"Reduce n_str_bp to avoid data leakage."
			)
		# Warn if 2 * n_str_bp is longer than smallest STR, though
		# should not cause leakage
		elif 2 * self.n_str_bp > min_str_length:
			print(
				f"WARNING: 2 * n_str_bp ({2 * self.n_str_bp}) is greater than "
				f"minimum STR length ({min_str_length}). "
				"Consider reducing n_str_bp to avoid overlapping STR regions."
			)

	def __getitem__(self, idx):
		"""Get item at index idx.

		Args:
			idx (int): Index of item to get.

		Returns:
			dict: Dictionary with keys 'input_ids' and 'label'.
		"""

		# Lazy Load Fasta
		if self.ref_genome is None:
			# sequence_always_upper=True is safer for tokenizers
			self.ref_genome = Fasta(self.ref_path, sequence_always_upper=True)

		row = self.str_df.iloc[idx]
		chrom = row["chrom"]
		start = row["str_start"]
		end = row["str_end"]

		# Get sequences
		pre_seq = self.ref_genome[
			chrom
		][
			start - self.n_flanking_bp : start + self.n_str_bp
		]
		post_seq = self.ref_genome[
			chrom
		][
			end - self.n_str_bp : end + self.n_flanking_bp
		]

		# Flip is reverse complement
		if self.str_df.iloc[idx]["rev_comp"]:
			final_pre_seq = post_seq.reverse.complement.seq
			final_post_seq = pre_seq.reverse.complement.seq
		else:
			final_pre_seq = pre_seq.seq
			final_post_seq = post_seq.seq

		# Combine and tokenize
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

		# Combine with prompt tokens in the middle and add label
		return {
			"input_ids": torch.cat(
				[
					start_tokens,
					self.prompt_token_ids,
					end_tokens
				],
				dim=0
			),
			"label": torch.tensor(
				self.str_df.iloc[idx]["copy_number"],
				dtype=torch.float32
			)
		}
	

class STRLengthDataModule(pl.LightningDataModule):
	""" DataModule for STR length prediction with soft prompting. """

	def __init__(
		self,
		data_path,
		ref_path,
		tokenizer_checkpoint,
		batch_size,
		n_flanking_bp,
		n_str_bp,
		n_prompt_tokens,
		num_workers=4,
	):
		super().__init__()
		self.save_hyperparameters()

		self.tokenizer = None

	def setup(self, stage=None):

		# Load tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(
			self.hparams.tokenizer_checkpoint,
			trust_remote_code=True
		)

		# Load data and create df by split
		full_df = pd.read_csv(
			self.hparams.data_path,
			sep="\t"
		)

		train_df = full_df[full_df["split"] == "train"].reset_index(drop=True)
		val_df = full_df[full_df["split"] == "val"].reset_index(drop=True)
		test_df = full_df[full_df["split"] == "test"].reset_index(drop=True)

		dataset_args = {
			"ref_path": self.hparams.ref_path,
			"tokenizer": self.tokenizer,
			"n_flanking_bp": self.hparams.n_flanking_bp,
			"n_str_bp": self.hparams.n_str_bp,
			"n_prompt_tokens": self.hparams.n_prompt_tokens,
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
			persistent_workers=True,
			pin_memory=True,
		)
	
	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			persistent_workers=True,
			pin_memory=True,
		)
	
	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			persistent_workers=True,
			pin_memory=True,
		)