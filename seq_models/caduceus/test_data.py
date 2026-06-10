"""Smoke tests for seq_models/caduceus/data.py.

Run with: pytest seq_models/caduceus/test_data.py -v

The pure helpers (tile_motif, reverse_complement) and per-item sequence
construction are tested offline. The single DataModule/dataloader smoke test
needs the Caduceus tokenizer; it is skipped gracefully if it cannot be loaded
(e.g. no network access).
"""

import os
import sys

import pandas as pd
import pytest
import torch

# Make the repo root importable when running pytest from anywhere.
_REPO_ROOT = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)

from seq_models.caduceus.data import (
	create_data_module,
	reverse_complement,
	tile_motif,
)


# --- Synthetic genome / labels -------------------------------------------------

# chr1 = 20 'C' (left flank) + "ATATATA" STR (motif AT) + 20 'G' (right flank).
# STR occupies [20, 27); it starts with "AT" and ends with "TA".
LEFT_FLANK = "C" * 20
STR_SEQ = "ATATATA"
RIGHT_FLANK = "G" * 20
CHROM_SEQ = LEFT_FLANK + STR_SEQ + RIGHT_FLANK
STR_START = len(LEFT_FLANK)          # 20
STR_END = STR_START + len(STR_SEQ)   # 27

N_FLANKING_BP = 10
N_STR_BP = 6
N_PREFIX, N_STR_TOK, N_SUFFIX = 2, 3, 4

# Real on-disk labeled STR files + reference genome (for the integration test).
_REAL_DATA_DIR = os.path.join(
	_REPO_ROOT, "data", "STR_data", "HipSTR_labeled_STRs"
)
_REAL_REF = os.path.join(
	_REPO_ROOT, "data", "STR_data", "reference_genome",
	"GRCh38_full_analysis_set_plus_decoy_hla.fa",
)
REAL_LABELED_FILES = [
	"str_len_1_n_flanking_10000.tsv",
	"str_len_2_n_flanking_10000.tsv",
]


def _write_fixtures(tmp_path):
	"""Write a tiny FASTA + TSV and return (ref_path, data_path)."""
	ref_path = tmp_path / "tiny.fa"
	ref_path.write_text(f">chr1\n{CHROM_SEQ}\n")

	df = pd.DataFrame(
		[
			{
				"ID": "STR_fwd",
				"chrom": "chr1",
				"str_start": STR_START,
				"str_end": STR_END,
				"ref_copy_number": 3.5,  # 7 bp tract / motif_len 2
				"motif": "AT",
				"mode_copy_number": 5.0,
				"heterozygosity": 0.25,
				"split": "train",
				"rev_comp": False,
			},
			{
				"ID": "STR_fwd",  # same locus, reverse-complement orientation
				"chrom": "chr1",
				"str_start": STR_START,
				"str_end": STR_END,
				"ref_copy_number": 3.5,  # 7 bp tract / motif_len 2
				"motif": "AT",
				"mode_copy_number": 5.0,
				"heterozygosity": 0.25,
				"split": "train",
				"rev_comp": True,
			},
		]
	)
	data_path = tmp_path / "tiny.tsv"
	df.to_csv(data_path, sep="\t", index=False)
	return str(ref_path), str(data_path)


def _config(ref_path, data_path):
	return {
		"data_path": data_path,
		"ref_path": ref_path,
		"batch_size": 2,
		"n_flanking_bp": N_FLANKING_BP,
		"n_str_bp": N_STR_BP,
		"n_prefix_prompt_tokens": N_PREFIX,
		"n_str_prompt_tokens": N_STR_TOK,
		"n_suffix_prompt_tokens": N_SUFFIX,
		"dataloader_num_workers": 0,
	}


# --- Pure helpers (offline) ----------------------------------------------------

def test_tile_motif_start_and_end_anchors():
	# Matches the plan example for STR "ATATATA" with n_str_bp = 6.
	assert tile_motif("AT", 6, "left") == "ATATAT"
	assert tile_motif("TA", 6, "right") == "TATATA"
	# Non-multiple lengths stay phase-correct.
	assert tile_motif("AT", 5, "left") == "ATATA"
	assert tile_motif("TA", 5, "right") == "ATATA"
	assert tile_motif("A", 4, "left") == "AAAA"
	assert tile_motif("AT", 0, "left") == ""


def test_reverse_complement():
	assert reverse_complement("ACGT") == "ACGT"
	assert reverse_complement("AAAA") == "TTTT"
	assert reverse_complement("ATATAT") == "ATATAT"
	assert reverse_complement("CCCCCCCCCCATATAT") == "ATATATGGGGGGGGGG"


# --- DataModule smoke test (needs the Caduceus tokenizer) ----------------------

def test_datamodule_builds_expected_input_ids_targets_and_revcomp(tmp_path):
	ref_path, data_path = _write_fixtures(tmp_path)
	dm = create_data_module(_config(ref_path, data_path))
	try:
		dm.setup()
	except Exception as exc:  # tokenizer download / load failure -> skip
		pytest.skip(f"Could not load Caduceus tokenizer: {exc}")

	expected_keys = {"input_ids", "length", "variation", "id", "rev_comp"}
	# Caduceus uses a character tokenizer: 1 token per base (no special tokens).
	expected_len = (
		N_PREFIX
		+ (N_FLANKING_BP + N_STR_BP)   # pre_seq: left flank + start fill
		+ N_STR_TOK
		+ (N_STR_BP + N_FLANKING_BP)   # post_seq: end fill + right flank
		+ N_SUFFIX
	)
	n_total_learnable = N_PREFIX + N_STR_TOK + N_SUFFIX

	# Per-item checks (deterministic; indexes the dataset directly).
	item = dm.train_dataset[0]
	assert expected_keys == set(item.keys())
	assert item["input_ids"].dtype == torch.long
	assert item["input_ids"].shape[0] == expected_len
	assert item["length"].dtype == torch.float32
	assert item["variation"].dtype == torch.float32
	assert item["id"] == "STR_fwd"

	# Learnable token ids occupy the block just past the tokenizer vocab.
	start = dm.prompt_start_id
	learnable = item["input_ids"][
		torch.tensor(
			[0, 1]  # prefix
			+ list(range(2 + (N_FLANKING_BP + N_STR_BP),
						 2 + (N_FLANKING_BP + N_STR_BP) + N_STR_TOK))  # str gap
			+ list(range(expected_len - N_SUFFIX, expected_len))  # suffix
		)
	]
	assert learnable.min().item() >= start
	assert learnable.max().item() < start + n_total_learnable
	# DNA tokens stay within the real vocab.
	assert item["input_ids"].max().item() < start + n_total_learnable

	# --- Correctness: prefix / str-gap / suffix tokens are in the right spot
	# with the right ids. The id block is contiguous from `start`:
	# prefix = [start, start+N_PREFIX), str gap continues, suffix continues.
	ids = item["input_ids"]
	gap_start = N_PREFIX + (N_FLANKING_BP + N_STR_BP)
	expected_prefix = torch.arange(start, start + N_PREFIX)
	expected_str = torch.arange(start + N_PREFIX, start + N_PREFIX + N_STR_TOK)
	expected_suffix = torch.arange(
		start + N_PREFIX + N_STR_TOK,
		start + N_PREFIX + N_STR_TOK + N_SUFFIX,
	)
	assert torch.equal(ids[:N_PREFIX], expected_prefix)
	assert torch.equal(ids[gap_start:gap_start + N_STR_TOK], expected_str)
	assert torch.equal(ids[expected_len - N_SUFFIX:], expected_suffix)
	# And the slots between/around them are real DNA tokens (below `start`).
	assert ids[N_PREFIX:gap_start].max().item() < start
	assert ids[gap_start + N_STR_TOK:expected_len - N_SUFFIX].max().item() < start

	# --- Correctness: actual sequence content (re-encode expected strings) ---
	def enc(seq):
		return dm.tokenizer(
			seq, add_special_tokens=False, return_tensors="pt"
		).input_ids.squeeze(0)

	pre_len = N_FLANKING_BP + N_STR_BP   # left flank + start fill
	post_len = N_STR_BP + N_FLANKING_BP  # end fill + right flank
	pre_start = N_PREFIX
	post_start = N_PREFIX + pre_len + N_STR_TOK

	# Forward: flank stays out of the STR; motif tiling lands correctly.
	expected_pre = "C" * N_FLANKING_BP + "ATATAT"   # left flank + tile('AT')
	expected_post = "TATATA" + "G" * N_FLANKING_BP   # tile('TA') + right flank
	pre_ids = item["input_ids"][pre_start:pre_start + pre_len]
	post_ids = item["input_ids"][post_start:post_start + post_len]
	assert torch.equal(pre_ids, enc(expected_pre))
	assert torch.equal(post_ids, enc(expected_post))

	# Target values map to the right columns (not just dtype).
	assert item["length"].item() == pytest.approx(5.0)
	assert item["variation"].item() == pytest.approx(0.25)

	# --- Correctness: reverse complement (DNA(rc) == revcomp(DNA(fwd))) ---
	rc_item = dm.train_dataset[1]
	fwd_dna = expected_pre + expected_post
	rc_pre_ids = rc_item["input_ids"][pre_start:pre_start + pre_len]
	rc_post_ids = rc_item["input_ids"][post_start:post_start + post_len]
	rc_dna_ids = torch.cat([rc_pre_ids, rc_post_ids], dim=0)
	assert torch.equal(rc_dna_ids, enc(reverse_complement(fwd_dna)))

	# RC pair shares the same locus id.
	assert item["id"] == rc_item["id"] == "STR_fwd"

	# Batch / collation check.
	batch = next(iter(dm.train_dataloader()))
	assert expected_keys == set(batch.keys())
	assert batch["input_ids"].shape == (2, expected_len)
	assert batch["length"].shape == (2,)
	assert batch["rev_comp"].dtype == torch.bool
	assert len(batch["id"]) == 2


def test_single_objective_targets_override_emits_only_that_target(tmp_path):
	"""A single-objective `targets` mapping emits only that target."""
	ref_path, data_path = _write_fixtures(tmp_path)
	config = _config(ref_path, data_path)
	config["targets"] = {"length": "mode_copy_number"}
	dm = create_data_module(config)
	try:
		dm.setup()
	except Exception as exc:  # tokenizer download / load failure -> skip
		pytest.skip(f"Could not load Caduceus tokenizer: {exc}")

	item = dm.train_dataset[0]
	assert set(item.keys()) == {"input_ids", "length", "id", "rev_comp"}
	assert item["length"].item() == pytest.approx(5.0)


# --- Integration test against real on-disk labeled files -----------------------

@pytest.mark.parametrize("fname", REAL_LABELED_FILES)
def test_real_labeled_file_is_compatible_with_datamodule(fname):
	"""A real HipSTR_labeled_STRs TSV parses, splits, and (if the reference
	genome is present) yields well-formed items."""
	data_path = os.path.join(_REAL_DATA_DIR, fname)
	if not os.path.exists(data_path):
		pytest.skip(f"Real labeled file not available: {data_path}")

	# The schema our Dataset/DataModule depends on (default target columns).
	df = pd.read_csv(data_path, sep="\t")
	required = {
		"ID", "chrom", "str_start", "str_end", "motif", "ref_copy_number",
		"mode_copy_number", "heterozygosity", "split", "rev_comp",
	}
	assert required <= set(df.columns)
	assert set(df["split"].unique()) <= {"train", "val", "test"}

	config = {
		"data_path": data_path,
		"ref_path": _REAL_REF,
		"batch_size": 4,
		"n_flanking_bp": 50,
		"n_str_bp": N_STR_BP,
		"n_prefix_prompt_tokens": N_PREFIX,
		"n_str_prompt_tokens": N_STR_TOK,
		"n_suffix_prompt_tokens": N_SUFFIX,
		"dataloader_num_workers": 0,
	}
	dm = create_data_module(config)
	try:
		dm.setup()  # parses + splits the real df; does not touch the FASTA yet
	except Exception as exc:  # tokenizer download / load failure -> skip
		pytest.skip(f"Could not load Caduceus tokenizer: {exc}")

	# Splits partition the file exactly.
	assert len(dm.train_dataset) == int((df["split"] == "train").sum())
	assert len(dm.val_dataset) == int((df["split"] == "val").sum())
	assert len(dm.test_dataset) == int((df["split"] == "test").sum())

	# Item extraction needs the reference genome; run it only if present.
	if not os.path.exists(_REAL_REF):
		return

	expected_len = (
		N_PREFIX
		+ (config["n_flanking_bp"] + N_STR_BP)
		+ N_STR_TOK
		+ (N_STR_BP + config["n_flanking_bp"])
		+ N_SUFFIX
	)
	item = dm.train_dataset[0]
	assert set(item.keys()) == {"input_ids", "length", "variation", "id", "rev_comp"}
	assert item["input_ids"].dtype == torch.long
	assert item["input_ids"].shape[0] == expected_len
	assert item["length"].dtype == torch.float32
	assert item["variation"].dtype == torch.float32
