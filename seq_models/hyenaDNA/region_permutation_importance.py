""" Region-level permutation importance for flanking sequences.

Breaks each flank into contiguous regions of a fixed width and measures
the impact of shuffling the bases within each region on model predictions.
Regions are indexed starting adjacent to the STR and moving outward; the
remainder region (if the flank length is not evenly divisible by the region
width) is placed at the far end of the flank.

Outputs an .npz file containing all predictions so that downstream
analysis (local per-sample deltas, global metric drops, plotting) can be
done separately without re-running inference.

Args:
	--config: Path to YAML config file.
	--output_dir: Directory to save outputs.
	--cpu: Force use of CPU, even if CUDA/MPS is available.

Config keys:
	data_path (str): Path to STR data file.
	ref_path (str): Path to human reference genome file.
	model_path (str): Path to model checkpoint.
	desc (str): Output name/path relative to output_dir.
	split (str): Split field value in data file to run on.
	batch_size (int): Batch size for forward passes.
	num_workers (int): DataLoader num_workers.
	region_width (int): Number of base pairs per region.
	n_permutations (int): Number of independent shuffles per region.
	seed (int): Random seed for reproducibility.
"""

import argparse
import datetime
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from seq_models.hyenaDNA.data import STRLengthDataset
from seq_models.hyenaDNA.model import STRLengthModel


# ---------------------------------------------------------------------------
# Region construction
# ---------------------------------------------------------------------------

def build_region_definitions(
	n_flanking_bp,
	n_prefix_prompt,
	n_str_bp,
	n_str_prompt,
	region_width,
):
	"""Define contiguous regions for both flanks.

	Regions are ordered from STR-adjacent outward.  If the flank length
	is not evenly divisible by region_width the smaller remainder region
	is placed at the far (non-STR) end.

	Sequence layout (token indices):
		[prefix_prompt | left_flank | left_str | str_prompt | right_str | right_flank]

	Args:
		n_flanking_bp: Number of flanking base pairs per side.
		n_prefix_prompt: Number of prefix prompt tokens.
		n_str_bp: Number of STR base pairs shown on each side.
		n_str_prompt: Number of STR gap prompt tokens.
		region_width: Number of bp per region.

	Returns:
		List[dict] where each dict has keys:
			flank (str): 'left' or 'right'.
			region_idx (int): 0 = closest to STR.
			token_start (int): Start index in input_ids (inclusive).
			token_end (int): End index in input_ids (exclusive).
			distance_bp_start (int): Distance in bp from STR to the
				near edge of this region (0 = adjacent).
			distance_bp_end (int): Distance in bp from STR to the
				far edge of this region.
	"""
	regions = []

	n_full = n_flanking_bp // region_width
	remainder = n_flanking_bp % region_width

	# --- Left flank ---
	# Token range: [n_prefix_prompt, n_prefix_prompt + n_flanking_bp)
	# STR-adjacent end is the right edge of this block.
	lf_token_end = n_prefix_prompt + n_flanking_bp  # one past last token

	for i in range(n_full):
		t_end = lf_token_end - i * region_width
		t_start = t_end - region_width
		regions.append({
			"flank": "left",
			"region_idx": i,
			"token_start": t_start,
			"token_end": t_end,
			"distance_bp_start": i * region_width,
			"distance_bp_end": (i + 1) * region_width,
		})

	if remainder > 0:
		regions.append({
			"flank": "left",
			"region_idx": n_full,
			"token_start": n_prefix_prompt,
			"token_end": n_prefix_prompt + remainder,
			"distance_bp_start": n_full * region_width,
			"distance_bp_end": n_full * region_width + remainder,
		})

	# --- Right flank ---
	# Token range: [rf_start, rf_start + n_flanking_bp)
	# STR-adjacent end is the left edge of this block.
	rf_start = (
		n_prefix_prompt + n_flanking_bp + n_str_bp
		+ n_str_prompt + n_str_bp
	)

	for i in range(n_full):
		t_start = rf_start + i * region_width
		t_end = t_start + region_width
		regions.append({
			"flank": "right",
			"region_idx": i,
			"token_start": t_start,
			"token_end": t_end,
			"distance_bp_start": i * region_width,
			"distance_bp_end": (i + 1) * region_width,
		})

	if remainder > 0:
		t_start = rf_start + n_full * region_width
		regions.append({
			"flank": "right",
			"region_idx": n_full,
			"token_start": t_start,
			"token_end": t_start + remainder,
			"distance_bp_start": n_full * region_width,
			"distance_bp_end": n_full * region_width + remainder,
		})

	return regions


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------

def shuffle_region(input_ids, token_start, token_end, rng):
	"""Shuffle tokens within a region independently per sample.

	Args:
		input_ids: (N, seq_len) tensor of token IDs.  Modified in-place.
		token_start: Start index (inclusive).
		token_end: End index (exclusive).
		rng: numpy Generator for reproducibility.
	"""
	region = input_ids[:, token_start:token_end].cpu().numpy()

	for i in range(region.shape[0]):
		rng.shuffle(region[i])

	input_ids[:, token_start:token_end] = torch.from_numpy(region)


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batched(model, input_ids, batch_size, device):
	"""Run model forward pass in batches.

	Args:
		model: STRLengthModel in eval mode.
		input_ids: (N, seq_len) tensor on CPU.
		batch_size: Batch size for forward passes.
		device: torch.device for inference.

	Returns:
		(N,) numpy array of raw predictions (log-space if log_transform).
	"""
	preds = []
	n = input_ids.shape[0]

	for start in range(0, n, batch_size):
		batch = input_ids[start:start + batch_size].to(device)
		out = model(batch)
		preds.append(out.cpu().numpy())

	return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(
		description="Region-level permutation importance for flanking "
		"sequences in the STR length model.",
	)
	parser.add_argument(
		"--config", type=str, required=True,
		help="Path to the configuration YAML file.",
	)
	parser.add_argument(
		"--output_dir", type=str, default=".",
		help="Directory to save outputs.",
	)
	parser.add_argument(
		"--cpu", action="store_true",
		help="Force use of CPU, even if CUDA/MPS is available.",
	)
	args = parser.parse_args()

	# ------------------------------------------------------------------
	# Load config
	# ------------------------------------------------------------------
	print(f"Loading config from {args.config}")
	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	region_width = config["region_width"]
	n_permutations = config["n_permutations"]
	seed = config.get("seed", 42)
	batch_size = config.get("batch_size", 64)

	# ------------------------------------------------------------------
	# Device selection
	# ------------------------------------------------------------------
	if args.cpu:
		device = torch.device("cpu")
		print("CPU override flag set. Using CPU.")
	elif torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"Using CUDA ({torch.cuda.get_device_name(0)})")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
		print("Using MPS")
	else:
		device = torch.device("cpu")
		print("No GPU detected. Using CPU.")

	# ------------------------------------------------------------------
	# Set up output directory
	# ------------------------------------------------------------------
	run_dir = os.path.join(args.output_dir, config["desc"])
	os.makedirs(run_dir, exist_ok=True)

	config_save_path = os.path.join(run_dir, "config.yaml")
	with open(config_save_path, "w") as f:
		yaml.dump(config, f)
	print(f"Outputs will be saved to: {run_dir}")

	# ------------------------------------------------------------------
	# Load model checkpoint & extract saved hyper-parameters
	# ------------------------------------------------------------------
	model_path = config["model_path"]
	print(f"Loading model from {model_path}")

	model = STRLengthModel.load_from_checkpoint(
		model_path, map_location="cpu"
	)
	model.eval()
	model.to(device)

	model_hp = model.hparams

	raw_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
	dm_hp = raw_ckpt.get("datamodule_hyper_parameters", {})

	# ------------------------------------------------------------------
	# Resolve sequence layout dimensions from saved hyper-parameters
	# ------------------------------------------------------------------
	n_prefix_prompt = model_hp.get(
		"n_prefix_prompt_tokens",
		dm_hp.get("n_prefix_prompt_tokens", 0),
	)
	n_str_prompt = model_hp.get(
		"n_str_prompt_tokens",
		dm_hp.get("n_str_prompt_tokens", 0),
	)
	n_flanking_bp = dm_hp.get("n_flanking_bp", model_hp.get("n_flanking_bp"))
	n_str_bp = dm_hp.get("n_str_bp", model_hp.get("n_str_bp"))
	tokenizer_checkpoint = dm_hp.get(
		"tokenizer_checkpoint",
		model_hp.get("hyenaDNA_checkpoint"),
	)

	seq_len = (
		n_prefix_prompt
		+ n_flanking_bp + n_str_bp
		+ n_str_prompt
		+ n_str_bp + n_flanking_bp
	)

	print(f"Sequence layout: "
		  f"prefix_prompt={n_prefix_prompt}, "
		  f"left_flank={n_flanking_bp}, "
		  f"left_str={n_str_bp}, "
		  f"str_prompt={n_str_prompt}, "
		  f"right_str={n_str_bp}, "
		  f"right_flank={n_flanking_bp}, "
		  f"total={seq_len}")

	# ------------------------------------------------------------------
	# Build region definitions
	# ------------------------------------------------------------------
	regions = build_region_definitions(
		n_flanking_bp, n_prefix_prompt, n_str_bp, n_str_prompt,
		region_width,
	)

	n_regions = len(regions)
	n_left = sum(1 for r in regions if r["flank"] == "left")
	n_right = sum(1 for r in regions if r["flank"] == "right")

	print(f"Region width: {region_width} bp")
	print(f"Total regions: {n_regions} "
		  f"(left={n_left}, right={n_right})")
	print(f"Permutations per region: {n_permutations}")

	# ------------------------------------------------------------------
	# Load dataset and collect all input_ids / labels into memory
	# ------------------------------------------------------------------
	print(f"Loading tokenizer from {tokenizer_checkpoint}")
	tokenizer = AutoTokenizer.from_pretrained(
		tokenizer_checkpoint, trust_remote_code=True
	)

	print(f"Loading data from {config['data_path']} "
		  f"(split={config['split']})")
	full_df = pd.read_csv(config["data_path"], sep="\t")
	split_df = full_df[
		full_df["split"] == config["split"]
	].reset_index(drop=True)
	print(f"  {len(split_df)} samples in '{config['split']}' split")

	dataset = STRLengthDataset(
		str_df=split_df,
		ref_path=config["ref_path"],
		tokenizer=tokenizer,
		n_flanking_bp=n_flanking_bp,
		n_str_bp=n_str_bp,
		n_prefix_prompt_tokens=n_prefix_prompt,
		n_str_prompt_tokens=n_str_prompt,
	)

	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=config.get("num_workers", 4),
		pin_memory=False,
	)

	print("Loading all samples into memory ...")
	all_input_ids = []
	all_labels = []

	for batch in tqdm(loader, desc="Loading data"):
		all_input_ids.append(batch["input_ids"])
		all_labels.append(batch["label"])

	all_input_ids = torch.cat(all_input_ids, dim=0)  # (N, seq_len)
	all_labels = torch.cat(all_labels, dim=0)          # (N,)
	n_loci = all_input_ids.shape[0]

	print(f"Loaded {n_loci} samples (input_ids shape: {all_input_ids.shape})")

	# ------------------------------------------------------------------
	# Baseline predictions (unperturbed)
	# ------------------------------------------------------------------
	print("Computing baseline predictions ...")
	baseline_preds = predict_batched(
		model, all_input_ids, batch_size, device
	)

	# ------------------------------------------------------------------
	# Permuted predictions
	# ------------------------------------------------------------------
	rng = np.random.default_rng(seed)

	# Pre-allocate output array: (n_regions, n_permutations, n_loci)
	permuted_preds = np.empty(
		(n_regions, n_permutations, n_loci), dtype=np.float32
	)

	total_iters = n_regions * n_permutations
	print(f"--- Running permutation importance "
		  f"({n_regions} regions x {n_permutations} permutations "
		  f"= {total_iters} iterations) ---")

	pbar = tqdm(total=total_iters, desc="Permutation importance")

	for r_idx, region in enumerate(regions):
		t_start = region["token_start"]
		t_end = region["token_end"]

		for s in range(n_permutations):
			# Clone and shuffle the target region
			shuffled_ids = all_input_ids.clone()
			shuffle_region(shuffled_ids, t_start, t_end, rng)

			# Forward pass
			preds = predict_batched(
				model, shuffled_ids, batch_size, device
			)
			permuted_preds[r_idx, s, :] = preds

			pbar.update(1)

	pbar.close()

	# ------------------------------------------------------------------
	# Save results
	# ------------------------------------------------------------------
	print("Saving results ...")

	# Convert region definitions to structured arrays for npz
	region_flanks = np.array([r["flank"] for r in regions])
	region_idxs = np.array([r["region_idx"] for r in regions])
	region_token_starts = np.array([r["token_start"] for r in regions])
	region_token_ends = np.array([r["token_end"] for r in regions])
	region_dist_starts = np.array([r["distance_bp_start"] for r in regions])
	region_dist_ends = np.array([r["distance_bp_end"] for r in regions])

	npz_path = os.path.join(run_dir, "permutation_results.npz")
	np.savez_compressed(
		npz_path,
		# Predictions
		baseline_predictions=baseline_preds,           # (N,)
		permuted_predictions=permuted_preds,           # (R, S, N)
		true_labels=all_labels.numpy(),                # (N,)
		# Per-locus metadata (all length N, same order as predictions)
		hipstr_names=split_df["HipSTR_name"].values,
		rev_comp=split_df["rev_comp"].values,
		# Region info arrays (all length R)
		region_flanks=region_flanks,
		region_idxs=region_idxs,
		region_token_starts=region_token_starts,
		region_token_ends=region_token_ends,
		region_distance_bp_starts=region_dist_starts,
		region_distance_bp_ends=region_dist_ends,
	)
	print(f"Results saved to {npz_path}")

	# ------------------------------------------------------------------
	# Save run metadata
	# ------------------------------------------------------------------
	log_transform = model_hp.get("log_transform", False)

	meta = {
		"config": config,
		"model_hyperparameters": {
			k: v for k, v in dict(model_hp).items()
			if isinstance(v, (int, float, str, bool, type(None)))
		},
		"datamodule_hyperparameters": {
			k: v for k, v in dm_hp.items()
			if isinstance(v, (int, float, str, bool, type(None)))
		},
		"sequence_layout": {
			"n_prefix_prompt": n_prefix_prompt,
			"n_flanking_bp": n_flanking_bp,
			"n_str_bp": n_str_bp,
			"n_str_prompt": n_str_prompt,
			"seq_len": seq_len,
			"order": [
				"prefix_prompt", "left_flank", "left_str",
				"str_prompt", "right_str", "right_flank",
			],
		},
		"permutation_config": {
			"region_width": region_width,
			"n_permutations": n_permutations,
			"seed": seed,
			"n_regions": n_regions,
			"n_regions_left": n_left,
			"n_regions_right": n_right,
		},
		"log_transform": log_transform,
		"n_loci": n_loci,
		"device": str(device),
		"timestamp": datetime.datetime.now().isoformat(),
	}

	meta_path = os.path.join(run_dir, "meta.json")
	with open(meta_path, "w") as f:
		json.dump(meta, f, indent=4)
	print(f"Metadata saved to {meta_path}")

	# ------------------------------------------------------------------
	# Quick summary stats (for sanity checking on HPC)
	# ------------------------------------------------------------------
	mean_abs_delta = np.mean(
		np.abs(permuted_preds - baseline_preds[np.newaxis, np.newaxis, :])
	)
	max_abs_delta = np.max(
		np.abs(permuted_preds - baseline_preds[np.newaxis, np.newaxis, :])
	)

	print(f"\n--- Quick summary (raw {'log-' if log_transform else ''}space) ---")
	print(f"  Mean |delta| across all regions/perms/loci: {mean_abs_delta:.6f}")
	print(f"  Max  |delta|: {max_abs_delta:.6f}")

	print("\n--- Done ---")


if __name__ == "__main__":
	main()