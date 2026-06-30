""" Region-level permutation importance for flanking sequences (Caduceus-Ph).

Breaks each flank into contiguous regions of a fixed width and measures the impact
of shuffling the bases within each region on the model's predictions. Regions are
indexed starting adjacent to the STR and moving outward; the remainder region (if
the flank length is not evenly divisible by the region width) is placed at the far
end of the flank.

This module:
- Loads a full fine-tuned Caduceus STRLengthModel from a model_dir (the predict.py
  pattern: read config.yaml, find the best checkpoint, rebuild the data module).
- Handles one OR multiple regression targets (e.g. length-only / variation-only
  single-task models, or the multi-task length+heterozygosity 'fiv' models). One
  forward pass yields every active target.
- Stores predictions in NATIVE units (the per-task inverse transform is applied
  during inference), so the downstream analyzer never has to know the transforms.

Outputs an .npz file with all baseline / permuted predictions so that downstream
analysis (local per-locus deltas, global metric drops, RC averaging, clustering,
plotting) can be done separately without re-running inference.

Args:
	--config: Path to the RPI YAML config file.
	--output_dir: Directory to save outputs (run dir is output_dir/<desc>).
	--cpu: Force use of CPU, even if CUDA/MPS is available.

Config keys:
	model_dir (str): Trained model directory, relative to
		scripts/training/output/caduceus/ (contains checkpoints/ and config.yaml).
	desc (str): Output name/path relative to output_dir.
	split (str): Split to run on ('test', 'val', 'train').
	batch_size (int): Batch size for forward passes.
	num_workers (int): DataLoader num_workers.
	region_width (int): Number of base pairs per region.
	n_permutations (int): Number of independent shuffles per region.
	seed (int): Random seed for reproducibility.
	subsample_loci (int, optional): If set, deterministically keep this many unique
		loci (both orientations of each), to cap runtime.
	max_flank_bp (int, optional): If set and < n_flanking_bp, only the inner
		max_flank_bp of each flank is partitioned into regions (the model still
		runs the full flank); caps the number of regions.
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from seq_models.caduceus.data import create_data_module
from seq_models.caduceus.model import STRLengthModel, inverse_transform
from seq_models.caduceus.predict import find_best_checkpoint


# Base directory holding trained Caduceus models (matches make_predictions.sh).
CADUCEUS_MODEL_DIR = "scripts/training/output/caduceus"


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

	Regions are ordered from STR-adjacent outward.  If the flank length is not
	evenly divisible by region_width the smaller remainder region is placed at the
	far (non-STR) end.

	Sequence layout (token indices):
		[prefix_prompt | left_flank | left_str | str_prompt | right_str | right_flank]
	(A trailing suffix_prompt, if any, sits after right_flank and does not shift
	any flank index, so it is irrelevant here.)

	Args:
		n_flanking_bp: Number of flanking base pairs per side.
		n_prefix_prompt: Number of prefix prompt tokens.
		n_str_bp: Number of STR base pairs shown on each side.
		n_str_prompt: Number of STR gap prompt tokens.
		region_width: Number of bp per region.

	Returns:
		List[dict] where each dict has keys: flank ('left'/'right'), region_idx
		(0 = closest to STR), token_start (inclusive), token_end (exclusive),
		distance_bp_start, distance_bp_end (bp from STR to near/far region edge).
	"""
	regions = []

	n_full = n_flanking_bp // region_width
	remainder = n_flanking_bp % region_width

	# --- Left flank ---
	# Token range: [n_prefix_prompt, n_prefix_prompt + n_flanking_bp).
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
	# Token range: [rf_start, rf_start + n_flanking_bp).
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
# Model / data loading (predict.py pattern)
# ---------------------------------------------------------------------------

def load_model_and_data(rpi_config):
	"""Load the trained model and build the dataset for the requested split.

	Mirrors seq_models/caduceus/predict.py: read the model's own config.yaml,
	override a few keys from the RPI config, build the data module, and pick the
	dataset for the split.

	Args:
		rpi_config (dict): Parsed RPI config.

	Returns:
		Tuple (model, model_config, dataset, str_df) where dataset is the
		STRLengthDataset for the split and str_df is its underlying DataFrame.
	"""
	full_model_dir = os.path.join(CADUCEUS_MODEL_DIR, rpi_config["model_dir"])

	model_config_path = os.path.join(full_model_dir, "config.yaml")
	print(f"Loading model config from {model_config_path}")
	with open(model_config_path, "r") as f:
		model_config = yaml.safe_load(f)

	# Override the loader-relevant keys from the RPI config.
	model_config["batch_size"] = rpi_config.get(
		"batch_size", model_config.get("batch_size", 16)
	)
	model_config["dataloader_num_workers"] = rpi_config.get(
		"num_workers", model_config.get("dataloader_num_workers", 4)
	)

	data_module = create_data_module(model_config)
	data_module.setup()

	split = rpi_config["split"]
	if split == "test":
		dataset = data_module.test_dataset
	elif split == "val":
		dataset = data_module.val_dataset
	elif split == "train":
		dataset = data_module.train_dataset
	else:
		raise ValueError(
			f"Invalid split: {split}. Must be 'test', 'val', or 'train'."
		)

	ckpt_path = find_best_checkpoint(full_model_dir)
	print(f"Loading model from checkpoint {ckpt_path}")
	model = STRLengthModel.load_from_checkpoint(ckpt_path, map_location="cpu")
	model.eval()

	return model, model_config, dataset, dataset.str_df


def subsample_dataset(dataset, n_loci, seed):
	"""Restrict a dataset to n_loci unique loci, keeping BOTH orientations.

	Deterministically (seeded) selects n_loci unique ``ID`` values and filters the
	dataset's str_df to the rows for those ids (forward + reverse complement). The
	dataset is filtered in place and its index reset.

	Args:
		dataset (STRLengthDataset): Dataset to subsample (modified in place).
		n_loci (int): Number of unique loci to keep.
		seed (int): Random seed.

	Returns:
		The same dataset, with a reduced str_df.
	"""
	df = dataset.str_df
	unique_ids = np.sort(df["ID"].unique())
	if n_loci >= len(unique_ids):
		print(
			f"  subsample_loci={n_loci} >= {len(unique_ids)} unique loci; "
			f"using all."
		)
		return dataset

	rng = np.random.default_rng(seed)
	chosen = rng.choice(unique_ids, size=n_loci, replace=False)
	chosen_set = set(chosen.tolist())

	dataset.str_df = (
		df[df["ID"].isin(chosen_set)].reset_index(drop=True)
	)
	print(
		f"  Subsampled to {n_loci} loci "
		f"({len(dataset.str_df)} orientation rows)."
	)
	return dataset


# ---------------------------------------------------------------------------
# Batched multi-task inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batched(model, input_ids, batch_size, device, task_names, transforms):
	"""Run model forward in batches, returning native-space preds per task.

	Args:
		model: STRLengthModel in eval mode, on ``device``.
		input_ids: (N, seq_len) tensor on CPU.
		batch_size: Batch size for forward passes.
		device: torch.device for inference.
		task_names: List of active target names.
		transforms: Dict of task -> transform name (for inverse transform).

	Returns:
		Dict task -> (N,) numpy array of native-space predictions.
	"""
	preds = {name: [] for name in task_names}
	n = input_ids.shape[0]

	for start in range(0, n, batch_size):
		batch = input_ids[start:start + batch_size].to(device)
		out_t = model(batch)  # dict task -> (B,) transformed space
		for name in task_names:
			native = inverse_transform(transforms[name], out_t[name])
			preds[name].append(native.float().cpu().numpy())

	return {name: np.concatenate(preds[name]) for name in task_names}


def collect_inputs(dataset, batch_size, num_workers, task_names):
	"""Load all input_ids, per-task labels, ids and rev_comp into memory.

	Args:
		dataset: STRLengthDataset for the split.
		batch_size: DataLoader batch size.
		num_workers: DataLoader num_workers.
		task_names: List of active target names.

	Returns:
		Tuple (input_ids (N, L) tensor, labels dict task -> (N,) numpy array,
		ids (N,) str numpy array, rev_comp (N,) bool numpy array).
	"""
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=False,
	)

	all_input_ids = []
	all_labels = {name: [] for name in task_names}
	all_ids = []
	all_rev_comp = []

	for batch in tqdm(loader, desc="Loading data"):
		all_input_ids.append(batch["input_ids"])
		all_ids.extend(list(batch["id"]))
		all_rev_comp.append(batch["rev_comp"])
		for name in task_names:
			all_labels[name].append(batch[name])

	input_ids = torch.cat(all_input_ids, dim=0)
	rev_comp = torch.cat(all_rev_comp, dim=0).numpy().astype(bool)
	ids = np.array([str(x) for x in all_ids])
	labels = {
		name: torch.cat(all_labels[name], dim=0).numpy()
		for name in task_names
	}
	return input_ids, labels, ids, rev_comp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(
		description="Region-level permutation importance for flanking "
		"sequences with the Caduceus STR length/variation model.",
	)
	parser.add_argument(
		"--config", type=str, required=True,
		help="Path to the RPI configuration YAML file.",
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

	# pytorch_lightning sets __spec__ assumptions; mirror predict.py guard.
	global __spec__
	__spec__ = None

	# ------------------------------------------------------------------
	# Load config
	# ------------------------------------------------------------------
	print(f"Loading config from {args.config}")
	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	region_width = config["region_width"]
	n_permutations = config["n_permutations"]
	seed = config.get("seed", 42)
	batch_size = config.get("batch_size", 16)
	subsample_loci = config.get("subsample_loci", None)
	max_flank_bp = config.get("max_flank_bp", None)

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

	with open(os.path.join(run_dir, "config.yaml"), "w") as f:
		yaml.dump(config, f)
	print(f"Outputs will be saved to: {run_dir}")

	# ------------------------------------------------------------------
	# Load model + data
	# ------------------------------------------------------------------
	model, model_config, dataset, _ = load_model_and_data(config)
	model.to(device)

	task_names = list(model.task_names)
	transforms = model.transforms
	targets = model.targets
	print(f"Active targets: {task_names}")
	print(f"Transforms: {dict(transforms)}")

	if subsample_loci is not None:
		dataset = subsample_dataset(dataset, int(subsample_loci), seed)

	# ------------------------------------------------------------------
	# Resolve sequence layout from the model's data config
	# ------------------------------------------------------------------
	n_prefix_prompt = model_config.get("n_prefix_prompt_tokens", 0)
	n_str_prompt = model_config.get("n_str_prompt_tokens", 0)
	n_suffix_prompt = model_config.get("n_suffix_prompt_tokens", 0)
	n_flanking_bp = model_config["n_flanking_bp"]
	n_str_bp = model_config["n_str_bp"]

	seq_len = (
		n_prefix_prompt
		+ n_flanking_bp + n_str_bp
		+ n_str_prompt
		+ n_str_bp + n_flanking_bp
		+ n_suffix_prompt
	)

	print(
		f"Sequence layout: prefix_prompt={n_prefix_prompt}, "
		f"left_flank={n_flanking_bp}, left_str={n_str_bp}, "
		f"str_prompt={n_str_prompt}, right_str={n_str_bp}, "
		f"right_flank={n_flanking_bp}, suffix_prompt={n_suffix_prompt}, "
		f"total={seq_len}"
	)

	# ------------------------------------------------------------------
	# Build region definitions
	# ------------------------------------------------------------------
	# Always build regions over the FULL flank so token indices land correctly
	# (build_region_definitions assumes the flank arg is the true flank length),
	# then optionally drop far regions to cap the inner extent that is permuted.
	regions = build_region_definitions(
		n_flanking_bp, n_prefix_prompt, n_str_bp, n_str_prompt,
		region_width,
	)
	region_flank_bp = n_flanking_bp
	if max_flank_bp is not None and max_flank_bp < n_flanking_bp:
		region_flank_bp = int(max_flank_bp)
		regions = [
			r for r in regions if r["distance_bp_end"] <= region_flank_bp
		]

	n_regions = len(regions)
	n_left = sum(1 for r in regions if r["flank"] == "left")
	n_right = sum(1 for r in regions if r["flank"] == "right")

	print(f"Region width: {region_width} bp "
		  f"(partitioning inner {region_flank_bp} bp of each flank)")
	print(f"Total regions: {n_regions} (left={n_left}, right={n_right})")
	print(f"Permutations per region: {n_permutations}")

	# ------------------------------------------------------------------
	# Collect inputs / labels into memory
	# ------------------------------------------------------------------
	num_workers = config.get("num_workers", 4)
	all_input_ids, all_labels, ids, rev_comp = collect_inputs(
		dataset, batch_size, num_workers, task_names
	)
	n_loci = all_input_ids.shape[0]
	print(f"Loaded {n_loci} orientation rows "
		  f"(input_ids shape: {tuple(all_input_ids.shape)})")

	# ------------------------------------------------------------------
	# Baseline predictions (unperturbed)
	# ------------------------------------------------------------------
	print("Computing baseline predictions ...")
	baseline_preds = predict_batched(
		model, all_input_ids, batch_size, device, task_names, transforms
	)

	# ------------------------------------------------------------------
	# Permuted predictions: one (R, S, N) array per task
	# ------------------------------------------------------------------
	rng = np.random.default_rng(seed)
	permuted_preds = {
		name: np.empty((n_regions, n_permutations, n_loci), dtype=np.float32)
		for name in task_names
	}

	total_iters = n_regions * n_permutations
	print(f"--- Running permutation importance "
		  f"({n_regions} regions x {n_permutations} permutations "
		  f"= {total_iters} iterations) ---")

	pbar = tqdm(total=total_iters, desc="Permutation importance")
	for r_idx, region in enumerate(regions):
		t_start = region["token_start"]
		t_end = region["token_end"]
		for s in range(n_permutations):
			shuffled_ids = all_input_ids.clone()
			shuffle_region(shuffled_ids, t_start, t_end, rng)
			preds = predict_batched(
				model, shuffled_ids, batch_size, device, task_names, transforms
			)
			for name in task_names:
				permuted_preds[name][r_idx, s, :] = preds[name]
			pbar.update(1)
	pbar.close()

	# ------------------------------------------------------------------
	# Save results
	# ------------------------------------------------------------------
	print("Saving results ...")
	region_flanks = np.array([r["flank"] for r in regions])
	region_idxs = np.array([r["region_idx"] for r in regions])
	region_token_starts = np.array([r["token_start"] for r in regions])
	region_token_ends = np.array([r["token_end"] for r in regions])
	region_dist_starts = np.array([r["distance_bp_start"] for r in regions])
	region_dist_ends = np.array([r["distance_bp_end"] for r in regions])

	save_arrays = {
		"ids": ids,
		"rev_comp": rev_comp,
		"task_names": np.array(task_names),
		"region_flanks": region_flanks,
		"region_idxs": region_idxs,
		"region_token_starts": region_token_starts,
		"region_token_ends": region_token_ends,
		"region_distance_bp_starts": region_dist_starts,
		"region_distance_bp_ends": region_dist_ends,
	}
	for name in task_names:
		save_arrays[f"baseline_predictions_{name}"] = baseline_preds[name]
		save_arrays[f"permuted_predictions_{name}"] = permuted_preds[name]
		save_arrays[f"true_labels_{name}"] = all_labels[name]

	npz_path = os.path.join(run_dir, "permutation_results.npz")
	np.savez_compressed(npz_path, **save_arrays)
	print(f"Results saved to {npz_path}")

	# ------------------------------------------------------------------
	# Save run metadata
	# ------------------------------------------------------------------
	meta = {
		"config": config,
		"model_dir": config["model_dir"],
		"task_names": task_names,
		"transforms": {n: transforms[n] for n in task_names},
		"targets": {n: targets[n] for n in task_names},
		"predictions_space": "native",
		"sequence_layout": {
			"n_prefix_prompt": n_prefix_prompt,
			"n_flanking_bp": n_flanking_bp,
			"n_str_bp": n_str_bp,
			"n_str_prompt": n_str_prompt,
			"n_suffix_prompt": n_suffix_prompt,
			"seq_len": seq_len,
			"order": [
				"prefix_prompt", "left_flank", "left_str",
				"str_prompt", "right_str", "right_flank", "suffix_prompt",
			],
		},
		"permutation_config": {
			"region_width": region_width,
			"region_max_flank_bp": region_flank_bp,
			"n_permutations": n_permutations,
			"seed": seed,
			"n_regions": n_regions,
			"n_regions_left": n_left,
			"n_regions_right": n_right,
			"subsample_loci": subsample_loci,
		},
		"n_loci_rows": int(n_loci),
		"n_unique_ids": int(len(np.unique(ids))),
		"device": str(device),
		"timestamp": datetime.datetime.now().isoformat(),
	}

	meta_path = os.path.join(run_dir, "meta.json")
	with open(meta_path, "w") as f:
		json.dump(meta, f, indent=4)
	print(f"Metadata saved to {meta_path}")

	# ------------------------------------------------------------------
	# Quick summary stats (sanity check on HPC)
	# ------------------------------------------------------------------
	print("\n--- Quick summary (native space) ---")
	for name in task_names:
		bl = baseline_preds[name]
		delta = np.abs(permuted_preds[name] - bl[np.newaxis, np.newaxis, :])
		print(f"  [{name}] mean |delta|={delta.mean():.6f}  "
			  f"max |delta|={delta.max():.6f}")

	print("\n--- Done ---")


if __name__ == "__main__":
	main()
