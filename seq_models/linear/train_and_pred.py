""" Train and evaluate a linear baseline model for STR length/variation.

Fits a ridge regression on k-mer count features extracted from flanking
sequences (optionally stratified by flank side and/or distance from the STR),
selects the regularization strength (alpha) per target using the validation
split, and saves test-set predictions in the same per-locus format as the
Caduceus predict script.

The targets follow the Caduceus standard (multi-objective): `length`
(mode_copy_number, trained in log1p space) and `variation` (heterozygosity,
trained in arcsin-sqrt space). Each locus appears in both orientations
(augmentation); native-space test predictions are averaged per locus (RC
conjoining), matching seq_models/caduceus/predict.py.

Args:
	--config: Path to YAML config file.
	--output_dir: Parent directory for outputs.

Required config keys:
	data_path (str): Path to STR TSV file (Caduceus standard).
	ref_path (str): Path to reference genome FASTA (.fa with .fai).
	experiment_name (str): Output subdirectory relative to output_dir
		(e.g. "lin/str2/str2_f5000_sep_dist").
	n_flanking_bp (int): Number of flanking base pairs per side.

Optional config keys:
	k_values (list[int]): K-mer sizes to count. Default: [3, 4, 5].
	flank_mode (str): "separate" (left/right distinct) or "pooled" (summed).
		Default: "separate".
	distance_bins (list[int]): Upper edges (bp from STR) for distance
		stratification. Default: None (single whole-flank bin).
	targets (dict): output_name -> source column. Default: both length and
		variation. Provide a subset for single-objective runs.
	alphas (list[float]): Ridge alpha values to search over on the validation
		set. Default: [1e-4, 1e-2, 1, 1e2, 1e4].
	seed (int): Random seed for reproducibility. Default: 42.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from seq_models.linear.data import create_kmer_data


# Targets and training-space transforms, mirroring DEFAULT_TARGETS /
# DEFAULT_TARGET_TRANSFORMS in seq_models/caduceus/model.py (NumPy versions here
# to avoid importing the heavy caduceus/torch stack).
DEFAULT_TARGETS = {
	"length": "mode_copy_number",
	"variation": "heterozygosity",
}
DEFAULT_TARGET_TRANSFORMS = {
	"length": "log1p",
	"variation": "arcsin_sqrt",
}

_HALF_PI = np.pi / 2.0

# name -> (forward transform, inverse transform), both operating on np arrays.
TARGET_TRANSFORMS = {
	"none": (lambda y: y, lambda p: p),
	"log1p": (np.log1p, np.expm1),
	"arcsin_sqrt": (
		lambda y: np.arcsin(np.sqrt(np.clip(y, 0.0, 1.0))),
		lambda p: np.sin(np.clip(p, 0.0, _HALF_PI)) ** 2,
	),
}


def get_args():
	parser = argparse.ArgumentParser(
		description="Train and evaluate linear baseline for STR length/variation."
	)
	parser.add_argument(
		"--config",
		type=str,
		required=True,
		help="Path to the configuration YAML file.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=".",
		help="Parent directory for outputs.",
	)
	return parser.parse_args()


def fit_target(X_train, X_val, X_test, y_raw, transform_name, alphas, seed):
	"""Train ridge for one target: alpha search on val, refit, predict test.

	Args:
		X_train/X_val/X_test: scaled feature matrices.
		y_raw: dict with 'train'/'val' raw (native) target arrays.
		transform_name: training-space transform name.
		alphas: list of ridge alphas to search.
		seed: random seed.

	Returns:
		(test_pred_native, info) where test_pred_native is the native-space
		prediction array and info is a dict of alpha-search results.
	"""
	fwd, inv = TARGET_TRANSFORMS[transform_name]
	y_train = fwd(np.asarray(y_raw["train"], dtype=np.float64))
	y_val = fwd(np.asarray(y_raw["val"], dtype=np.float64))

	best_alpha = None
	best_val_mse = np.inf
	val_results = []
	for alpha in tqdm(alphas, desc=f"Alpha search ({transform_name})"):
		model = Ridge(alpha=alpha, random_state=seed)
		model.fit(X_train, y_train)
		val_preds = model.predict(X_val)
		val_mse = float(np.mean((val_preds - y_val) ** 2))
		val_results.append({"alpha": alpha, "val_mse": val_mse})
		tqdm.write(f"  alpha={alpha:<10g}  val_mse={val_mse:.6f}")
		if val_mse < best_val_mse:
			best_val_mse = val_mse
			best_alpha = alpha

	tqdm.write(f"  best alpha={best_alpha}  (val MSE={best_val_mse:.6f})")

	model = Ridge(alpha=best_alpha, random_state=seed)
	model.fit(X_train, y_train)
	test_pred_native = inv(model.predict(X_test))

	info = {
		"transform": transform_name,
		"best_alpha": best_alpha,
		"best_val_mse": best_val_mse,
		"alpha_search": val_results,
	}
	return test_pred_native, info


if __name__ == "__main__":

	args = get_args()

	# ------------------------------------------------------------------
	# Load config
	# ------------------------------------------------------------------

	print(f"Loading config from {args.config}")
	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	n_flanking_bp = config["n_flanking_bp"]
	k_values = config.get("k_values", [3, 4, 5])
	flank_mode = config.get("flank_mode", "separate")
	distance_bins = config.get("distance_bins", None)
	targets = config.get("targets", DEFAULT_TARGETS)
	alphas = [
		float(a) for a in config.get("alphas", [1e-4, 1e-2, 1.0, 1e2, 1e4])
	]
	seed = config.get("seed", 42)

	# Resolve a transform per target (defaults; "none" if unknown target).
	transforms = {
		name: DEFAULT_TARGET_TRANSFORMS.get(name, "none") for name in targets
	}

	# ------------------------------------------------------------------
	# Set up output directory
	# ------------------------------------------------------------------

	experiment_name = config["experiment_name"]
	experiment_path = os.path.join(args.output_dir, experiment_name)
	os.makedirs(experiment_path, exist_ok=True)

	with open(os.path.join(experiment_path, "config.yaml"), "w") as f:
		yaml.dump(config, f)

	# ------------------------------------------------------------------
	# Build k-mer dataset
	# ------------------------------------------------------------------

	splits, feature_names = create_kmer_data(
		data_path=config["data_path"],
		ref_path=config["ref_path"],
		n_flanking_bp=n_flanking_bp,
		k_values=k_values,
		flank_mode=flank_mode,
		distance_bins=distance_bins,
	)

	X_train = splits["train"]["X"]
	X_val = splits["val"]["X"]
	X_test = splits["test"]["X"]

	print(f"Train: {X_train.shape[0]} samples, "
		  f"Val: {X_val.shape[0]} samples, "
		  f"Test: {X_test.shape[0]} samples")
	print(f"Features: {X_train.shape[1]}")

	# ------------------------------------------------------------------
	# Scale features
	# ------------------------------------------------------------------

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_val = scaler.transform(X_val)
	X_test = scaler.transform(X_test)

	# ------------------------------------------------------------------
	# Fit one ridge per target and collect native-space test predictions
	# ------------------------------------------------------------------

	train_meta = splits["train"]["metadata"]
	val_meta = splits["val"]["metadata"]
	test_meta = splits["test"]["metadata"]

	target_info = {}
	test_preds = {}  # out_name -> native-space prediction array
	for out_name, col in targets.items():
		print(f"\n=== Target '{out_name}' (column '{col}', "
			  f"transform '{transforms[out_name]}') ===")
		pred_native, info = fit_target(
			X_train, X_val, X_test,
			y_raw={
				"train": train_meta[col].values,
				"val": val_meta[col].values,
			},
			transform_name=transforms[out_name],
			alphas=alphas,
			seed=seed,
		)
		test_preds[out_name] = pred_native
		info["source_column"] = col
		target_info[out_name] = info

	# ------------------------------------------------------------------
	# Assemble per-orientation table, then RC-average per locus
	# (parallel to seq_models/caduceus/predict.py aggregate_predictions)
	# ------------------------------------------------------------------

	per_orientation = pd.DataFrame({
		"id": test_meta["ID"].astype(str).values,
		"rev_comp": test_meta["rev_comp"].astype(bool).values,
	})
	for out_name, col in targets.items():
		per_orientation[f"pred_{out_name}"] = test_preds[out_name]
		per_orientation[f"true_{out_name}"] = test_meta[col].values

	agg = {f"pred_{t}": "mean" for t in targets}
	agg.update({f"true_{t}": "first" for t in targets})
	per_locus = per_orientation.groupby("id", as_index=False).agg(agg)

	# Merge locus metadata (one row per ID) into the per-locus table.
	meta_cols = [
		c for c in ["chrom", "str_start", "str_end", "motif", "split"]
		if c in test_meta.columns
	]
	if meta_cols:
		meta = (
			test_meta[["ID"] + meta_cols]
			.drop_duplicates("ID")
			.rename(columns={"ID": "id"})
		)
		per_locus = per_locus.merge(meta, on="id", how="left")

	# ------------------------------------------------------------------
	# Save predictions (matches Caduceus predict.py output format)
	# ------------------------------------------------------------------

	orient_path = os.path.join(experiment_path, "predictions_test_by_orientation.tsv")
	per_orientation.to_csv(orient_path, sep="\t", index=False)
	print(f"\nSaved {len(per_orientation)} per-orientation rows to {orient_path}")

	pred_path = os.path.join(experiment_path, "predictions_test.tsv")
	per_locus.to_csv(pred_path, sep="\t", index=False)
	print(f"Saved {len(per_locus)} per-locus (RC-averaged) rows to {pred_path}")

	# ------------------------------------------------------------------
	# Save run summary
	# ------------------------------------------------------------------

	summary = {
		"n_flanking_bp": n_flanking_bp,
		"k_values": k_values,
		"flank_mode": flank_mode,
		"distance_bins": distance_bins,
		"n_features": len(feature_names),
		"n_train": int(X_train.shape[0]),
		"n_val": int(X_val.shape[0]),
		"n_test": int(X_test.shape[0]),
		"n_test_loci": int(len(per_locus)),
		"targets": target_info,
	}

	summary_path = os.path.join(experiment_path, "summary.json")
	with open(summary_path, "w") as f:
		json.dump(summary, f, indent=4)
	print(f"Saved run summary to {summary_path}")
