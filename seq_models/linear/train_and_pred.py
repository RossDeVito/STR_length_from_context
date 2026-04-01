""" Train and evaluate a linear baseline model for STR length prediction.

Fits a ridge regression on k-mer count features extracted from flanking
sequences, selects the regularization strength (alpha) using the
validation split, and saves test-set predictions in the same TSV format
as the HyenaDNA predict script.

Args:
	--config: Path to YAML config file.
	--output_dir: Parent directory for outputs.

Required config keys:
	data_path (str): Path to filtered STR TSV file.
	ref_path (str): Path to reference genome FASTA (.fa with .fai).
	experiment_name (str): Output subdirectory relative to output_dir
		(e.g. "str2/v1/str2_ridge_f2000").
	n_flanking_bp (int): Number of flanking base pairs per side.

Optional config keys:
	k_values (list[int]): K-mer sizes to count. Default: [3, 4, 5].
	log_transform (bool): If true, train in log1p(copy_number) space.
		Predictions are always converted back to copy-number space
		for the output TSV. Default: true.
	alphas (list[float]): Ridge alpha values to search over on the
		validation set. Default: [1e-4, 1e-2, 1, 1e2, 1e4].
	seed (int): Random seed for reproducibility. Default: 42.
"""

import argparse
import json
import os

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from seq_models.linear.data import create_kmer_data


def get_args():
	parser = argparse.ArgumentParser(
		description="Train and evaluate linear baseline for STR length."
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
	log_transform = config.get("log_transform", True)
	alphas = [
		float(a) for a in config.get("alphas", [1e-4, 1e-2, 1.0, 1e2, 1e4])
	]
	seed = config.get("seed", 42)

	# ------------------------------------------------------------------
	# Set up output directory
	# ------------------------------------------------------------------

	experiment_name = config["experiment_name"]
	experiment_path = os.path.join(args.output_dir, experiment_name)
	os.makedirs(experiment_path, exist_ok=True)

	# Save config copy
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
	)

	X_train = splits["train"]["X"]
	X_val = splits["val"]["X"]
	X_test = splits["test"]["X"]

	if log_transform:
		# create_kmer_data already returns log1p(copy_number) as y
		y_train = splits["train"]["y"]
		y_val = splits["val"]["y"]
		y_test = splits["test"]["y"]
		print("Training in log1p(copy_number) space.")
	else:
		# Use raw copy number
		y_train = splits["train"]["metadata"]["copy_number"].values
		y_val = splits["val"]["metadata"]["copy_number"].values
		y_test = splits["test"]["metadata"]["copy_number"].values
		print("Training in raw copy-number space.")

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
	# Alpha selection on validation set
	# ------------------------------------------------------------------

	print(f"\nSearching over {len(alphas)} alpha values...")

	best_alpha = None
	best_val_mse = np.inf
	val_results = []

	for alpha in tqdm(alphas, desc="Alpha search"):
		model = Ridge(alpha=alpha, random_state=seed)

		tqdm.write(f"  alpha={alpha:<10g}  fitting on {X_train.shape[0]} samples...")
		model.fit(X_train, y_train)

		val_preds = model.predict(X_val)
		val_mse = np.mean((val_preds - y_val) ** 2)

		val_results.append({"alpha": alpha, "val_mse": float(val_mse)})
		tqdm.write(f"  alpha={alpha:<10g}  val_mse={val_mse:.6f}")

		if val_mse < best_val_mse:
			best_val_mse = val_mse
			best_alpha = alpha

	print(f"\nBest alpha: {best_alpha}  (val MSE: {best_val_mse:.6f})")

	# ------------------------------------------------------------------
	# Refit with best alpha and predict on test
	# ------------------------------------------------------------------

	print(f"\nRefitting with best alpha={best_alpha} on training set...")
	model = Ridge(alpha=best_alpha, random_state=seed)
	model.fit(X_train, y_train)

	print(f"Predicting on {X_test.shape[0]} test samples...")
	test_preds = model.predict(X_test)

	# ------------------------------------------------------------------
	# Convert predictions to copy-number space for output
	# ------------------------------------------------------------------

	if log_transform:
		# Model predicted in log1p space -> expm1 to get copy number
		test_pred_cn = np.expm1(test_preds)
	else:
		test_pred_cn = test_preds

	true_cn = splits["test"]["metadata"]["copy_number"].values

	# ------------------------------------------------------------------
	# Save predictions (matches HyenaDNA predict.py output format)
	# ------------------------------------------------------------------

	output_df = splits["test"]["metadata"].copy()
	output_df["pred_length"] = test_pred_cn
	output_df["true_length"] = true_cn

	# Sanity check
	if not np.allclose(output_df["true_length"], output_df["copy_number"]):
		raise ValueError("Mismatch between true_length and copy_number.")

	pred_path = os.path.join(experiment_path, "predictions_test.tsv")
	output_df.to_csv(pred_path, sep="\t", index=False)
	print(f"\nSaved {len(output_df)} test predictions to {pred_path}")

	# ------------------------------------------------------------------
	# Save run summary
	# ------------------------------------------------------------------

	summary = {
		"best_alpha": best_alpha,
		"best_val_mse": float(best_val_mse),
		"log_transform": log_transform,
		"n_flanking_bp": n_flanking_bp,
		"k_values": k_values,
		"n_features": len(feature_names),
		"n_train": X_train.shape[0],
		"n_val": X_val.shape[0],
		"n_test": X_test.shape[0],
		"alpha_search": val_results,
	}

	summary_path = os.path.join(experiment_path, "summary.json")
	with open(summary_path, "w") as f:
		json.dump(summary, f, indent=4)
	print(f"Saved run summary to {summary_path}")