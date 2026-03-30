""" Analyze region permutation importance results.

Computes local (per-locus) and global (dataset-level) effect metrics for
each flanking region, with bootstrap confidence intervals.  Produces a
summary DataFrame and spatial profile plots.

Local metric:
	Mean |percent change| per region, computed in copy-number space:
		pct_change = (cn_shuffled - cn_baseline) / cn_baseline * 100
	Bootstrap CIs are computed by resampling loci.

Global metrics:
	For each region, the baseline dataset-level metrics (MSE, MAE, MAPE,
	R2, Pearson r, Spearman r) are compared to the mean metrics across
	S permutations.  The "delta" for each metric is:
		baseline_metric - mean_shuffled_metric
	so a positive delta means shuffling that region *hurt* performance.
	All global metrics are computed in copy-number space.

Outputs:
	- region_df DataFrame
	- Two-panel spatial profile figure (local + global)
	- Printed summary tables
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sk_metrics
from scipy import stats
from tqdm import trange


# ===========================================================================
# Configuration
# ===========================================================================

RUN_DIR = "output/str2/dev/dev1"

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
SEED = 42

SAVE_DIR = None		# Set to a path to save figures, or None to plt.show()


# ===========================================================================
# Loading
# ===========================================================================

def load_results(run_dir):
	"""Load permutation importance results and metadata.

	Args:
		run_dir: Path to the output directory of a permutation run.

	Returns:
		Tuple of (data_dict, meta_dict) where data_dict contains the
		numpy arrays from the npz file.
	"""
	data = np.load(
		os.path.join(run_dir, "permutation_results.npz"),
		allow_pickle=True,
	)

	with open(os.path.join(run_dir, "meta.json"), "r") as f:
		meta = json.load(f)

	return data, meta


# ===========================================================================
# Shared metric computation
# ===========================================================================

def get_metrics(pred, true):
	"""Compute regression metrics in copy-number space.

	Matches the metric suite used in eval_multiple_preds.py.

	Args:
		pred: (N,) array of predictions.
		true: (N,) array of true values.

	Returns:
		Dict of metric name -> value.
	"""
	return {
		"MSE": sk_metrics.mean_squared_error(true, pred),
		"MAE": sk_metrics.mean_absolute_error(true, pred),
		"MAPE": sk_metrics.mean_absolute_percentage_error(true, pred),
		"R2": sk_metrics.r2_score(true, pred),
		"Pearson_r": stats.pearsonr(true, pred).statistic,
		"Spearman_r": stats.spearmanr(true, pred).statistic,
	}


# ===========================================================================
# Local metric: per-region mean |percent change| with bootstrap CI
# ===========================================================================

def compute_local_metrics(
	baseline_preds,
	permuted_preds,
	log_transform,
	n_bootstrap,
	ci_level,
	rng,
):
	"""Compute per-region mean |percent change| with bootstrap CIs.

	Percent change is computed in copy-number space:
		pct_change = (cn_shuffled - cn_baseline) / cn_baseline * 100

	For each region, the per-locus percent change is averaged across
	permutations first (to reduce shuffle noise), then absolute value
	is taken.  The bootstrap resamples loci to get a CI on the mean.

	Args:
		baseline_preds: (N,) array of unperturbed predictions.
		permuted_preds: (R, S, N) array of permuted predictions.
		log_transform: Whether predictions are in log1p space.
		n_bootstrap: Number of bootstrap iterations.
		ci_level: Confidence level (e.g. 0.95).
		rng: numpy Generator.

	Returns:
		Dict with keys:
			mean_abs_pct_change: (R,) mean |percent change| per region.
			ci_lower: (R,) lower CI bound.
			ci_upper: (R,) upper CI bound.
			per_locus_abs_pct_change: (R, N) per-locus |percent change|
				(useful for downstream exploration).
	"""
	n_regions, n_perms, n_loci = permuted_preds.shape

	# Convert to copy-number space
	if log_transform:
		bl_cn = np.expm1(baseline_preds)                # (N,)
		pp_cn = np.expm1(permuted_preds)                # (R, S, N)
	else:
		bl_cn = baseline_preds
		pp_cn = permuted_preds

	# Percent change per permutation per locus: (R, S, N)
	pct_change = (
		(pp_cn - bl_cn[np.newaxis, np.newaxis, :])
		/ bl_cn[np.newaxis, np.newaxis, :]
		* 100.0
	)

	# Average across permutations first, then take abs: (R, N)
	# Taking abs after averaging preserves information about systematic
	# shifts vs random noise.  Alternatively, |pct| per perm then average
	# would inflate near-zero regions.  Both are defensible; this choice
	# is more conservative.
	mean_pct_per_locus = pct_change.mean(axis=1)            # (R, N)
	abs_pct_per_locus = np.abs(mean_pct_per_locus)          # (R, N)

	# Point estimate
	mean_abs_pct = abs_pct_per_locus.mean(axis=1)  # (R,)

	# Bootstrap CI by resampling loci
	alpha = 1 - ci_level
	boot_means = np.empty((n_bootstrap, n_regions), dtype=np.float32)

	for b in trange(n_bootstrap, desc="Bootstrap (local)"):
		idx = rng.integers(0, n_loci, size=n_loci)
		boot_means[b] = abs_pct_per_locus[:, idx].mean(axis=1)

	ci_lower = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
	ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)

	return {
		"mean_abs_pct_change": mean_abs_pct,
		"ci_lower": ci_lower,
		"ci_upper": ci_upper,
		"per_locus_abs_pct_change": abs_pct_per_locus,
	}


# ===========================================================================
# Global metrics: per-region dataset-level metric changes
# ===========================================================================

def compute_global_metrics(
	baseline_preds,
	permuted_preds,
	true_labels,
	log_transform,
):
	"""Compute per-region dataset-level metric changes.

	All metrics are computed in copy-number space to match the project's
	standard evaluation pipeline.  For each region, each of the S
	permutations yields one set of metrics.  The reported delta is:
		delta_metric = baseline_metric - mean_shuffled_metric
	so a positive delta means shuffling hurt performance (for metrics where
	higher is better: R2, Pearson, Spearman) or improved it (for error
	metrics: MSE, MAE, MAPE -- but we negate these so that positive
	always means "this region matters").

	Args:
		baseline_preds: (N,) array of baseline predictions (log1p space
			if log_transform is True).
		permuted_preds: (R, S, N) array.
		true_labels: (N,) array of true copy numbers.
		log_transform: Whether predictions are in log1p space.

	Returns:
		Dict with keys:
			baseline_metrics: Dict of metric_name -> scalar.
			per_perm_metrics: Dict of metric_name -> (R, S) array.
			mean_shuffled_metrics: Dict of metric_name -> (R,) array.
			delta_metrics: Dict of metric_name -> (R,) array.
				For all metrics, positive = region matters.
	"""
	n_regions, n_perms, n_loci = permuted_preds.shape

	# Convert to copy-number space
	if log_transform:
		bl_cn = np.expm1(baseline_preds)
		pp_cn = np.expm1(permuted_preds)
	else:
		bl_cn = baseline_preds
		pp_cn = permuted_preds

	# Baseline metrics
	baseline_metrics = get_metrics(bl_cn, true_labels)

	# Per-permutation metrics
	metric_names = list(baseline_metrics.keys())
	per_perm_metrics = {
		name: np.empty((n_regions, n_perms), dtype=np.float64)
		for name in metric_names
	}

	for r in range(n_regions):
		for s in range(n_perms):
			m = get_metrics(pp_cn[r, s, :], true_labels)
			for name in metric_names:
				per_perm_metrics[name][r, s] = m[name]

	# Mean across permutations and delta
	mean_shuffled_metrics = {}
	delta_metrics = {}

	# For error metrics (lower is better), delta = shuffled - baseline
	# so that positive means "shuffling made it worse" (region matters).
	# For correlation/R2 metrics (higher is better), delta = baseline - shuffled.
	error_metrics = {"MSE", "MAE", "MAPE"}

	for name in metric_names:
		mean_shuffled_metrics[name] = per_perm_metrics[name].mean(axis=1)

		if name in error_metrics:
			delta_metrics[name] = (
				mean_shuffled_metrics[name] - baseline_metrics[name]
			)
		else:
			delta_metrics[name] = (
				baseline_metrics[name] - mean_shuffled_metrics[name]
			)

	return {
		"baseline_metrics": baseline_metrics,
		"per_perm_metrics": per_perm_metrics,
		"mean_shuffled_metrics": mean_shuffled_metrics,
		"delta_metrics": delta_metrics,
	}


# ===========================================================================
# Build results DataFrame
# ===========================================================================

def build_results_df(data, local_metrics, global_metrics):
	"""Combine region info with computed metrics into a DataFrame.

	Args:
		data: npz data dict.
		local_metrics: Output of compute_local_metrics().
		global_metrics: Output of compute_global_metrics().

	Returns:
		pd.DataFrame with one row per region.
	"""
	df = pd.DataFrame({
		"flank": data["region_flanks"],
		"region_idx": data["region_idxs"],
		"token_start": data["region_token_starts"],
		"token_end": data["region_token_ends"],
		"dist_bp_start": data["region_distance_bp_starts"],
		"dist_bp_end": data["region_distance_bp_ends"],
		"mean_abs_pct_change": local_metrics["mean_abs_pct_change"],
		"ci_lower": local_metrics["ci_lower"],
		"ci_upper": local_metrics["ci_upper"],
	})

	# Add all global delta metrics
	for name, values in global_metrics["delta_metrics"].items():
		df[f"delta_{name}"] = values

	# Add mean shuffled metrics for reference
	for name, values in global_metrics["mean_shuffled_metrics"].items():
		df[f"shuffled_{name}"] = values

	# Midpoint distance for plotting
	df["dist_bp_mid"] = (df["dist_bp_start"] + df["dist_bp_end"]) / 2

	return df


# ===========================================================================
# Plotting
# ===========================================================================

def plot_spatial_profiles(
	region_df,
	baseline_metrics,
	save_dir=None,
):
	"""Two-panel spatial profile: local |percent change| and global delta Spearman.

	Left and right flanks are shown on the same axes with different
	colors.  The x-axis is distance from the STR in bp.

	Args:
		region_df: DataFrame from build_results_df().
		baseline_metrics: Dict of baseline metric values.
		save_dir: If set, save figure to this directory.
	"""
	sns.set_theme(style="whitegrid")

	fig, (ax_local, ax_global) = plt.subplots(
		2, 1, figsize=(12, 8), sharex=True
	)

	flank_colors = {"left": "#3170ad", "right": "#e8443a"}
	flank_labels = {"left": "Upstream", "right": "Downstream"}

	# --- Panel 1: Local mean |percent change| ---
	for flank, color in flank_colors.items():
		sub = region_df[region_df["flank"] == flank].sort_values("dist_bp_mid")

		ax_local.plot(
			sub["dist_bp_mid"], sub["mean_abs_pct_change"],
			color=color, linewidth=1.0, marker="o", markersize=3,
			label=flank_labels[flank],
		)
		ax_local.fill_between(
			sub["dist_bp_mid"], sub["ci_lower"], sub["ci_upper"],
			alpha=0.25, color=color,
		)

	ax_local.set_ylabel("Mean |percent change| (%)")
	ax_local.set_title(
		f"Local effect: mean |percent change| per region "
		f"({int(CI_LEVEL * 100)}% bootstrap CI, "
		f"n_boot={N_BOOTSTRAP})"
	)
	ax_local.legend(loc="upper right")
	ax_local.set_ylim(bottom=0)

	# --- Panel 2: Global delta Spearman ---
	for flank, color in flank_colors.items():
		sub = region_df[region_df["flank"] == flank].sort_values("dist_bp_mid")

		ax_global.plot(
			sub["dist_bp_mid"], sub["delta_Spearman_r"],
			color=color, linewidth=1.0, marker="o", markersize=3,
			label=flank_labels[flank],
		)

	ax_global.axhline(0, color="gray", linewidth=0.5, linestyle="--")
	ax_global.set_ylabel("delta Spearman (baseline - shuffled)")
	ax_global.set_xlabel("Distance from STR (bp)")
	ax_global.set_title(
		f"Global effect: Spearman correlation drop per region "
		f"(baseline Spearman = "
		f"{baseline_metrics['Spearman_r']:.4f})"
	)
	ax_global.legend(loc="upper right")
	ax_global.set_ylim(bottom=0)

	plt.tight_layout()

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "spatial_profiles.png")
		fig.savefig(path, dpi=150, bbox_inches="tight")
		print(f"Saved figure to {path}")
	else:
		plt.show()

	return fig


def plot_global_metrics_grid(
	region_df,
	baseline_metrics,
	save_dir=None,
):
	"""Grid of spatial profiles for all global metrics.

	Args:
		region_df: DataFrame from build_results_df().
		baseline_metrics: Dict of baseline metric values.
		save_dir: If set, save figure to this directory.
	"""
	metric_names = list(baseline_metrics.keys())
	n_metrics = len(metric_names)

	sns.set_theme(style="whitegrid")
	fig, axes = plt.subplots(
		n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True
	)

	flank_colors = {"left": "#3170ad", "right": "#e8443a"}
	flank_labels = {"left": "Upstream", "right": "Downstream"}

	for ax, name in zip(axes, metric_names):
		col = f"delta_{name}"

		for flank, color in flank_colors.items():
			sub = region_df[
				region_df["flank"] == flank
			].sort_values("dist_bp_mid")

			ax.plot(
				sub["dist_bp_mid"], sub[col],
				color=color, linewidth=1.0, marker="o", markersize=3,
				label=flank_labels[flank],
			)

		ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
		ax.set_ylabel(f"delta {name}")
		ax.set_title(
			f"{name}  (baseline = {baseline_metrics[name]:.4f})"
		)
		ax.legend(loc="upper right", fontsize=8)

	axes[-1].set_xlabel("Distance from STR (bp)")

	plt.tight_layout()

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "global_metrics_grid.png")
		fig.savefig(path, dpi=150, bbox_inches="tight")
		print(f"Saved figure to {path}")
	else:
		plt.show()

	return fig


def plot_left_right_comparison(region_df, save_dir=None):
	"""Scatter plot comparing left vs right flank effect at matched distances.

	Only includes regions that exist in both flanks at the same distance.

	Args:
		region_df: DataFrame from build_results_df().
		save_dir: If set, save figure to this directory.
	"""
	left = region_df[region_df["flank"] == "left"].set_index("region_idx")
	right = region_df[region_df["flank"] == "right"].set_index("region_idx")

	common_idx = left.index.intersection(right.index)
	if len(common_idx) == 0:
		print("No matching region indices for left/right comparison.")
		return None

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

	# Local
	ax1.scatter(
		left.loc[common_idx, "mean_abs_pct_change"],
		right.loc[common_idx, "mean_abs_pct_change"],
		alpha=0.5, s=20, color="#3170ad",
	)
	for idx in common_idx:
		ax1.annotate(
			str(idx),
			(left.loc[idx, "mean_abs_pct_change"],
			 right.loc[idx, "mean_abs_pct_change"]),
			fontsize=5, alpha=0.7, ha="left", va="bottom",
			xytext=(2, 2), textcoords="offset points",
		)
	lim = max(
		left.loc[common_idx, "mean_abs_pct_change"].max(),
		right.loc[common_idx, "mean_abs_pct_change"].max(),
	) * 1.1
	ax1.plot([0, lim], [0, lim], "r--", alpha=0.5, linewidth=0.5)
	ax1.set_xlabel("Upstream mean |pct change| (%)")
	ax1.set_ylabel("Downstream mean |pct change| (%)")
	ax1.set_title("Local effect: upstream vs downstream")
	ax1.set_aspect("equal", adjustable="box")

	# Global (Spearman)
	ax2.scatter(
		left.loc[common_idx, "delta_Spearman_r"],
		right.loc[common_idx, "delta_Spearman_r"],
		alpha=0.5, s=20, color="#e8443a",
	)
	for idx in common_idx:
		ax2.annotate(
			str(idx),
			(left.loc[idx, "delta_Spearman_r"],
			 right.loc[idx, "delta_Spearman_r"]),
			fontsize=5, alpha=0.7, ha="left", va="bottom",
			xytext=(2, 2), textcoords="offset points",
		)
	lim = max(
		left.loc[common_idx, "delta_Spearman_r"].max(),
		right.loc[common_idx, "delta_Spearman_r"].max(),
	) * 1.1
	ax2.plot([0, lim], [0, lim], "r--", alpha=0.5, linewidth=0.5)
	ax2.set_xlabel("Upstream delta Spearman")
	ax2.set_ylabel("Downstream delta Spearman")
	ax2.set_title("Global effect: upstream vs downstream")
	ax2.set_aspect("equal", adjustable="box")

	plt.tight_layout()

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "left_right_comparison.png")
		fig.savefig(path, dpi=150, bbox_inches="tight")
		print(f"Saved figure to {path}")
	else:
		plt.show()

	return fig


# ===========================================================================
# Print summary
# ===========================================================================

def print_summary(region_df, global_metrics):
	"""Print summary tables to stdout."""

	baseline_metrics = global_metrics["baseline_metrics"]

	print("=" * 70)
	print("Baseline metrics (copy-number space)")
	print("=" * 70)
	for name, val in baseline_metrics.items():
		print(f"  {name:>12}: {val:.4f}")

	for flank in ["left", "right"]:
		flank_label = "Upstream" if flank == "left" else "Downstream"
		sub = region_df[region_df["flank"] == flank].sort_values("region_idx")

		print(f"\n{'=' * 70}")
		print(f"{flank_label} flank ({len(sub)} regions)")
		print("=" * 70)

		print(f"  Local mean |percent change|:")
		idx_max = sub["mean_abs_pct_change"].idxmax()
		print(f"    max:    {sub['mean_abs_pct_change'].max():.4f}%  "
		      f"(region {sub.loc[idx_max, 'region_idx']}, "
		      f"{sub.loc[idx_max, 'dist_bp_start']}-"
		      f"{sub.loc[idx_max, 'dist_bp_end']} bp)")
		print(f"    mean:   {sub['mean_abs_pct_change'].mean():.4f}%")
		print(f"    min:    {sub['mean_abs_pct_change'].min():.4f}%")

		print(f"  Global delta Spearman:")
		idx_max = sub["delta_Spearman_r"].idxmax()
		print(f"    max:    {sub['delta_Spearman_r'].max():.6f}  "
		      f"(region {sub.loc[idx_max, 'region_idx']}, "
		      f"{sub.loc[idx_max, 'dist_bp_start']}-"
		      f"{sub.loc[idx_max, 'dist_bp_end']} bp)")
		print(f"    mean:   {sub['delta_Spearman_r'].mean():.6f}")
		print(f"    min:    {sub['delta_Spearman_r'].min():.6f}")

	# Top 10 by local effect
	print(f"\n{'=' * 70}")
	print("Top 10 regions by local mean |percent change|")
	print("=" * 70)
	top = region_df.nlargest(10, "mean_abs_pct_change")
	for _, row in top.iterrows():
		print(f"  {row['flank']:>5} region {int(row['region_idx']):>3}  "
		      f"({int(row['dist_bp_start']):>5}-"
		      f"{int(row['dist_bp_end']):>5} bp)  "
		      f"mean|pct|={row['mean_abs_pct_change']:.4f}%  "
		      f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]  "
		      f"delta_Spearman={row['delta_Spearman_r']:.6f}")

	# Top 10 by global Spearman
	print(f"\n{'=' * 70}")
	print("Top 10 regions by global delta Spearman")
	print("=" * 70)
	top = region_df.nlargest(10, "delta_Spearman_r")
	for _, row in top.iterrows():
		print(f"  {row['flank']:>5} region {int(row['region_idx']):>3}  "
		      f"({int(row['dist_bp_start']):>5}-"
		      f"{int(row['dist_bp_end']):>5} bp)  "
		      f"delta_Spearman={row['delta_Spearman_r']:.6f}  "
		      f"mean|pct|={row['mean_abs_pct_change']:.4f}%")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

	# ------------------------------------------------------------------
	# Load
	# ------------------------------------------------------------------
	print(f"Loading results from {RUN_DIR}")
	data, meta = load_results(RUN_DIR)

	log_transform = meta["log_transform"]
	baseline_preds = data["baseline_predictions"]
	permuted_preds = data["permuted_predictions"]
	true_labels = data["true_labels"]

	n_regions, n_perms, n_loci = permuted_preds.shape
	print(f"  {n_loci} loci, {n_regions} regions, "
	      f"{n_perms} permutations per region")
	print(f"  log_transform: {log_transform}")

	# ------------------------------------------------------------------
	# Compute metrics
	# ------------------------------------------------------------------
	rng = np.random.default_rng(SEED)

	print("\nComputing local metrics ...")
	local_metrics = compute_local_metrics(
		baseline_preds, permuted_preds, log_transform,
		n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, rng=rng,
	)

	print("Computing global metrics ...")
	global_metrics = compute_global_metrics(
		baseline_preds, permuted_preds, true_labels, log_transform,
	)

	# ------------------------------------------------------------------
	# Build results DataFrame
	# ------------------------------------------------------------------
	region_df = build_results_df(data, local_metrics, global_metrics)

	# ------------------------------------------------------------------
	# Print summary
	# ------------------------------------------------------------------
	print_summary(region_df, global_metrics)

	# ------------------------------------------------------------------
	# Plot
	# ------------------------------------------------------------------
	fig_spatial = plot_spatial_profiles(
		region_df, global_metrics["baseline_metrics"],
		save_dir=SAVE_DIR,
	)

	fig_metrics_grid = plot_global_metrics_grid(
		region_df, global_metrics["baseline_metrics"],
		save_dir=SAVE_DIR,
	)

	fig_lr = plot_left_right_comparison(region_df, save_dir=SAVE_DIR)