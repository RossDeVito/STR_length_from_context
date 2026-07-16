""" Evaluate predictions.

Prints metrics for the mean baselines, the linear baselines, and the Caduceus
predictions (one table per target), then plots the best baseline next to the
best model side by side.

Current Caduceus / linear prediction TSVs (predictions_test.tsv, one row per
locus after RC-averaging) have columns:

	id  pred_length  pred_variation  true_length  true_variation
	chrom  str_start  str_end  motif  split

Models may predict only one of the two targets (length, variation); the script
detects which `pred_{task}`/`true_{task}` columns are present and adapts.
"""

import os
import json
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import scipy.stats as stats
from matplotlib.colors import LogNorm


# Multi-objective targets (output_name -> data column), mirroring the Caduceus
# config. Used for display ordering; the tasks actually present are detected
# per prediction file.
TARGETS = {
	"length": "mode_copy_number",
	"variation": "heterozygosity",
}


def get_metrics(pred, true):
	"""Regression metrics for a set of predictions (replicated from
	eval_multiple_preds.get_metrics so this script is self-contained)."""
	mse = sk_metrics.mean_squared_error(true, pred)
	mape = sk_metrics.mean_absolute_percentage_error(true, pred)
	mae = sk_metrics.mean_absolute_error(true, pred)
	r2 = sk_metrics.r2_score(true, pred)
	pearson_r, pearson_p = stats.pearsonr(true, pred)
	spearman_r, spearman_p = stats.spearmanr(true, pred)

	return {
		"MSE": mse,
		"MAE": mae,
		"MAPE": mape,
		"R2": r2,
		"Pearson_r": pearson_r,
		"Pearson_p": pearson_p,
		"Spearman_r": spearman_r,
		"Spearman_p": spearman_p,
	}


def detect_tasks(df):
	"""Return the targets for which both pred_{task} and true_{task} exist."""
	return [
		task for task in TARGETS
		if f"pred_{task}" in df.columns and f"true_{task}" in df.columns
	]


def force_square_limits(ax):
	"""
	Adjusts x and y limits to be the same span so that
	aspect='equal' results in a square plot.
	"""
	# Get current limits
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	# Calculate current spans
	x_span = xlim[1] - xlim[0]
	y_span = ylim[1] - ylim[0]

	# Determine the larger span to ensure nothing is cut off
	max_span = max(x_span, y_span)

	# Re-center x
	x_mid = (xlim[0] + xlim[1]) / 2
	ax.set_xlim(x_mid - max_span / 2, x_mid + max_span / 2)

	# Re-center y
	y_mid = (ylim[0] + ylim[1]) / 2
	ax.set_ylim(y_mid - max_span / 2, y_mid + max_span / 2)

	# Now set aspect to equal
	ax.set_aspect('equal', adjustable='box')


def plot_density_scatter(ax, df, true_col, pred_col, title, lims=None,
						 norm=None, add_colorbar=True):
	"""Plot 1: 2D Histogram/Density Plot.

	If ``lims`` (mn, mx) is given, use it for both axes instead of computing
	limits from this df (lets side-by-side panels share x/y limits).

	If ``norm`` is given it is used for the color mapping (lets side-by-side
	panels share a common color scale); otherwise a per-panel ``LogNorm`` is
	used. Set ``add_colorbar=False`` to skip this panel's own colorbar (e.g.
	when a single shared colorbar is drawn for both panels). Returns the
	``QuadMesh`` mappable so callers can build a shared colorbar."""

	if lims is not None:
		mn, mx = lims
	else:
		# 1. Calculate square limits first
		mn = min(df[true_col].min(), df[pred_col].min())
		mx = max(df[true_col].max(), df[pred_col].max())

		# Add buffer
		buff = (mx - mn) * 0.05
		mn -= buff
		mx += buff

	# 2. Pass 'range' to hist2d.
	# This forces the bins AND the plot limits to be this exact square.
	h = ax.hist2d(
		df[true_col],
		df[pred_col],
		bins=50,
		cmap='Blues',
		cmin=1,
		norm=norm if norm is not None else LogNorm(),
		range=[[mn, mx], [mn, mx]]  # <--- CRITICAL FIX
	)

	# 3. Add Colorbar without distorting the square
	# "fraction=0.046, pad=0.04" aligns the colorbar height with a square plot
	if add_colorbar:
		plt.colorbar(h[3], ax=ax, label='Count', fraction=0.046, pad=0.04)

	# 4. Identity line
	ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.75, zorder=10, label='Perfect Prediction')

	ax.set_title(title)
	ax.set_xlabel('True')
	ax.set_ylabel('Predicted')
	ax.legend()
	ax.grid(True, alpha=0.3)

	# 5. Force square aspect
	ax.set_xlim(mn, mx) # Ensure limits stick
	ax.set_ylim(mn, mx)
	ax.set_aspect('equal', adjustable='box')

	return h[3]


def plot_box_by_length(ax, df, true_col, pred_col):
	"""Plot 2: Boxplot grouped by True value"""
	sns.boxplot(x=true_col, y=pred_col, data=df, ax=ax,
				color='lightblue', showfliers=False) # Hide outliers to see the median better

	# Add identity line approximation (diagonal across boxes)
	# Since x-axis is categorical 0..N, we map true values to 0..N indices
	unique_lens = sorted(df[true_col].unique())
	ax.plot(range(len(unique_lens)), unique_lens, 'r--', lw=2, label='Target')

	# Clean up x-labels if too crowded
	if len(unique_lens) > 20:
		for ind, label in enumerate(ax.get_xticklabels()):
			if ind % 5 == 0:  # every 5th label is kept
				label.set_visible(True)
			else:
				label.set_visible(False)

	ax.set_title('Distribution of Predictions per True value')
	ax.set_ylabel('Predicted')

def plot_residuals(ax, df, true_col, pred_col):
	"""Plot 3: Residuals (Bias Analysis)"""
	residuals = df[pred_col] - df[true_col]

	sns.scatterplot(x=df[true_col], y=residuals, ax=ax, alpha=0.3)
	ax.axhline(0, color='r', linestyle='--')

	ax.set_title('Residuals (Pred - True)')
	ax.set_xlabel('True')
	ax.set_ylabel('Error')
	ax.grid(True, alpha=0.3)
	force_square_limits(ax)


def load_predictions(pred_dir):
	"""Load a model's per-locus predictions_test.tsv (or None if missing)."""
	pred_path = os.path.join(pred_dir, "predictions_test.tsv")
	if not os.path.exists(pred_path):
		print(f"  [skip] missing {pred_path}")
		return None
	return pd.read_csv(pred_path, sep="\t")


if __name__ == "__main__":

	# ------------------------------------------------------------------
	# Config
	# ------------------------------------------------------------------
	str_len = 3                 # which STR length to evaluate (1 or 2)
	plot_metric = "Spearman_r"  # metric used to pick the "best" baseline/model

	caduceus_pred_dir = f"predictions/caduceus/caduceus_v0/str{str_len}"
	# Default: evaluate every run subdir found. Replace with an explicit list
	# of run names to curate which Caduceus runs are evaluated.
	caduceus_model_names = sorted(
		d for d in os.listdir(caduceus_pred_dir)
		if os.path.isdir(os.path.join(caduceus_pred_dir, d))
	) if os.path.isdir(caduceus_pred_dir) else []

	lin_pred_dir = f"predictions/baseline/lin/str{str_len}"
	# Auto-discover all linear baseline models for this STR length.
	lin_model_names = sorted(
		d for d in os.listdir(lin_pred_dir)
		if os.path.isdir(os.path.join(lin_pred_dir, d))
	) if os.path.isdir(lin_pred_dir) else []

	mean_perf_path = "predictions/baseline/mean_performance.json"

	# ------------------------------------------------------------------
	# Build a unified results table: one row per (model, model_type, target).
	# Per-locus prediction frames are kept for plotting (mean baselines have
	# no per-locus predictions, only JSON metrics).
	# ------------------------------------------------------------------
	results = []
	pred_frames = {}  # (model_type, model_name) -> DataFrame

	# Mean baselines (metrics-only, from JSON)
	with open(mean_perf_path, "r") as f:
		mean_perf = json.load(f)

	mean_perf_str = mean_perf.get(str(str_len), {})
	for target, by_type in mean_perf_str.items():
		for baseline_type, metrics in by_type.items():  # overall_mean / motif_mean
			results.append({
				"model": baseline_type,
				"model_type": "mean",
				"target": target,
				**metrics,
			})

	# Linear baselines
	for model_name in lin_model_names:
		pred_df = load_predictions(os.path.join(lin_pred_dir, model_name))
		if pred_df is None:
			continue
		pred_frames[("linear", model_name)] = pred_df
		for task in detect_tasks(pred_df):
			results.append({
				"model": model_name,
				"model_type": "linear",
				"target": task,
				**get_metrics(pred=pred_df[f"pred_{task}"], true=pred_df[f"true_{task}"]),
			})

	# Caduceus predictions. Distinguish single- vs multi-objective runs by how
	# many targets the run actually predicts.
	for model_name in caduceus_model_names:
		pred_df = load_predictions(os.path.join(caduceus_pred_dir, model_name))
		if pred_df is None:
			continue
		pred_frames[("caduceus", model_name)] = pred_df
		tasks = detect_tasks(pred_df)
		model_type = "caduceus_multi" if len(tasks) > 1 else "caduceus_single"
		for task in tasks:
			results.append({
				"model": model_name,
				"model_type": model_type,
				"target": task,
				**get_metrics(pred=pred_df[f"pred_{task}"], true=pred_df[f"true_{task}"]),
			})

	results_df = pd.DataFrame(results)

	# ------------------------------------------------------------------
	# Print metrics, one table per target
	# ------------------------------------------------------------------
	display_cols = [
		"model", "model_type",
		"MSE", "MAE", "MAPE", "R2",
		"Pearson_r", "Pearson_p",
		"Spearman_r", "Spearman_p",
	]
	# Sort each target's table best-first by the chosen metric. Lower is better
	# for error metrics, higher is better for everything else.
	metric_ascending = plot_metric in ("MSE", "MAE", "MAPE")

	print(f"\n===== STR length {str_len} results =====")
	for target in TARGETS:
		target_df = results_df[results_df["target"] == target]
		if len(target_df) == 0:
			continue
		target_df = target_df.sort_values(
			by=plot_metric, ascending=metric_ascending
		)
		print(f"\n--- Target: {target} ---")
		print(target_df[display_cols].to_string(index=False, max_colwidth=40))

	# ------------------------------------------------------------------
	# Plot: best baseline vs best model, side by side, per target.
	# Best model  = Caduceus run with max plot_metric for that target.
	# Best baseline = best *linear* run (only linear baselines have per-locus
	# predictions; mean baselines are constant/by-motif and reported above).
	# ------------------------------------------------------------------
	sns.set_theme(style="whitegrid")

	for target in TARGETS:
		true_col, pred_col = f"true_{target}", f"pred_{target}"

		def best_of(model_types):
			sub = results_df[
				(results_df["target"] == target)
				& (results_df["model_type"].isin(model_types))
			]
			if len(sub) == 0:
				return None
			best = sub.loc[sub[plot_metric].idxmax()]
			return best["model"], best[plot_metric]

		best_lin = best_of(["linear"])
		best_cad = best_of(["caduceus_single", "caduceus_multi"])

		if best_lin is None or best_cad is None:
			print(
				f"\n[plot skipped for '{target}']: need both a linear baseline "
				f"and a Caduceus model with this target."
			)
			continue

		lin_name, lin_score = best_lin
		cad_name, cad_score = best_cad
		lin_df = pred_frames[("linear", lin_name)]
		cad_df = pred_frames[("caduceus", cad_name)]

		# Shared square limits across both panels.
		mn = min(
			lin_df[true_col].min(), lin_df[pred_col].min(),
			cad_df[true_col].min(), cad_df[pred_col].min(),
		)
		mx = max(
			lin_df[true_col].max(), lin_df[pred_col].max(),
			cad_df[true_col].max(), cad_df[pred_col].max(),
		)
		buff = (mx - mn) * 0.05
		lims = (mn - buff, mx + buff)

		# Shared color scale across both panels: use the same LogNorm with a
		# common vmax = max bin count over both histograms (computed with the
		# same bins/range used for plotting).
		hist_range = [[lims[0], lims[1]], [lims[0], lims[1]]]
		vmax = max(
			np.histogram2d(lin_df[true_col], lin_df[pred_col],
						   bins=50, range=hist_range)[0].max(),
			np.histogram2d(cad_df[true_col], cad_df[pred_col],
						   bins=50, range=hist_range)[0].max(),
		)
		shared_norm = LogNorm(vmin=1, vmax=vmax)

		fig, axes = plt.subplots(1, 2, figsize=(14, 6))
		plot_density_scatter(
			axes[0], lin_df, true_col, pred_col,
			f"Best baseline: {lin_name}\n{plot_metric}={lin_score:.3f}",
			lims=lims, norm=shared_norm, add_colorbar=False,
		)
		mappable = plot_density_scatter(
			axes[1], cad_df, true_col, pred_col,
			f"Best model: {cad_name}\n{plot_metric}={cad_score:.3f}",
			lims=lims, norm=shared_norm, add_colorbar=False,
		)
		fig.suptitle(f"STR length {str_len} | target: {target}", fontsize=14)
		plt.tight_layout()
		# Single shared colorbar spanning both panels.
		fig.colorbar(mappable, ax=axes, label='Count', fraction=0.046, pad=0.04)
		plt.show()
