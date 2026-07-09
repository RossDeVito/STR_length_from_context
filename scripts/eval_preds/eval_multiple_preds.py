""" Compare the best model at each STR length, across STR lengths.

For a configurable list of STR lengths (``str_lens``), this picks the best
Caduceus model and the best linear baseline at each length and plots how the
chosen metrics change with repeat-unit length. Mean baselines (overall / by
motif), where available in ``predictions/baseline/mean_performance.json``, are
drawn as dashed reference lines.

Two selection knobs let you curate the comparison:

  * ``model_overrides`` -- pin a specific Caduceus run per length, replacing the
	auto best-by-metric pick. Handy to compare one config family across lengths
	(e.g. all the ``f5000_fiv_v4`` runs) rather than the per-length best.
  * ``baseline_type`` -- pin a single linear-baseline variant (pooled / sep /
	pooled_dist / sep_dist) for every length instead of the per-length best.

Prediction TSVs (predictions_test.tsv, one row per locus after RC-averaging)
have columns:

	id  pred_length  pred_variation  true_length  true_variation
	chrom  str_start  str_end  motif  split

Models may predict only one of the two targets (length, variation); the tasks
present are detected per prediction file.
"""

import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import scipy.stats as stats


# Multi-objective targets (output_name -> data column), mirroring the Caduceus
# config. Used for display ordering; the tasks actually present are detected
# per prediction file.
TARGETS = {
	"length": "mode_copy_number",
	"variation": "heterozygosity",
}

# Error metrics: lower is better (ascending sort / idxmin). Everything else:
# higher is better.
ERROR_METRICS = ("MSE", "MAE", "MAPE")


def get_metrics(pred, true):
	"""Regression metrics for a set of predictions (replicated from
	eval_preds.get_metrics so this script is self-contained)."""
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


def load_predictions(pred_dir):
	"""Load a model's per-locus predictions_test.tsv (or None if missing)."""
	pred_path = os.path.join(pred_dir, "predictions_test.tsv")
	if not os.path.exists(pred_path):
		print(f"  [skip] missing {pred_path}")
		return None
	return pd.read_csv(pred_path, sep="\t")


def list_run_dirs(parent):
	"""Sorted run subdirectory names under ``parent`` (empty if not a dir)."""
	if not os.path.isdir(parent):
		return []
	return sorted(
		d for d in os.listdir(parent)
		if os.path.isdir(os.path.join(parent, d))
	)


def best_by_metric(rows, metric):
	"""Pick the best of ``rows`` (list of metric dicts) by ``metric``. Lower is
	better for error metrics, higher otherwise. Rows with a NaN in ``metric``
	are ignored. Returns the winning row, or None if none are comparable."""
	comparable = [r for r in rows if not pd.isna(r.get(metric, np.nan))]
	if not comparable:
		return None
	key = lambda r: r[metric]
	return (min if metric in ERROR_METRICS else max)(comparable, key=key)


if __name__ == "__main__":

	# ------------------------------------------------------------------
	# Config
	# ------------------------------------------------------------------
	str_lens = [1, 2, 3, 4, 5, 6]

	plot_metrics = ["Spearman_r", "R2"]  # one figure per metric per target
	select_metric = "Spearman_r"         # metric used to auto-pick the best run

	# --- Model (Caduceus) selection ---
	caduceus_root = "predictions/caduceus/caduceus_v0"
	# Optionally pin specific Caduceus runs per length (REPLACES auto-best-by-
	# metric). Lets you compare one config family across lengths, e.g. the
	# f5000_fiv_v4 runs:
	# model_overrides = {
	#     1: "str1_f5000_fiv_v4_2026-06-26_17-20-55",
	#     2: "str2_f5000_fiv_v4_2026-06-26_17-20-42",
	#     4: "str4_f5000_fiv_v4_2026-06-26_17-20-49",
	# }
	model_overrides = {}

	# --- Linear baseline selection ---
	lin_root = "predictions/baseline/lin"
	# None => best linear variant per length by select_metric. Or pin ONE
	# variant for every length: "pooled" | "sep" | "pooled_dist" | "sep_dist".
	baseline_type = None

	mean_perf_path = "predictions/baseline/mean_performance.json"

	# ------------------------------------------------------------------
	# Build a unified results table: one row per (str_len, target, series).
	# series in {"model", "linear", "overall_mean", "motif_mean"}.
	# ------------------------------------------------------------------
	results = []

	# Mean baselines (metrics-only, from JSON).
	with open(mean_perf_path, "r") as f:
		mean_perf = json.load(f)

	for str_len in str_lens:

		# --- Caduceus model: pin override or auto-best per target ---
		cad_dir = os.path.join(caduceus_root, f"str{str_len}")
		if str_len in model_overrides:
			cad_runs = [model_overrides[str_len]]
		else:
			cad_runs = list_run_dirs(cad_dir)

		cad_rows = []  # per (run, target) metric dicts
		for run in cad_runs:
			pred_df = load_predictions(os.path.join(cad_dir, run))
			if pred_df is None:
				continue
			for task in detect_tasks(pred_df):
				cad_rows.append({
					"target": task,
					"model": run,
					**get_metrics(pred=pred_df[f"pred_{task}"], true=pred_df[f"true_{task}"]),
				})

		# --- Linear baseline: pin variant or auto-best per target ---
		lin_dir = os.path.join(lin_root, f"str{str_len}")
		if baseline_type is not None:
			lin_runs = [f"str{str_len}_f5000_{baseline_type}"]
		else:
			lin_runs = list_run_dirs(lin_dir)

		lin_rows = []
		for run in lin_runs:
			pred_df = load_predictions(os.path.join(lin_dir, run))
			if pred_df is None:
				continue
			for task in detect_tasks(pred_df):
				lin_rows.append({
					"target": task,
					"model": run,
					**get_metrics(pred=pred_df[f"pred_{task}"], true=pred_df[f"true_{task}"]),
				})

		# Keep the best run per target for each series.
		for target in TARGETS:
			best_cad = best_by_metric(
				[r for r in cad_rows if r["target"] == target], select_metric
			)
			if best_cad is not None:
				results.append({"str_len": str_len, "series": "model", **best_cad})

			best_lin = best_by_metric(
				[r for r in lin_rows if r["target"] == target], select_metric
			)
			if best_lin is not None:
				results.append({"str_len": str_len, "series": "linear", **best_lin})

		# --- Mean baselines from JSON (available for a subset of lengths) ---
		mean_perf_len = mean_perf.get(str(str_len), {})
		for target, by_type in mean_perf_len.items():
			for baseline_type_name, metrics in by_type.items():  # overall_mean / motif_mean
				results.append({
					"str_len": str_len,
					"series": baseline_type_name,
					"target": target,
					"model": baseline_type_name,
					**metrics,
				})

	results_df = pd.DataFrame(results)

	# ------------------------------------------------------------------
	# Print metrics, one table per target (rows sorted by STR length).
	# ------------------------------------------------------------------
	display_cols = [
		"str_len", "series", "model",
		"MSE", "MAE", "MAPE", "R2",
		"Pearson_r", "Pearson_p",
		"Spearman_r", "Spearman_p",
	]
	series_order = {"model": 0, "linear": 1, "overall_mean": 2, "motif_mean": 3}

	for target in TARGETS:
		target_df = results_df[results_df["target"] == target].copy()
		if len(target_df) == 0:
			continue
		target_df["_series_ord"] = target_df["series"].map(series_order)
		target_df = target_df.sort_values(by=["str_len", "_series_ord"])
		print(f"\n===== Target: {target} =====")
		print(target_df[display_cols].to_string(index=False, max_colwidth=40))

	# ------------------------------------------------------------------
	# Plot: metric vs STR length, one figure per (target, metric). Lines for
	# the best model and best linear baseline; dashed reference lines for the
	# mean baselines where available.
	# ------------------------------------------------------------------
	sns.set_theme(style="whitegrid")

	model_label = "Caduceus (pinned)" if model_overrides else "Caduceus (best)"
	linear_label = (
		f"Linear ({baseline_type})" if baseline_type is not None else "Linear (best)"
	)
	series_style = {
		"model": dict(label=model_label, marker="o", linestyle="-", color="C0"),
		"linear": dict(label=linear_label, marker="s", linestyle="-", color="C1"),
		"overall_mean": dict(label="Overall Mean", marker="x", linestyle="--", color="red"),
		"motif_mean": dict(label="STR Motif Mean", marker="^", linestyle="--", color="orange"),
	}

	for target in TARGETS:
		target_df = results_df[results_df["target"] == target]
		if len(target_df) == 0:
			continue

		for metric in plot_metrics:
			plt.figure(figsize=(8, 6))
			plotted_any = False

			for series, style in series_style.items():
				sub = target_df[target_df["series"] == series]
				# Align to str_lens ordering and drop lengths with no / NaN value.
				sub = sub[sub["str_len"].isin(str_lens)].sort_values("str_len")
				xs = sub["str_len"].to_numpy()
				ys = sub[metric].to_numpy(dtype=float)
				mask = ~np.isnan(ys)
				if not mask.any():
					continue
				plt.plot(xs[mask], ys[mask], **style, zorder=5)
				plotted_any = True

			if not plotted_any:
				plt.close()
				continue

			plt.title(f"{target.capitalize()} prediction performance across STR lengths")
			plt.xlabel("STR motif length (bp)")
			plt.ylabel(metric)
			plt.xticks(str_lens)
			plt.grid(True)
			plt.legend(loc="best")
			plt.tight_layout()
			plt.show()
