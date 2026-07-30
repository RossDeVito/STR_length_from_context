""" Compare model types head-to-head at a single STR length, with bootstrap CIs.

For one STR length (``str_len``), loads one pinned run per model type (linear
k-mer baseline, linear GC baseline, Caduceus multi-task under fixed-inverse-
variance and learned-uncertainty loss weighting, Caduceus single-task, plus
the two mean baselines from predictions/baseline/mean_performance.json) and
plots them side by side as a bar chart with percentile-bootstrap confidence
intervals -- one figure per (target, metric). Unlike eval_multiple_preds.py
(best model per length, across lengths) or eval_by_flank_size.py (one model
family, across flank sizes), STR length and flank size are both fixed here;
only model *type* varies.

Prediction TSVs (predictions_test.tsv, one row per locus after RC-averaging)
have columns:

	id  pred_length  pred_variation  true_length  true_variation
	chrom  str_start  str_end  motif  split

Models may predict only one of the two targets (length, variation); the tasks
present are detected per prediction file -- the two single-task Caduceus runs
(length-only, variation-only) each only ever contribute to one target's plot.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sklearn.metrics as sk_metrics


# Multi-objective targets (output_name -> data column), mirroring the Caduceus
# config. Used for display ordering; the tasks actually present are detected
# per prediction file.
TARGETS = {
	"length": "mode_copy_number",
	"variation": "heterozygosity",
}

# Floor on the bootstrap resample size: scipy's pearsonr/spearmanr raise below
# n=2 and are degenerate (|r|=1) at n=2, so keep replicates large enough to be
# statistically meaningful.
MIN_RESAMPLE_N = 10


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


def bootstrap_ci(pred, true, n_boot, ci_level, rng, resample_frac=1.0):
	"""Percentile bootstrap CI (resampling loci with replacement) for every
	metric in get_metrics. resample_frac=1.0 (default) matches the standard
	full-size bootstrap in eval_by_flank_size.py; other values draw
	round(resample_frac * n) loci per replicate (with replacement) as a
	sensitivity knob -- a heuristic, not a calibrated m-out-of-n correction --
	floored at MIN_RESAMPLE_N with a printed warning if clamped. Returns
	{"{metric}_lo": v, "{metric}_hi": v, ...}."""
	pred, true = np.asarray(pred), np.asarray(true)
	n = len(pred)
	requested_n = round(resample_frac * n)
	if requested_n < MIN_RESAMPLE_N:
		print(f"  [warn] resample_frac={resample_frac} implies resample_n="
			  f"{requested_n} (< {MIN_RESAMPLE_N}); clamping to {MIN_RESAMPLE_N}")
	resample_n = max(MIN_RESAMPLE_N, requested_n)
	boot_df = pd.DataFrame([
		get_metrics(pred[idx], true[idx])
		for idx in (rng.integers(0, n, size=resample_n) for _ in range(n_boot))
	])
	alpha = 1 - ci_level
	lo, hi = boot_df.quantile(alpha / 2), boot_df.quantile(1 - alpha / 2)
	return {
		**{f"{m}_lo": lo[m] for m in boot_df.columns},
		**{f"{m}_hi": hi[m] for m in boot_df.columns},
	}


if __name__ == "__main__":

	# ------------------------------------------------------------------
	# Config
	# ------------------------------------------------------------------
	str_len = 4  # which STR length to evaluate; must be a key of MODEL_CONFIGS

	plot_metrics = ["Spearman_r", "R2"]  # one bar-chart figure per metric per target

	n_boot = 1000
	resample_frac = 1.0   # fraction of n loci drawn (w/ replacement) per bootstrap
						   # replicate; 1.0 = standard bootstrap (see bootstrap_ci docstring)
	ci_level = 0.95
	bootstrap_seed = 42
	rng = np.random.default_rng(bootstrap_seed)

	caduceus_root = "predictions/caduceus/caduceus_v0"
	lin_root = "predictions/baseline/lin"
	mean_perf_path = "predictions/baseline/mean_performance.json"

	# Per-str-length model selection: config key -> run directory name. Add
	# more lengths here as more baselines/models become available.
	MODEL_CONFIGS = {
		3: {
			"linear_kmer": "str3_f5000_sep",
			"linear_gc": "str3_f5000_gc_100_pooled",
			"caduceus": "str3_f5000_fiv_v4_2026-07-06_16-11-21",
			"caduceus_uncertainty": "str3_f5000_v4_2026-07-10_14-44-59",
			"caduceus_len_only": "str3_f5000_len_v4_2026-07-17_18-17-08",
			"caduceus_var_only": "str3_f5000_var_v4_2026-07-17_08-38-43",
		},
		4: {
			"linear_kmer": "str4_f5000_sep_dist",
			"linear_gc": "str4_f5000_gc_100_pooled",
			"caduceus": "str4_f5000_fiv_v4_2026-06-26_17-20-49",
			"caduceus_uncertainty": "str4_f5000_v4_2026-07-10_14-32-42",
			"caduceus_len_only": "str4_f5000_len_v4_2026-07-16_00-13-06",
			"caduceus_var_only": "str4_f5000_var_v4_2026-07-16_13-21-38",
		},
	}

	if str_len not in MODEL_CONFIGS:
		raise SystemExit(f"str_len={str_len} has no entry in MODEL_CONFIGS "
						  f"(available: {sorted(MODEL_CONFIGS)})")
	model_paths = MODEL_CONFIGS[str_len]

	# config key -> display series key. The two single-task Caduceus runs
	# share one series/style since they never both appear in the same
	# target's plot (length-only only predicts length, variation-only only
	# predicts variation).
	SERIES_SOURCE_KEY = {
		"linear_kmer": "linear_kmer",
		"linear_gc": "linear_gc",
		"caduceus": "caduceus",
		"caduceus_uncertainty": "caduceus_uncertainty",
		"caduceus_len_only": "caduceus_single_task",
		"caduceus_var_only": "caduceus_single_task",
	}
	# Which root each config key's run directory lives under.
	SERIES_ROOT = {
		"linear_kmer": lin_root,
		"linear_gc": lin_root,
		"caduceus": caduceus_root,
		"caduceus_uncertainty": caduceus_root,
		"caduceus_len_only": caduceus_root,
		"caduceus_var_only": caduceus_root,
	}

	# Left-to-right bar order (and printed-table row order) follows this dict's
	# insertion order.
	SERIES_STYLE = {
		"overall_mean":         dict(label="Overall Mean", color="0.5"),
		"motif_mean":           dict(label="STR Motif Mean", color="0.7"),
		"linear_gc":            dict(label="Linear (GC)", color="C2"),
		"linear_kmer":          dict(label="Linear (k-mer)", color="C1"),
		"caduceus_single_task": dict(label="Caduceus (single-task)", color="C3"),
		"caduceus":             dict(label="Caduceus", color="C0"),
		"caduceus_uncertainty": dict(label="Caduceus (uncertainty weighting)", color="C4"),
	}

	# ------------------------------------------------------------------
	# Build results: one row per (series, target).
	# ------------------------------------------------------------------
	results = []

	for config_key, run_name in model_paths.items():
		run_dir = os.path.join(SERIES_ROOT[config_key], f"str{str_len}", run_name)
		pred_df = load_predictions(run_dir)
		if pred_df is None:
			continue
		for task in detect_tasks(pred_df):
			pred_arr = pred_df[f"pred_{task}"].to_numpy()
			true_arr = pred_df[f"true_{task}"].to_numpy()
			row = {
				"series": SERIES_SOURCE_KEY[config_key],
				"target": task,
				"model": run_name,
				**get_metrics(pred=pred_arr, true=true_arr),
			}
			if n_boot > 0:
				row.update(bootstrap_ci(
					pred_arr, true_arr, n_boot, ci_level, rng,
					resample_frac=resample_frac,
				))
			results.append(row)

	# Mean baselines (no per-locus predictions available -> no bootstrap CI).
	with open(mean_perf_path, "r") as f:
		mean_perf = json.load(f)
	mean_perf_len = mean_perf.get(str(str_len), {})
	for target, by_type in mean_perf_len.items():
		for baseline_name, metrics in by_type.items():  # overall_mean / motif_mean
			results.append({
				"series": baseline_name,
				"target": target,
				"model": baseline_name,
				**metrics,
			})

	results_df = pd.DataFrame(results)
	if len(results_df) == 0:
		raise SystemExit("No matching predictions found for the given config.")

	# ------------------------------------------------------------------
	# Print metrics, one full table + one compact CI table per target.
	# ------------------------------------------------------------------
	display_cols = [
		"series", "model",
		"MSE", "MAE", "MAPE", "R2",
		"Pearson_r", "Pearson_p",
		"Spearman_r", "Spearman_p",
	]
	series_order = {s: i for i, s in enumerate(SERIES_STYLE)}

	ci_cols = ["series", "model"]
	for metric in plot_metrics:
		ci_cols += [metric, f"{metric}_lo", f"{metric}_hi"]

	for target in TARGETS:
		target_df = results_df[results_df["target"] == target].copy()
		if len(target_df) == 0:
			continue
		target_df["_series_ord"] = target_df["series"].map(series_order)
		target_df = target_df.sort_values(by="_series_ord")

		print(f"\n===== Target: {target} =====")
		print(target_df[display_cols].to_string(index=False, max_colwidth=40))

		if n_boot > 0:
			present_ci_cols = [c for c in ci_cols if c in target_df.columns]
			print(f"\n----- Target: {target} -- {int(ci_level * 100)}% bootstrap CI -----")
			print(target_df[present_ci_cols].to_string(index=False, max_colwidth=40))

	# ------------------------------------------------------------------
	# Plot: one bar chart per (target, metric). Bars = series present for
	# that target; error bars = bootstrap CI (mean baselines get none).
	# ------------------------------------------------------------------
	sns.set_theme(style="whitegrid")

	for target in TARGETS:
		target_df = results_df[results_df["target"] == target]
		if len(target_df) == 0:
			continue

		for metric in plot_metrics:
			# Series present for this target AND with a non-NaN value for this
			# metric (e.g. overall_mean's Spearman_r/Pearson_r are NaN -- a
			# constant prediction has undefined correlation -- so it's simply
			# left out of those figures rather than drawn as an empty bar).
			present = [
				s for s in SERIES_STYLE
				if s in target_df["series"].values
				and pd.notna(target_df[target_df["series"] == s].iloc[0][metric])
			]
			if not present:
				continue
			xs = np.arange(len(present))

			heights, lower, upper, colors = [], [], [], []
			for s in present:
				row = target_df[target_df["series"] == s].iloc[0]
				val = row[metric]
				heights.append(val)
				colors.append(SERIES_STYLE[s]["color"])
				lo_col, hi_col = f"{metric}_lo", f"{metric}_hi"
				if lo_col in row and pd.notna(row.get(lo_col)):
					lower.append(val - row[lo_col])
					upper.append(row[hi_col] - val)
				else:
					lower.append(0.0)  # mean baselines: no CI available
					upper.append(0.0)

			fig, ax = plt.subplots(figsize=(9, 6))
			ax.bar(xs, heights, color=colors, yerr=[lower, upper], capsize=5)

			ax.axhline(0, color="black", linewidth=0.8)
			ax.set_xticks(xs)
			ax.set_xticklabels(
				[SERIES_STYLE[s]["label"] for s in present], rotation=30, ha="right"
			)
			ax.set_title(
				f"{target.capitalize()} prediction performance ({metric}) "
				f"-- STR length {str_len}"
			)
			ax.set_ylabel(metric)
			ax.grid(True, axis="y", alpha=0.3)
			plt.tight_layout()
			plt.show()
