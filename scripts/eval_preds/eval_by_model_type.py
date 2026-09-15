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

# Default number of equal-frequency bins for the quintile-split evaluation.
DEFAULT_N_QUANTILES = 5


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


def compute_quantile_edges(values, n_quantiles=DEFAULT_N_QUANTILES):
	"""Global bin edges for n_quantiles equal-frequency bins over `values`
	(e.g. true label values pooled across all model series for one target),
	via pd.qcut(..., duplicates="drop"). Returns the sorted edge array
	(len = n_bins + 1), or None if ties collapse it below 2 bins."""
	values = np.asarray(values)
	_, edges = pd.qcut(values, q=n_quantiles, retbins=True, duplicates="drop")
	n_bins = len(edges) - 1
	if n_bins < 2:
		print(f"  [warn] quantile edges degenerate ({n_bins} bin(s) from "
			  f"{len(values)} pooled values); skipping quantile eval")
		return None
	if n_bins < n_quantiles:
		print(f"  [warn] only {n_bins}/{n_quantiles} distinct quantile bins "
			  f"(tied values); continuing with {n_bins} bins")
	return edges


def assign_quantile_bin(values, edges):
	"""Assign each value to a 0-indexed bin using pre-computed global `edges`
	(from compute_quantile_edges) via pd.cut -- NOT a fresh qcut -- so every
	series is binned against the same boundaries. Edges are extended to
	+/-inf at the extremes so a series whose local min/max falls outside the
	pooled range still lands in the outermost bin rather than becoming NaN."""
	edges = np.array(edges, dtype=float)
	edges[0], edges[-1] = -np.inf, np.inf
	return pd.cut(values, bins=edges, labels=False, include_lowest=True)


def compute_locus_errors(pred, true, kinds):
	"""Per-locus error values, computed once over the full series (never
	rescored on a subset). Used so the quantile breakdown can show the
	*distribution* of an already-computed per-locus quantity split by bin,
	rather than recomputing a whole-array metric (R2, Spearman, ...) on each
	narrow subset -- which is unstable for R2 in particular, since a bin's
	local variance (its R2 denominator) shrinks as the bin narrows.

	kinds: subset of {"squared_error", "abs_error", "signed_error",
	"pct_error"} ("pct_error" is on a 0-100 scale, e.g. 50.0 = 50%, not
	0.5). Returns {kind: np.ndarray} aligned with pred/true."""
	pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
	diff = pred - true
	out = {}
	if "squared_error" in kinds:
		out["squared_error"] = diff ** 2
	if "abs_error" in kinds:
		out["abs_error"] = np.abs(diff)
	if "signed_error" in kinds:
		out["signed_error"] = diff
	if "pct_error" in kinds:
		with np.errstate(divide="ignore", invalid="ignore"):
			out["pct_error"] = 100 * np.abs(diff) / np.abs(true)
	return out


if __name__ == "__main__":

	# ------------------------------------------------------------------
	# Config
	# ------------------------------------------------------------------
	str_len = 4  # which STR length to evaluate; must be a key of MODEL_CONFIGS

	plot_metrics = ["Spearman_r"] #, "R2"]  # one bar-chart figure per metric per target

	# Additive: also split each target's loci into quantile bins by its own
	# true value (see compute_quantile_edges / assign_quantile_bin) and show
	# how a per-locus error, computed once over the whole series, is
	# distributed within each bin -- no metric (R2/Spearman/...) is rescored
	# per bin. Set False to reproduce pre-quintile behavior.
	enable_quintile_eval = True
	n_quantiles = 5
	# Per-locus error kind(s) to show the distribution of; see
	# compute_locus_errors for the available options.
	quantile_error_metrics = ["squared_error", "pct_error"]

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
	loaded_preds = {}  # config_key -> pred_df, reused below by quintile pass

	for config_key, run_name in model_paths.items():
		run_dir = os.path.join(SERIES_ROOT[config_key], f"str{str_len}", run_name)
		pred_df = load_predictions(run_dir)
		if pred_df is None:
			continue
		loaded_preds[config_key] = pred_df
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
	# Quintile-split evaluation: bin each target's loci into n_quantiles
	# equal-frequency bins by that target's own true value (bin edges shared
	# across all series -- pooled + deduped by locus id, then every series is
	# assigned into those same edges). Each per-locus error in
	# quantile_error_metrics is computed ONCE over the full series (see
	# compute_locus_errors) -- no metric is recomputed on a subset -- and we
	# simply show how that already-computed per-locus quantity is
	# distributed within each bin. Mean baselines have no per-locus
	# predictions, so they never appear here.
	# ------------------------------------------------------------------
	quintile_frames = []
	quintile_edges = {}  # target -> edges, for bin-range annotations below

	if enable_quintile_eval:
		for target in TARGETS:
			target_frames = [
				df for df in loaded_preds.values() if target in detect_tasks(df)
			]
			if not target_frames:
				continue
			pooled = pd.concat(
				[df[["id", f"true_{target}"]] for df in target_frames],
				ignore_index=True,
			).drop_duplicates(subset="id")

			edges = compute_quantile_edges(
				pooled[f"true_{target}"].to_numpy(), n_quantiles
			)
			if edges is None:
				continue
			quintile_edges[target] = edges

			for config_key, run_name in model_paths.items():
				pred_df = loaded_preds.get(config_key)
				if pred_df is None or target not in detect_tasks(pred_df):
					continue
				pred_arr = pred_df[f"pred_{target}"].to_numpy()
				true_arr = pred_df[f"true_{target}"].to_numpy()
				errors = compute_locus_errors(
					pred_arr, true_arr, quantile_error_metrics
				)
				bins = assign_quantile_bin(true_arr, edges)
				quintile_frames.append(pd.DataFrame({
					"series": SERIES_SOURCE_KEY[config_key],
					"target": target,
					"model": run_name,
					"quintile": bins.astype(int) + 1,  # 1-indexed, Q1 = lowest true
					**errors,
				}))

	quintile_df = (
		pd.concat(quintile_frames, ignore_index=True)
		if quintile_frames else pd.DataFrame()
	)

	if enable_quintile_eval and len(quintile_df) > 0:
		for target in TARGETS:
			target_qdf = quintile_df[quintile_df["target"] == target]
			if len(target_qdf) == 0:
				continue
			edges = quintile_edges[target]
			n_bins = len(edges) - 1

			print(f"\n===== Target: {target} -- per-locus error distribution "
				  f"by quantile bin ({n_bins} bins) =====")
			for kind in quantile_error_metrics:
				summary = (
					target_qdf.groupby(["quintile", "series"])[kind]
					.agg(n="count", mean="mean", median="median", std="std",
						 q25=lambda s: s.quantile(0.25),
						 q75=lambda s: s.quantile(0.75))
					.reset_index()
				)
				summary["bin_lo"] = summary["quintile"].map(
					lambda q: edges[q - 1]
				)
				summary["bin_hi"] = summary["quintile"].map(
					lambda q: edges[q]
				)
				summary["_series_ord"] = summary["series"].map(series_order)
				summary = summary.sort_values(
					by=["quintile", "_series_ord"]
				).drop(columns="_series_ord")
				print(f"\n----- {kind} -----")
				print(summary[[
					"quintile", "bin_lo", "bin_hi", "series", "n",
					"mean", "median", "std", "q25", "q75",
				]].to_string(index=False, max_colwidth=40))

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

	# ------------------------------------------------------------------
	# Plot: one point+CI plot per (target, error kind) for the quintile-split
	# evaluation. X-axis = quintile bin (low-to-high true value); within each
	# bin, one point per series = mean of that per-locus error (computed once
	# over the whole series, see compute_locus_errors -- never rescored on
	# the subset), with a seaborn-bootstrapped 95% CI on the mean.
	# ------------------------------------------------------------------
	if enable_quintile_eval and len(quintile_df) > 0:
		for target in TARGETS:
			target_qdf = quintile_df[quintile_df["target"] == target]
			if len(target_qdf) == 0:
				continue

			present_series = [s for s in SERIES_STYLE if s in target_qdf["series"].values]
			if not present_series:
				continue
			plot_df = target_qdf.copy()
			plot_df["series_label"] = plot_df["series"].map(
				lambda s: SERIES_STYLE[s]["label"]
			)
			hue_order = [SERIES_STYLE[s]["label"] for s in present_series]
			palette = {SERIES_STYLE[s]["label"]: SERIES_STYLE[s]["color"] for s in present_series}

			for kind in quantile_error_metrics:
				kdf = plot_df.dropna(subset=[kind])
				if len(kdf) == 0:
					continue

				fig, ax = plt.subplots(figsize=(11, 6))
				sns.pointplot(
					data=kdf, x="quintile", y=kind, hue="series_label",
					hue_order=hue_order, palette=palette, ax=ax,
					estimator="mean", errorbar=("ci", 95),
					dodge=0.4, linestyle="none", capsize=0.05, markersize=5,
				)
				ax.set_xlabel(f"True {target} quantile bin (low -> high)")
				ylabel = kind.replace("_", " ").title()
				if kind == "pct_error":
					ylabel += " (%)"
				ax.set_ylabel(ylabel)
				ax.set_title(
					f"{target.capitalize()} per-locus {kind.replace('_', ' ')} "
					f"distribution by true-value quantile -- STR length {str_len}"
				)
				ax.grid(True, axis="y", alpha=0.3)
				ax.legend(loc="best", fontsize=8, title=None)
				plt.tight_layout()
				plt.show()
