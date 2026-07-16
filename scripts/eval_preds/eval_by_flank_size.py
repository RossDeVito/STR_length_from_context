""" Compare a model config across flank sizes, for one or more STR lengths.

Given a set of STR lengths (``str_lens``), a set of flank sizes
(``flank_sizes``), and a run-directory template (``run_template``, e.g.
``"str{}_f{}_fiv_v4"`` formatted with (str_len, flank)), this loads the matching
Caduceus run at each (length, flank) and plots how performance changes with
flank size -- one line per STR length, one figure per (target, metric).

Linear baselines are intentionally left out for now; only Caduceus model
predictions are evaluated.

Prediction TSVs (predictions_test.tsv, one row per locus after RC-averaging)
have columns:

	id  pred_length  pred_variation  true_length  true_variation
	chrom  str_start  str_end  motif  split

Models may predict only one of the two targets (length, variation); the tasks
present are detected per prediction file.
"""

import os

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

# Error metrics: lower is better (idxmin). Everything else: higher is better.
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


def find_runs(parent, prefix):
	"""Run subdirs under ``parent`` that are exactly ``prefix`` or begin with
	``prefix`` + "_" (the timestamp separator). Guards against e.g. an f1000
	prefix matching an f10000 run. Returns sorted names (empty if none)."""
	if not os.path.isdir(parent):
		return []
	return sorted(
		d for d in os.listdir(parent)
		if os.path.isdir(os.path.join(parent, d))
		and (d == prefix or d.startswith(prefix + "_"))
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


def bootstrap_ci(pred, true, n_boot, ci_level, rng):
	"""Percentile bootstrap CI (resampling loci with replacement) for every
	metric in get_metrics. Returns {"{metric}_lo": v, "{metric}_hi": v, ...}."""
	pred, true = np.asarray(pred), np.asarray(true)
	n = len(pred)
	boot_df = pd.DataFrame([
		get_metrics(pred[idx], true[idx])
		for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))
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
	str_lens = [3, 4]
	flank_sizes = [1000, 2500, 5000, 7500, 10000]

	# Run-directory template, formatted with (str_len, flank) to build the run
	# prefix, e.g. "str4_f2500_fiv_v4". The actual run dir has a timestamp (and
	# optional _resumed_epochN) appended.
	run_template = "str{}_f{}_fiv_v4"

	plot_metrics = ["Spearman_r", "Pearson_r", "R2"]  # one figure per metric per target
	# Tie-break metric when more than one run dir matches a (length, flank)
	# prefix (e.g. a resumed run): keep the best by this metric per target.
	select_metric = "Spearman_r"

	caduceus_root = "predictions/caduceus/caduceus_v0"

	n_boot = 1000        # bootstrap resamples for CI bands (0 disables CI entirely)
	ci_level = 0.95      # confidence level as a fraction, e.g. 0.95 = 95% CI
	bootstrap_seed = 42  # RNG seed, for reproducible bootstrap resamples
	rng = np.random.default_rng(bootstrap_seed)

	# ------------------------------------------------------------------
	# Build results: one row per (str_len, flank, target).
	# ------------------------------------------------------------------
	results = []

	for str_len in str_lens:
		cad_dir = os.path.join(caduceus_root, f"str{str_len}")
		for flank in flank_sizes:
			prefix = run_template.format(str_len, flank)
			runs = find_runs(cad_dir, prefix)
			if not runs:
				print(f"[skip] no run matching {prefix!r} under {cad_dir}")
				continue

			rows = []  # per (run, target) metric dicts
			for run in runs:
				pred_df = load_predictions(os.path.join(cad_dir, run))
				if pred_df is None:
					continue
				for task in detect_tasks(pred_df):
					pred_arr = pred_df[f"pred_{task}"].to_numpy()
					true_arr = pred_df[f"true_{task}"].to_numpy()
					rows.append({
						"target": task,
						"model": run,
						"_pred": pred_arr,
						"_true": true_arr,
						**get_metrics(pred=pred_arr, true=true_arr),
					})

			# Keep the best matching run per target (usually just one), and
			# bootstrap CI only for that winner.
			for target in TARGETS:
				best = best_by_metric(
					[r for r in rows if r["target"] == target], select_metric
				)
				if best is None:
					continue
				entry = {"str_len": str_len, "flank": flank, **best}
				if n_boot > 0:
					entry.update(bootstrap_ci(
						entry["_pred"], entry["_true"], n_boot, ci_level, rng
					))
				entry.pop("_pred", None)
				entry.pop("_true", None)
				results.append(entry)

	results_df = pd.DataFrame(results)
	if len(results_df) == 0:
		raise SystemExit("No matching predictions found for the given config.")

	# ------------------------------------------------------------------
	# Print metrics, one table per target (rows sorted by str_len then flank).
	# ------------------------------------------------------------------
	display_cols = [
		"str_len", "flank", "model",
		"MSE", "MAE", "MAPE", "R2",
		"Pearson_r", "Pearson_p",
		"Spearman_r", "Spearman_p",
	]
	if n_boot > 0:
		for metric in plot_metrics:
			display_cols += [f"{metric}_lo", f"{metric}_hi"]
	for target in TARGETS:
		target_df = results_df[results_df["target"] == target]
		if len(target_df) == 0:
			continue
		target_df = target_df.sort_values(by=["str_len", "flank"])
		print(f"\n===== Target: {target} =====")
		print(target_df[display_cols].to_string(index=False, max_colwidth=40))

	# ------------------------------------------------------------------
	# Plot: metric vs flank size, one line per STR length, one figure per
	# (target, metric). Lengths/flanks with no matching run simply have gaps.
	# ------------------------------------------------------------------
	sns.set_theme(style="whitegrid")

	for target in TARGETS:
		target_df = results_df[results_df["target"] == target]
		if len(target_df) == 0:
			continue

		for metric in plot_metrics:
			plt.figure(figsize=(8, 6))
			plotted_any = False

			for str_len in str_lens:
				sub = target_df[target_df["str_len"] == str_len]
				sub = sub[sub["flank"].isin(flank_sizes)].sort_values("flank")
				xs = sub["flank"].to_numpy()
				ys = sub[metric].to_numpy(dtype=float)
				mask = ~np.isnan(ys)
				if not mask.any():
					continue
				line, = plt.plot(xs[mask], ys[mask], marker="o", label=f"STR len {str_len}")
				if n_boot > 0 and f"{metric}_lo" in sub.columns:
					los = sub[f"{metric}_lo"].to_numpy(dtype=float)
					his = sub[f"{metric}_hi"].to_numpy(dtype=float)
					plt.fill_between(
						xs[mask], los[mask], his[mask],
						color=line.get_color(), alpha=0.2,
					)
				plotted_any = True

			if not plotted_any:
				plt.close()
				continue

			plt.title(f"{target.capitalize()} prediction performance by flank size")
			plt.xlabel("Flank size (bp)")
			plt.ylabel(metric)
			plt.xticks(flank_sizes)
			plt.grid(True)
			plt.legend(loc="best")
			plt.tight_layout()
			plt.show()
