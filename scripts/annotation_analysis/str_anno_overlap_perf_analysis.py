""" Look at how Caduceus model *prediction error* (not the raw label) varies
with the genomic annotation an STR overlaps, for any annotation registered in
annotation_config.yaml (bed or bigwig, discrete or continuous).

This is the performance-vs-annotation analog of str_anno_overlap_analysis.py:
same annotation-overlap machinery, but the quantity being related to the
annotation is per-locus prediction error for one pinned Caduceus run rather
than the raw label (mode_copy_number / heterozygosity). All discrete/
continuous overlap logic, statistics, and plotting are reused unchanged from
str_anno_overlap_analysis.py (imported as `saoa`) -- see that module's
docstring for the full statistical description (Kruskal-Wallis effect sizes,
Cliff's delta with multiple-testing correction, Spearman/Kendall for
continuous annotations, etc.).

Predictions come from a single pinned Caduceus run's predictions_test.tsv
(PRED_DIR, built from STR_LEN + CADUCEUS_RUN_NAME), which already carries
chrom/str_start/str_end per locus -- so no reload of the HipSTR label TSV is
needed; the predictions file *is* the per-locus data frame. That file is one
row per locus **after RC-averaging** (documented in scripts/eval_preds'
docstrings): pred_length/pred_variation are already the average of the
forward-strand and reverse-complement predictions, not one orientation's raw
output. There's a separate predictions_test_by_orientation.tsv (one row per
locus per strand, pre-averaging) that this script deliberately does not use.

For each task the pinned run predicts (length and/or heterozygosity --
detected via detect_tasks, mirroring scripts/eval_preds), two per-locus error metrics
are computed against true_{task}: abs_error (|pred - true|) and signed_error
(pred - true, i.e. bias/direction). squared_error is intentionally omitted:
squaring a nonnegative quantity is a monotonic transform, so the rank-based
statistics used throughout (Spearman, Kendall tau-b, Kruskal-Wallis, Cliff's
delta) would give identical results to abs_error -- redundant computation,
not new signal.

Two of the reused plotting functions (plot_boxplots_by_category,
plot_cliffs_delta_summary) read SHOW_FLIERS/SHOW_CLIFFS_DELTA/
CLIFFS_DELTA_CORRECTION/CLIFFS_DELTA_ALPHA as module globals of
str_anno_overlap_analysis, not as parameters. Defining same-named constants
here would NOT affect their behavior -- they're synced onto the imported
module explicitly in __main__ (`saoa.SHOW_FLIERS = SHOW_FLIERS`, etc.)
before those functions are called.

Configuration is via the editable constants below (no argparse), mirroring
the rest of this directory.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import annotation_utils as au
import str_anno_overlap_analysis as saoa


# ===========================================================================
# Configuration
# ===========================================================================

STR_LEN = 3

# STR_LEN -> pinned Caduceus run (multi-task, fixed-inverse-variance) -- same
# defaults eval_by_model_type.py's MODEL_CONFIGS uses for "caduceus" at each
# STR length. Add an entry here before switching STR_LEN to a length that
# isn't yet covered.
CADUCEUS_RUN_BY_STR_LEN = {
	3: "str3_f5000_fiv_v4_2026-07-06_16-11-21",
	4: "str4_f5000_fiv_v4_2026-06-26_17-20-49",
}
CADUCEUS_RUN_NAME = CADUCEUS_RUN_BY_STR_LEN[STR_LEN]
PRED_DIR = f"../eval_preds/predictions/caduceus/caduceus_v0/str{STR_LEN}/{CADUCEUS_RUN_NAME}"

# ANNOTATION_SELECTION = "ChromHMM"
# ANNOTATION_SELECTION = "ChromHMM_group"
ANNOTATION_SELECTION = "phyloP"

# Prediction-column suffix (pred_{task}/true_{task}) -> display name, mirroring
# TARGETS in scripts/eval_preds/ (there "variation" is heterozygosity's task name).
TASK_DISPLAY_NAMES = {"length": "length", "variation": "heterozygosity"}

# Per-locus error metrics computed against true_{task}. See module docstring
# for why squared_error is omitted.
PERFORMANCE_METRICS = {
	"abs_error": lambda pred, true: (pred - true).abs(),
	# "signed_error": lambda pred, true: pred - true,
}

SHOW_FLIERS = False  # hide outlier points in the box plots (can get busy with ~100 states)
SHOW_CLIFFS_DELTA = True  # annotate each box with its one-vs-rest Cliff's delta

# Multiple-testing correction across the k one-vs-rest Cliff's delta tests run
# per plot (k = number of categories: 16 for ChromHMM_group, 99 for _state).
# None | "bonferroni" | "fdr_bh" (Benjamini-Hochberg). None means the raw
# per-test p-value is compared directly to CLIFFS_DELTA_ALPHA.
CLIFFS_DELTA_CORRECTION = "bonferroni"
CLIFFS_DELTA_ALPHA = 0.01  # significance threshold (applied post-correction) for the "*" marker

CONTINUOUS_METRIC = "mean"  # "mean" | "median" | "max" | "min", aggregated over each STR's own span
CONTINUOUS_ABS = False  # abs() applied per-bp before aggregation

MAX_LEGEND_CATEGORIES = 20  # standalone color legend only drawn for families with <= this many categories

SAVE_DIR = None  # None -> plt.show(); else save one PNG per figure into this dir


# ===========================================================================
# Loading
# ===========================================================================

def detect_tasks(df):
	"""Which of length/heterozygosity this run predicts (replicated from
	scripts/eval_preds' detect_tasks so this script doesn't need a
	cross-directory import)."""
	return [t for t in TASK_DISPLAY_NAMES if f"pred_{t}" in df.columns and f"true_{t}" in df.columns]


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

	sns.set_theme(style="whitegrid")
	config = au.load_config()

	# saoa.plot_boxplots_by_category / plot_cliffs_delta_summary read these as
	# module globals, not params -- sync this script's constants onto the
	# imported module before calling them (see module docstring).
	saoa.SHOW_FLIERS = SHOW_FLIERS
	saoa.SHOW_CLIFFS_DELTA = SHOW_CLIFFS_DELTA
	saoa.CLIFFS_DELTA_CORRECTION = CLIFFS_DELTA_CORRECTION
	saoa.CLIFFS_DELTA_ALPHA = CLIFFS_DELTA_ALPHA

	print(f"Loading Caduceus predictions from {PRED_DIR}...")
	pred_df = pd.read_csv(os.path.join(PRED_DIR, "predictions_test.tsv"), sep="\t")
	tasks = detect_tasks(pred_df)
	print(f"  {len(pred_df):,} loci, tasks present: {[TASK_DISPLAY_NAMES[t] for t in tasks]}")

	for task in tasks:
		for metric_name, metric_fn in PERFORMANCE_METRICS.items():
			pred_df[f"{task}_{metric_name}"] = metric_fn(pred_df[f"pred_{task}"], pred_df[f"true_{task}"])

	label_cols = [
		(f"{task}_{metric_name}", f"{TASK_DISPLAY_NAMES[task]} {metric_name}")
		for task in tasks for metric_name in PERFORMANCE_METRICS
	]

	discrete_results = []
	continuous_results = []

	members = saoa.resolve_family(ANNOTATION_SELECTION, config)
	value_type = config[members[0]]["value_type"]
	print(f"\n{ANNOTATION_SELECTION!r} -> members {members} (value_type={value_type})")

	if value_type == "discrete":
		for member in members:
			entry = au.get_entry(member, config=config)
			value_column = entry["value_column"]

			print(f"  Loading {member}...")
			bed_df = au.load_annotation(member, config=config)

			assigned = saoa.assign_discrete_category(pred_df, bed_df, value_column)
			assigned = saoa.drop_unassigned(assigned, value_column)
			print(
				f"  {len(assigned):,} / {len(pred_df):,} loci assigned, "
				f"{assigned[value_column].nunique()} categories represented"
			)

			color_map = saoa.build_color_map(bed_df, value_column, MAX_LEGEND_CATEGORIES)
			if len(color_map) <= MAX_LEGEND_CATEGORIES:
				saoa.plot_category_legend(color_map, title=member, save_dir=SAVE_DIR)

			for label_col, label_name in label_cols:
				print(f"  Plotting {label_name} by {member}...")
				box_title = f"{label_name} by {member} (STR length {STR_LEN}, {CADUCEUS_RUN_NAME})"
				kw, order, deltas = saoa.plot_boxplots_by_category(
					assigned, label_col, label_name, value_column, color_map,
					title=box_title, save_dir=SAVE_DIR,
				)
				discrete_results.append({**kw, "label_name": label_name, "category_col": member})
				if deltas:
					saoa.plot_cliffs_delta_summary(order, deltas, color_map, title=box_title, save_dir=SAVE_DIR)

	elif value_type == "continuous":
		for member in members:
			entry = au.get_entry(member, config=config)

			print(f"  Loading {member}...")
			bw = au.load_bigwig(entry)
			try:
				agg = saoa.aggregate_continuous(bw, pred_df, CONTINUOUS_METRIC, CONTINUOUS_ABS)
			finally:
				bw.close()
			n_valid = int((~np.isnan(agg)).sum())
			print(f"  {n_valid:,} / {len(pred_df):,} loci have a valid aggregated value")

			for label_col, label_name in label_cols:
				labels = pred_df[label_col].to_numpy()
				corr = saoa.compute_continuous_correlation(agg, labels)
				print(
					f"  {label_name} vs {member}: n={corr['n']:,}, "
					f"rho={corr['rho']:.3f} (p={saoa.format_p_value(corr['p_rho'])}), "
					f"tau={corr['tau']:.3f} (p={saoa.format_p_value(corr['p_tau'])})"
				)
				saoa.plot_hexbin(
					agg, labels, member, label_name, corr,
					title=f"{label_name} vs {member} (STR length {STR_LEN}, {CADUCEUS_RUN_NAME}, "
						f"metric={CONTINUOUS_METRIC}, abs={CONTINUOUS_ABS})",
					save_dir=SAVE_DIR,
				)
				continuous_results.append({**corr, "annotation_type": member, "label_name": label_name})

	else:
		raise ValueError(f"Unknown value_type {value_type!r} for selection {ANNOTATION_SELECTION!r}")

	if discrete_results:
		saoa.plot_effect_size_summary(
			discrete_results, metric="eps2", ylabel=r"Kruskal-Wallis $\epsilon^2$",
			title=r"Kruskal-Wallis effect size ($\epsilon^2$) by annotation", save_dir=SAVE_DIR,
		)
		saoa.plot_effect_size_summary(
			discrete_results, metric="eta2_h", ylabel=r"Kruskal-Wallis $\eta^2_H$",
			title=r"Kruskal-Wallis effect size ($\eta^2_H$) by annotation", save_dir=SAVE_DIR,
		)
	if continuous_results:
		saoa.plot_continuous_effect_size_summary(
			continuous_results, title="Correlation strength by annotation", save_dir=SAVE_DIR,
		)
		saoa.plot_continuous_effect_size_summary(
			continuous_results, title="Correlation strength by annotation (signed)", save_dir=SAVE_DIR,
			use_abs=False,
		)
