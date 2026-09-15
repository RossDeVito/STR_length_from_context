""" Look at how STR labels (length / heterozygosity) vary with the genomic
annotation an STR overlaps, for any annotation registered in
annotation_config.yaml (bed or bigwig, discrete or continuous).

ANNOTATION_SELECTION is either an exact config key (e.g. "phyloP") or a
family base name (e.g. "ChromHMM", which expands via prefix-match to every
config key starting with "ChromHMM_" -- currently ChromHMM_state and
ChromHMM_group).

Discrete annotations (e.g. ChromHMM_state/_group): each STR is assigned the
category covering the most total bp of its span. Overlap bp is summed across
all candidate segments sharing the same category value before taking the
max, so two adjacent same-category segments correctly combine instead of
spuriously tying; only a genuine tie between >=2 distinct category totals
excludes the STR. For each label, produces one horizontal box plot per
family member -- one box per category, sorted by median, colored by the
annotation's own `color` column if it has one (else an auto-generated
palette), with n and (if SHOW_CLIFFS_DELTA) each category's one-vs-rest
Cliff's delta annotated per box and a reference line at the overall median,
plus a standalone companion bar chart of delta per category -- plus a
cross-family Kruskal-Wallis effect size (epsilon^2, eta^2_H) comparison bar
chart. Cliff's delta is computed via the Mann-Whitney U identity
(delta = 2*U/(n_x*n_y) - 1), which stays exact under ties -- common for the
length target (mode_copy_number). Since each family member runs k
one-vs-rest tests (k = number of categories, up to 99 for ChromHMM_state),
CLIFFS_DELTA_CORRECTION applies a multiple-testing correction
(None/"bonferroni"/"fdr_bh") across those k tests before the "*"
significance marker (at CLIFFS_DELTA_ALPHA) is decided.

Continuous annotations (e.g. phyloP): each STR's span is aggregated with
CONTINUOUS_METRIC (mean/median/max/min, nan-aware), optionally abs()'d
per-bp first via CONTINUOUS_ABS (abs is applied before aggregating, not
after, so positive/negative signal doesn't cancel over the span). For each
label, produces a hexbin plot of the aggregated value vs. the label with
Spearman rho and Kendall tau-b (tau-b handles ties better than tau-a) in the
title, plus a cross-annotation comparison bar chart in both an unsigned
(|rho|/|tau|, magnitude only) and signed (rho/tau, direction included)
version.

Configuration is via the editable constants below (no argparse), mirroring
the rest of this directory.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import kruskal, spearmanr, kendalltau, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import annotation_utils as au


# ===========================================================================
# Configuration
# ===========================================================================

STR_LEN = 3
N_FLANKING = 10000

STR_DATA_PATH = f"../../data/STR_data/HipSTR_labeled_STRs/str_len_{STR_LEN}_n_flanking_{N_FLANKING}.tsv"

# ANNOTATION_SELECTION = "ChromHMM"  # a base name (prefix-expanded) or an exact config key
# ANNOTATION_SELECTION = "ChromHMM_group"
ANNOTATION_SELECTION = "phyloP"

# Targets, mirroring scripts/eval_preds/get_mean_baseline_performance.py
TARGETS = {
	"length": "mode_copy_number",
	"heterozygosity": "heterozygosity",
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

def load_str_data(path):
	"""Load HipSTR-labeled STRs, keeping one row per locus (drop the
	reverse-complement duplicate; it shares the same position/labels)."""
	df = pd.read_csv(path, sep="\t")
	df = df[~df["rev_comp"]].reset_index(drop=True)
	return df


def resolve_family(selection, config):
	"""Resolve a selection to a list of config keys. An exact key match
	wins outright; otherwise expand via prefix-match on "{selection}_" (e.g.
	"ChromHMM" -> ["ChromHMM_group", "ChromHMM_state"]). All resolved
	members must share one value_type (discrete xor continuous)."""
	if selection in config:
		return [selection]

	prefix = f"{selection}_"
	members = sorted(k for k in config if k.startswith(prefix))
	if not members:
		raise KeyError(
			f"No exact match or {prefix!r}* members for selection {selection!r}. "
			f"Available: {sorted(config.keys())}"
		)

	value_types = {config[m]["value_type"] for m in members}
	if len(value_types) > 1:
		raise ValueError(
			f"Family {selection!r} mixes value_types across members: "
			f"{ {m: config[m]['value_type'] for m in members} } -- "
			f"a family must be all-discrete or all-continuous"
		)
	return members


# ===========================================================================
# Discrete: assign each STR its best-overlap category
# ===========================================================================

def assign_discrete_category(str_df, bed_df, value_column):
	"""Adds `<value_column>` / `<value_column>_overlap_bp` columns to a copy
	of str_df: the category each STR overlaps the most total bp, and that
	total.

	Assumes bed_df is a non-overlapping, sorted-by-start partition per
	chromosome (true for ChromHMM; a constraint for any future discrete bed
	added to the config). Bed rows with a null value_column (unjoined) are
	dropped first, treated as gap.

	Tie handling: overlap bp is summed across all candidate segments sharing
	the same category value before taking the max, so two adjacent
	same-category segments correctly combine instead of spuriously tying. A
	genuine tie between >=2 distinct category totals leaves the STR
	unassigned (NaN), same handling as an unannotated gap.
	"""
	bed_df = bed_df.dropna(subset=[value_column])

	n = len(str_df)
	out_cat = np.full(n, None, dtype=object)
	out_overlap = np.zeros(n, dtype=np.int64)

	chrom_arr = str_df["chrom"].to_numpy()
	start_arr = str_df["str_start"].to_numpy()
	end_arr = str_df["str_end"].to_numpy()

	for chrom in np.unique(chrom_arr):
		str_positions = np.nonzero(chrom_arr == chrom)[0]

		bed_sub = bed_df[bed_df["chrom"] == chrom].sort_values("start")
		if bed_sub.empty or len(str_positions) == 0:
			continue
		b_starts = bed_sub["start"].to_numpy()
		b_ends = bed_sub["end"].to_numpy()
		b_cats = bed_sub[value_column].to_numpy()

		qs = start_arr[str_positions]
		qe = end_arr[str_positions]

		# Segments overlapping [qs, qe) form a contiguous index range since
		# segments are sorted and non-overlapping.
		lo = np.searchsorted(b_ends, qs, side="right")  # first segment with end > qs
		hi = np.searchsorted(b_starts, qe, side="left")  # first segment with start >= qe

		for k in range(len(str_positions)):
			l, h = lo[k], hi[k]
			if h <= l:
				continue  # STR falls in an unannotated gap

			if h - l == 1:
				# Single candidate: no tie possible regardless of category grouping.
				j = l
				out_cat[str_positions[k]] = b_cats[j]
				out_overlap[str_positions[k]] = min(b_ends[j], qe[k]) - max(b_starts[j], qs[k])
				continue

			js = np.arange(l, h)
			ovs = np.minimum(b_ends[js], qe[k]) - np.maximum(b_starts[js], qs[k])
			cats = b_cats[js]

			totals = {}
			for c, ov in zip(cats, ovs):
				totals[c] = totals.get(c, 0) + ov
			max_total = max(totals.values())
			winners = [c for c, v in totals.items() if v == max_total]
			if len(winners) > 1:
				continue  # tie across distinct categories -- exclude

			out_cat[str_positions[k]] = winners[0]
			out_overlap[str_positions[k]] = max_total

	out = str_df.copy()
	out[value_column] = out_cat
	out[f"{value_column}_overlap_bp"] = out_overlap
	return out


def drop_unassigned(str_df, value_column):
	n_unassigned = str_df[value_column].isna().sum()
	if n_unassigned:
		print(f"  Dropping {n_unassigned} / {len(str_df)} STRs with no unambiguous {value_column!r} assignment (gap or tie)")
	return str_df.dropna(subset=[value_column]).reset_index(drop=True)


# ===========================================================================
# Discrete: color mapping
# ===========================================================================

def build_color_map(df, value_column, max_legend_categories):
	"""If the loaded annotation dataframe has its own `color` column (true
	for ChromHMM via its join), use it directly (one color per category,
	first occurrence). Otherwise auto-generate a categorical palette."""
	if "color" in df.columns:
		cmap_df = df.dropna(subset=[value_column, "color"]).drop_duplicates(value_column)
		return dict(zip(cmap_df[value_column], cmap_df["color"]))

	categories = sorted(df[value_column].dropna().unique())
	if len(categories) <= 20:
		palette = sns.color_palette("tab20", len(categories))
	else:
		palette = sns.color_palette("husl", len(categories))
	return dict(zip(categories, palette))


def plot_category_legend(color_map, title, save_dir):
	"""Standalone legend mapping category -> color."""
	fig, ax = plt.subplots(figsize=(4, 0.25 * len(color_map) + 0.5))
	handles = [
		Patch(facecolor=color, edgecolor="black", linewidth=0.6, label=cat)
		for cat, color in color_map.items()
	]
	ax.legend(handles=handles, loc="center", frameon=False, title=title, fontsize=8)
	ax.axis("off")
	plt.tight_layout()
	_show_or_save(fig, f"legend_{title}", save_dir)


# ===========================================================================
# Statistics
# ===========================================================================

def kruskal_wallis_effect_size(df, label_col, category_col):
	"""Kruskal-Wallis H-test for whether `label_col` differs across
	`category_col` groups, with two rank-based effect sizes.

	Both effect sizes follow Tomczak, M., & Tomczak, E. (2014). "The need
	to report effect size estimates revisited. An overview of some
	recommended measures of effect size." Trends in Sport Sciences,
	1(21), 19-25. T&T attribute the eta-squared form to their ref [26]
	(Cohen, B. H., 2008, Explaining Psychological Statistics, 3rd ed.)
	and the epsilon-squared form to their ref [1].

	eps2 -- epsilon-squared (E^2_R in T&T):

		eps2 = H / ((n^2 - 1) / (n + 1))   ==   H / (n - 1)

	Equals SS_between / SS_total computed on midranks, i.e. the proportion
	of rank variance in the label explained by group membership. Exact
	under ties: scipy's tie correction and the tie-shrunken midrank
	denominator cancel. Ranges 0 (no relationship) to 1.

	eta2_h -- eta-squared based on the H-statistic:

		eta2_h = (H - k + 1) / (n - k)

	eps2 is not corrected for the number of groups. Under the null,
	H ~ chi2(k-1), so E[eps2] ~ (k-1)/(n-1): eps2 rises with k even when
	group membership is pure noise, as R^2 rises with predictor count.
	eta2_h subtracts that null expectation and rescales by residual df,
	giving expectation ~0 under the null at any k. The two agree to
	<0.1% when n/k is large and diverge sharply when it is small.

	NOTE: T&T describe eta2_h as bounded [0, 1]. It is not. When H < k-1
	the value is negative, meaning the observed effect is at or below
	chance. This is informative, not a bug -- do not clamp it. Expect it
	in covariate-adjusted runs at k=100.

	Raises on NaN in either column. Filtering is the caller's decision and
	must be applied identically across labels and motif lengths, otherwise
	the effect sizes are not comparable.
	"""
	sub = df[[label_col, category_col]]
	if sub.isna().any().any():
		n_bad = int(sub.isna().any(axis=1).sum())
		raise ValueError(
			f"{n_bad} rows have NaN in {label_col!r} or {category_col!r}; "
			f"filter upstream"
		)

	groups = [
		g[label_col].to_numpy()
		for _, g in sub.groupby(category_col, observed=True)
	]
	if len(groups) < 2:
		raise ValueError(f"need >=2 groups, got {len(groups)}")

	h_stat, p_value = kruskal(*groups)
	n, k = len(sub), len(groups)

	return {
		"h_stat": h_stat,
		"p_value": p_value,
		"eps2": h_stat / (n - 1),
		"eta2_h": (h_stat - k + 1) / (n - k),
		"n": n,
		"k": k,
	}


def format_p_value(p_value):
	return f"{p_value:.2e}" if p_value > 0 else "< 1e-300"


def correction_label(correction, alpha):
	method_names = {"bonferroni": "Bonferroni", "fdr_bh": "BH-FDR"}
	method = method_names.get(correction, "uncorrected") if correction else "uncorrected"
	return f"{method}, α={alpha}"


def cliffs_delta(x, y):
	"""Cliff's delta effect size (P(X>Y) - P(X<Y)) via the Mann-Whitney U
	identity delta = 2*U/(n_x*n_y) - 1. U already gives tied pairs half
	credit, so this is exact under ties -- important since `length`
	(mode_copy_number) has many repeated values across loci. Returns
	(delta, p_value) from the same two-sided Mann-Whitney U test.

	Interpretation thresholds (Romano et al. 2006): |delta| < 0.147
	negligible, < 0.33 small, < 0.474 medium, >= 0.474 large.
	"""
	u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
	delta = 2 * u_stat / (len(x) * len(y)) - 1
	return delta, p_value


def one_vs_rest_cliffs_delta(df, label_col, category_col, correction=None, alpha=0.05):
	"""For each category value, Cliff's delta comparing that category's
	label_col values against all other rows' (one-vs-rest). Assumes df has
	already been validated NaN-free in label_col/category_col by the caller
	(kruskal_wallis_effect_size already does this check earlier in
	plot_boxplots_by_category).

	`correction` applies a multiple-testing correction jointly across the k
	one-vs-rest tests run here (k = number of categories), since each is an
	independent test and running many of them (e.g. 99 for ChromHMM_state)
	inflates the false-positive rate at any fixed per-test alpha:
	None -> no correction, raw p compared directly to `alpha`;
	"bonferroni" -> family-wise error rate control (conservative at large k);
	"fdr_bh" -> Benjamini-Hochberg false discovery rate control.

	Returns {category: {"delta", "p_raw", "p_corrected", "significant"}}.
	"""
	cats, deltas, p_raws = [], [], []
	for cat, group in df.groupby(category_col, observed=True):
		rest = df.loc[df[category_col] != cat, label_col].to_numpy()
		d, p = cliffs_delta(group[label_col].to_numpy(), rest)
		cats.append(cat)
		deltas.append(d)
		p_raws.append(p)

	if correction:
		reject, p_corrected, _, _ = multipletests(p_raws, alpha=alpha, method=correction)
	else:
		p_corrected = p_raws
		reject = [p <= alpha for p in p_raws]

	return {
		cat: {"delta": d, "p_raw": p_raw, "p_corrected": p_corr, "significant": bool(sig)}
		for cat, d, p_raw, p_corr, sig in zip(cats, deltas, p_raws, p_corrected, reject)
	}


def compute_continuous_correlation(agg_values, labels):
	mask = ~np.isnan(agg_values)
	rho, p_rho = spearmanr(agg_values[mask], labels[mask])
	tau, p_tau = kendalltau(agg_values[mask], labels[mask], variant="b")  # explicit even though it's scipy's default
	return {"rho": rho, "p_rho": p_rho, "tau": tau, "p_tau": p_tau, "n": int(mask.sum())}


# ===========================================================================
# Continuous: aggregate each STR's span
# ===========================================================================

METRIC_FNS = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax, "min": np.nanmin}


def aggregate_continuous(bw, str_df, metric, use_abs):
	"""Per-locus metric_fn over bw.values(chrom, str_start, str_end,
	numpy=True) (abs()'d first if use_abs), nan-aware. A locus whose entire
	span is missing is left NaN (excluded downstream, never a hard error)."""
	metric_fn = METRIC_FNS[metric]
	chrom_lengths = bw.chroms()

	chrom_arr = str_df["chrom"].to_numpy()
	start_arr = str_df["str_start"].to_numpy()
	end_arr = str_df["str_end"].to_numpy()

	agg = np.full(len(str_df), np.nan, dtype=np.float64)
	n_skipped_chrom = 0
	for i in range(len(str_df)):
		chrom = chrom_arr[i]
		if chrom not in chrom_lengths:
			n_skipped_chrom += 1
			continue

		vals = bw.values(chrom, int(start_arr[i]), int(end_arr[i]), numpy=True)
		if use_abs:
			vals = np.abs(vals)
		if np.all(np.isnan(vals)):
			continue  # entire span missing -- leave NaN, excluded downstream

		agg[i] = metric_fn(vals)

	if n_skipped_chrom:
		print(f"  Skipped {n_skipped_chrom} loci on chromosomes absent from the bigwig")
	return agg


# ===========================================================================
# Plotting
# ===========================================================================

def _show_or_save(fig, name, save_dir):
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, f"{name}.png")
		fig.savefig(path, dpi=150, bbox_inches="tight")
		plt.close(fig)
		print(f"  saved {path}")
	else:
		plt.show()


def plot_boxplots_by_category(df, label_col, label_name, category_col, color_map, title, save_dir):
	"""Horizontal box plots of `label_col`, one box per `category_col` value,
	sorted by median (highest at top), colored by `color_map`, with n (and,
	if SHOW_CLIFFS_DELTA, each category's one-vs-rest Cliff's delta)
	annotated, a reference line at the overall median, and the omnibus
	Kruskal-Wallis test (H, p, epsilon-squared, eta-squared_H) across all
	categories shown in the title. Returns (kw, order, deltas): the test
	result dict from kruskal_wallis_effect_size, the category order (sorted
	by median, matching the plot's y-axis), and the one-vs-rest Cliff's
	delta dict from one_vs_rest_cliffs_delta (None if SHOW_CLIFFS_DELTA is
	False) -- `order`/`deltas` are meant to be handed to
	plot_cliffs_delta_summary for a companion plot in the same category
	order/coloring."""
	medians = df.groupby(category_col)[label_col].median().sort_values()
	order = medians.index.tolist()
	counts = df.groupby(category_col)[label_col].size()
	data = [df.loc[df[category_col] == cat, label_col].to_numpy() for cat in order]

	kw = kruskal_wallis_effect_size(df, label_col, category_col)
	p_str = format_p_value(kw["p_value"])
	print(
		f"  Kruskal-Wallis ({label_name} by {category_col}): "
		f"H={kw['h_stat']:.1f}, df={kw['k'] - 1}, n={kw['n']:,}, "
		f"p={p_str}, eps^2={kw['eps2']:.3f}, eta^2_H={kw['eta2_h']:.3f}"
	)

	deltas = (
		one_vs_rest_cliffs_delta(
			df, label_col, category_col,
			correction=CLIFFS_DELTA_CORRECTION, alpha=CLIFFS_DELTA_ALPHA,
		) if SHOW_CLIFFS_DELTA else None
	)
	if deltas:
		top = sorted(deltas.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True)[:3]
		top_str = ", ".join(
			f"{cat} δ={v['delta']:+.2f} (p={format_p_value(v['p_corrected'])}"
			f"{'*' if v['significant'] else ''})"
			for cat, v in top
		)
		print(
			f"  Cliff's delta (one-vs-rest, {correction_label(CLIFFS_DELTA_CORRECTION, CLIFFS_DELTA_ALPHA)}), "
			f"top 3 by |delta|: {top_str}"
		)

	fig_height = max(4, 0.28 * len(order) + 1.5)
	fig, ax = plt.subplots(figsize=(10, fig_height))

	bp = ax.boxplot(
		data, vert=False, patch_artist=True, showfliers=SHOW_FLIERS,
		widths=0.7, flierprops=dict(marker="o", markersize=2, alpha=0.3),
	)
	for patch, cat in zip(bp["boxes"], order):
		patch.set_facecolor(color_map.get(cat, "#cccccc"))
		patch.set_edgecolor("black")
		patch.set_linewidth(0.6)
	for median_line in bp["medians"]:
		median_line.set_color("black")

	ax.set_yticks(range(1, len(order) + 1))
	ax.set_yticklabels(order, fontsize=8)
	ax.set_ylim(0.3, len(order) + 0.7)

	overall_median = df[label_col].median()
	ax.axvline(
		overall_median, color="red", linestyle="--", linewidth=1,
		label=f"Overall median ({overall_median:.3g})",
	)

	for i, cat in enumerate(order, start=1):
		label = f"n={counts[cat]:,}"
		if deltas:
			star = "*" if deltas[cat]["significant"] else ""
			label += f"  δ={deltas[cat]['delta']:+.2f}{star}"
		ax.text(
			1.01, i, label, va="center", ha="left",
			fontsize=7, transform=ax.get_yaxis_transform(),
		)

	ax.set_xlabel(label_name)
	ax.set_title(
		f"{title}  (n = {len(df):,} STRs)\n"
		f"Kruskal-Wallis: H={kw['h_stat']:.1f}, df={kw['k'] - 1}, p={p_str}, "
		rf"$\epsilon^2$={kw['eps2']:.3f}, $\eta^2_H$={kw['eta2_h']:.3f}"
	)
	ax.legend(loc="lower right", fontsize=8)
	plt.tight_layout()
	_show_or_save(fig, f"box_{title}", save_dir)

	return kw, order, deltas


def plot_cliffs_delta_summary(order, deltas, color_map, title, save_dir):
	"""Standalone horizontal bar chart of one-vs-rest Cliff's delta per
	category -- a companion to plot_boxplots_by_category using the same
	category order (sorted by label median) and colors, so the two figures
	line up visually. Bars extend from 0; each is labeled with its signed
	delta, starred if significant (post multiple-testing correction, per
	CLIFFS_DELTA_CORRECTION/CLIFFS_DELTA_ALPHA). x-axis fixed to [-1, 1]
	(Cliff's delta's natural range), so magnitudes are comparable across
	plots."""
	delta_vals = [deltas[cat]["delta"] for cat in order]
	sig_flags = [deltas[cat]["significant"] for cat in order]

	fig_height = max(4, 0.28 * len(order) + 1.5)
	fig, ax = plt.subplots(figsize=(8, fig_height))

	colors = [color_map.get(cat, "#cccccc") for cat in order]
	ax.barh(
		range(1, len(order) + 1), delta_vals, color=colors,
		edgecolor="black", linewidth=0.6, height=0.7,
	)

	for i, (d, sig) in enumerate(zip(delta_vals, sig_flags), start=1):
		star = "*" if sig else ""
		ax.text(
			d + (0.01 if d >= 0 else -0.01), i, f"{d:+.2f}{star}",
			va="center", ha="left" if d >= 0 else "right", fontsize=7,
		)

	ax.axvline(0, color="black", linewidth=0.8)
	ax.set_yticks(range(1, len(order) + 1))
	ax.set_yticklabels(order, fontsize=8)
	ax.set_ylim(0.3, len(order) + 0.7)
	ax.set_xlim(-1, 1)
	ax.set_xlabel("Cliff's delta (one-vs-rest)")
	ax.set_title(
		f"{title}\nCliff's delta by category "
		f"(* = significant, {correction_label(CLIFFS_DELTA_CORRECTION, CLIFFS_DELTA_ALPHA)})"
	)
	plt.tight_layout()
	_show_or_save(fig, f"cliffs_delta_{title}", save_dir)


def plot_effect_size_summary(results, metric, ylabel, title, save_dir):
	"""Bar plot comparing effect size `metric` ("eps2" or "eta2_h") across
	the (label, category_col) combinations in `results` (each a dict as
	returned by plot_boxplots_by_category / kruskal_wallis_effect_size, plus
	`label_name` and `category_col` keys), with p-values annotated.

	eta2_h can be negative (see kruskal_wallis_effect_size), so bars are
	drawn from a zero baseline and p-value labels flip above/below the bar
	depending on sign.
	"""
	bar_labels = [f"{r['label_name']}\nby {r['category_col']}" for r in results]
	vals = [r[metric] for r in results]

	fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(results)), 5))
	bars = ax.bar(bar_labels, vals, color="#4C72B0", edgecolor="black", linewidth=0.6)
	ax.axhline(0, color="black", linewidth=0.8)

	for bar, r in zip(bars, results):
		height = bar.get_height()
		ax.text(
			bar.get_x() + bar.get_width() / 2, height,
			f"p={format_p_value(r['p_value'])}",
			ha="center", va="bottom" if height >= 0 else "top", fontsize=8,
		)

	ax.set_ylabel(ylabel)
	ax.set_title(title)
	plt.tight_layout()
	_show_or_save(fig, f"effect_size_summary_{metric}", save_dir)


def plot_hexbin(agg_values, labels, annotation_type, label_name, corr, title, save_dir):
	mask = ~np.isnan(agg_values)
	fig, ax = plt.subplots(figsize=(7, 6))
	hb = ax.hexbin(agg_values[mask], labels[mask], gridsize=40, cmap="viridis", mincnt=1)
	fig.colorbar(hb, ax=ax, label="count")
	ax.set_xlabel(annotation_type)
	ax.set_ylabel(label_name)
	ax.set_title(
		f"{title}  (n = {corr['n']:,} STRs)\n"
		f"Spearman $\\rho$={corr['rho']:.3f} (p={format_p_value(corr['p_rho'])})   "
		f"Kendall $\\tau_b$={corr['tau']:.3f} (p={format_p_value(corr['p_tau'])})"
	)
	plt.tight_layout()
	_show_or_save(fig, f"hexbin_{annotation_type}_{label_name}", save_dir)


def plot_continuous_effect_size_summary(results, title, save_dir, use_abs=True):
	"""Grouped bar plot comparing Spearman rho and Kendall tau-b across the
	(label, annotation_type) combinations in `results` (each a dict as
	returned by compute_continuous_correlation, plus `annotation_type` and
	`label_name` keys), with p-values annotated.

	If use_abs (default), bars show |rho|/|tau| -- both then always >=0, so
	no zero-baseline sign-flip logic is needed. If False, bars show the
	signed correlation (direction as well as strength), can go negative, and
	are drawn from a zero baseline with p-value labels flipped above/below
	per sign, mirroring plot_effect_size_summary's eta2_h handling."""
	bar_labels = [f"{r['label_name']}\nby {r['annotation_type']}" for r in results]
	x = np.arange(len(results))
	width = 0.35
	transform = abs if use_abs else (lambda v: v)

	fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(results)), 5))
	bars_rho = ax.bar(
		x - width / 2, [transform(r["rho"]) for r in results], width,
		label=r"$|\rho|$ (Spearman)" if use_abs else r"$\rho$ (Spearman)",
		color="#4C72B0", edgecolor="black", linewidth=0.6,
	)
	bars_tau = ax.bar(
		x + width / 2, [transform(r["tau"]) for r in results], width,
		label=r"$|\tau_b|$ (Kendall)" if use_abs else r"$\tau_b$ (Kendall)",
		color="#DD8452", edgecolor="black", linewidth=0.6,
	)
	if not use_abs:
		ax.axhline(0, color="black", linewidth=0.8)

	for bar, r in zip(bars_rho, results):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2, height, f"p={format_p_value(r['p_rho'])}",
			ha="center", va="bottom" if height >= 0 else "top", fontsize=7)
	for bar, r in zip(bars_tau, results):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2, height, f"p={format_p_value(r['p_tau'])}",
			ha="center", va="bottom" if height >= 0 else "top", fontsize=7)

	ax.set_xticks(x)
	ax.set_xticklabels(bar_labels)
	ax.set_ylabel("|correlation|" if use_abs else "correlation")
	ax.set_title(title)
	ax.legend(loc="best", fontsize=8)
	plt.tight_layout()
	name = "continuous_effect_size_summary" if use_abs else "continuous_effect_size_summary_signed"
	_show_or_save(fig, name, save_dir)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

	sns.set_theme(style="whitegrid")
	config = au.load_config()

	print(f"Loading STR data (motif length {STR_LEN})...")
	str_df = load_str_data(STR_DATA_PATH)
	print(f"  {len(str_df):,} loci")

	discrete_results = []
	continuous_results = []

	members = resolve_family(ANNOTATION_SELECTION, config)
	value_type = config[members[0]]["value_type"]
	print(f"\n{ANNOTATION_SELECTION!r} -> members {members} (value_type={value_type})")

	if value_type == "discrete":
		for member in members:
			entry = au.get_entry(member, config=config)
			value_column = entry["value_column"]

			print(f"  Loading {member}...")
			bed_df = au.load_annotation(member, config=config)

			assigned = assign_discrete_category(str_df, bed_df, value_column)
			assigned = drop_unassigned(assigned, value_column)
			print(
				f"  {len(assigned):,} / {len(str_df):,} loci assigned, "
				f"{assigned[value_column].nunique()} categories represented"
			)

			color_map = build_color_map(bed_df, value_column, MAX_LEGEND_CATEGORIES)
			if len(color_map) <= MAX_LEGEND_CATEGORIES:
				plot_category_legend(color_map, title=member, save_dir=SAVE_DIR)

			for target_name, label_col in TARGETS.items():
				print(f"  Plotting {target_name} ({label_col}) by {member}...")
				box_title = f"{label_col} by {member} (STR length {STR_LEN})"
				kw, order, deltas = plot_boxplots_by_category(
					assigned, label_col, label_col, value_column, color_map,
					title=box_title, save_dir=SAVE_DIR,
				)
				discrete_results.append({**kw, "label_name": label_col, "category_col": member})
				if deltas:
					plot_cliffs_delta_summary(order, deltas, color_map, title=box_title, save_dir=SAVE_DIR)

	elif value_type == "continuous":
		for member in members:
			entry = au.get_entry(member, config=config)

			print(f"  Loading {member}...")
			bw = au.load_bigwig(entry)
			try:
				agg = aggregate_continuous(bw, str_df, CONTINUOUS_METRIC, CONTINUOUS_ABS)
			finally:
				bw.close()
			n_valid = int((~np.isnan(agg)).sum())
			print(f"  {n_valid:,} / {len(str_df):,} loci have a valid aggregated value")

			for target_name, label_col in TARGETS.items():
				labels = str_df[label_col].to_numpy()
				corr = compute_continuous_correlation(agg, labels)
				print(
					f"  {label_col} vs {member}: n={corr['n']:,}, "
					f"rho={corr['rho']:.3f} (p={format_p_value(corr['p_rho'])}), "
					f"tau={corr['tau']:.3f} (p={format_p_value(corr['p_tau'])})"
				)
				plot_hexbin(
					agg, labels, member, label_col, corr,
					title=f"{label_col} vs {member} (STR length {STR_LEN}, "
						f"metric={CONTINUOUS_METRIC}, abs={CONTINUOUS_ABS})",
					save_dir=SAVE_DIR,
				)
				continuous_results.append({**corr, "annotation_type": member, "label_name": label_col})

	else:
		raise ValueError(f"Unknown value_type {value_type!r} for selection {ANNOTATION_SELECTION!r}")

	if discrete_results:
		plot_effect_size_summary(
			discrete_results, metric="eps2", ylabel=r"Kruskal-Wallis $\epsilon^2$",
			title=r"Kruskal-Wallis effect size ($\epsilon^2$) by annotation", save_dir=SAVE_DIR,
		)
		plot_effect_size_summary(
			discrete_results, metric="eta2_h", ylabel=r"Kruskal-Wallis $\eta^2_H$",
			title=r"Kruskal-Wallis effect size ($\eta^2_H$) by annotation", save_dir=SAVE_DIR,
		)
	if continuous_results:
		plot_continuous_effect_size_summary(
			continuous_results, title="Correlation strength by annotation", save_dir=SAVE_DIR,
		)
		plot_continuous_effect_size_summary(
			continuous_results, title="Correlation strength by annotation (signed)", save_dir=SAVE_DIR,
			use_abs=False,
		)
