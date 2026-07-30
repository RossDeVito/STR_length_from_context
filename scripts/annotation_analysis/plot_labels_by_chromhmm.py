""" Look at how STR labels (length / heterozygosity) vary with the genomic
annotation an STR overlaps.

For a configurable STR length, loads the HipSTR-labeled STRs and assigns each
locus the full-stack ChromHMM (https://doi.org/10.1186/s13059-021-02572-z)
segment it overlaps the most (by bp). Segments are mapped to a short state
name and one of 16 coarser state groups via state_annotations_processed.csv.

For each label (length = mode_copy_number, variation = heterozygosity),
produces two horizontal box plots -- one box per state, one box per group --
sorted by median, colored by group, with n annotated per box and a reference
line at the overall median. Plots are shown, not saved.

Configuration is via the editable constants below (no argparse), mirroring
scripts/linear_baseline/interpret_coefficients.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import kruskal


# ===========================================================================
# Configuration
# ===========================================================================

STR_LEN = 4
N_FLANKING = 10000

STR_DATA_PATH = f"../../data/STR_data/HipSTR_labeled_STRs/str_len_{STR_LEN}_n_flanking_{N_FLANKING}.tsv"
CHROMHMM_BED_PATH = "../../data/annotations/full_stack_ChromHMM/hg38_genome_100_segments.bed.gz"
STATE_ANNOTATIONS_PATH = "../../data/annotations/full_stack_ChromHMM/state_annotations_processed.csv"

# Targets, mirroring scripts/eval_preds/get_mean_baseline_performance.py
TARGETS = {
	"length": "mode_copy_number",
	"variation": "heterozygosity",
}

SHOW_FLIERS = False  # hide outlier points in the box plots (can get busy with ~100 states)


# ===========================================================================
# Loading
# ===========================================================================

def load_str_data(path):
	"""Load HipSTR-labeled STRs, keeping one row per locus (drop the
	reverse-complement duplicate; it shares the same position/labels)."""
	df = pd.read_csv(path, sep="\t")
	df = df[~df["rev_comp"]].reset_index(drop=True)
	return df


def load_chromhmm_segments(path):
	"""Load the ChromHMM bed segments. Segment names are
	"{ChromHMM state number}_{mneumonic}", but that leading number is the
	model's internal state order -- it does NOT match the `state` column in
	state_annotations_processed.csv (which uses a different, group-sorted
	numbering). Only the mneumonic is a stable join key between the two
	files, so pull it out here.
	"""
	df = pd.read_csv(
		path, sep="\t", header=None, names=["chrom", "start", "end", "segment_name"],
		compression="gzip",
	)
	df["mneumonics"] = df["segment_name"].str.split("_", n=1).str[1]
	return df


def load_state_annotations(path):
	return pd.read_csv(path)


# ===========================================================================
# Assign each STR its best-overlap ChromHMM segment
# ===========================================================================

def assign_chromhmm_segment(str_df, bed_df):
	"""Add `state` / `chromhmm_overlap_bp` columns: the mneumonic of the
	ChromHMM segment each STR overlaps the most (by bp), and that overlap
	length.

	ChromHMM segments are a non-overlapping, sorted-by-start partition per
	chromosome (with occasional unannotated gaps), so for each STR we can
	binary-search for the (small) range of segments it could touch instead of
	scanning the whole 6.1M-row bed table.
	"""
	n = len(str_df)
	seg_states = np.full(n, None, dtype=object)
	seg_overlap = np.zeros(n, dtype=np.int64)

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
		b_states = bed_sub["mneumonics"].to_numpy()

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
				j = l
				ov = min(b_ends[j], qe[k]) - max(b_starts[j], qs[k])
			else:
				js = np.arange(l, h)
				ovs = np.minimum(b_ends[js], qe[k]) - np.maximum(b_starts[js], qs[k])
				j = js[np.argmax(ovs)]
				ov = ovs.max()
			seg_states[str_positions[k]] = b_states[j]
			seg_overlap[str_positions[k]] = ov

	out = str_df.copy()
	out["state"] = seg_states
	out["chromhmm_overlap_bp"] = seg_overlap
	return out


def annotate_with_states(str_df, state_df):
	"""Merge in group metadata for each STR's assigned state. Drops STRs that
	couldn't be assigned a state (fell in an unannotated gap)."""
	n_unassigned = str_df["state"].isna().sum()
	if n_unassigned:
		print(f"  Dropping {n_unassigned} / {len(str_df)} STRs with no overlapping ChromHMM segment")
	df = str_df.dropna(subset=["state"]).reset_index(drop=True)

	group_lookup = state_df.set_index("mneumonics")["Group"]
	df = df.join(group_lookup, on="state")
	df = df.rename(columns={"Group": "group"})
	return df


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


# ===========================================================================
# Plotting
# ===========================================================================

def plot_boxplots_by_category(df, label_col, label_name, category_col, color_map, title):
	"""Horizontal box plots of `label_col`, one box per `category_col` value,
	sorted by median (highest at top), colored by `color_map`, with n
	annotated, a reference line at the overall median, and the omnibus
	Kruskal-Wallis test (H, p, epsilon-squared, eta-squared_H) across all
	categories shown in the title. Returns the test result dict from
	kruskal_wallis_effect_size."""
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
		ax.text(
			1.01, i, f"n={counts[cat]:,}", va="center", ha="left",
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
	plt.show()

	return kw


def plot_effect_size_summary(results, metric, ylabel, title):
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
	plt.show()


def plot_group_legend(group_color_map, title="ChromHMM state group"):
	"""Standalone legend mapping group name -> color, since the state-level
	plot's y-tick labels are individual states, not groups."""
	fig, ax = plt.subplots(figsize=(4, 0.25 * len(group_color_map) + 0.5))
	handles = [
		Patch(facecolor=color, edgecolor="black", linewidth=0.6, label=group)
		for group, color in group_color_map.items()
	]
	ax.legend(handles=handles, loc="center", frameon=False, title=title, fontsize=8)
	ax.axis("off")
	plt.tight_layout()
	plt.show()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

	print(f"Loading STR data (motif length {STR_LEN})...")
	str_df = load_str_data(STR_DATA_PATH)
	print(f"  {len(str_df):,} loci")

	print("Loading ChromHMM segments...")
	bed_df = load_chromhmm_segments(CHROMHMM_BED_PATH)
	state_df = load_state_annotations(STATE_ANNOTATIONS_PATH)

	print("Assigning each STR its best-overlap ChromHMM segment...")
	str_df = assign_chromhmm_segment(str_df, bed_df)
	str_df = annotate_with_states(str_df, state_df)
	print(f"  {len(str_df):,} loci assigned, {str_df['state'].nunique()} states / "
		f"{str_df['group'].nunique()} groups represented")

	# One color per group (a few states share a near-identical group color in
	# the source csv with a 1-bit hex difference; canonicalize to one).
	group_color_map = state_df.drop_duplicates("Group").set_index("Group")["color"].to_dict()
	state_color_map = state_df.drop_duplicates("mneumonics").set_index("mneumonics")["Group"].map(group_color_map).to_dict()

	plot_group_legend(group_color_map)

	effect_size_results = []
	for target_name, label_col in TARGETS.items():
		print(f"\nPlotting {target_name} ({label_col})...")
		kw_state = plot_boxplots_by_category(
			str_df, label_col, label_col, "state", state_color_map,
			title=f"{label_col} by ChromHMM state (STR length {STR_LEN})",
		)
		kw_group = plot_boxplots_by_category(
			str_df, label_col, label_col, "group", group_color_map,
			title=f"{label_col} by ChromHMM state group (STR length {STR_LEN})",
		)
		effect_size_results.append({**kw_state, "label_name": label_col, "category_col": "state"})
		effect_size_results.append({**kw_group, "label_name": label_col, "category_col": "group"})

	plot_effect_size_summary(
		effect_size_results, metric="eps2", ylabel=r"Kruskal-Wallis $\epsilon^2$",
		title=r"Kruskal-Wallis effect size ($\epsilon^2$) by grouping",
	)
	plot_effect_size_summary(
		effect_size_results, metric="eta2_h", ylabel=r"Kruskal-Wallis $\eta^2_H$",
		title=r"Kruskal-Wallis effect size ($\eta^2_H$) by grouping",
	)
