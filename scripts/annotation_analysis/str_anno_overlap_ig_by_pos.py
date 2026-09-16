"""Plot per-position IG attribution magnitude, split into lines by an
annotation-derived group, with a per-position Kruskal-Wallis test of whether
the groups differ.

Combines scripts/ig/ig_by_pos.py (per-position |IG attribution| with a
bootstrap CI, for the RC-symmetrized per-locus average) with
str_anno_overlap_analysis.py's per-locus annotation-group assignment
(discrete: dominant-overlap ChromHMM category; continuous: an aggregated
bigwig value over the STR's own span, quantile-binned). The annotation only
determines group membership -- the plotted/tested quantity is always IG
attribution.

Only the RC-symmetrized per-locus population is used (one independent
observation per locus): the raw fwd+RC-combined population has each locus
twice, and those two rows are correlated, which would violate the
independence assumption of the per-position Kruskal-Wallis test and understate
the true p-values.

Three figures are produced, each an independent main-panel + Kruskal-Wallis
significance-strip pair (own KW test run on whatever matrix that panel
actually plots, so the strip always describes the data next to it):

1. Raw magnitude (SAVE_PATH): |normalized attribution| by position.

2. Share, per-position normalized (SHOW_SHARE_NORMALIZED / SAVE_PATH_SHARE):
   the same quantity on the same position axis, but each group's
   per-position mean is divided by the pooled (all-groups) per-position
   mean. The raw profile is dominated by a shared shape (decay with
   distance from the STR, prompt-proximity artifacts) that has nothing to
   do with the annotation and would dwarf any group difference in panel
   1; dividing it out isolates whether a group carries more or less
   attribution than average at each position, so the line is centered
   near 1.0. Because that rescaling is a per-position positive constant,
   it leaves the Kruskal-Wallis ranks untouched -- panel 2's significance
   strip is panel 1's, reused rather than recomputed.

3. Folded by distance from STR (SHOW_FOLDED / SAVE_PATH_FOLDED): same
   quantity as panel 2, but the upstream and downstream flanks are summed
   (not averaged -- they're different bases being aggregated, not repeat
   measurements of one, so summing is what keeps a locus's total mass
   unchanged) into a single distance-from-STR axis. Folding upstream and
   downstream together was validated as safe in ig_by_pos_label_strat.py
   (profiles correlate at 0.9999, no significant asymmetry) and that
   conclusion is a property of the sequence layout, not of the grouping
   variable, so it carries over unchanged here.

Normalization and the forward/RC combination are the shared ones in
seq_models/ig_attribution_norm.py. NORM_SCHEME picks between "l1_mass"
(each orientation rescaled to unit absolute attribution mass across the
flanks) and "pred_diff" (each orientation divided by the absolute
prediction shift). "l1_mass" matters here for the same reason it does in
scripts/ig/ig_by_pos_label_strat.py: taking an absolute value breaks IG's
sum-to-the-prediction-shift completeness property, so total |attribution|
mass varies locus to locus and can itself correlate with the grouping
annotation -- left in, that shows up as a flat per-group vertical offset
in panels 2 and 3, which would misread as a positional effect when it's
really just a mass difference. The mean mass per group is printed to the
console as a diagnostic: if it varies a lot across groups, the choice of
scheme is doing real work.

IG samples are joined to genomic coordinates via attributions.npz's `ids`
array, which holds the source tsv's `ID` column (shared by a locus's fwd/RC
pair) -- see seq_models/caduceus/data.py / integrated_gradients.py. Set
STR_DATA_PATH to the tsv used to train the checkpoint behind ATTR_DIR.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from tqdm import trange

import seq_models.ig_attribution_norm as ign

import annotation_utils as au
import str_anno_overlap_analysis as saoa

# ============================================================================
# Configuration
# ============================================================================

ATTR_DIR = "../ig/output/caduceus_v0/str3/str3_f5000_fiv_v4"

# Which task's attributions/predictions to use (must be one of the
# checkpoint's active targets -- see {ATTR_DIR}/meta.json "task_names").
# TASK = "length"
TASK = "variation"

# tsv used to train the checkpoint behind ATTR_DIR (see its config.yaml
# "data_path"). Must be the exact file so attributions.npz's `ids` can be
# joined to chrom/str_start/str_end.
STR_DATA_PATH = "../../data/STR_data/HipSTR_labeled_STRs/str_len_3_n_flanking_10000.tsv"

# Annotation to group loci by: an exact annotation_config.yaml key, or a
# family base name (e.g. "ChromHMM" -> expands to ChromHMM_group/_state; all
# members must share one value_type). See resolve_family in
# str_anno_overlap_analysis.py.
# ANNOTATION_SELECTION = "ChromHMM_group"
ANNOTATION_SELECTION = "phyloP"

# Continuous annotations only: number of quantile groups to split loci into,
# by their own STR-span-aggregated value (see CONTINUOUS_METRIC/_ABS below).
N_QUANTILES = 5

# Continuous annotations only: how to aggregate each STR's own span, and
# whether to abs() per-bp before aggregating. Mirrors
# str_anno_overlap_analysis.py's CONTINUOUS_METRIC/CONTINUOUS_ABS.
CONTINUOUS_METRIC = "mean"  # "mean" | "median" | "max" | "min"
CONTINUOUS_ABS = False

# Confidence interval level (0-1) and bootstrap iterations, per group.
CI_LEVEL = 0.95
N_BOOTSTRAP = 10

# Sample filters, mirroring ig_by_pos.py.
PRED_DIFF_THRESHOLD = 0.05
REL_DELTA_THRESHOLD = 0.05

# Per-orientation attribution normalization: "l1_mass" or "pred_diff".
# See the module docstring and seq_models/ig_attribution_norm.py.
NORM_SCHEME = "l1_mass"

# Per-position Kruskal-Wallis multiple-testing correction across all tested
# flanking positions. None | "bonferroni" | "fdr_bh" (Benjamini-Hochberg).
KW_CORRECTION = "bonferroni"
KW_ALPHA = 0.01

# Smoothing: rolling average window applied to plotted mean/CI lines only
# (never to the p-values behind the significance strip). Set to None or 1 to
# disable.
SMOOTHING_WINDOW = 25

# Panel 2: per-position share, normalized by the pooled per-position mean.
SHOW_SHARE_NORMALIZED = True

# Panel 3: same share quantity, folded by distance from STR.
SHOW_FOLDED = True

FIG_WIDTH = 20
FIG_HEIGHT = 6

# None -> plt.show() for that figure; else save PNG here. Each panel saves
# independently; whichever figure(s) have no path still get shown.
SAVE_PATH = None
SAVE_PATH_SHARE = None
SAVE_PATH_FOLDED = None

# ============================================================================
# Load IG attributions and metadata
# ============================================================================

data = np.load(os.path.join(ATTR_DIR, "attributions.npz"), allow_pickle=True)

with open(os.path.join(ATTR_DIR, "meta.json"), "r") as f:
	meta = json.load(f)

layout = meta["sequence_layout"]

attributions = data[f"attributions_{TASK}"]
raw_predictions = data[f"raw_predictions_{TASK}"]
raw_baseline_predictions = data[f"raw_baseline_predictions_{TASK}"]
relative_deltas = data[f"relative_convergence_deltas_{TASK}"]
rev_comp = data["rev_comp"]
ids = data["ids"]

seq_len = attributions.shape[1]

# ============================================================================
# Region boundaries (mirrors ig_by_pos.py)
# ============================================================================

flanks = ign.FlankLayout.from_layout(layout)

n_prefix = flanks.n_prefix
n_left_flank = flanks.n_left_flank
n_left_str = flanks.n_left_str
n_str_prompt_region = flanks.n_str_prompt
n_right_str = flanks.n_right_str
n_right_flank = flanks.n_right_flank

lf_start = flanks.lf_start
lf_end = flanks.lf_end
rf_start = flanks.rf_start
rf_end = flanks.rf_end

region_blocks = flanks.region_blocks

# ============================================================================
# Filter samples, pair forward/RC by locus, symmetrize (mirrors ig_by_pos.py)
# ============================================================================

# Filter, normalize per orientation, flip the RC flanks into forward genomic
# coordinates, average the SIGNED profiles, then take the absolute value.
# See seq_models/ig_attribution_norm.py.
loci = ign.per_locus_attributions(
	attributions, raw_predictions, raw_baseline_predictions, relative_deltas,
	rev_comp, flanks, ids=ids, scheme=NORM_SCHEME,
	pred_diff_threshold=PRED_DIFF_THRESHOLD,
	rel_delta_threshold=REL_DELTA_THRESHOLD,
)

print(f"{loci.n_loci} / {loci.n_loci_total} loci kept "
      f"(both fwd and RC passed filters)")

ids_kept = loci.locus_ids

# ============================================================================
# Join to genomic coordinates and assign per-locus annotation group
# ============================================================================

print(f"\nLoading STR data from {STR_DATA_PATH}...")
tsv = pd.read_csv(STR_DATA_PATH, sep="\t").drop_duplicates("ID").set_index("ID")

missing = pd.Index(ids_kept).difference(tsv.index)
if len(missing) > 0:
	raise ValueError(
		f"{len(missing)} locus id(s) from attributions.npz not found in "
		f"{STR_DATA_PATH!r} (e.g. {missing[:5].tolist()}). Is STR_DATA_PATH "
		f"the tsv that trained the checkpoint behind {ATTR_DIR!r}?"
	)

str_df_kept = tsv.loc[ids_kept, ["chrom", "str_start", "str_end"]].reset_index()

config = au.load_config()
members = saoa.resolve_family(ANNOTATION_SELECTION, config)
value_type = config[members[0]]["value_type"]
if len(members) != 1:
	raise ValueError(
		f"{ANNOTATION_SELECTION!r} resolved to multiple annotation config "
		f"entries {members}; pick one exact key for this plot."
	)
member = members[0]
entry = au.get_entry(member, config=config)
print(f"\nGrouping by {member!r} (value_type={value_type})")


def build_quantile_color_map(bin_labels):
	"""Sequential colormap for ordinal quantile bins (low -> high)."""
	cmap = plt.get_cmap("viridis", len(bin_labels))
	return {label: cmap(i) for i, label in enumerate(bin_labels)}


if value_type == "discrete":
	value_column = entry["value_column"]
	bed_df = au.load_annotation(member, config=config)
	assigned = saoa.assign_discrete_category(str_df_kept, bed_df, value_column)
	keep = ~assigned[value_column].isna().to_numpy()
	n_dropped = int((~keep).sum())
	if n_dropped:
		print(f"  Dropping {n_dropped} / {len(assigned)} loci with no unambiguous "
		      f"{value_column!r} assignment (gap or tie)")
	group_labels = assigned.loc[keep, value_column].to_numpy()
	loci = loci.select(keep)
	color_map = saoa.build_color_map(bed_df, value_column, max_legend_categories=len(set(group_labels)))
	group_order = sorted(set(group_labels))

elif value_type == "continuous":
	bw = au.load_bigwig(entry)
	try:
		agg = saoa.aggregate_continuous(bw, str_df_kept, CONTINUOUS_METRIC, CONTINUOUS_ABS)
	finally:
		bw.close()
	keep = ~np.isnan(agg)
	n_dropped = int((~keep).sum())
	if n_dropped:
		print(f"  Dropping {n_dropped} / {len(agg)} loci with no valid aggregated "
		      f"{member!r} value over their span")
	agg_kept = agg[keep]
	loci = loci.select(keep)

	bin_codes, bin_edges = pd.qcut(agg_kept, q=N_QUANTILES, retbins=True, duplicates="drop")
	bin_labels = [
		f"Q{i+1} [{bin_edges[i]:.3g}, {bin_edges[i+1]:.3g}]"
		for i in range(len(bin_edges) - 1)
	]
	print(f"  Quantile bin edges ({CONTINUOUS_METRIC} {member}, abs={CONTINUOUS_ABS}): "
	      f"{np.round(bin_edges, 3).tolist()}")
	code_to_label = dict(enumerate(bin_labels))
	group_labels = np.array([code_to_label[c] for c in bin_codes.codes])
	group_order = bin_labels
	color_map = build_quantile_color_map(bin_labels)

else:
	raise ValueError(f"Unknown value_type {value_type!r} for {member!r}")

n_final = len(group_labels)
print(f"  {n_final} loci with an assigned group")
for g in group_order:
	print(f"    {g}: n={int(np.sum(group_labels == g))}")

# The forward/RC-averaged |attribution| profile all three panels plot.
avg_abs_norm = loci.avg_abs

# Total |attribution| per locus in units of the prediction shift -- the
# scalar NORM_SCHEME="l1_mass" divides out. Reported, never plotted.
locus_mass = loci.mass_ratio

print(f"\n  Mean total |attribution| mass by group (diagnostic -- if this "
      f"varies a lot across groups, NORM_SCHEME is doing real work; "
      f"currently {NORM_SCHEME!r}):")
for g in group_order:
	print(f"    {g}: mean mass = {locus_mass[group_labels == g].mean():.2f}")

# ============================================================================
# Per-position/per-distance bootstrap CI (per group) and Kruskal-Wallis tests
# ============================================================================


def bootstrap_ci(abs_norm_matrix, n_boot, ci_level, rng):
	"""Compute mean and percentile bootstrap CI per position."""
	n, seq = abs_norm_matrix.shape
	mean = np.mean(abs_norm_matrix, axis=0)

	boot_means = np.zeros((n_boot, seq), dtype=np.float32)
	for b in range(n_boot):
		idx = rng.integers(0, n, size=n)
		boot_means[b] = np.mean(abs_norm_matrix[idx], axis=0)

	alpha = 1 - ci_level
	ci_lo = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
	ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
	return mean, ci_lo, ci_hi


def build_group_stats(matrix, normalize_by_overall_mean):
	"""Bootstrap mean+CI per group; optionally divide by the pooled
	per-column mean so lines show relative deviation from the average
	profile (see module docstring) rather than absolute magnitude.
	"""
	overall_mean = matrix.mean(axis=0)
	stats = {}
	for g in group_order:
		g_mask = group_labels == g
		mean, ci_lo, ci_hi = bootstrap_ci(matrix[g_mask], N_BOOTSTRAP, CI_LEVEL, rng)
		if normalize_by_overall_mean:
			with np.errstate(divide="ignore", invalid="ignore"):
				mean = mean / overall_mean
				ci_lo = ci_lo / overall_mean
				ci_hi = ci_hi / overall_mean
		stats[g] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(g_mask.sum())}
	return stats


def run_kw_test(matrix, cols_to_test, array_len):
	"""Per-column Kruskal-Wallis across group_order, multiple-testing
	corrected across cols_to_test. Returns (tested_cols, significant,
	n_sig).
	"""
	p = np.full(array_len, np.nan)
	for i in trange(len(cols_to_test)):
		col = cols_to_test[i]
		groups_at_col = [matrix[group_labels == g, col] for g in group_order]
		_, p_value = kruskal(*groups_at_col)
		p[col] = p_value

	tested = np.array(cols_to_test)
	if KW_CORRECTION:
		reject, p_corrected, _, _ = multipletests(p[tested], alpha=KW_ALPHA, method=KW_CORRECTION)
	else:
		p_corrected = p[tested]
		reject = p_corrected <= KW_ALPHA

	significant = np.zeros(array_len, dtype=bool)
	significant[tested] = reject
	n_sig = int(reject.sum())
	return tested, significant, n_sig


rng = np.random.default_rng(seed=42)
flank_positions = sorted(set(range(lf_start, lf_end)) | set(range(rf_start, rf_end)))

print(f"\nBootstrapping {len(group_order)} group(s) ({N_BOOTSTRAP} iterations each)...")
group_stats = build_group_stats(avg_abs_norm, normalize_by_overall_mean=False)

print(f"Running per-position Kruskal-Wallis (raw magnitude) across {len(group_order)} "
      f"groups at {len(flank_positions)} flanking positions...")
tested_cols, significant, n_sig = run_kw_test(avg_abs_norm, flank_positions, seq_len)
print(f"  {n_sig} / {len(tested_cols)} positions significant "
      f"({100 * n_sig / len(tested_cols):.1f}%) at alpha={KW_ALPHA} "
      f"(correction={KW_CORRECTION})")

if SHOW_SHARE_NORMALIZED:
	print(f"\nBootstrapping {len(group_order)} group(s) for share-normalized panel...")
	share_group_stats = build_group_stats(avg_abs_norm, normalize_by_overall_mean=True)

	# Panel 2 plots the same matrix as panel 1, rescaled per position by the
	# pooled mean. Dividing a column by a positive constant leaves the
	# within-column ranks untouched, so the Kruskal-Wallis result is
	# identical to panel 1's by construction -- reuse it rather than
	# recomputing seq_len tests for the same answer.
	share_tested_cols, share_significant, share_n_sig = (
		tested_cols, significant, n_sig
	)


def fold_flanks(matrix):
	"""Collapse upstream+downstream flanks into one distance-from-STR
	profile (column d = distance d+1 bp). SUMMED, not averaged -- these
	are two different bases being aggregated, not repeat measurements of
	one, so summing is what leaves a locus's total mass unchanged. See
	module docstring.
	"""
	upstream = matrix[:, lf_start:lf_end][:, ::-1]
	downstream = matrix[:, rf_start:rf_end]
	return upstream + downstream


if SHOW_FOLDED:
	folded_share = fold_flanks(avg_abs_norm)

	print(f"\nBootstrapping {len(group_order)} group(s) for folded panel...")
	folded_group_stats = build_group_stats(folded_share, normalize_by_overall_mean=True)

	print(f"Running per-distance Kruskal-Wallis (folded, mass-normalized share) across "
	      f"{len(group_order)} groups at {n_left_flank} distances...")
	folded_tested_cols, folded_significant, folded_n_sig = run_kw_test(
		folded_share, list(range(n_left_flank)), n_left_flank
	)
	print(f"  {folded_n_sig} / {n_left_flank} distances significant "
	      f"({100 * folded_n_sig / n_left_flank:.1f}%) at alpha={KW_ALPHA} "
	      f"(correction={KW_CORRECTION})")

# ============================================================================
# Relative positions (mirrors ig_by_pos.py)
# ============================================================================

STR_GAP_WIDTH = n_left_str + n_str_prompt_region + n_right_str
gap_start = 0
gap_end = STR_GAP_WIDTH

relative_positions = np.full(seq_len, np.nan)
for i in range(n_left_flank):
	relative_positions[lf_start + i] = -(n_left_flank - i)
for i in range(n_right_flank):
	relative_positions[rf_start + i] = gap_end + (i + 1)

prompt_end = gap_start + n_left_str
right_str_start = gap_end - n_right_str

region_blocks_rel = [
	("prefix_prompt", -(n_left_flank + n_prefix),  -n_left_flank),
	("left_flank",    -n_left_flank,                gap_start),
	("left_str",       gap_start,                   prompt_end),
	("str_prompt",     prompt_end,                   right_str_start),
	("right_str",      right_str_start,              gap_end),
	("right_flank",    gap_end,                      gap_end + n_right_flank + 1),
]

# ============================================================================
# Optional smoothing of plotted lines only (mirrors ig_by_pos.py)
# ============================================================================


def apply_smoothing(mean_arr, ci_lo_arr, ci_hi_arr, window):
	if not window or window <= 1:
		return
	kernel = np.ones(window) / window
	for label, start, end in region_blocks:
		if label in ("left_flank", "right_flank"):
			mean_arr[start:end] = np.convolve(mean_arr[start:end], kernel, mode="same")
			ci_lo_arr[start:end] = np.convolve(ci_lo_arr[start:end], kernel, mode="same")
			ci_hi_arr[start:end] = np.convolve(ci_hi_arr[start:end], kernel, mode="same")


for g in group_order:
	apply_smoothing(group_stats[g]["mean"], group_stats[g]["ci_lo"], group_stats[g]["ci_hi"],
	                 SMOOTHING_WINDOW)

if SHOW_SHARE_NORMALIZED:
	for g in group_order:
		apply_smoothing(share_group_stats[g]["mean"], share_group_stats[g]["ci_lo"],
		                 share_group_stats[g]["ci_hi"], SMOOTHING_WINDOW)

if SHOW_FOLDED:
	def smooth_1d(segment, window):
		"""Rolling average with edge-value padding (not np.convolve's
		implicit zero-padding under mode="same"), so the mean isn't pulled
		toward zero at the segment's ends. ig_by_pos_label_strat.py hit a
		visible dip artifact right at the STR-adjacent edge from exactly
		this zero-padding once smoothing was turned on for a folded panel;
		this avoids reintroducing it here.
		"""
		if not window or window <= 1:
			return segment
		kernel = np.ones(window) / window
		half = window // 2
		padded = np.pad(segment, (half, window - 1 - half), mode="edge")
		return np.convolve(padded, kernel, mode="valid")

	for g in group_order:
		s = folded_group_stats[g]
		s["mean"] = smooth_1d(s["mean"], SMOOTHING_WINDOW)
		s["ci_lo"] = smooth_1d(s["ci_lo"], SMOOTHING_WINDOW)
		s["ci_hi"] = smooth_1d(s["ci_hi"], SMOOTHING_WINDOW)

# ============================================================================
# Plot
# ============================================================================

region_colors = {
	"prefix_prompt": "#d0d0d0",
	"left_flank":    "#ffffff",
	"left_str":      "#e8443a",
	"str_prompt":    "#d0d0d0",
	"right_str":     "#e8443a",
	"right_flank":   "#ffffff",
}

region_display_names = {
	"prefix_prompt": "Prefix Prompt",
	"left_flank":    "Upstream Flank",
	"left_str":      "STR Start",
	"str_prompt":    "STR Prompt",
	"right_str":     "STR End",
	"right_flank":   "Downstream Flank",
}


def plot_position_panel(ax, ax_sig, panel_group_stats, title_str, ylabel_str,
                         panel_tested_cols, panel_significant, show_reference_line):
	"""Plot one main+significance-strip pair, position-indexed (used for
	both the raw-magnitude and share-normalized panels).
	"""

	for label, start, end in region_blocks_rel:
		color = region_colors.get(label, "#ffffff")
		ax.axvspan(start, end, alpha=0.2, color=color, zorder=0)

	if show_reference_line:
		ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

	for g in group_order:
		color = color_map.get(g, "#888888")
		stats = panel_group_stats[g]
		ax.fill_between(relative_positions, stats["ci_lo"], stats["ci_hi"],
		                 alpha=0.25, color=color, zorder=2)
		ax.plot(relative_positions, stats["mean"], color=color, linewidth=1.0,
		        label=f"{g} (n={stats['n']})", zorder=3)

	for label, start, end in region_blocks_rel:
		if start > -(n_left_flank + n_prefix):
			ax.axvline(start, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)

	# Finalize the y-range (with headroom for the region-name labels) now,
	# before placing those labels, so their placement is correct regardless
	# of whether the axis floor sits at 0 (raw magnitude) or floats above/
	# below it (normalized share) -- see ig_by_pos_label_strat.py, where
	# computing labels from the pre-headroom ylim caused them to overlap
	# the title once a panel's y-range didn't start at 0.
	data_bottom, data_top = ax.get_ylim()
	if not show_reference_line:
		data_bottom = 0
	headroom = (data_top - data_bottom) * 0.15
	ax.set_ylim(data_bottom, data_top + headroom)

	label_y_top = data_top + headroom * 0.1
	label_y_text = data_top + headroom * 0.55

	middle_labels = ["left_str", "str_prompt", "right_str"]
	middle_blocks = [(l, s, e) for l, s, e in region_blocks_rel if l in middle_labels]
	if middle_blocks:
		cluster_center = (middle_blocks[0][1] + middle_blocks[-1][2]) / 2
	x_range = (n_left_flank + n_prefix) + STR_GAP_WIDTH + n_right_flank
	fan_offset = x_range * 0.06
	middle_text_x = {
		"left_str":   cluster_center - fan_offset,
		"str_prompt": cluster_center,
		"right_str":  cluster_center + fan_offset,
	}
	line_labels = {"left_str", "right_str"}
	for label, start, end in region_blocks_rel:
		mid = (start + end) / 2
		display_name = region_display_names.get(label, label)
		if label in line_labels:
			text_x = middle_text_x[label]
			ax.annotate(display_name, xy=(mid, label_y_top), xytext=(text_x, label_y_text),
			            ha="center", va="bottom", fontsize=7, alpha=0.7, fontweight="bold",
			            annotation_clip=False,
			            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.5))
		elif label in middle_labels:
			text_x = middle_text_x[label]
			ax.text(text_x, label_y_text, display_name, ha="center", va="bottom",
			        fontsize=7, alpha=0.7, fontweight="bold", clip_on=False)
		else:
			ax.text(mid, label_y_text, display_name, ha="center", va="bottom",
			         fontsize=7, alpha=0.7, fontweight="bold", clip_on=False)

	ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2 if len(group_order) > 8 else 1)
	ax.set_ylabel(ylabel_str)
	smoothing_str = f", smoothing={SMOOTHING_WINDOW}" if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1 else ""
	ax.set_title(title_str + smoothing_str)

	# --- significance strip ---
	# panel_tested_cols only spans left_flank/right_flank columns, and
	# relative_positions is monotonic within each of those regions, so a run
	# of consecutive significant *columns* is also a contiguous run in
	# relative-position space (never straddles the STR gap).
	sig_cols = panel_tested_cols[panel_significant[panel_tested_cols]]
	if len(sig_cols) > 0:
		run_breaks = np.where(np.diff(sig_cols) > 1)[0]
		run_starts = np.concatenate(([0], run_breaks + 1))
		run_ends = np.concatenate((run_breaks, [len(sig_cols) - 1]))
		for s, e in zip(run_starts, run_ends):
			lo = relative_positions[sig_cols[s]]
			hi = relative_positions[sig_cols[e]]
			width = hi - lo if hi > lo else 1.0
			ax_sig.axvspan(lo, lo + width, color="black", alpha=0.8)

	ax_sig.set_yticks([])
	ax_sig.set_ylabel(f"p<{KW_ALPHA}\n({KW_CORRECTION})", fontsize=7, rotation=0,
	                   ha="right", va="center")
	ax_sig.set_xlabel("Position relative to STR (bp)")

	xmin = -(n_left_flank + n_prefix)
	xmax = gap_end + n_right_flank
	tick_step = 500
	left_bp = list(range(tick_step, n_left_flank + 1, tick_step))
	right_bp = list(range(tick_step, n_right_flank + 1, tick_step))
	left_ticks = [-bp for bp in left_bp]
	right_ticks = [gap_end + bp for bp in right_bp]
	all_ticks = left_ticks + right_ticks
	all_labels = [str(-bp) for bp in left_bp] + [str(bp) for bp in right_bp]
	ax_sig.set_xticks(all_ticks)
	ax_sig.set_xticklabels(all_labels)
	ax_sig.set_xlim(xmin, xmax)


def plot_folded_panel(ax, ax_sig, panel_group_stats, title_str, ylabel_str,
                       panel_tested_cols, panel_significant, show_reference_line):
	"""Plot one main+significance-strip pair, distance-from-STR indexed
	(upstream and downstream folded together). Simpler than
	plot_position_panel: no STR gap to straddle, no region blocks to shade.
	"""
	distances = np.arange(1, n_left_flank + 1)

	if show_reference_line:
		ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

	for g in group_order:
		color = color_map.get(g, "#888888")
		stats = panel_group_stats[g]
		ax.fill_between(distances, stats["ci_lo"], stats["ci_hi"], alpha=0.25, color=color, zorder=2)
		ax.plot(distances, stats["mean"], color=color, linewidth=1.0,
		        label=f"{g} (n={stats['n']})", zorder=3)

	if not show_reference_line:
		ax.set_ylim(bottom=0)

	ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2 if len(group_order) > 8 else 1)
	ax.set_ylabel(ylabel_str)
	smoothing_str = f", smoothing={SMOOTHING_WINDOW}" if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1 else ""
	ax.set_title(title_str + smoothing_str)

	# --- significance strip ---
	sig_cols = panel_tested_cols[panel_significant[panel_tested_cols]]
	if len(sig_cols) > 0:
		run_breaks = np.where(np.diff(sig_cols) > 1)[0]
		run_starts = np.concatenate(([0], run_breaks + 1))
		run_ends = np.concatenate((run_breaks, [len(sig_cols) - 1]))
		for s, e in zip(run_starts, run_ends):
			lo = distances[sig_cols[s]]
			hi = distances[sig_cols[e]]
			width = hi - lo if hi > lo else 1.0
			ax_sig.axvspan(lo, lo + width, color="black", alpha=0.8)

	ax_sig.set_yticks([])
	ax_sig.set_ylabel(f"p<{KW_ALPHA}\n({KW_CORRECTION})", fontsize=7, rotation=0,
	                   ha="right", va="center")
	ax_sig.set_xlabel("Distance from STR (bp, upstream and downstream folded)")
	ax_sig.set_xticks(list(range(0, n_left_flank + 1, 500)))
	ax_sig.set_xlim(1, n_left_flank)


mass_str = ign.scheme_description(NORM_SCHEME)

# --- Panel 1: raw magnitude ---

fig1, (ax1, ax1_sig) = plt.subplots(
	2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True,
	gridspec_kw={"height_ratios": [8, 1]},
)
title1 = (
	f"[{TASK}] IG attribution magnitude by {member} ({n_final} loci, "
	f"{int(CI_LEVEL*100)}% CI)\n"
	f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, rel_delta_threshold={REL_DELTA_THRESHOLD}, "
	f"{mass_str}, Kruskal-Wallis {n_sig}/{len(tested_cols)} positions significant "
	f"(alpha={KW_ALPHA}, {KW_CORRECTION})"
)
plot_position_panel(
	ax1, ax1_sig, group_stats, title1,
	ign.magnitude_axis_label(NORM_SCHEME),
	tested_cols, significant, show_reference_line=False,
)
fig1.tight_layout()
if SAVE_PATH:
	fig1.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
	print(f"\nSaved to {SAVE_PATH}")

# --- Panel 2: share, per-position normalized ---

if SHOW_SHARE_NORMALIZED:
	fig2, (ax2, ax2_sig) = plt.subplots(
		2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True,
		gridspec_kw={"height_ratios": [8, 1]},
	)
	title2 = (
		f"[{TASK}] IG attribution share by {member}, per-position normalized "
		f"({n_final} loci, {int(CI_LEVEL*100)}% CI)\n"
		f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, rel_delta_threshold={REL_DELTA_THRESHOLD}, "
		f"{mass_str}, Kruskal-Wallis {share_n_sig}/{len(share_tested_cols)} positions "
		f"significant (alpha={KW_ALPHA}, {KW_CORRECTION})"
	)
	plot_position_panel(
		ax2, ax2_sig, share_group_stats, title2,
		"Relative share of attribution mass\n(group mean / overall mean)",
		share_tested_cols, share_significant, show_reference_line=True,
	)
	fig2.tight_layout()
	if SAVE_PATH_SHARE:
		fig2.savefig(SAVE_PATH_SHARE, dpi=150, bbox_inches="tight")
		print(f"Saved to {SAVE_PATH_SHARE}")

# --- Panel 3: share, folded by distance from STR ---

if SHOW_FOLDED:
	fig3, (ax3, ax3_sig) = plt.subplots(
		2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True,
		gridspec_kw={"height_ratios": [8, 1]},
	)
	title3 = (
		f"[{TASK}] IG attribution share by {member}, folded by distance from STR "
		f"(upstream + downstream combined, {n_final} loci, {int(CI_LEVEL*100)}% CI)\n"
		f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, rel_delta_threshold={REL_DELTA_THRESHOLD}, "
		f"{mass_str}, Kruskal-Wallis {folded_n_sig}/{n_left_flank} distances significant "
		f"(alpha={KW_ALPHA}, {KW_CORRECTION})"
	)
	plot_folded_panel(
		ax3, ax3_sig, folded_group_stats, title3,
		"Relative share of attribution mass\n(group mean / overall mean)",
		folded_tested_cols, folded_significant, show_reference_line=True,
	)
	fig3.tight_layout()
	if SAVE_PATH_FOLDED:
		fig3.savefig(SAVE_PATH_FOLDED, dpi=150, bbox_inches="tight")
		print(f"Saved to {SAVE_PATH_FOLDED}")

need_show = (
	not SAVE_PATH
	or (SHOW_SHARE_NORMALIZED and not SAVE_PATH_SHARE)
	or (SHOW_FOLDED and not SAVE_PATH_FOLDED)
)
if need_show:
	plt.show()
