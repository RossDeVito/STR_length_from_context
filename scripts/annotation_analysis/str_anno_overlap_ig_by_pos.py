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

For each flanking position (STR/prompt positions are skipped, matching
ig_by_pos.py's gap convention), a Kruskal-Wallis test compares the groups'
attribution values at that position; p-values are FDR-BH corrected across all
tested positions (Bonferroni would be far too conservative given adjacent
positions are highly spatially correlated). Significant positions are shown
as a shaded strip below the main line plot.

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

# Per-position Kruskal-Wallis multiple-testing correction across all tested
# flanking positions. None | "bonferroni" | "fdr_bh" (Benjamini-Hochberg).
KW_CORRECTION = "bonferroni"
KW_ALPHA = 0.01

# Smoothing: rolling average window applied to plotted mean/CI lines only
# (never to the p-values behind the significance strip). Set to None or 1 to
# disable.
SMOOTHING_WINDOW = None

FIG_WIDTH = 20
FIG_HEIGHT = 6

SAVE_PATH = None  # None -> plt.show(); else save PNG here

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

n_samples, seq_len = attributions.shape

# ============================================================================
# Region boundaries (mirrors ig_by_pos.py)
# ============================================================================

n_prefix = layout["n_prefix_prompt"]
n_left_flank = layout["n_flanking_bp"]
n_left_str = layout["n_str_bp"]
n_str_prompt_region = layout["n_str_prompt"]
n_right_str = layout["n_str_bp"]
n_right_flank = layout["n_flanking_bp"]

lf_start = n_prefix
lf_end = n_prefix + n_left_flank
rf_start = n_prefix + n_left_flank + n_left_str + n_str_prompt_region + n_right_str
rf_end = rf_start + n_right_flank

region_blocks = [
	("prefix_prompt", 0, n_prefix),
	("left_flank",    lf_start, lf_end),
	("left_str",      lf_end, lf_end + n_left_str),
	("str_prompt",    lf_end + n_left_str,
	                   lf_end + n_left_str + n_str_prompt_region),
	("right_str",     lf_end + n_left_str + n_str_prompt_region,
	                   rf_start),
	("right_flank",   rf_start, rf_end),
]

# ============================================================================
# Filter samples, pair forward/RC by locus, symmetrize (mirrors ig_by_pos.py)
# ============================================================================

raw_pred_diffs = raw_predictions - raw_baseline_predictions
abs_pred_diffs = np.abs(raw_pred_diffs)

mask = np.ones(n_samples, dtype=bool)
if PRED_DIFF_THRESHOLD > 0:
	mask &= abs_pred_diffs > PRED_DIFF_THRESHOLD
if REL_DELTA_THRESHOLD is not None:
	mask &= relative_deltas <= REL_DELTA_THRESHOLD

fwd_indices = np.where(~rev_comp)[0]
rc_indices = np.where(rev_comp)[0]
assert len(fwd_indices) == len(rc_indices), (
	f"Unequal forward ({len(fwd_indices)}) and RC ({len(rc_indices)}) "
	f"sample counts. Cannot form 1:1 locus pairs."
)
assert np.array_equal(ids[fwd_indices], ids[rc_indices]), (
	"Forward/RC index pairing does not line up by id -- fwd_indices[k] and "
	"rc_indices[k] must be the same locus."
)

pair_mask = mask[fwd_indices] & mask[rc_indices]
n_paired_kept = int(np.sum(pair_mask))
print(f"{n_paired_kept} / {len(fwd_indices)} loci kept (both fwd and RC passed filters)")
if n_paired_kept == 0:
	raise ValueError("No locus pairs passed filters. Adjust thresholds.")

kept_fwd_idx = fwd_indices[pair_mask]
kept_rc_idx = rc_indices[pair_mask]
ids_kept = ids[kept_fwd_idx]

fwd_abs_norm = np.abs(attributions[kept_fwd_idx] / raw_pred_diffs[kept_fwd_idx, None])
rc_abs_norm = np.abs(attributions[kept_rc_idx] / raw_pred_diffs[kept_rc_idx, None])

avg_abs_norm = np.zeros_like(fwd_abs_norm)
avg_abs_norm[:, lf_start:lf_end] = (
	fwd_abs_norm[:, lf_start:lf_end] + rc_abs_norm[:, rf_start:rf_end][:, ::-1]
) / 2
avg_abs_norm[:, rf_start:rf_end] = (
	fwd_abs_norm[:, rf_start:rf_end] + rc_abs_norm[:, lf_start:lf_end][:, ::-1]
) / 2

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
	avg_abs_norm = avg_abs_norm[keep]
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
	avg_abs_norm = avg_abs_norm[keep]

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

# ============================================================================
# Per-position bootstrap CI (per group) and Kruskal-Wallis test
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


rng = np.random.default_rng(seed=42)

print(f"\nBootstrapping {len(group_order)} group(s) ({N_BOOTSTRAP} iterations each)...")
group_stats = {}
for g in group_order:
	g_mask = group_labels == g
	mean, ci_lo, ci_hi = bootstrap_ci(avg_abs_norm[g_mask], N_BOOTSTRAP, CI_LEVEL, rng)
	group_stats[g] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(g_mask.sum())}

flank_positions = sorted(set(range(lf_start, lf_end)) | set(range(rf_start, rf_end)))

print(f"Running per-position Kruskal-Wallis across {len(group_order)} groups "
      f"at {len(flank_positions)} flanking positions...")
kw_p = np.full(seq_len, np.nan)
kw_h = np.full(seq_len, np.nan)
for pos in trange(len(flank_positions)):
	col = flank_positions[pos]
	groups_at_pos = [avg_abs_norm[group_labels == g, col] for g in group_order]
	h_stat, p_value = kruskal(*groups_at_pos)
	kw_h[col] = h_stat
	kw_p[col] = p_value

tested_cols = np.array(flank_positions)
if KW_CORRECTION:
	reject, p_corrected, _, _ = multipletests(
		kw_p[tested_cols], alpha=KW_ALPHA, method=KW_CORRECTION
	)
else:
	p_corrected = kw_p[tested_cols]
	reject = p_corrected <= KW_ALPHA

kw_p_corrected = np.full(seq_len, np.nan)
kw_p_corrected[tested_cols] = p_corrected
significant = np.zeros(seq_len, dtype=bool)
significant[tested_cols] = reject

n_sig = int(reject.sum())
print(f"  {n_sig} / {len(tested_cols)} positions significant "
      f"({100 * n_sig / len(tested_cols):.1f}%) at alpha={KW_ALPHA} "
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


fig, (ax, ax_sig) = plt.subplots(
	2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True,
	gridspec_kw={"height_ratios": [8, 1]},
)

for label, start, end in region_blocks_rel:
	color = region_colors.get(label, "#ffffff")
	ax.axvspan(start, end, alpha=0.2, color=color, zorder=0)

for g in group_order:
	color = color_map.get(g, "#888888")
	stats = group_stats[g]
	ax.fill_between(relative_positions, stats["ci_lo"], stats["ci_hi"],
	                 alpha=0.25, color=color, zorder=1)
	ax.plot(relative_positions, stats["mean"], color=color, linewidth=1.0,
	        label=f"{g} (n={stats['n']})", zorder=2)

for label, start, end in region_blocks_rel:
	if start > -(n_left_flank + n_prefix):
		ax.axvline(start, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)

ylim = ax.get_ylim()
label_y_top = ylim[1] * 0.98
label_y_text = label_y_top * 1.12
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
ax.set_ylabel("|Normalized attribution|\n(fraction of total prediction shift)")
smoothing_str = f", smoothing={SMOOTHING_WINDOW}" if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1 else ""
ax.set_title(
	f"[{TASK}] IG attribution magnitude by {member} ({n_final} loci, "
	f"{int(CI_LEVEL*100)}% CI){smoothing_str}\n"
	f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, rel_delta_threshold={REL_DELTA_THRESHOLD}, "
	f"Kruskal-Wallis {n_sig}/{len(tested_cols)} positions significant "
	f"(alpha={KW_ALPHA}, {KW_CORRECTION})"
)
ax.set_ylim(bottom=0)
ylim = ax.get_ylim()
ax.set_ylim(bottom=0, top=ylim[1] * 1.15)

# --- significance strip ---
# tested_cols only spans left_flank/right_flank columns, and relative_positions
# is monotonic within each of those regions, so a run of consecutive
# significant *columns* is also a contiguous run in relative-position space
# (never straddles the STR gap). Draw one axvspan per run, mirroring the
# region-shading axvspans above.
sig_cols = tested_cols[significant[tested_cols]]
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

# Symmetric tick labels despite the right-side offset (mirrors ig_by_pos.py)
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

plt.tight_layout()

if SAVE_PATH:
	fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
	print(f"\nSaved to {SAVE_PATH}")
else:
	plt.show()
