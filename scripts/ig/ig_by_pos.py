"""Plot per-position normalized IG attribution magnitudes with confidence intervals.

For each position, computes the mean absolute normalized attribution across
samples and plots with a confidence interval band.

Two subplots:
  1. All samples (forward and reverse complement) plotted as-is.
  2. Per-locus average of forward and reverse complement attributions.
     For each locus, the RC flank attributions are flipped to align with
     forward genomic positions, then averaged with the forward attributions.
     This symmetrizes the signal and halves the prompt-proximity artifact,
     which is model-position-dependent rather than genomic-position-dependent.

Normalization: each sample's attributions are divided by the raw prediction
difference (F(input) - F(baseline)), so attributions sum to ~1 per sample
and are comparable across loci with different prediction magnitudes. The
absolute value is then taken as a measure of importance regardless of
direction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
from tqdm import trange

# ============================================================================
# Configuration
# ============================================================================

ATTR_DIR = "output/str2/v1/2000_p128_300steps_rt"

# Confidence interval level (0-1), e.g. 0.95 for 95% CI
CI_LEVEL = 0.95

# Number of bootstrap iterations for CI
N_BOOTSTRAP = 10

# Filter: minimum raw prediction difference to include a sample.
# Samples below this threshold have uninformative flanking sequences
# and inflated relative convergence deltas. Set to 0 to include all.
PRED_DIFF_THRESHOLD = 0.05

# Filter: maximum relative convergence delta to include a sample.
# Set to None to skip this filter.
REL_DELTA_THRESHOLD = None

# Figure size (per subplot)
FIG_WIDTH = 20
FIG_HEIGHT = 5

# Smoothing: rolling average window size. Set to None or 1 to disable.
SMOOTHING_WINDOW = None

# Output path (set to None to just display)
SAVE_PATH = None

# ============================================================================
# Load data and metadata
# ============================================================================

data = np.load(os.path.join(ATTR_DIR, "attributions.npz"), allow_pickle=True)

with open(os.path.join(ATTR_DIR, "meta.json"), "r") as f:
	meta = json.load(f)

layout = meta["sequence_layout"]

attributions = data["attributions"]
raw_predictions = data["raw_predictions"]
raw_baseline_predictions = data["raw_baseline_predictions"]
relative_deltas = data["relative_convergence_deltas"]
labels = data["labels"]
rev_comp = data["rev_comp"]  # boolean: True for reverse complement samples

n_samples, seq_len = attributions.shape

# ============================================================================
# Compute region boundaries
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

# Region blocks in array-index space (for smoothing)
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
# Filter samples
# ============================================================================

raw_pred_diffs = raw_predictions - raw_baseline_predictions
abs_pred_diffs = np.abs(raw_pred_diffs)

mask = np.ones(n_samples, dtype=bool)

excluded_pred_diff = 0
excluded_rel_delta = 0

if PRED_DIFF_THRESHOLD > 0:
	fail_pred = abs_pred_diffs <= PRED_DIFF_THRESHOLD
	excluded_pred_diff = int(np.sum(fail_pred))
	mask &= ~fail_pred

if REL_DELTA_THRESHOLD is not None:
	fail_delta = relative_deltas > REL_DELTA_THRESHOLD
	excluded_rel_delta = int(np.sum(mask & fail_delta))
	mask &= ~fail_delta

n_kept = int(np.sum(mask))
n_filtered = n_samples - n_kept
print(f"Subplot 1 (all samples): {n_kept} kept, {n_filtered} filtered "
      f"(of {n_samples} total)")
print(f"  Excluded by pred_diff <= {PRED_DIFF_THRESHOLD}: {excluded_pred_diff}")
print(f"  Excluded by rel_delta > {REL_DELTA_THRESHOLD}: {excluded_rel_delta}")

if n_kept == 0:
	raise ValueError("No samples passed filters. Adjust thresholds.")

# ============================================================================
# Normalize and take absolute value (all filtered samples, for subplot 1)
# ============================================================================

all_norm = attributions[mask] / raw_pred_diffs[mask, None]
all_abs_norm = np.abs(all_norm)

# ============================================================================
# Pair forward and RC samples by locus for subplot 2
# ============================================================================

# Identify forward and RC indices among ALL samples (before filtering)
fwd_indices = np.where(~rev_comp)[0]
rc_indices = np.where(rev_comp)[0]

assert len(fwd_indices) == len(rc_indices), (
	f"Unequal forward ({len(fwd_indices)}) and RC ({len(rc_indices)}) "
	f"sample counts. Cannot form 1:1 locus pairs."
)

n_loci = len(fwd_indices)
print(f"\nFound {n_loci} loci with forward + RC pairs")

# Pair filter: both forward AND RC must pass for the locus to be included
pair_mask = mask[fwd_indices] & mask[rc_indices]
n_paired_kept = int(np.sum(pair_mask))
print(f"Subplot 2 (averaged): {n_paired_kept} loci kept "
      f"(both fwd and RC passed filters)")

if n_paired_kept == 0:
	raise ValueError("No locus pairs passed filters. Adjust thresholds.")

kept_fwd_idx = fwd_indices[pair_mask]
kept_rc_idx = rc_indices[pair_mask]

# Normalize each sample by its own pred_diff, take abs
fwd_abs_norm = np.abs(attributions[kept_fwd_idx] / raw_pred_diffs[kept_fwd_idx, None])
rc_abs_norm = np.abs(attributions[kept_rc_idx] / raw_pred_diffs[kept_rc_idx, None])

# ============================================================================
# Average forward and flipped-RC attributions at corresponding genomic
# positions
# ============================================================================

# In the RC run the model sees:
#   prefix | RC(genomic_right_flank) | RC(STR) | RC(genomic_left_flank)
#
# So the RC's left-flank positions (array indices lf_start:lf_end) contain
# the reverse complement of the original RIGHT flank, with the STR-adjacent
# end at lf_end-1 and the distal end at lf_start.
#
# Flipping the RC's right flank ([::-1]) aligns it with the forward's left
# flank by genomic distance from the STR, and vice versa.

avg_abs_norm = np.zeros_like(fwd_abs_norm)

# Left flank: average forward left with flipped RC right
avg_abs_norm[:, lf_start:lf_end] = (
	fwd_abs_norm[:, lf_start:lf_end] +
	rc_abs_norm[:, rf_start:rf_end][:, ::-1]
) / 2

# Right flank: average forward right with flipped RC left
avg_abs_norm[:, rf_start:rf_end] = (
	fwd_abs_norm[:, rf_start:rf_end] +
	rc_abs_norm[:, lf_start:lf_end][:, ::-1]
) / 2

# Non-flank regions (prompt tokens, STR bases) are ~zero by construction;
# leave as zeros.

# ============================================================================
# Compute means and bootstrap CIs
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

print(f"\nBootstrapping all samples ({N_BOOTSTRAP} iterations)...")
all_mean, all_ci_lo, all_ci_hi = bootstrap_ci(
	all_abs_norm, N_BOOTSTRAP, CI_LEVEL, rng
)

print(f"Bootstrapping averaged ({N_BOOTSTRAP} iterations)...")
avg_mean, avg_ci_lo, avg_ci_hi = bootstrap_ci(
	avg_abs_norm, N_BOOTSTRAP, CI_LEVEL, rng
)

# ============================================================================
# Compute relative positions from metadata
# ============================================================================

# Visual width of the STR gap region in coordinate units
STR_GAP_WIDTH = n_left_str + n_str_prompt_region + n_right_str
gap_start = 0
gap_end = STR_GAP_WIDTH

# Build relative positions: left flank at -n_left_flank to -1,
# a visual gap from 0 to STR_GAP_WIDTH for the STR region,
# right flank at STR_GAP_WIDTH+1 to STR_GAP_WIDTH+n_right_flank.
# Non-flank positions are NaN so the plot line breaks cleanly.
relative_positions = np.full(seq_len, np.nan)

for i in range(n_left_flank):
	relative_positions[lf_start + i] = -(n_left_flank - i)

for i in range(n_right_flank):
	relative_positions[rf_start + i] = gap_end + (i + 1)

# Region shading boundaries in relative coordinates
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
# Optional smoothing (flanking regions only, clipped at boundaries)
# ============================================================================

def apply_smoothing(mean_arr, ci_lo_arr, ci_hi_arr, window):
	"""Apply rolling average to flanking regions in-place."""
	if not window or window <= 1:
		return
	kernel = np.ones(window) / window
	for label, start, end in region_blocks:
		if label in ("left_flank", "right_flank"):
			mean_arr[start:end] = np.convolve(
				mean_arr[start:end], kernel, mode="same"
			)
			ci_lo_arr[start:end] = np.convolve(
				ci_lo_arr[start:end], kernel, mode="same"
			)
			ci_hi_arr[start:end] = np.convolve(
				ci_hi_arr[start:end], kernel, mode="same"
			)


apply_smoothing(all_mean, all_ci_lo, all_ci_hi, SMOOTHING_WINDOW)
apply_smoothing(avg_mean, avg_ci_lo, avg_ci_hi, SMOOTHING_WINDOW)

if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1:
	print(f"Applied rolling average smoothing (window={SMOOTHING_WINDOW}, "
	      f"flanking regions only)")

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

DATA_COLOR = "#3170ad"


def plot_attribution_panel(ax, mean_attr, ci_lower, ci_upper, title_str):
	"""Plot one attribution panel on the given axes."""

	# Background shading for regions
	for label, start, end in region_blocks_rel:
		color = region_colors.get(label, "#ffffff")
		ax.axvspan(start, end, alpha=0.2, color=color, zorder=0)

	# CI band
	ax.fill_between(
		relative_positions, ci_lower, ci_upper,
		alpha=0.3, color=DATA_COLOR, label=f"{int(CI_LEVEL*100)}% CI",
	)

	# Mean line
	ax.plot(
		relative_positions, mean_attr,
		color=DATA_COLOR, linewidth=0.5, label="Mean |attribution|",
	)

	# Region boundary lines (skip the very first edge)
	for label, start, end in region_blocks_rel:
		if start > -(n_left_flank + n_prefix):
			ax.axvline(start, color="gray", linewidth=0.5, linestyle=":",
			           alpha=0.7)

	# Region name labels
	ylim = ax.get_ylim()
	label_y_top = ylim[1] * 0.98
	label_y_text = label_y_top * 1.12

	middle_labels = ["left_str", "str_prompt", "right_str"]
	middle_blocks = [
		(l, s, e) for l, s, e in region_blocks_rel if l in middle_labels
	]
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
			ax.annotate(
				display_name,
				xy=(mid, label_y_top),
				xytext=(text_x, label_y_text),
				ha="center", va="bottom", fontsize=7, alpha=0.7,
				fontweight="bold",
				annotation_clip=False,
				arrowprops=dict(arrowstyle="-", color="gray",
				                alpha=0.4, lw=0.5),
			)
		elif label in middle_labels:
			text_x = middle_text_x[label]
			ax.text(text_x, label_y_text, display_name,
			        ha="center", va="bottom", fontsize=7, alpha=0.7,
			        fontweight="bold", clip_on=False)
		else:
			ax.text(mid, label_y_text, display_name,
			        ha="center", va="bottom", fontsize=7, alpha=0.7,
			        fontweight="bold", clip_on=False)

	# Legend
	legend_patches = [
		mpatches.Patch(facecolor="#d0d0d0", alpha=0.3, label="Prompt tokens"),
		mpatches.Patch(facecolor="#e8443a", alpha=0.3, label="STR bases"),
		mpatches.Patch(facecolor=DATA_COLOR, alpha=0.3,
		               label=f"{int(CI_LEVEL*100)}% CI"),
	]
	ax.legend(
		handles=legend_patches, loc="upper right", fontsize=8,
		framealpha=0.9,
	)

	ax.set_xlabel("Position relative to STR (bp)")
	ax.set_ylabel("|Normalized attribution|\n(fraction of total prediction shift)")

	smoothing_str = (f", smoothing={SMOOTHING_WINDOW}"
	                 if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1 else "")
	ax.set_title(title_str + smoothing_str)
	ax.set_ylim(bottom=0)

	# Symmetric tick labels despite the right-side offset
	xmin = -(n_left_flank + n_prefix)
	xmax = gap_end + n_right_flank
	tick_step = 500
	left_ticks = list(range(-2000, 0, tick_step))
	right_bp = list(range(tick_step, n_right_flank + 1, tick_step))
	right_ticks = [gap_end + bp for bp in right_bp]

	all_ticks = left_ticks + right_ticks
	all_labels = [str(t) for t in left_ticks] + [str(bp) for bp in right_bp]

	ax.set_xticks(all_ticks)
	ax.set_xticklabels(all_labels)
	ax.set_xlim(xmin, xmax)

	# Headroom for staggered labels
	ylim = ax.get_ylim()
	ax.set_ylim(bottom=0, top=ylim[1] * 1.15)


# --- Create figure with two subplots ---

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 2),
                                sharex=True)

plot_attribution_panel(
	ax1, all_mean, all_ci_lo, all_ci_hi,
	f"All samples IG attribution magnitude ({n_kept} samples, "
	f"{int(CI_LEVEL*100)}% CI)\n"
	f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, "
	f"rel_delta_threshold={REL_DELTA_THRESHOLD}"
)

plot_attribution_panel(
	ax2, avg_mean, avg_ci_lo, avg_ci_hi,
	f"Forward + RC averaged IG attribution magnitude ({n_paired_kept} loci, "
	f"{int(CI_LEVEL*100)}% CI)\n"
	f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, "
	f"rel_delta_threshold={REL_DELTA_THRESHOLD}"
)

# Only show x-axis label on the bottom subplot
ax1.set_xlabel("")

plt.tight_layout()

if SAVE_PATH:
	fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
	print(f"Saved to {SAVE_PATH}")
else:
	plt.show()