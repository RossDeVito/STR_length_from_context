"""Plot per-position normalized IG attribution magnitudes with confidence intervals.

For each position, computes the mean absolute normalized attribution across
samples and plots with a confidence interval band.

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

# Figure size
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

n_samples, seq_len = attributions.shape

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
	excluded_rel_delta = int(np.sum(mask & fail_delta))  # only count those not already excluded
	mask &= ~fail_delta

n_kept = np.sum(mask)
n_filtered = n_samples - n_kept
print(f"Samples: {n_kept} kept, {n_filtered} filtered "
      f"(of {n_samples} total)")
print(f"  Excluded by pred_diff <= {PRED_DIFF_THRESHOLD}: {excluded_pred_diff}")
print(f"  Excluded by rel_delta > {REL_DELTA_THRESHOLD}: {excluded_rel_delta}")

if n_kept == 0:
	raise ValueError("No samples passed filters. Adjust thresholds.")

attributions = attributions[mask]
raw_pred_diffs = raw_pred_diffs[mask]
labels = labels[mask]

# ============================================================================
# Normalize and take absolute value
# ============================================================================

# Divide by signed pred_diff so attributions sum to ~1, then take abs
norm_attributions = attributions / raw_pred_diffs[:, None]
abs_norm_attributions = np.abs(norm_attributions)

# ============================================================================
# Compute mean and bootstrap CI
# ============================================================================

mean_attr = np.mean(abs_norm_attributions, axis=0)           # (seq_len,)

# Bootstrap: resample samples (rows), compute mean per position each iteration
rng = np.random.default_rng(seed=42)
boot_means = np.zeros((N_BOOTSTRAP, seq_len), dtype=np.float32)

print(f"Bootstrapping ({N_BOOTSTRAP} iterations)...")
for b in trange(N_BOOTSTRAP, desc="Bootstrap iterations"):
	idx = rng.integers(0, n_kept, size=n_kept)
	boot_means[b] = np.mean(abs_norm_attributions[idx], axis=0)

# Percentile CI (no normality assumption)
alpha = 1 - CI_LEVEL
ci_lower = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)

# ============================================================================
# Compute relative positions from metadata
# ============================================================================

n_prefix = layout["n_prefix_prompt"]
n_left_flank = layout["n_flanking_bp"]
n_left_str = layout["n_str_bp"]
n_str_prompt_region = layout["n_str_prompt"]
n_right_str = layout["n_str_bp"]
n_right_flank = layout["n_flanking_bp"]

# Region blocks in array-index space (for smoothing)
region_blocks = [
	("prefix_prompt", 0, n_prefix),
	("left_flank",    n_prefix, n_prefix + n_left_flank),
	("left_str",      n_prefix + n_left_flank,
	                   n_prefix + n_left_flank + n_left_str),
	("str_prompt",    n_prefix + n_left_flank + n_left_str,
	                   n_prefix + n_left_flank + n_left_str + n_str_prompt_region),
	("right_str",     n_prefix + n_left_flank + n_left_str + n_str_prompt_region,
	                   n_prefix + n_left_flank + n_left_str + n_str_prompt_region + n_right_str),
	("right_flank",   n_prefix + n_left_flank + n_left_str + n_str_prompt_region + n_right_str,
	                   seq_len),
]

# Build relative positions: left flank at -n_left_flank to -1,
# a visual gap from 0 to STR_GAP_WIDTH for the STR region,
# right flank at STR_GAP_WIDTH+1 to STR_GAP_WIDTH+n_right_flank.
# Middle positions are NaN so the plot line breaks cleanly.
relative_positions = np.full(seq_len, np.nan)

# Visual width of the STR gap region in coordinate units
# Set to n_middle so 1 position = 1 coordinate unit, matching the prefix prompt
STR_GAP_WIDTH = n_left_str + n_str_prompt_region + n_right_str
gap_start = 0
gap_end = STR_GAP_WIDTH

# Prefix prompt: leave as NaN (prompt tokens, not flanking bases)

# Left flank: -n_left_flank to -1
lf_start = n_prefix
for i in range(n_left_flank):
	relative_positions[lf_start + i] = -(n_left_flank - i)

# Middle (left_str, str_prompt, right_str): leave as NaN

# Right flank: STR_GAP_WIDTH+1 to STR_GAP_WIDTH+n_right_flank
rf_start = n_prefix + n_left_flank + n_left_str + n_str_prompt_region + n_right_str
for i in range(n_right_flank):
	relative_positions[rf_start + i] = gap_end + (i + 1)

# Region shading boundaries
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

if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1:
	kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW

	for label, start, end in region_blocks:
		if label in ("left_flank", "right_flank"):
			mean_attr[start:end] = np.convolve(
				mean_attr[start:end], kernel, mode="same"
			)
			ci_lower[start:end] = np.convolve(
				ci_lower[start:end], kernel, mode="same"
			)
			ci_upper[start:end] = np.convolve(
				ci_upper[start:end], kernel, mode="same"
			)

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

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

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

# Region name labels — all at the same height above the plot
# Fan out the three narrow middle regions so they don't overlap
ylim = ax.get_ylim()
label_y_top = ylim[1] * 0.98
label_y_text = label_y_top * 1.12

# Precompute the center of the middle cluster for fanning
middle_labels = ["left_str", "str_prompt", "right_str"]
middle_blocks = [
	(l, s, e) for l, s, e in region_blocks_rel if l in middle_labels
]
if middle_blocks:
	cluster_center = (middle_blocks[0][1] + middle_blocks[-1][2]) / 2

# Fan offsets for narrow middle regions
x_range = (n_left_flank + n_prefix) + STR_GAP_WIDTH + n_right_flank
fan_offset = x_range * 0.06
middle_text_x = {
	"left_str":   cluster_center - fan_offset,
	"str_prompt": cluster_center,
	"right_str":  cluster_center + fan_offset,
}

# Labels that need connecting lines (narrow, fanned out from their region)
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
ax.set_title(
	f"Per-position IG attribution magnitude ({n_kept} samples, "
	f"{int(CI_LEVEL*100)}% CI{smoothing_str})\n"
	f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, "
	f"rel_delta_threshold={REL_DELTA_THRESHOLD}"
)
ax.set_ylim(bottom=0)

# Build symmetric tick labels despite the right-side offset
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

# Add headroom for staggered labels
ylim = ax.get_ylim()
ax.set_ylim(bottom=0, top=ylim[1] * 1.15)

plt.tight_layout()

if SAVE_PATH:
	fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
	print(f"Saved to {SAVE_PATH}")
else:
	plt.show()