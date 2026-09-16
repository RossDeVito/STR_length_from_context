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

Normalization and forward/RC combination are the shared ones defined in
seq_models/ig_attribution_norm.py; NORM_SCHEME selects between "l1_mass" (each
orientation rescaled to unit absolute attribution mass over the flanks)
and "pred_diff" (each orientation divided by |F(input) - F(baseline)|).
Both keep the sign through the forward/RC average, and the absolute value
is taken last -- abs-of-mean, not mean-of-abs, so positions where the two
orientations disagree in sign are down-weighted rather than summed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

import seq_models.ig_attribution_norm as ign

# ============================================================================
# Configuration
# ============================================================================

ATTR_DIR = "output/caduceus_v0/str3/str3_f5000_fiv_v4"

# Which task's attributions/predictions to plot (must be one of the
# checkpoint's active targets, e.g. "length" or "variation" -- see
# {ATTR_DIR}/meta.json "task_names").
# TASK = "length"
TASK = "variation"

# Per-orientation attribution normalization: "l1_mass" (unit absolute
# attribution mass over the flanks) or "pred_diff" (divide by the absolute
# prediction shift from baseline). See seq_models/ig_attribution_norm.py.
NORM_SCHEME = "l1_mass"

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
REL_DELTA_THRESHOLD = 0.05

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

attributions = data[f"attributions_{TASK}"]
raw_predictions = data[f"raw_predictions_{TASK}"]
raw_baseline_predictions = data[f"raw_baseline_predictions_{TASK}"]
relative_deltas = data[f"relative_convergence_deltas_{TASK}"]
labels = data[f"labels_{TASK}"]
rev_comp = data["rev_comp"]  # boolean: True for reverse complement samples

seq_len = attributions.shape[1]

# ============================================================================
# Compute region boundaries
# ============================================================================

flanks = ign.FlankLayout.from_layout(layout)

n_prefix = flanks.n_prefix
n_left_flank = flanks.n_left_flank
n_left_str = flanks.n_left_str
n_str_prompt_region = flanks.n_str_prompt
n_right_str = flanks.n_right_str
n_right_flank = flanks.n_right_flank
n_suffix_prompt = flanks.n_suffix_prompt

lf_start = flanks.lf_start
lf_end = flanks.lf_end
rf_start = flanks.rf_start
rf_end = flanks.rf_end

# Region blocks in array-index space (for smoothing)
region_blocks = flanks.region_blocks

# ============================================================================
# Filter samples
# ============================================================================

raw_pred_diffs = raw_predictions - raw_baseline_predictions

samp_filter = ign.filter_samples(
	raw_pred_diffs, relative_deltas, PRED_DIFF_THRESHOLD, REL_DELTA_THRESHOLD
)
mask = samp_filter.mask
n_kept = samp_filter.n_kept

print(f"Subplot 1 (all samples): {n_kept} kept, {samp_filter.n_filtered} "
      f"filtered (of {samp_filter.n_total} total)")
print(f"  Excluded by pred_diff <= {PRED_DIFF_THRESHOLD}: "
      f"{samp_filter.n_excluded_pred_diff}")
print(f"  Excluded by rel_delta > {REL_DELTA_THRESHOLD}: "
      f"{samp_filter.n_excluded_rel_delta}")

if n_kept == 0:
	raise ValueError("No samples passed filters. Adjust thresholds.")

# ============================================================================
# Normalize and take absolute value (all filtered samples, for subplot 1)
# ============================================================================

all_abs_norm = np.abs(ign.normalize_signed(
	attributions[mask], raw_pred_diffs[mask], flanks, NORM_SCHEME
))

# ============================================================================
# Pair forward and RC samples by locus for subplot 2
# ============================================================================

# Normalize each orientation, flip the RC flanks into forward genomic
# coordinates, average the SIGNED profiles, then take the absolute value
# (abs-of-mean). See seq_models/ig_attribution_norm.py for the full definition.
loci = ign.per_locus_attributions(
	attributions, raw_predictions, raw_baseline_predictions, relative_deltas,
	rev_comp, flanks, ids=data["ids"], scheme=NORM_SCHEME,
	sample_filter=samp_filter,
)

avg_abs_norm = loci.avg_abs
n_paired_kept = loci.n_loci

print(f"\nFound {loci.n_loci_total} loci with forward + RC pairs")
print(f"Subplot 2 (averaged): {n_paired_kept} loci kept "
      f"(both fwd and RC passed filters)")

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
	ax.set_ylabel(ign.magnitude_axis_label(NORM_SCHEME))

	smoothing_str = (f", smoothing={SMOOTHING_WINDOW}"
	                 if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1 else "")
	ax.set_title(title_str + smoothing_str)
	ax.set_ylim(bottom=0)

	# Symmetric tick labels despite the right-side offset
	xmin = -(n_left_flank + n_prefix)
	xmax = gap_end + n_right_flank
	tick_step = 500
	left_bp = list(range(tick_step, n_left_flank + 1, tick_step))
	right_bp = list(range(tick_step, n_right_flank + 1, tick_step))
	left_ticks = [-bp for bp in left_bp]
	right_ticks = [gap_end + bp for bp in right_bp]

	all_ticks = left_ticks + right_ticks
	all_labels = [str(-bp) for bp in left_bp] + [str(bp) for bp in right_bp]

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
	f"[{TASK}] All samples IG attribution magnitude ({n_kept} samples, "
	f"{int(CI_LEVEL*100)}% CI)\n"
	f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, "
	f"rel_delta_threshold={REL_DELTA_THRESHOLD}, "
	f"{ign.scheme_description(NORM_SCHEME)}"
)

plot_attribution_panel(
	ax2, avg_mean, avg_ci_lo, avg_ci_hi,
	f"[{TASK}] Forward + RC averaged IG attribution magnitude ({n_paired_kept} loci, "
	f"{int(CI_LEVEL*100)}% CI)\n"
	f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, "
	f"rel_delta_threshold={REL_DELTA_THRESHOLD}, "
	f"{ign.scheme_description(NORM_SCHEME)}"
)

# Only show x-axis label on the bottom subplot
ax1.set_xlabel("")

plt.tight_layout()

if SAVE_PATH:
	fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
	print(f"Saved to {SAVE_PATH}")
else:
	plt.show()