"""Plot per-position IG attribution magnitude stratified by label quantile.

Like the forward+RC-averaged panel in ig_by_pos.py, but instead of one
pooled line, loci are split into N_QUANTILES groups by their own label
value (mode_copy_number for "length", heterozygosity for "variation") and
each group gets its own line. Two subplots: length on top, heterozygosity
on the bottom.

For each locus, the RC flank attributions are flipped to align with
forward genomic positions, then averaged with the forward attributions
(see ig_by_pos.py for the rationale).

Normalization happens in two stages:

1. Per-orientation normalization, shared with every other IG-by-position
   script (see seq_models/ig_attribution_norm.py). NORM_SCHEME picks the scheme:

   "l1_mass" (default) rescales each orientation to unit absolute
   attribution mass across the flanks. This is what makes a stratified
   plot readable. IG's completeness axiom makes the SIGNED attributions
   sum to the prediction shift, but taking an absolute value breaks that
   property, because positive and negative attributions no longer cancel.
   Total |attribution| per locus varies ~3x (1.6 to 5.3 prediction-shift
   units on str3_f5000_fiv_v4) and is itself strongly label-correlated
   (r = -0.56 against mode copy number: short STRs have far more
   cancellation, so more absolute mass). Left uncorrected, that per-locus
   scalar shows up as a flat vertical offset for each quantile group,
   burying any actual positional difference. Under "l1_mass" every locus
   contributes equally, so the plot answers "where does this group put
   its attribution" rather than "how much total mass does this group
   have". The mass effect itself is real and interesting, so per-group
   mean mass (in units of the prediction shift) is printed to the console
   rather than discarded.

   "pred_diff" instead divides each orientation by the absolute
   prediction shift, leaving that label-correlated mass scalar in the
   profiles -- useful for inspecting the mass effect directly, but see
   above for how it reads in a stratified plot.

   The forward and RC profiles are then averaged while still SIGNED, and
   the absolute value is taken last (abs-of-mean, not mean-of-abs), so
   positions where the two orientations disagree in sign are
   down-weighted instead of being summed as if they agreed.

2. Per-position group normalization (NORMALIZE_BY_OVERALL_MEAN): the raw
   per-position attribution profile is dominated by a shared shape (decay
   with distance from the STR, prompt-proximity artifacts) that has nothing
   to do with the label. That shared shape would dwarf any label-dependent
   differences if plotted directly. Dividing each group's per-position mean
   by the overall (all-groups-pooled) per-position mean at that same
   position removes the shared shape and isolates whether a group carries
   more or less attribution than average at each position -- the plotted
   line is centered near 1.0, with deviations showing label-dependent
   positional differences. This is a first-order approximation (it does
   not propagate the overall mean's own sampling uncertainty into the CI
   band).

Under "l1_mass" each orientation carries unit mass, but the forward/RC
average does not: where the two orientations disagree in sign they
partially cancel, so a locus's post-average mass is <= 1 and equals the
fraction of its mass on which the strands agree. Group lines therefore
still sit near 1.0 in stage 2 but are not guaranteed to cross it exactly;
a group offset that is flat across all positions now means that group's
strands agree more (or less) than average, which is worth knowing in its
own right rather than a normalization failure.

SHOW_FOLDED adds a second panel per task that collapses the upstream and
downstream flanks into a single distance-from-STR axis. The fold happens
between stages 1 and 2: the two sides are SUMMED per distance bin (they
are different bases being aggregated, not repeat measurements of one
base, which is why the forward/RC step averages but this one sums), so a
locus's total mass is unchanged by folding. Folding is justified on this
data because upstream and downstream are statistically indistinguishable
-- profiles correlate at 0.9999, the
mass split is 0.4998/0.5002, and the observed up/down gap is smaller than
a null that randomly swaps which side is called upstream. Because the
fold is linear and the stage-1 rescaling is a per-locus scalar, folding
and normalizing commute; the order between them doesn't matter.
"""

import colorsys
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import seq_models.ig_attribution_norm as ign

# ============================================================================
# Configuration
# ============================================================================

ATTR_DIR = "output/caduceus_v0/str3/str3_f5000_fiv_v4"

# Number of quantile groups to split loci into, by label value.
N_QUANTILES = 5

# Confidence interval level (0-1), e.g. 0.95 for 95% CI
CI_LEVEL = 0.95

# Number of bootstrap iterations for CI
N_BOOTSTRAP = 10

# Whether to draw the bootstrap CI band around each group's line.
# With N_QUANTILES lines per panel, bands can get busy -- set False for a
# cleaner mean-only plot.
SHOW_CI = False

# Filter: minimum raw prediction difference to include a sample.
PRED_DIFF_THRESHOLD = 0.05

# Filter: maximum relative convergence delta to include a sample.
# Set to None to skip this filter.
REL_DELTA_THRESHOLD = 0.05

# Per-orientation attribution normalization: "l1_mass" (each orientation
# rescaled to unit absolute attribution mass across the flanks, so a
# locus's total amount of attribution -- which is label-correlated --
# can't masquerade as a positional effect) or "pred_diff" (divide by the
# absolute prediction shift, leaving that mass scalar in). See the module
# docstring and seq_models/ig_attribution_norm.py.
NORM_SCHEME = "l1_mass"

# Divide each group's per-position mean by the overall (pooled) per-position
# mean, so lines show relative deviation from the average positional
# profile rather than absolute magnitude. See module docstring.
NORMALIZE_BY_OVERALL_MEAN = True

# Add a second panel per task that folds the upstream and downstream flanks
# together, showing attribution purely as a function of distance from the
# STR. Upstream/downstream are statistically indistinguishable on this data
# (profiles correlate at 0.9999; the observed gap is smaller than a null
# that randomly swaps sides), so folding is a ~2x gain in loci per distance
# bin rather than a loss of signal.
SHOW_FOLDED = True

# Figure size (per subplot)
FIG_WIDTH = 20
FIG_HEIGHT = 5

# Colormap used to color the N_QUANTILES lines, low group to high. Sampled
# only up to CMAP_MAX_FRAC of the map's range (not the full 0-1) so the
# highest-quantile line doesn't land on a near-white/light endpoint that
# disappears against the white flank background -- try lowering
# CMAP_MAX_FRAC further, or a different qualitative/sequential map name
# (e.g. "plasma", "Spectral", "turbo", "cool", "tab10"), to taste.
CMAP_NAME = "Spectral"
CMAP_MAX_FRAC = 1.00

# Diverging colormaps like RdYlGn are lightest at their midpoint by
# construction (that's what makes them read as diverging), which is where
# RdYlGn turns into a pale, hard-to-see yellow. CMAP_MID_DARKEN reduces
# lightness (in HLS) for colors sampled near CMAP_MID_DARKEN_CENTER, with
# the reduction tapering off (Gaussian, width CMAP_MID_DARKEN_SIGMA) away
# from that center -- so only the washed-out middle is affected, not the
# already-saturated red/green ends. Set CMAP_MID_DARKEN to 0 to disable.
CMAP_MID_DARKEN = 0.60
CMAP_MID_DARKEN_CENTER = 0.5
CMAP_MID_DARKEN_SIGMA = 0.1

# Line width for the per-group mean lines.
LINE_WIDTH = 1.4

# Smoothing: rolling average window size. Set to None or 1 to disable.
SMOOTHING_WINDOW = 51

# Output paths (set to None to just display). Two separate figures are
# produced: the full-window (signed position) figure and, if SHOW_FOLDED,
# the folded (distance-from-STR) figure.
SAVE_PATH = None
SAVE_PATH_FOLDED = None

# Tasks to plot, top to bottom, with display formatting.
TASK_CONFIGS = [
	{"task": "length", "display_name": "Copy Number",
	 "label_fmt": "{:.1f}"},
	{"task": "variation", "display_name": "Heterozygosity",
	 "label_fmt": "{:.3f}"},
]

# ============================================================================
# Load data and metadata (shared across tasks)
# ============================================================================

data = np.load(os.path.join(ATTR_DIR, "attributions.npz"), allow_pickle=True)

with open(os.path.join(ATTR_DIR, "meta.json"), "r") as f:
	meta = json.load(f)

layout = meta["sequence_layout"]
rev_comp = data["rev_comp"]  # boolean: True for reverse complement samples

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

region_blocks = flanks.region_blocks

# Identify forward and RC indices among ALL samples (before filtering)
fwd_indices, rc_indices = ign.pair_fwd_rc(rev_comp, ids=data["ids"])
n_loci = len(fwd_indices)

# ============================================================================
# Relative positions (shared across tasks -- layout is the same)
# ============================================================================

STR_GAP_WIDTH = n_left_str + n_str_prompt_region + n_right_str
gap_start = 0
gap_end = STR_GAP_WIDTH

seq_len = layout["seq_len"]
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
# Helper functions
# ============================================================================


def get_group_colors(n, cmap_name, max_frac, mid_darken=0.0,
                      mid_darken_center=0.5, mid_darken_sigma=0.25):
	"""Return n colors sampled from cmap_name, low group to high.

	If mid_darken > 0, colors sampled close to mid_darken_center (in the
	colormap's own 0-1 domain, before max_frac scaling) get their HLS
	lightness reduced, tapering off away from the center -- see
	CMAP_MID_DARKEN in the config section for why.
	"""
	base_cmap = plt.get_cmap(cmap_name)
	colors = []
	for i in range(n):
		frac = i / max(n - 1, 1) * max_frac
		r, g, b, a = base_cmap(frac)
		if mid_darken > 0:
			dist = (frac - mid_darken_center) / mid_darken_sigma
			amount = mid_darken * np.exp(-0.5 * dist ** 2)
			h, l, s = colorsys.rgb_to_hls(r, g, b)
			r, g, b = colorsys.hls_to_rgb(h, l * (1 - amount), s)
		colors.append((r, g, b, a))
	return colors


def fold_flanks(abs_norm_matrix):
	"""Collapse the two flanks into one profile indexed by distance from
	the STR: column d holds distance d+1 bp, counting outward.

	The two sides are SUMMED, not averaged. Averaging is right for the
	forward/RC pair because those are two measurements of the same
	genomic base; folding instead aggregates two genuinely different
	bases (one upstream, one downstream) into a single distance bin, so
	summing is what keeps "share of total attribution mass" intact -- a
	locus at unit mass across both flanks stays at unit mass here.

	Upstream is stored distal-to-proximal, so it is reversed to put
	distance 1 first; downstream is already proximal-to-distal.
	"""
	upstream = abs_norm_matrix[:, lf_start:lf_end][:, ::-1]
	downstream = abs_norm_matrix[:, rf_start:rf_end]
	return upstream + downstream


def assign_quantile_groups(values, n_quantiles):
	"""Split values into n_quantiles equal-sized groups by rank.

	Uses rank (not raw quantile edges) so groups stay equal-sized even
	when many values tie (e.g. heterozygosity is zero-inflated), at the
	cost of splitting ties arbitrarily across adjacent groups.
	"""
	n = len(values)
	order = np.argsort(values, kind="stable")
	ranks = np.empty(n, dtype=int)
	ranks[order] = np.arange(n)
	groups = np.clip((ranks * n_quantiles) // n, 0, n_quantiles - 1)
	return groups


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


def smooth_1d(segment, window):
	"""Rolling average of a 1-D segment, same length out as in.

	Pads with edge-value replication (rather than np.convolve's implicit
	zero-padding under mode="same") so the rolling mean doesn't get pulled
	toward zero at the segment's ends.
	"""
	if not window or window <= 1:
		return segment
	kernel = np.ones(window) / window
	half = window // 2
	padded = np.pad(segment, (half, window - 1 - half), mode="edge")
	return np.convolve(padded, kernel, mode="valid")


def apply_smoothing_1d(arr, window):
	"""Smooth the flanking regions of a full-length array in-place.

	Each flank is smoothed independently so the rolling window never spans
	the STR gap or a region boundary. Returns arr for convenience.
	"""
	if not window or window <= 1:
		return arr
	for label, start, end in region_blocks:
		if label in ("left_flank", "right_flank"):
			arr[start:end] = smooth_1d(arr[start:end], window)
	return arr


def build_group_lines(profile_matrix, groups, locus_labels, label_fmt,
                       smoother):
	"""Bootstrap, normalize and smooth one plot line per quantile group.

	profile_matrix is (n_loci, n_positions) -- either the full-length
	signed-position profiles or the folded distance-from-STR profiles.
	smoother takes and returns a 1-D array, so each caller supplies the
	variant appropriate to its axis.
	"""
	overall_mean = np.mean(profile_matrix, axis=0)

	lines = []
	for g in range(N_QUANTILES):
		g_mask = groups == g
		g_values = locus_labels[g_mask]

		mean_g, ci_lo_g, ci_hi_g = bootstrap_ci(
			profile_matrix[g_mask], N_BOOTSTRAP, CI_LEVEL, rng
		)

		if NORMALIZE_BY_OVERALL_MEAN:
			with np.errstate(divide="ignore", invalid="ignore"):
				mean_g = mean_g / overall_mean
				ci_lo_g = ci_lo_g / overall_mean
				ci_hi_g = ci_hi_g / overall_mean

		range_str = (f"{label_fmt.format(g_values.min())}-"
		             f"{label_fmt.format(g_values.max())}")

		lines.append({
			"label": f"Q{g + 1} [{range_str}] (n={int(np.sum(g_mask))})",
			"mean": smoother(mean_g),
			"ci_lo": smoother(ci_lo_g),
			"ci_hi": smoother(ci_hi_g),
			"color": group_colors[g],
		})
	return lines


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


def plot_stratified_panel(ax, group_lines, title_str, ylabel_str,
                           show_reference_line):
	"""Plot one panel with one line (+ optional CI band) per group."""

	for label, start, end in region_blocks_rel:
		color = region_colors.get(label, "#ffffff")
		ax.axvspan(start, end, alpha=0.2, color=color, zorder=0)

	if show_reference_line:
		ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--",
		           alpha=0.5, zorder=1)

	for gl in group_lines:
		if SHOW_CI:
			ax.fill_between(
				relative_positions, gl["ci_lo"], gl["ci_hi"],
				alpha=0.15, color=gl["color"], zorder=2,
			)
		ax.plot(
			relative_positions, gl["mean"],
			color=gl["color"], linewidth=LINE_WIDTH, label=gl["label"], zorder=3,
		)

	for label, start, end in region_blocks_rel:
		if start > -(n_left_flank + n_prefix):
			ax.axvline(start, color="gray", linewidth=0.5, linestyle=":",
			           alpha=0.7)

	# Finalize the y-range (with headroom for the region-name labels) now,
	# before placing those labels, so their placement is correct regardless
	# of whether the axis floor sits at 0 (raw magnitude) or floats above/
	# below it (normalized ratio).
	data_bottom, data_top = ax.get_ylim()
	if not show_reference_line:
		data_bottom = 0
	headroom = (data_top - data_bottom) * 0.15
	ax.set_ylim(data_bottom, data_top + headroom)

	label_y_top = data_top + headroom * 0.1
	label_y_text = data_top + headroom * 0.55

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
		if label == "prefix_prompt":
			continue

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

	ax.legend(loc="upper right", fontsize=7, framealpha=0.9,
	          title="Label quantile (low -> high)", title_fontsize=7)

	ax.set_xlabel("Position relative to STR (bp)")
	ax.set_ylabel(ylabel_str)

	smoothing_str = (f", smoothing={SMOOTHING_WINDOW}"
	                 if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1 else "")
	ax.set_title(title_str + smoothing_str)

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


def plot_folded_panel(ax, group_lines, title_str, ylabel_str,
                       show_reference_line):
	"""Plot one distance-from-STR panel (upstream and downstream folded).

	Simpler than plot_stratified_panel: the x axis is a single distance
	axis with no STR gap to straddle and no region blocks to shade.
	"""
	distances = np.arange(1, n_left_flank + 1)

	if show_reference_line:
		ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--",
		           alpha=0.5, zorder=1)

	for gl in group_lines:
		if SHOW_CI:
			ax.fill_between(
				distances, gl["ci_lo"], gl["ci_hi"],
				alpha=0.15, color=gl["color"], zorder=2,
			)
		ax.plot(
			distances, gl["mean"],
			color=gl["color"], linewidth=LINE_WIDTH, label=gl["label"],
			zorder=3,
		)

	if not show_reference_line:
		ax.set_ylim(bottom=0)

	ax.legend(loc="upper right", fontsize=7, framealpha=0.9,
	          title="Label quantile (low -> high)", title_fontsize=7)

	ax.set_xlabel("Distance from STR (bp, upstream and downstream folded)")
	ax.set_ylabel(ylabel_str)

	smoothing_str = (f", smoothing={SMOOTHING_WINDOW}"
	                 if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1 else "")
	ax.set_title(title_str + smoothing_str)

	ax.set_xticks(list(range(0, n_left_flank + 1, 500)))
	ax.set_xlim(1, n_left_flank)


# ============================================================================
# Per-task processing and plotting
# ============================================================================

n_tasks = len(TASK_CONFIGS)

fig, axes = plt.subplots(n_tasks, 1,
                          figsize=(FIG_WIDTH, FIG_HEIGHT * n_tasks),
                          sharex=True)
if n_tasks == 1:
	axes = [axes]

if SHOW_FOLDED:
	fig_folded, axes_folded = plt.subplots(
		n_tasks, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * n_tasks), sharex=True
	)
	if n_tasks == 1:
		axes_folded = [axes_folded]

rng = np.random.default_rng(seed=42)
group_colors = get_group_colors(
	N_QUANTILES, CMAP_NAME, CMAP_MAX_FRAC,
	mid_darken=CMAP_MID_DARKEN,
	mid_darken_center=CMAP_MID_DARKEN_CENTER,
	mid_darken_sigma=CMAP_MID_DARKEN_SIGMA,
)

for task_idx, task_cfg in enumerate(TASK_CONFIGS):
	task = task_cfg["task"]
	ax = axes[task_idx]

	attributions = data[f"attributions_{task}"]
	raw_predictions = data[f"raw_predictions_{task}"]
	raw_baseline_predictions = data[f"raw_baseline_predictions_{task}"]
	relative_deltas = data[f"relative_convergence_deltas_{task}"]
	labels = data[f"labels_{task}"]

	# Filter, normalize per orientation, flip the RC flanks into forward
	# genomic coordinates, average the SIGNED profiles, then take the
	# absolute value. See seq_models/ig_attribution_norm.py.
	loci = ign.per_locus_attributions(
		attributions, raw_predictions, raw_baseline_predictions,
		relative_deltas, rev_comp, flanks, ids=data["ids"],
		scheme=NORM_SCHEME,
		pred_diff_threshold=PRED_DIFF_THRESHOLD,
		rel_delta_threshold=REL_DELTA_THRESHOLD,
	)

	n_paired_kept = loci.n_loci
	print(f"[{task}] {n_paired_kept} / {n_loci} loci kept "
	      f"(both fwd and RC passed filters)")

	avg_abs_norm = loci.avg_abs

	# Total absolute attribution mass per locus, in units of the prediction
	# shift. Kept for reporting even when NORM_SCHEME divides it out.
	locus_mass = loci.mass_ratio

	locus_labels = labels[loci.fwd_idx]
	groups = assign_quantile_groups(locus_labels, N_QUANTILES)
	label_fmt = task_cfg["label_fmt"]

	for g in range(N_QUANTILES):
		g_mask = groups == g
		g_values = locus_labels[g_mask]
		range_str = (f"{label_fmt.format(g_values.min())}-"
		             f"{label_fmt.format(g_values.max())}")

		# Report the scalar that NORM_SCHEME="l1_mass" divides out: total
		# |attribution| per locus in units of the prediction shift, which is
		# label-correlated in its own right but is not a positional effect.
		print(f"  [{task}] Q{g + 1} [{range_str}] n={int(np.sum(g_mask))}  "
		      f"mean total |attribution| mass = {locus_mass[g_mask].mean():.2f}")

		if g_values.min() == g_values.max():
			print(f"    warning: quantile group {g + 1}/{N_QUANTILES} "
			      f"collapses to a single label value; "
			      f"likely due to many tied label values.")

	group_lines = build_group_lines(
		avg_abs_norm, groups, locus_labels, label_fmt,
		lambda a: apply_smoothing_1d(a, SMOOTHING_WINDOW),
	)

	if NORMALIZE_BY_OVERALL_MEAN:
		share_str = ("share of attribution mass"
		             if NORM_SCHEME == "l1_mass" else "attribution")
		ylabel_str = f"Relative {share_str}\n(group mean / overall mean)"
	else:
		ylabel_str = ign.magnitude_axis_label(NORM_SCHEME)

	mass_str = ign.scheme_description(NORM_SCHEME)
	title_str = (
		f"[{task_cfg['display_name']}] Forward+RC averaged IG attribution, "
		f"stratified by label quantile ({N_QUANTILES} groups, {n_paired_kept} loci, "
		f"{int(CI_LEVEL*100)}% CI)\n"
		f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, "
		f"rel_delta_threshold={REL_DELTA_THRESHOLD}, {mass_str}"
	)

	plot_stratified_panel(ax, group_lines, title_str, ylabel_str,
	                       show_reference_line=NORMALIZE_BY_OVERALL_MEAN)

	if SHOW_FOLDED:
		folded_lines = build_group_lines(
			fold_flanks(avg_abs_norm), groups, locus_labels, label_fmt,
			lambda a: smooth_1d(a, SMOOTHING_WINDOW),
		)
		folded_title = (
			f"[{task_cfg['display_name']}] Forward+RC averaged IG attribution, "
			f"folded by distance from STR (upstream + downstream combined, "
			f"{N_QUANTILES} groups, {n_paired_kept} loci, {int(CI_LEVEL*100)}% CI)\n"
			f"pred_diff_threshold={PRED_DIFF_THRESHOLD}, "
			f"rel_delta_threshold={REL_DELTA_THRESHOLD}, {mass_str}"
		)
		plot_folded_panel(
			axes_folded[task_idx], folded_lines,
			folded_title, ylabel_str,
			show_reference_line=NORMALIZE_BY_OVERALL_MEAN,
		)

fig.tight_layout()
if SAVE_PATH:
	fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
	print(f"Saved to {SAVE_PATH}")

if SHOW_FOLDED:
	fig_folded.tight_layout()
	if SAVE_PATH_FOLDED:
		fig_folded.savefig(SAVE_PATH_FOLDED, dpi=150, bbox_inches="tight")
		print(f"Saved to {SAVE_PATH_FOLDED}")

# Show whichever figure(s) weren't saved to disk.
if not SAVE_PATH or (SHOW_FOLDED and not SAVE_PATH_FOLDED):
	plt.show()
