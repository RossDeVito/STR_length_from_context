"""Single-locus IG attribution visualization using sequence logos.

Visualizes a specific locus (selected from ig_single_example.py ranking)
as logomaker sequence logos showing per-base signed normalized attributions.

Three subplots:
  1. Forward strand attributions.
  2. RC strand attributions, flipped to align with forward genomic positions.
     Bases shown are the forward strand bases so letters align vertically
     with subplot 1.
  3. Mean of forward and aligned RC attributions.

Positive attribution = pushes prediction upward (toward higher copy number).
Negative attribution = pushes prediction downward.
Letter height encodes attribution magnitude; letter identity encodes the
actual genomic base at that position.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logomaker
import json
import os

from seq_models.caduceus.model import inverse_transform


def to_native(transform_name, value):
	"""Apply a model target's inverse transform to a raw scalar."""
	return float(inverse_transform(transform_name, torch.tensor(float(value))))


# ============================================================================
# Helper functions
# ============================================================================

def build_logo_df(sequence, attributions, positions):
	"""Build a logomaker-compatible DataFrame.

	Each position's attribution goes into the column of the actual base;
	other bases get 0.

	Parameters
	----------
	sequence : str
		Nucleotide sequence (A/C/G/T characters).
	attributions : array-like
		Attribution values, same length as sequence.
	positions : list of int
		Position labels for the DataFrame index.

	Returns
	-------
	pd.DataFrame
		Columns A, C, G, T with index = positions.
	"""
	records = []
	for base, attr in zip(sequence, attributions):
		row = {b: 0.0 for b in "ACGT"}
		b = base.upper()
		if b in row:
			row[b] = float(attr)
		records.append(row)
	return pd.DataFrame(records, index=positions)


# ============================================================================

if __name__ == "__main__":

	# ====================================================================
	# Configuration
	# ====================================================================

	ATTR_DIR = "output/caduceus_v0/str3/str3_f5000_fiv_v4"

	# Which task's attributions/predictions to plot (must be one of the
	# checkpoint's active targets, e.g. "length" or "variation" -- see
	# {ATTR_DIR}/meta.json "task_names").
	TASK = "length"

	LOCUS_ID = "Human_STR_422778"
	MAX_DIST = 200  # max bp from STR to include in each direction

	# Figure
	FIG_WIDTH = 30
	FIG_HEIGHT_PER_ROW = 2.5
	SAVE_PATH = None  # set to a filepath to save instead of display

	# ====================================================================
	# Load data
	# ====================================================================

	data = np.load(
		os.path.join(ATTR_DIR, "attributions.npz"), allow_pickle=True
	)
	with open(os.path.join(ATTR_DIR, "meta.json"), "r") as f:
		meta = json.load(f)

	layout = meta["sequence_layout"]
	transform_name = meta["transforms"][TASK]

	attributions = data[f"attributions_{TASK}"]
	raw_predictions = data[f"raw_predictions_{TASK}"]
	raw_baseline_predictions = data[f"raw_baseline_predictions_{TASK}"]
	labels = data[f"labels_{TASK}"]
	rev_comp = data["rev_comp"]
	ids = data["ids"]
	sequences = data["sequences"]

	# ====================================================================
	# Region boundaries
	# ====================================================================

	n_prefix = layout["n_prefix_prompt"]
	n_left_flank = layout["n_flanking_bp"]
	n_left_str = layout["n_str_bp"]
	n_str_prompt = layout["n_str_prompt"]
	n_right_str = layout["n_str_bp"]
	n_right_flank = layout["n_flanking_bp"]

	lf_start = n_prefix
	lf_end = lf_start + n_left_flank
	rf_start = lf_end + n_left_str + n_str_prompt + n_right_str
	rf_end = rf_start + n_right_flank

	# ====================================================================
	# Find forward and RC samples for this locus
	# ====================================================================

	fwd_idx = None
	rc_idx = None

	for i in range(len(ids)):
		if str(ids[i]) == LOCUS_ID:
			if rev_comp[i]:
				rc_idx = i
			else:
				fwd_idx = i

	assert fwd_idx is not None, f"No forward sample found for {LOCUS_ID}"
	assert rc_idx is not None, f"No RC sample found for {LOCUS_ID}"

	# ====================================================================
	# Normalize attributions (signed, divide by pred_diff)
	# ====================================================================

	fwd_pred_diff = (
		raw_predictions[fwd_idx] - raw_baseline_predictions[fwd_idx]
	)
	rc_pred_diff = (
		raw_predictions[rc_idx] - raw_baseline_predictions[rc_idx]
	)

	fwd_norm = attributions[fwd_idx] / fwd_pred_diff
	rc_norm = attributions[rc_idx] / rc_pred_diff

	# ====================================================================
	# Extract flanking attributions and sequences
	# ====================================================================

	dist = min(MAX_DIST, n_left_flank)  # cap at actual flank length

	fwd_seq = str(sequences[fwd_idx])

	# Forward flanks (model positions map directly to genomic positions)
	fwd_left_attr = fwd_norm[lf_end - dist : lf_end]
	fwd_right_attr = fwd_norm[rf_start : rf_start + dist]
	fwd_left_bases = fwd_seq[lf_end - dist : lf_end]
	fwd_right_bases = fwd_seq[rf_start : rf_start + dist]

	# RC aligned to genomic positions.
	#
	# RC model layout:
	#   [prefix] [RC(genomic_right)] [RC(STR)] [RC(genomic_left)]
	#
	# RC left flank (lf_start:lf_end) = RC(genomic_right_flank):
	#   model index lf_end-1 -> genomic position +1 (STR-adjacent)
	#   model index lf_end-2 -> genomic position +2
	#   ...
	#
	# RC right flank (rf_start:rf_end) = RC(genomic_left_flank):
	#   model index rf_start   -> genomic position -1 (STR-adjacent)
	#   model index rf_start+1 -> genomic position -2
	#   ...
	#
	# Reversing gives genomic order for each flank.

	# Genomic left flank (-dist to -1): from RC's right flank, reversed
	rc_aligned_left_attr = rc_norm[rf_start : rf_start + dist][::-1]

	# Genomic right flank (+1 to +dist): from RC's left flank, reversed
	rc_aligned_right_attr = rc_norm[lf_end - dist : lf_end][::-1]

	# ====================================================================
	# Build logomaker DataFrames
	# ====================================================================

	# Use forward strand bases for all three plots (alignment).
	left_positions = list(range(-dist, 0))
	right_positions = list(range(1, dist + 1))

	# Insert a zero-gap position at 0 for visual STR separation
	gap_row = pd.DataFrame(
		{"A": [0.0], "C": [0.0], "G": [0.0], "T": [0.0]}, index=[0]
	)

	def make_full_df(left_attr, right_attr):
		"""Concatenate left flank, gap, right flank into one DataFrame."""
		left_df = build_logo_df(fwd_left_bases, left_attr, left_positions)
		right_df = build_logo_df(
			fwd_right_bases, right_attr, right_positions
		)
		return pd.concat([left_df, gap_row, right_df])

	fwd_df = make_full_df(fwd_left_attr, fwd_right_attr)
	rc_df = make_full_df(rc_aligned_left_attr, rc_aligned_right_attr)
	mean_df = (fwd_df + rc_df) / 2

	# ====================================================================
	# Metadata for titles
	# ====================================================================

	true_cn = float(labels[fwd_idx])
	fwd_pred_cn = to_native(transform_name, raw_predictions[fwd_idx])
	rc_pred_cn = to_native(transform_name, raw_predictions[rc_idx])

	fwd_pred_diff_val = float(fwd_pred_diff)
	rc_pred_diff_val = float(rc_pred_diff)

	# Extract repeat unit from the start of the STR region.
	# STR motifs are at most 2 bp, so the first 1-2 non-X bases give the unit.
	str_left = fwd_seq[lf_end : lf_end + n_left_str]
	str_left_clean = str_left.replace("X", "").replace("x", "")

	# Detect repeat unit length (1 or 2) by checking periodicity
	if len(str_left_clean) >= 2 and str_left_clean[0] != str_left_clean[1]:
		repeat_unit = str_left_clean[:2]
	else:
		repeat_unit = str_left_clean[:1]

	print(f"Locus: {LOCUS_ID}  (task: {TASK}, transform: {transform_name})")
	print(f"  Repeat unit: {repeat_unit}")
	print(f"  True:       {true_cn:.4g}")
	print(f"  Fwd pred:   {fwd_pred_cn:.4g}  "
	      f"({transform_name} pred_diff: {fwd_pred_diff_val:.4f})")
	print(f"  RC  pred:   {rc_pred_cn:.4g}  "
	      f"({transform_name} pred_diff: {rc_pred_diff_val:.4f})")
	print(f"  Plotting ±{dist} bp from STR")

	# ====================================================================
	# Plot
	# ====================================================================

	fig, axes = plt.subplots(
		3, 1,
		figsize=(FIG_WIDTH, FIG_HEIGHT_PER_ROW * 3),
		sharex=True,
	)

	subplot_info = [
		(fwd_df, f"Forward strand  (pred: {fwd_pred_cn:.4g})"),
		(rc_df, f"RC aligned to genomic  (pred: {rc_pred_cn:.4g})"),
		(mean_df, f"Mean (forward + RC aligned)"),
	]

	# Use consistent y-axis limits across all three subplots
	all_vals = pd.concat([fwd_df, rc_df, mean_df]).values
	y_max = np.max(np.abs(all_vals)) * 1.1
	y_lim = (-y_max, y_max)

	for ax, (df, title) in zip(axes, subplot_info):
		logo = logomaker.Logo(
			df,
			ax=ax,
			color_scheme="classic",
			font_name="DejaVu Sans Mono",
		)

		ax.set_ylabel("Normalized\nattribution")
		ax.set_title(title, fontsize=11)
		ax.set_ylim(y_lim)

		# STR boundary marker
		ax.axvline(0, color="red", linewidth=1.5, linestyle="--", alpha=0.6)
		ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

		# Label the STR gap
		ax.text(
			0, y_lim[1] * 0.9, "STR",
			ha="center", va="top", fontsize=9, fontweight="bold",
			color="red", alpha=0.6,
		)

	axes[-1].set_xlabel("Position relative to STR (bp)")

	fig.suptitle(
		f"{LOCUS_ID}  [{TASK}]    |    STR: [{repeat_unit}]{true_cn:.4g}"
		f"    |    Fwd pred: {fwd_pred_cn:.4g}"
		f"    |    RC pred: {rc_pred_cn:.4g}",
		fontsize=13, fontweight="bold",
	)

	plt.tight_layout(rect=[0, 0, 1, 0.96])

	if SAVE_PATH:
		fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
		print(f"Saved to {SAVE_PATH}")
	else:
		plt.show()