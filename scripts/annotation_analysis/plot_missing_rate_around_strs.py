""" Compute and plot the missing rate of a genomic annotation (bed or
bigwig) at STR loci and at positions flanking each STR on either side.

For a configurable STR length, loads the HipSTR-labeled STRs and, for each
configured annotation type, queries a window of MAX_FLANK_BP on either side
of every STR to determine which positions have no annotation data ("missing"
-- e.g. an unscored gap in a bigwig, or a position with no covering row in a
bed file; a bed row that happens to represent a low-signal/artifact state is
still present data, not missing data).

Produces, per annotation type, one figure with two panels:
  - Per-bp resolution: missing rate at each individual bp distance from the
    STR boundary.
  - Binned: the same, with flank distances grouped into BIN_WIDTH_BP bins.
Both panels show the STR itself as a single aggregate point (STR span length
varies per locus, so there's no consistent relative-position scale for it),
plus a printed summary table of missing rates by region.

ChromHMM_state and ChromHMM_group share the same underlying bed file, so
their missingness (which depends only on bed row coverage, not the joined
value) is computed once and reused.

Configuration is via the editable constants below (no argparse), mirroring
plot_labels_by_chromhmm.py.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import annotation_utils as au


# ===========================================================================
# Configuration
# ===========================================================================

STR_LEN = 4
N_FLANKING = 10000

STR_DATA_PATH = f"../../data/STR_data/HipSTR_labeled_STRs/str_len_{STR_LEN}_n_flanking_{N_FLANKING}.tsv"
SPLIT = "test"  # which `split` column value to keep (train/val/test)

ANNOTATION_TYPES = ["phyloP", "ChromHMM_state", "ChromHMM_group"]

MAX_FLANK_BP = 5000  # distance (bp) analyzed on each side of the STR
BIN_WIDTH_BP = 100  # bin width (bp) for the binned panel

SAVE_DIR = None  # None -> plt.show(); else save one PNG per annotation type

FLANK_COLORS = {"left": "#3170ad", "right": "#e8443a"}
FLANK_LABELS = {"left": "Upstream (left flank)", "right": "Downstream (right flank)"}
STR_COLOR = "#2b2b2b"


# ===========================================================================
# Loading
# ===========================================================================

def load_str_data(path, split):
	"""Load HipSTR-labeled STRs, keeping one row per locus (drop the
	reverse-complement duplicate; it shares the same position/labels) and
	restricting to the given `split` (train/val/test)."""
	df = pd.read_csv(path, sep="\t")
	df = df[~df["rev_comp"]].reset_index(drop=True)
	df = df[df["split"] == split].reset_index(drop=True)
	return df


# ===========================================================================
# Missingness accumulation
# ===========================================================================

def locus_window_bounds(str_start, str_end, chrom_len, max_flank_bp):
	"""Clip [str_start-max_flank_bp, str_end+max_flank_bp) to [0, chrom_len)."""
	return max(0, str_start - max_flank_bp), min(chrom_len, str_end + max_flank_bp)


def new_accumulators(max_flank_bp):
	return {
		"left_missing": np.zeros(max_flank_bp, dtype=np.int64),
		"left_total": np.zeros(max_flank_bp, dtype=np.int64),
		"right_missing": np.zeros(max_flank_bp, dtype=np.int64),
		"right_total": np.zeros(max_flank_bp, dtype=np.int64),
		"str_missing": 0,
		"str_total": 0,
	}


def accumulate_missing(missing_mask, window_start, str_start, str_end, max_flank_bp,
		left_missing, left_total, right_missing, right_total):
	"""missing_mask: bool array aligned to [window_start, window_start+len(mask)).

	Mutates left_missing/left_total/right_missing/right_total in place; index
	d (0-based) holds distance d+1 bp from the STR boundary. Returns
	(str_missing_delta, str_total_delta) for the caller to accumulate.

	A locus whose window was clipped near a chromosome end simply contributes
	fewer entries (only the near offsets get +1 to *_total) rather than being
	excluded entirely or padded as falsely covered.
	"""
	s_lo = str_start - window_start
	s_hi = str_end - window_start

	str_missing_delta = int(missing_mask[s_lo:s_hi].sum())
	str_total_delta = s_hi - s_lo

	left_len = s_lo
	if left_len > 0:
		# missing_mask[:s_lo] is ordered far -> near; reverse so index 0 =
		# distance 1 (the position immediately upstream of the STR).
		left_missing[:left_len] += missing_mask[:s_lo][::-1]
		left_total[:left_len] += 1

	right_len = len(missing_mask) - s_hi
	if right_len > 0:
		right_missing[:right_len] += missing_mask[s_hi:]
		right_total[:right_len] += 1

	return str_missing_delta, str_total_delta


def compute_missingness_bigwig(bw, str_df, chrom_lengths, max_flank_bp):
	result = new_accumulators(max_flank_bp)

	chrom_arr = str_df["chrom"].to_numpy()
	start_arr = str_df["str_start"].to_numpy()
	end_arr = str_df["str_end"].to_numpy()

	n_skipped = 0
	for i in range(len(str_df)):
		chrom, s, e = chrom_arr[i], int(start_arr[i]), int(end_arr[i])
		chrom_len = chrom_lengths.get(chrom)
		if chrom_len is None:
			n_skipped += 1
			continue

		w_start, w_end = locus_window_bounds(s, e, chrom_len, max_flank_bp)
		vals = bw.values(chrom, w_start, w_end, numpy=True)
		missing = np.isnan(vals)

		dm, dt = accumulate_missing(
			missing, w_start, s, e, max_flank_bp,
			result["left_missing"], result["left_total"],
			result["right_missing"], result["right_total"],
		)
		result["str_missing"] += dm
		result["str_total"] += dt

	if n_skipped:
		print(f"  Skipped {n_skipped} loci on chromosomes absent from the bigwig")

	return result


def compute_missingness_bed(bed_df, str_df, chrom_lengths, max_flank_bp):
	result = new_accumulators(max_flank_bp)

	chrom_arr = str_df["chrom"].to_numpy()
	start_arr = str_df["str_start"].to_numpy()
	end_arr = str_df["str_end"].to_numpy()

	for chrom in np.unique(chrom_arr):
		chrom_len = chrom_lengths.get(chrom)
		if chrom_len is None:
			continue
		idxs = np.nonzero(chrom_arr == chrom)[0]

		bed_sub = bed_df[bed_df["chrom"] == chrom].sort_values("start")
		b_starts = bed_sub["start"].to_numpy()
		b_ends = bed_sub["end"].to_numpy()

		qs = np.maximum(0, start_arr[idxs] - max_flank_bp)
		qe = np.minimum(chrom_len, end_arr[idxs] + max_flank_bp)

		if len(b_starts) == 0:
			lo = np.zeros(len(idxs), dtype=np.int64)
			hi = np.zeros(len(idxs), dtype=np.int64)
		else:
			lo = np.searchsorted(b_ends, qs, side="right")  # first segment with end > qs
			hi = np.searchsorted(b_starts, qe, side="left")  # first segment with start >= qe

		for k in range(len(idxs)):
			w_start, w_end = int(qs[k]), int(qe[k])
			missing = np.ones(w_end - w_start, dtype=bool)
			for j in range(lo[k], hi[k]):
				seg_s = max(b_starts[j], w_start)
				seg_e = min(b_ends[j], w_end)
				if seg_e > seg_s:
					missing[seg_s - w_start:seg_e - w_start] = False

			i = idxs[k]
			dm, dt = accumulate_missing(
				missing, w_start, int(start_arr[i]), int(end_arr[i]), max_flank_bp,
				result["left_missing"], result["left_total"],
				result["right_missing"], result["right_total"],
			)
			result["str_missing"] += dm
			result["str_total"] += dt

	return result


# ===========================================================================
# Stats
# ===========================================================================

def print_stats_table(annotation_type, result):
	rows = [
		("STR", result["str_total"], result["str_missing"]),
		("Left flank (upstream)", int(result["left_total"].sum()), int(result["left_missing"].sum())),
		("Right flank (downstream)", int(result["right_total"].sum()), int(result["right_missing"].sum())),
	]
	rows.append(("Combined flank", rows[1][1] + rows[2][1], rows[1][2] + rows[2][2]))

	df = pd.DataFrame(rows, columns=["region", "n_positions", "n_missing"])
	df["missing_rate"] = df["n_missing"] / df["n_positions"]
	print(f"\n{annotation_type} missing-rate summary:")
	print(df.to_string(index=False, formatters={
		"n_positions": "{:,}".format,
		"n_missing": "{:,}".format,
		"missing_rate": "{:.4%}".format,
	}))


# ===========================================================================
# Plotting
# ===========================================================================

def missing_rate(missing, total):
	missing = missing.astype(float)
	total = total.astype(float)
	return np.divide(missing, total, out=np.full(missing.shape, np.nan), where=total > 0)


def bin_counts(missing_arr, total_arr, bin_width):
	"""Sum missing/total within contiguous bin_width-bp bins over 1-indexed
	distances 1..len(missing_arr) (last bin may be short). Returns
	(bin_mid, binned_missing, binned_total)."""
	n = len(missing_arr)
	n_bins = -(-n // bin_width)  # ceil division
	bin_mid = np.empty(n_bins)
	binned_missing = np.zeros(n_bins, dtype=np.int64)
	binned_total = np.zeros(n_bins, dtype=np.int64)
	for b in range(n_bins):
		lo, hi = b * bin_width, min((b + 1) * bin_width, n)
		binned_missing[b] = missing_arr[lo:hi].sum()
		binned_total[b] = total_arr[lo:hi].sum()
		bin_mid[b] = (lo + 1 + hi) / 2.0
	return bin_mid, binned_missing, binned_total


def plot_missing_rate(annotation_type, result, n_loci, max_flank_bp, bin_width_bp, save_dir):
	distances = np.arange(1, max_flank_bp + 1)
	left_rate = missing_rate(result["left_missing"], result["left_total"])
	right_rate = missing_rate(result["right_missing"], result["right_total"])
	str_rate = result["str_missing"] / result["str_total"] if result["str_total"] else float("nan")

	bin_mid_l, bm_l, bt_l = bin_counts(result["left_missing"], result["left_total"], bin_width_bp)
	bin_mid_r, bm_r, bt_r = bin_counts(result["right_missing"], result["right_total"], bin_width_bp)
	left_rate_binned = missing_rate(bm_l, bt_l)
	right_rate_binned = missing_rate(bm_r, bt_r)

	sns.set_theme(style="whitegrid")
	fig, (ax_bp, ax_bin) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

	panels = [
		(ax_bp, -distances, left_rate, distances, right_rate, "Per-bp resolution"),
		(ax_bin, -bin_mid_l, left_rate_binned, bin_mid_r, right_rate_binned, f"Binned ({bin_width_bp}bp)"),
	]
	for ax, xl, yl, xr, yr, subtitle in panels:
		ax.plot(xl, yl, color=FLANK_COLORS["left"], linewidth=1.0, label=FLANK_LABELS["left"])
		ax.plot(xr, yr, color=FLANK_COLORS["right"], linewidth=1.0, label=FLANK_LABELS["right"])
		ax.axhline(str_rate, color=STR_COLOR, linestyle="--", linewidth=0.7, alpha=0.5)
		ax.scatter([0], [str_rate], color=STR_COLOR, marker="D", s=70, zorder=5, label="STR (single point)")
		ax.annotate(
			f"STR: {str_rate:.1%} (n={result['str_total']:,})",
			xy=(0, str_rate), xytext=(8, 8), textcoords="offset points", fontsize=8,
		)
		ax.set_ylabel("Missing rate")
		ax.set_ylim(bottom=0)
		ax.set_title(subtitle)
		ax.legend(loc="best", fontsize=8)

	ax_bin.set_xlabel("Distance from STR boundary (bp); negative = upstream, positive = downstream")
	fig.suptitle(
		f"{annotation_type} - missing rate around STRs "
		f"(n={n_loci:,} loci, max_flank={max_flank_bp}bp)",
		fontweight="bold",
	)
	plt.tight_layout(rect=[0, 0, 1, 0.96])

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, f"missing_rate_{annotation_type}.png")
		fig.savefig(path, dpi=150, bbox_inches="tight")
		plt.close(fig)
		print(f"  saved {path}")
	else:
		plt.show()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

	print(f"Loading STR data (motif length {STR_LEN}, split={SPLIT!r})...")
	str_df = load_str_data(STR_DATA_PATH, SPLIT)
	print(f"  {len(str_df):,} loci")

	phylop_entry = au.get_entry("phyloP")
	print("Opening phyloP bigwig for chromosome lengths...")
	phylop_bw = au.load_bigwig(phylop_entry)
	chrom_lengths = phylop_bw.chroms()

	cache = {}  # (file_type, path) -> result dict

	for annotation_type in ANNOTATION_TYPES:
		entry = au.get_entry(annotation_type)
		key = (entry["file_type"], entry["path"])

		if key not in cache:
			print(f"\nComputing missingness for {annotation_type} (path={entry['path']})...")
			if entry["file_type"] == "bigwig":
				bw = phylop_bw if entry["path"] == phylop_entry["path"] else au.load_bigwig(entry)
				cache[key] = compute_missingness_bigwig(bw, str_df, chrom_lengths, MAX_FLANK_BP)
				if bw is not phylop_bw:
					bw.close()
			elif entry["file_type"] == "bed":
				bed_df = au.load_bed(entry)
				cache[key] = compute_missingness_bed(bed_df, str_df, chrom_lengths, MAX_FLANK_BP)
			else:
				raise ValueError(f"Unknown file_type {entry['file_type']!r} for {annotation_type!r}")
		else:
			print(f"\nReusing missingness computed for path {entry['path']} (shared with another annotation type)")

		result = cache[key]
		print_stats_table(annotation_type, result)
		plot_missing_rate(annotation_type, result, len(str_df), MAX_FLANK_BP, BIN_WIDTH_BP, SAVE_DIR)

	phylop_bw.close()
