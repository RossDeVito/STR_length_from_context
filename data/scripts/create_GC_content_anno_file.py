"""Create a per-bp local-GC-content bigWig annotation for one window size,
covering the genomic regions used by the labeled HipSTR STR datasets.

For each STR locus (from the str_len_<k> tables selected by --str-lens,
default motif lengths 3 and 4), the covered region is [str_start -
n_flanking, str_end + n_flanking) -- i.e. the STR plus its flanking window,
matching how the STR tsvs themselves were built (see
create_STR_data_files/create_STR_data_files.py). Regions are unioned across
the selected motif-length tables and merged per chromosome before computing
GC, so overlapping loci (STRs are dense enough that +/-10kb windows overlap
heavily) aren't recomputed. This is NOT a genome-wide track -- positions
outside every locus's covered region have no data, matching how missing
bigWig data reads back as NaN via pyBigWig, same as any other continuous
annotation in this pipeline.

Per-bp value = GC fraction of a centered window of `--window-size` bases
(must be odd, so the window is symmetric around each output position: half =
(window_size - 1) // 2 bases on each side). The window's sequence is pulled
directly from the reference FASTA (not from the covered-region union), so
edge positions near a covered region's boundary still get a full window --
only true chromosome ends truncate the window, in which case the GC
fraction is computed over however many valid bases are actually available
(sum of G/C over sum of non-N bases in the truncated window).

GC fraction is reverse-complement invariant (G<->C and A<->T swaps preserve
the G+C count), so only the forward strand is computed here; it applies
unchanged to both a locus's fwd and rev_comp rows in the STR tables.

Output: one bigWig at <output-dir>/GC_content_<window_size>bp.bw, intended
to be registered in scripts/annotation_analysis/annotation_config.yaml as
`GC_content_<window_size>bp` (file_type bigwig, value_type continuous) so
the "GC_content" family resolves across window sizes.

CLI arguments:
	-w, --window-size: Centered window width in bp. Must be odd. Required.
	--str-data-dir: Directory of str_len_<k>_n_flanking_<n>.tsv tables.
		Default: "../STR_data/HipSTR_labeled_STRs".
	--str-lens: Motif lengths (str_len_<k> tables) to include. Default 3 4.
	--n-flanking: Flanking bases on each side of the STR to cover (must
		match the tables' filename suffix). Default 10000.
	--ref-genome: Path to reference genome FASTA.
		Default: "../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa".
	--output-dir: Directory to save the bigWig.
		Default: "../annotations/GC_content".

Example usage:
	python create_GC_content_anno_file.py --window-size 101
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta


# ======================================================================
# CLI
# ======================================================================

def get_args():
	parser = argparse.ArgumentParser(
		description=(
			"Create a per-bp local-GC-content bigWig for one window size, "
			"covering the STR +/- flanking regions used by the labeled "
			"HipSTR STR datasets."
		)
	)

	parser.add_argument(
		"-w", "--window-size",
		type=int, required=True,
		help="Centered window width in bp. Must be odd."
	)

	parser.add_argument(
		"--str-data-dir",
		type=str, default="../STR_data/HipSTR_labeled_STRs",
		help="Directory of str_len_<k>_n_flanking_<n>.tsv tables."
	)
	parser.add_argument(
		"--str-lens",
		type=int, nargs="+", default=[3, 4],
		help="Motif lengths (str_len_<k> tables) to include. Default 3 4."
	)
	parser.add_argument(
		"--n-flanking",
		type=int, default=10000,
		help=(
			"Flanking bases on each side of the STR to cover. Must match "
			"the tables' filename suffix. Default 10000."
		)
	)
	parser.add_argument(
		"--ref-genome",
		type=str,
		default="../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa",
	)
	parser.add_argument(
		"--output-dir",
		type=str, default="../annotations/GC_content",
	)

	return parser.parse_args()


# ======================================================================
# Regions
# ======================================================================

def load_covered_regions(str_data_dir, n_flanking, str_lens):
	"""Loads the str_len_<k>_n_flanking_<n_flanking>.tsv table for each k in
	str_lens and returns a dict chrom -> list of (start, end) 0-based
	half-open (str_start - n_flanking, str_end + n_flanking), one per locus
	(pre-merge)."""
	paths = []
	for str_len in str_lens:
		path = os.path.join(
			str_data_dir, f"str_len_{str_len}_n_flanking_{n_flanking}.tsv"
		)
		if not os.path.exists(path):
			raise FileNotFoundError(f"STR table not found: {path!r}")
		paths.append(path)

	regions = {}
	for path in paths:
		df = pd.read_csv(
			path, sep="\t", usecols=["chrom", "str_start", "str_end"]
		)
		print(f"  {os.path.basename(path)}: {len(df)} rows")
		for chrom, start, end in zip(
			df["chrom"], df["str_start"], df["str_end"]
		):
			regions.setdefault(chrom, []).append(
				(start - n_flanking, end + n_flanking)
			)
	return regions


def merge_intervals(intervals):
	"""Sorts and merges overlapping/touching 0-based half-open intervals."""
	if not intervals:
		return []
	intervals = sorted(intervals)
	merged = [list(intervals[0])]
	for start, end in intervals[1:]:
		if start <= merged[-1][1]:
			merged[-1][1] = max(merged[-1][1], end)
		else:
			merged.append([start, end])
	return [(s, e) for s, e in merged]


def natural_chrom_sort_key(chrom):
	num_part = chrom.replace("chr", "")
	return (0, int(num_part)) if num_part.isdigit() else (1, num_part)


# ======================================================================
# GC computation
# ======================================================================

def gc_fraction_track(seq, half):
	"""Given an uppercase sequence `seq` (str), returns a float array of
	length len(seq): for each position i, the GC fraction of the centered
	window [i - half, i + half] clipped to the sequence bounds (denominator
	is the count of non-N bases actually in the clipped window). NaN where
	the clipped window has zero valid bases."""
	arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
	is_gc = (arr == ord("G")) | (arr == ord("C"))
	is_valid = arr != ord("N")

	cs_gc = np.concatenate(([0], np.cumsum(is_gc, dtype=np.int64)))
	cs_valid = np.concatenate(([0], np.cumsum(is_valid, dtype=np.int64)))

	n = len(seq)
	idx = np.arange(n)
	win_start = np.clip(idx - half, 0, n)
	win_end = np.clip(idx + half + 1, 0, n)

	sum_gc = cs_gc[win_end] - cs_gc[win_start]
	sum_valid = cs_valid[win_end] - cs_valid[win_start]

	with np.errstate(invalid="ignore", divide="ignore"):
		frac = sum_gc / sum_valid
	frac[sum_valid == 0] = np.nan
	return frac


def write_chrom_values(bw, chrom, start, values):
	"""Writes `values` (may contain NaN) starting at 0-based `start`, split
	into contiguous finite runs (fixed-step, span=1, step=1 per run)."""
	finite = np.isfinite(values)
	if not finite.any():
		return
	# Indices where a finite run starts/ends.
	edges = np.flatnonzero(np.diff(np.concatenate(([0], finite.view(np.int8), [0]))))
	for run_start, run_end in edges.reshape(-1, 2):
		bw.addEntries(
			chrom, int(start + run_start),
			values=[float(v) for v in values[run_start:run_end]],
			span=1, step=1,
		)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
	args = get_args()

	if args.window_size <= 0 or args.window_size % 2 == 0:
		print(
			f"--window-size must be a positive odd number "
			f"(got {args.window_size}); a centered window needs equal "
			f"flanking bases on each side."
		)
		sys.exit(1)
	half = (args.window_size - 1) // 2

	print("Settings:")
	print(f"  Window size:   {args.window_size} (half={half})")
	print(f"  n_flanking:    {args.n_flanking}")
	print(f"  STR data dir:  {args.str_data_dir}")
	print(f"  STR lens:      {args.str_lens}")
	print(f"  Ref genome:    {args.ref_genome}")

	print("\nLoading covered regions from STR tables...")
	regions = load_covered_regions(args.str_data_dir, args.n_flanking, args.str_lens)
	n_loci = sum(len(v) for v in regions.values())
	print(f"  total loci (pre-merge, all lengths): {n_loci}")

	print("\nOpening reference genome...")
	ref_genome = Fasta(args.ref_genome, sequence_always_upper=True)

	print("\nMerging per-chromosome intervals and computing GC...")
	chrom_order = sorted(regions.keys(), key=natural_chrom_sort_key)
	header = [(c, len(ref_genome[c])) for c in chrom_order]

	os.makedirs(args.output_dir, exist_ok=True)
	out_path = os.path.join(
		args.output_dir, f"GC_content_{args.window_size}bp.bw"
	)
	bw = pyBigWig.open(out_path, "w")
	bw.addHeader(header)

	total_bp = 0
	total_intervals = 0
	gc_min, gc_max, gc_sum, gc_n = 1.0, 0.0, 0.0, 0

	for chrom, chrom_len in header:
		merged = merge_intervals(regions[chrom])
		merged = [
			(max(0, s), min(chrom_len, e)) for s, e in merged if s < chrom_len
		]
		total_intervals += len(merged)
		print(f"  {chrom}: {len(regions[chrom])} loci -> {len(merged)} merged intervals")

		for start, end in merged:
			fetch_start = max(0, start - half)
			fetch_end = min(chrom_len, end + half)
			seq = str(ref_genome[chrom][fetch_start:fetch_end])

			track = gc_fraction_track(seq, half)
			# Slice out just the covered [start, end) portion (drop the
			# extra `half` padding fetched only to compute edge windows).
			out_vals = track[start - fetch_start: end - fetch_start]

			write_chrom_values(bw, chrom, start, out_vals)

			total_bp += end - start
			finite_vals = out_vals[np.isfinite(out_vals)]
			if finite_vals.size:
				gc_min = min(gc_min, float(finite_vals.min()))
				gc_max = max(gc_max, float(finite_vals.max()))
				gc_sum += float(finite_vals.sum())
				gc_n += finite_vals.size

	bw.close()

	print(f"\nWrote {out_path}")
	print(f"  chromosomes:      {len(header)}")
	print(f"  merged intervals: {total_intervals}")
	print(f"  total bp covered: {total_bp}")
	if gc_n:
		print(f"  GC fraction min/mean/max: {gc_min:.4f} / {gc_sum / gc_n:.4f} / {gc_max:.4f}")
