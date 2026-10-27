"""Create a per-bp G4Hunter-score bigWig annotation for one window size,
covering the genomic regions used by the labeled HipSTR STR datasets.

G4Hunter (Bedrat, Lacroix & Mergny 2016, doi 10.1093/nar/gkw006) scores
G-quadruplex-forming propensity from sequence alone. Each base gets a
per-base score from the run of identical bases it sits in: a base inside a
run of k consecutive Gs scores +min(k, 4), inside a run of k consecutive Cs
scores -min(k, 4), and A/T/N score 0. The value reported for a position is
the mean of those per-base scores over a centered window of `--window-size`
bases (must be odd, so the window is symmetric: half = (window_size - 1) //
2 bases on each side). That raw score is signed: positive = G-rich (a G4
would form on the reference forward strand), negative = C-rich (G4 on the
reverse strand). What this script WRITES is the max of the two strands --
see STRAND below -- so the stored value is non-negative G4-forming
propensity, and the paper's thresholds apply to it directly (>= 1.2
G4-forming, >= 1.5 for higher specificity).

A base's run length is a property of the reference sequence, not of the
window, so runs are measured before windowing and a run straddling a window
edge keeps its true (capped) length. That needs up to G4H_MAX_RUN - 1 bases
of context beyond each window edge, which is fetched from the FASTA on top
of the `half` window padding.

STRAND: unlike GC content, the raw G4Hunter score is NOT reverse-complement
invariant -- it flips sign, since the G runs on one strand are C runs on the
other (score_revcomp == -score_fwd at the mirrored position). Rather than
leave that sign for every downstream consumer to get right, this track
stores the MAX OF THE TWO STRANDS at each position. At a fixed coordinate
the minus-strand score of a window is exactly the negation of its
plus-strand score (the window read on the other strand IS the reverse
complement of that same window), so

    max(plus, minus) == max(s, -s) == abs(s)

and the strand max is just abs() of the forward score. The stored value is
therefore non-negative, strand-agnostic G4 propensity: identical on a
locus's fwd and rev_comp rows, needing no negation anywhere downstream, and
safe to correlate directly against orientation-invariant labels.

The abs is taken on the WINDOW MEAN, not per base, and that distinction is
the whole point of the score: a window whose G runs and C runs offset each
other stays near zero, which is correct -- such a window is not G4-forming
on either strand. Taking abs per base first would instead measure G+C
run-richness, a different and far more GC-content-like quantity.

The directional score is still available in code as
g4hunter_signed_score_track() if a future analysis needs G4s oriented
against something meaningful (replication fork direction, transcription
direction); the reference strand itself is not that.

For each STR locus (from the str_len_<k> tables selected by --str-lens,
default motif lengths 3 and 4), the covered region is [str_start -
n_flanking, str_end + n_flanking) -- i.e. the STR plus its flanking window,
matching how the STR tsvs themselves were built (see
create_STR_data_files/create_STR_data_files.py) and matching the coverage of
the sibling create_GC_content_anno_file.py, so the two tracks have data at
exactly the same positions. Regions are unioned across the selected
motif-length tables and merged per chromosome before scoring, so overlapping
loci (STRs are dense enough that +/-10kb windows overlap heavily) aren't
recomputed. This is NOT a genome-wide track -- positions outside every
locus's covered region have no data, matching how missing bigWig data reads
back as NaN via pyBigWig, same as any other continuous annotation in this
pipeline.

N bases are excluded from the window's denominator (and contribute 0 to its
numerator), and a window with zero valid bases is written as no data
(NaN on read-back). This mirrors create_GC_content_anno_file.py and differs
from the reference G4Hunter implementation, which divides by the full window
width: excluding N keeps a window that happens to contain assembly gaps
comparable to an N-free one instead of diluting its score toward 0. N is
rare inside the covered regions. True chromosome ends likewise truncate the
window, and the mean is taken over however many valid bases are available.

Output: one bigWig at <output-dir>/G4Hunter_<window_size>bp.bw, intended to
be registered in scripts/annotation_analysis/annotation_config.yaml as
`G4Hunter_<window_size>bp` (file_type bigwig, value_type continuous) so the
"G4Hunter" family resolves across window sizes.

CLI arguments:
	-w, --window-size: Centered window width in bp. Must be odd. Required.
		G4Hunter's canonical window is 25.
	--str-data-dir: Directory of str_len_<k>_n_flanking_<n>.tsv tables.
		Default: "../STR_data/HipSTR_labeled_STRs".
	--str-lens: Motif lengths (str_len_<k> tables) to include. Default 3 4.
	--n-flanking: Flanking bases on each side of the STR to cover (must
		match the tables' filename suffix). Default 10000.
	--ref-genome: Path to reference genome FASTA.
		Default: "../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa".
	--output-dir: Directory to save the bigWig.
		Default: "../annotations/G4Hunter".

Example usage:
	python create_G4Hunter_anno_file.py --window-size 25
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta


# Per-base G4Hunter score caps at a run of this many identical bases.
G4H_MAX_RUN = 4

# Extra bases fetched beyond each window edge so that a G/C run straddling
# the edge still gets its true (capped) length: seeing G4H_MAX_RUN - 1 more
# bases is always enough to tell whether a run reaches the cap.
RUN_PAD = G4H_MAX_RUN - 1


# ======================================================================
# CLI
# ======================================================================

def get_args():
	parser = argparse.ArgumentParser(
		description=(
			"Create a per-bp G4Hunter-score bigWig for one window size, "
			"covering the STR +/- flanking regions used by the labeled "
			"HipSTR STR datasets."
		)
	)

	parser.add_argument(
		"-w", "--window-size",
		type=int, required=True,
		help="Centered window width in bp. Must be odd. G4Hunter's canonical "
		     "window is 25."
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
		type=str, default="../annotations/G4Hunter",
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
# G4Hunter computation
# ======================================================================

def run_lengths(mask):
	"""For a boolean array, an int array holding at each True position the
	length of the contiguous True run containing it (0 at False positions)."""
	out = np.zeros(mask.size, dtype=np.int64)
	true_idx = np.flatnonzero(mask)
	if true_idx.size == 0:
		return out
	# Run boundaries: indices where the padded mask changes value, so
	# edges[0::2] are run starts and edges[1::2] are run ends (exclusive).
	padded = np.concatenate(([False], mask, [False]))
	edges = np.flatnonzero(padded[1:] != padded[:-1])
	lengths = edges[1::2] - edges[0::2]
	# flatnonzero walks the True positions run by run in order, so repeating
	# each run's length by itself lines up one length per True position.
	out[true_idx] = np.repeat(lengths, lengths)
	return out


def g4hunter_base_scores(seq):
	"""(scores, is_valid) for an uppercase sequence string `seq`.

	scores[i] = +min(run, G4H_MAX_RUN) if base i sits in a run of `run`
	consecutive Gs, -min(run, G4H_MAX_RUN) if in a run of Cs, else 0.
	is_valid[i] = base i is not N.
	"""
	arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
	scores = np.zeros(arr.size, dtype=np.float64)
	# G runs and C runs are disjoint, so the two contributions just add.
	for base, sign in (("G", 1.0), ("C", -1.0)):
		runs = run_lengths(arr == ord(base))
		np.clip(runs, 0, G4H_MAX_RUN, out=runs)
		scores += sign * runs
	return scores, arr != ord("N")


def g4hunter_signed_score_track(seq, half):
	"""Given an uppercase sequence `seq` (str), returns a float array of
	length len(seq): for each position i, the mean per-base G4Hunter score
	over the centered window [i - half, i + half] clipped to the sequence
	bounds (denominator is the count of non-N bases actually in the clipped
	window; N bases contribute 0 to the numerator). NaN where the clipped
	window has zero valid bases.

	This is the raw, signed G4Hunter score (positive = G-rich on the strand
	given, negative = C-rich). The bigWig stores the strand max instead --
	see g4hunter_strand_max_score_track()."""
	scores, is_valid = g4hunter_base_scores(seq)

	cs_score = np.concatenate(([0.0], np.cumsum(scores)))
	cs_valid = np.concatenate(([0], np.cumsum(is_valid, dtype=np.int64)))

	n = len(seq)
	idx = np.arange(n)
	win_start = np.clip(idx - half, 0, n)
	win_end = np.clip(idx + half + 1, 0, n)

	sum_score = cs_score[win_end] - cs_score[win_start]
	sum_valid = cs_valid[win_end] - cs_valid[win_start]

	with np.errstate(invalid="ignore", divide="ignore"):
		mean = sum_score / sum_valid
	mean[sum_valid == 0] = np.nan
	return mean


def g4hunter_strand_max_score_track(seq, half):
	"""The value this script writes: at each position, the larger of the
	two strands' G4Hunter window scores.

	The minus-strand score of a window is the negation of its plus-strand
	score, so max(plus, minus) == abs(plus) exactly -- no second pass over
	the reverse complement is needed. Non-negative, and identical whichever
	strand a locus is read from. NaN propagates from the signed track.
	"""
	return np.abs(g4hunter_signed_score_track(seq, half))


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

	print("\nMerging per-chromosome intervals and computing G4Hunter scores...")
	chrom_order = sorted(regions.keys(), key=natural_chrom_sort_key)
	header = [(c, len(ref_genome[c])) for c in chrom_order]

	os.makedirs(args.output_dir, exist_ok=True)
	out_path = os.path.join(
		args.output_dir, f"G4Hunter_{args.window_size}bp.bw"
	)
	bw = pyBigWig.open(out_path, "w")
	bw.addHeader(header)

	total_bp = 0
	total_intervals = 0
	score_min, score_max = np.inf, -np.inf
	score_sum, score_n = 0.0, 0

	for chrom, chrom_len in header:
		merged = merge_intervals(regions[chrom])
		merged = [
			(max(0, s), min(chrom_len, e)) for s, e in merged if s < chrom_len
		]
		total_intervals += len(merged)
		print(f"  {chrom}: {len(regions[chrom])} loci -> {len(merged)} merged intervals")

		for start, end in merged:
			# RUN_PAD past the window padding so G/C runs crossing the
			# fetched sequence's edge aren't measured as shorter than they
			# are; those pad bases affect run lengths only, never a window.
			fetch_start = max(0, start - half - RUN_PAD)
			fetch_end = min(chrom_len, end + half + RUN_PAD)
			seq = str(ref_genome[chrom][fetch_start:fetch_end])

			track = g4hunter_strand_max_score_track(seq, half)
			# Slice out just the covered [start, end) portion (drop the
			# extra padding fetched only to compute edge windows and runs).
			out_vals = track[start - fetch_start: end - fetch_start]

			write_chrom_values(bw, chrom, start, out_vals)

			total_bp += end - start
			finite_vals = out_vals[np.isfinite(out_vals)]
			if finite_vals.size:
				score_min = min(score_min, float(finite_vals.min()))
				score_max = max(score_max, float(finite_vals.max()))
				score_sum += float(finite_vals.sum())
				score_n += finite_vals.size

	bw.close()

	print(f"\nWrote {out_path}")
	print(f"  chromosomes:      {len(header)}")
	print(f"  merged intervals: {total_intervals}")
	print(f"  total bp covered: {total_bp}")
	if score_n:
		print(
			f"  G4Hunter strand-max score min/mean/max: {score_min:.4f} / "
			f"{score_sum / score_n:.4f} / {score_max:.4f}"
		)
