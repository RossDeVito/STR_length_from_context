"""Test whether the TE-adjacency effect on STR copy number / heterozygosity is
driven by poly(A) tails.

Background
----------
nearby_repeat_elements.py found that STRs sitting at a RepeatMasker element
boundary have markedly higher copy number and heterozygosity than STRs that do
not (Cliff's delta ~ +0.78 at str_len=4), and that the shift survives
stratifying by motif. That script proposed ASCERTAINMENT as the mechanism --
an STR that is part of an element gets annotated as inside it and dropped by
create_STR_data_files.py, so the boundary loci that survive are enriched for
tracts that expanded past the alignable region.

This script tests a specific, mechanistic version of that story. L1, Alu and
SVA are poly(A) retrotransposons: they insert with a poly(A) tail at their 3'
end, and that tail decays over time into an A-rich short tandem repeat
(AAAT, AAAC, AAAAT, ...). If the effect is tails, it has a signature:

  * it concentrates DOWNSTREAM of poly(A) elements -- past the 3' end, where
    the tail is -- and is weak upstream, past the 5' end, where there is none;
  * it concentrates in STRs that are A-rich ON THE ELEMENT'S SENSE STRAND,
    because that is the strand the tail was written on;
  * it is weak for elements that do not insert with a tail (DNA transposons,
    LTRs, MIR, L2);
  * it shrinks once putative tails are removed from the adjacent set.

A generic "elements are near long STRs" effect predicts none of those
asymmetries. The comparisons below are arranged so that each of the four
signatures is a separate test with its own control.

Orientation is the whole analysis
---------------------------------
"Downstream" is defined in the ELEMENT's orientation, not the genome's. An STR
to the genomic LEFT of a + strand element is upstream of nothing -- it is past
nothing -- but an STR to the genomic left of a - strand element is past that
element's 5' end. The mapping (annotate_te_adjacency) is:

  TE strand   TE lies    facing end   gap              position
  +           left       3'           s0 - genoEnd     downstream
  +           right      5'           genoStart - e0   upstream
  -           right      3'           genoStart - e0   downstream
  -           left       5'           s0 - genoEnd     upstream

Because this is the one thing that, if flipped, would silently invert every
result, two checks guard it. RUN_STRAND_CHECK prints the sense-oriented
reference sequence (last STRAND_CHECK_FLANK bp of the element, then the gap,
then the STR) for a handful of P-down loci on each strand, so the A-tail can
be read off by eye. And comparison 7 is a pure count: among P-down loci whose
canonical motif has unequal A and T counts, the fraction whose sense motif is
the A-rich orientation. Under the tail hypothesis that fraction is well above
0.5; under flipped strand logic it is well below. If comparison 7 is not well
above 0.5, fix the strand logic before reading anything else.

Scores
------
Motif composition differs sharply between the adjacent and background sets
(AAAT alone is 34% of str_len=4 loci), and motif is the single largest
determinant of both labels, so nothing here is compared without holding it
fixed. Every locus gets a per-locus score against the background loci with the
SAME canonical motif:

  u_L(i) = [ #(B_m with L < L_i) + 0.5 * #(B_m with L == L_i) ] / |B_m|

u is the locus's motif-conditional percentile, tie-corrected the same way
Mann-Whitney U is. Its group mean has a direct reading: 2*mean(u) - 1 is the
motif-stratified Cliff's delta of that group against background, a
|G|-weighted average of the per-motif deltas. That lets every comparison pool
across motifs while staying free of composition effects, instead of running 32
separate underpowered tests. (The per-motif tests are still available, as the
secondary analysis.)

Comparison 9 repeats part of the analysis with a second score that also holds
LENGTH fixed: heterozygosity ranked against only the N_MATCH background loci
of the same canonical motif with the nearest copy number. Heterozygosity rises
with tract length almost mechanically, so without this a het result is only a
restatement of the copy-number result. What survives the length-matched score
is heterozygosity beyond what the length explains.

Groups
------
A locus is assigned by how many TE rows it is adjacent to (0 <= gap <= D):

  0 rows   -> B, the background
  1 row    -> P/N (poly(A) family or not) x down/up, four disjoint groups
  >1 rows  -> dropped from everything, including B: its flanks carry mixed
              evidence, and assigning it to one group would attribute the
              label to an arbitrary one of the elements
  overlaps -> dropped. create_STR_data_files.py already dropped every locus
              overlapping any row of mobile_elements.bed.gz, whose rows are
              exactly the rows read here, so this count should be 0; it is
              computed anyway as a check that the two files are in register.

P ("poly(A)-type") is LINE/L1, SINE/Alu and Retroposon/SVA -- the three
families that retrotranspose through a poly(A)-primed mechanism. N is every
other TE row: DNA, LTR, RC, and the non-L1/non-Alu LINEs and SINEs (L2, MIR),
which are ancient and tail-less in the annotation. repClass and repFamily are
kept per locus so N can be narrowed to DNA/LTR only by editing the group masks.

Significance
------------
  vs background    permutation p (group membership reshuffled WITHIN each
                   canonical motif, N_PERM draws -- the stratification is the
                   null's, not just the statistic's), percentile bootstrap CI
                   over focal loci.
  between groups   Cliff's delta on the u-scores, two-sided Mann-Whitney p,
                   bootstrap CI.
  comparison 7     two-sided binomial test against 0.5.

Every primary comparison in one run is one multiple-testing family, corrected
by CORRECTION. The secondary per-motif tests get their own, separate family --
they are a breakdown of the primary result, not additional independent
questions, and folding them in would dilute the primaries.

CORRECTION follows str_anno_overlap_analysis.py's CLIFFS_DELTA_CORRECTION:
"fdr_bh" (the default here) controls the expected false discovery rate, which
suits a battery of comparisons most of which are expected to be real;
"bonferroni" controls the family-wise error rate, which is the stricter claim
-- worth switching to when a single comparison is going to be quoted on its
own, at the cost of power on the small groups (SVA, the per-motif tests).
None skips correction and compares the raw p to CORRECTION_ALPHA.

Inputs
------
STR_DATA_PATH   the training/eval locus set, filtered to MOTIF_LEN. str_start
                and str_end in that file are already s0 = (HipSTR start - 1)
                and e0 = end_ref (see create_STR_data_files.py) -- 0-based
                half-open, built from end_ref and never from the data-adaptive
                end_call -- so they are used directly.
RMSK_BED_PATH   rmsk_stranded.bed.gz, from download_rmsk_stranded.sh. Same
                source, release and repClass filter as the mobile_elements.bed.gz
                that nearby_repeat_elements.py and the dataset build use --
                verified row-for-row identical on the shared columns -- plus
                strand, repName and repFamily, which that 4-column file drops
                and which this analysis cannot run without.
REF_GENOME_PATH hg38, for the plus-strand motif (the dataset's motif column is
                rotation-canonicalized, so its phase is not the locus's) and
                for the strand check.

Configuration is via the editable constants below (no argparse), mirroring the
rest of this directory. One run covers one MOTIF_LEN.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyfaidx import Fasta
from scipy.stats import mannwhitneyu, binomtest
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# ===========================================================================
# Configuration
# ===========================================================================

MOTIF_LEN = 3
N_FLANKING = 10000  # must match the STR tsv's filename suffix

STR_DATA_PATH = f"../../data/STR_data/HipSTR_labeled_STRs/str_len_{MOTIF_LEN}_n_flanking_{N_FLANKING}.tsv"
# 7-column bed (chrom, start, end, strand, repName, repClass, repFamily) from
# data/invalid_regions/download_rmsk_stranded.sh.
RMSK_BED_PATH = "../../data/invalid_regions/rmsk_stranded.bed.gz"
REF_GENOME_PATH = "../../data/reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa"

# Adjacency distance, bp. A locus is adjacent to a TE row when the gap between
# them, measured to the facing end, is 0 <= gap <= D.
D = 32

# A P-down locus is a putative decayed tail when its sense-strand motif is
# more than this fraction A. At MOTIF_LEN=4 that means >= 3 of 4 (AAAT, AAAC,
# AAAG, AAAA); at 5, >= 3 of 5 (AAAAT, AAATA, ...).
TAIL_MIN_A = 0.5

# Background loci per focal locus for the length-matched heterozygosity score.
N_MATCH = 99
# Minimum focal loci for a canonical motif to get its own secondary result.
MIN_N = 30

TARGETS = {
	"length": "mode_copy_number",
	"heterozygosity": "heterozygosity",
}

# repClass values (trailing "?" stripped) that count as a TE at all. Everything
# else in the bed -- Satellite, the RNA families -- is ignored entirely: it is
# neither a TE for the adjacency groups nor a reason to exclude a locus.
TE_CLASSES = {"SINE", "LINE", "LTR", "DNA", "Retroposon", "RC"}
# (repClass, repFamily) pairs, "?" stripped, that insert with a poly(A) tail.
POLYA_FAMILIES = {("LINE", "L1"), ("SINE", "Alu"), ("Retroposon", "SVA")}

# Permutation draws for the vs-background p-values. This also sets the
# smallest p those tests can report, 1/(N_PERM + 1), which has to survive the
# multiple-testing correction -- see check_permutation_floor, which warns when
# it does not. 10,000 leaves plenty of room for the ~50 and ~75 test families
# this script runs.
N_PERM = 10000
N_BOOT = 1000       # bootstrap resamples for every CI
CI_LEVEL = 0.95
# Multiple-testing correction, applied separately within the primary family
# and the secondary family. None | "bonferroni" | "fdr_bh" (Benjamini-
# Hochberg), mirroring CLIFFS_DELTA_CORRECTION in str_anno_overlap_analysis.py.
# None means the raw per-test p-value is compared directly to CORRECTION_ALPHA.
CORRECTION = "bonferroni"
CORRECTION_ALPHA = 0.01    # significance threshold, applied post-correction
# Permutations/bootstraps are generated in chunks of this many draws at a time
# so the random matrices stay small; it does not affect the results.
DRAW_CHUNK = 512

SEED = 147

# Strand-logic check: print the sense-oriented reference around a few P-down
# loci before any statistics, so the orientation can be verified by eye.
RUN_STRAND_CHECK = True
N_STRAND_CHECK = 5          # loci per TE strand
STRAND_CHECK_FLANK = 20     # bp of the element's 3' end to show

# Output. SAVE_DIR=None -> plt.show(); SAVE_RESULTS_PATH=None -> no json.
SAVE_DIR = None
SAVE_RESULTS_PATH = None


# Group labels, used as dict keys and in the report. "All down"/"All up" are
# unions of the P and N groups at that position, not separate assignments.
G_BACKGROUND = "B"
G_P_DOWN, G_P_UP = "P down", "P up"
G_N_DOWN, G_N_UP = "N down", "N up"
G_ALL_DOWN, G_ALL_UP = "All down", "All up"
G_DOWN_NO_TAIL = "All down, no tails"
# The whole near-TE effect: every locus adjacent to exactly one TE row. This
# is comparison 10's denominator -- the thing the poly(A) share is a share OF.
G_ALL_ADJ = "All adjacent"
# Comparison 10's two splits of the adjacent set. Each pair is disjoint and
# together covers G_ALL_ADJ, which is what makes the share exact and keeps the
# test from comparing a group with itself (P down alone is ~57% of adjacent).
G_TAILS = "Putative tails"
G_NOT_P_DOWN = "Adjacent, not P down"
G_NOT_TAIL = "Adjacent, not a tail"

ASSIGNED_GROUPS = [G_P_DOWN, G_P_UP, G_N_DOWN, G_N_UP]

# (version, T group, O group) for comparison 10. 10a is the a priori split --
# position and element class only. 10b is the mechanistic one, which also uses
# the sense-A threshold and so is the sharper test of the tail hypothesis.
COMPARISON_10_SPLITS = [
	("10a", G_P_DOWN, G_NOT_P_DOWN),
	("10b", G_TAILS, G_NOT_TAIL),
]


# ===========================================================================
# Loading
# ===========================================================================

def load_str_data(path):
	"""Load the labeled STR set, one row per locus.

	Each locus appears twice in the file, once with rev_comp=True; the two
	rows share a position and labels and differ only in how the model reads
	the sequence, so the duplicate is dropped (as in nearby_repeat_elements.py).
	str_start/str_end are already s0/e0 -- see the module docstring.
	"""
	df = pd.read_csv(path, sep="\t")
	df = df[~df["rev_comp"]].reset_index(drop=True)
	return df


def _strip_q(value):
	"""Drop RepeatMasker's "?" uncertainty suffix (LTR? -> LTR, L1? -> L1).

	A questioned call is still the best available call; keeping it separate
	would only split a family into two underpowered pieces.
	"""
	return value[:-1] if value.endswith("?") else value


def load_te_bed(path, keep_chroms):
	"""Load the stranded rmsk bed, restricted to TE_CLASSES and `keep_chroms`.

	Adds `te_class`/`te_family` ("?" stripped) and `is_polya`. Returns the
	TE rows and a count of the non-TE rows (Satellite, RNA) that were read
	and dropped, for the report.
	"""
	bed = pd.read_csv(
		path, sep="\t", header=None, compression="infer",
		names=["chrom", "start", "end", "strand", "rep_name",
		       "raw_class", "raw_family"],
	)
	bed = bed[bed["chrom"].isin(keep_chroms)]

	bed["te_class"] = [_strip_q(c) for c in bed["raw_class"]]
	bed["te_family"] = [_strip_q(f) for f in bed["raw_family"]]

	is_te = bed["te_class"].isin(TE_CLASSES)
	n_non_te = int((~is_te).sum())
	bed = bed[is_te].reset_index(drop=True)

	bed["is_polya"] = [
		(c, f) in POLYA_FAMILIES
		for c, f in zip(bed["te_class"], bed["te_family"])
	]
	return bed, n_non_te


# ===========================================================================
# Motifs
# ===========================================================================

_RC_TABLE = str.maketrans("ACGT", "TGCA")


def rev_comp(seq):
	return seq.translate(_RC_TABLE)[::-1]


def _rotations(motif):
	return [motif[i:] + motif[:i] for i in range(len(motif))]


def canonical_motif(motif):
	"""Lexicographically smallest rotation of the motif or its reverse
	complement.

	This is the stratum: it collapses both phase (AAAT/AATA/ATAA/TAAA) and
	strand (AAAT/ATTT), so the background pool for a locus is every
	background locus that is the same repeat, however the reference happened
	to write it down. Collapsing strand is what makes the pool big enough to
	rank against, and costs nothing here -- orientation is carried separately
	by the sense motif, which is the quantity the A-content comparisons use.
	"""
	return min(min(_rotations(motif)), min(_rotations(rev_comp(motif))))


def read_plus_strand_motifs(str_df, ref, motif_len):
	"""The first `motif_len` bases of each tract on the PLUS strand.

	Read from the reference rather than taken from the dataset's `motif`
	column: that column is one rotation-canonical representative per rotation
	class, so its phase is not the locus's (and it is NaN for a handful of
	loci). Orientation it does preserve -- verified: where the column is
	present it is always a rotation of the string read here -- but phase and
	completeness both matter, and the reference is the ground truth for both.

	Tracts are perfect repeats by construction (create_STR_data_files.py
	checks), so the first motif_len bases are the motif.
	"""
	motifs = np.empty(len(str_df), dtype=object)
	chroms = str_df["chrom"].to_numpy()
	starts = str_df["str_start"].to_numpy()

	for chrom in tqdm(np.unique(chroms), desc="  plus-strand motifs"):
		rows = np.nonzero(chroms == chrom)[0]
		contig = ref[chrom]
		for i in rows:
			motifs[i] = contig[starts[i]:starts[i] + motif_len].upper()
	return motifs


def a_fraction(motif):
	"""Fraction of the motif that is A. Rotation-invariant, so it is a
	property of the repeat's orientation and not of where the reference
	happened to cut the phase."""
	return motif.count("A") / len(motif)


# ===========================================================================
# TE adjacency and orientation
# ===========================================================================

def annotate_te_adjacency(str_df, te_df, d):
	"""Per-locus TE context: overlap, adjacent-row count, and -- when exactly
	one row is adjacent -- that row's side, strand, class, family and gap.

	Everything is vectorized per chromosome via searchsorted; the only Python
	loop is over chromosomes.

	  overlap       any TE row intersecting [s0, e0). Found with the standard
	                running-max-of-ends test over rows sorted by start: among
	                the rows starting before e0, an overlap exists iff the
	                largest end among them exceeds s0.
	  left adjacent genoEnd in [s0 - d, s0], i.e. gap = s0 - genoEnd in [0, d]
	  right adjacent genoStart in [e0, e0 + d], i.e. gap = genoStart - e0 in [0, d]

	A row cannot be both left- and right-adjacent (it would have to lie
	entirely on both sides), and an adjacent row cannot also overlap
	(adjacency requires it to end at or before s0, or start at or after e0),
	so the counts are clean partitions.
	"""
	n = len(str_df)
	chroms = str_df["chrom"].to_numpy()
	s0 = str_df["str_start"].to_numpy().astype(np.int64)
	e0 = str_df["str_end"].to_numpy().astype(np.int64)

	overlaps = np.zeros(n, dtype=bool)
	n_adjacent = np.zeros(n, dtype=np.int32)
	te_row = np.full(n, -1, dtype=np.int64)   # index into te_df; -1 = none
	side = np.full(n, "", dtype=object)

	for chrom in tqdm(np.unique(chroms), desc="  TE adjacency"):
		rows = np.nonzero(chroms == chrom)[0]
		sub = te_df[te_df["chrom"] == chrom]
		if sub.empty:
			continue

		te_idx = sub.index.to_numpy()
		starts = sub["start"].to_numpy().astype(np.int64)
		ends = sub["end"].to_numpy().astype(np.int64)
		ls0, le0 = s0[rows], e0[rows]

		# --- overlap ---
		by_start = np.argsort(starts, kind="stable")
		starts_sorted = starts[by_start]
		running_max_end = np.maximum.accumulate(ends[by_start])
		# j = how many rows start strictly before e0
		j = np.searchsorted(starts_sorted, le0, side="left")
		overlaps[rows] = (j > 0) & (running_max_end[np.maximum(j - 1, 0)] > ls0)

		# --- left-adjacent: genoEnd in [s0 - d, s0] ---
		by_end = np.argsort(ends, kind="stable")
		ends_sorted = ends[by_end]
		lo_l = np.searchsorted(ends_sorted, ls0 - d, side="left")
		hi_l = np.searchsorted(ends_sorted, ls0, side="right")
		n_left = hi_l - lo_l

		# --- right-adjacent: genoStart in [e0, e0 + d] ---
		lo_r = np.searchsorted(starts_sorted, le0, side="left")
		hi_r = np.searchsorted(starts_sorted, le0 + d, side="right")
		n_right = hi_r - lo_r

		n_adjacent[rows] = n_left + n_right

		only_left = (n_left == 1) & (n_right == 0)
		te_row[rows[only_left]] = te_idx[by_end[lo_l[only_left]]]
		side[rows[only_left]] = "left"

		only_right = (n_left == 0) & (n_right == 1)
		te_row[rows[only_right]] = te_idx[by_start[lo_r[only_right]]]
		side[rows[only_right]] = "right"

	out = pd.DataFrame({
		"te_overlap": overlaps,
		"n_adjacent": n_adjacent,
		"te_row": te_row,
		"te_side": side,
	}, index=str_df.index)

	# Attributes of the single adjacent row, NaN/empty where there isn't one.
	has_te = te_row >= 0
	picked = te_df.loc[te_row[has_te]]
	for col, out_col, dtype, fill in [
		("strand", "te_strand", object, ""),
		("rep_name", "te_rep_name", object, ""),
		("te_class", "te_class", object, ""),
		("te_family", "te_family", object, ""),
		("is_polya", "te_is_polya", bool, False),
		("start", "te_start", np.int64, -1),
		("end", "te_end", np.int64, -1),
	]:
		values = np.full(n, fill, dtype=dtype)
		values[has_te] = picked[col].to_numpy()
		out[out_col] = values

	# --- position, in the ELEMENT's orientation (module docstring table) ---
	# The facing end is set by the side; which end of the element that is --
	# and so whether the STR is past its 3' end (downstream) or before its 5'
	# end (upstream) -- is set by the strand.
	gap = np.full(n, -1, dtype=np.int64)
	left, right = out["te_side"].to_numpy() == "left", out["te_side"].to_numpy() == "right"
	gap[left] = s0[left] - out["te_end"].to_numpy()[left]
	gap[right] = out["te_start"].to_numpy()[right] - e0[right]

	plus = out["te_strand"].to_numpy() == "+"
	downstream = (left & plus) | (right & ~plus)
	upstream = (right & plus) | (left & ~plus)

	position = np.full(n, "", dtype=object)
	position[downstream] = "downstream"
	position[upstream] = "upstream"

	out["gap"] = gap
	out["position"] = position
	# Signed for plotting/inspection: + past the 3' end, - before the 5' end.
	out["signed_gap"] = np.where(downstream, gap, np.where(upstream, -gap, 0))
	return out


def assign_groups(df):
	"""Map each locus to its analysis group, or "" if it is excluded.

	Exclusions are deliberately strict and are reported as counts: a locus
	with more than one adjacent row carries mixed evidence and is dropped
	from the background too, not just from the TE groups -- leaving it in B
	would put TE-adjacent loci in the comparison baseline.
	"""
	group = np.full(len(df), "", dtype=object)

	usable = ~df["te_overlap"].to_numpy()
	background = usable & (df["n_adjacent"].to_numpy() == 0)
	single = usable & (df["n_adjacent"].to_numpy() == 1)

	group[background] = G_BACKGROUND
	polya = df["te_is_polya"].to_numpy()
	down = df["position"].to_numpy() == "downstream"
	up = df["position"].to_numpy() == "upstream"

	group[single & polya & down] = G_P_DOWN
	group[single & polya & up] = G_P_UP
	group[single & ~polya & down] = G_N_DOWN
	group[single & ~polya & up] = G_N_UP
	return group


# ===========================================================================
# Scores
# ===========================================================================

def motif_matched_u(focal_labels, focal_canon, bg_labels, bg_canon,
                    leave_one_out=False):
	"""Per-locus motif-conditional percentile against same-motif background.

	  u(i) = [#(B_m with L < L_i) + 0.5 * #(B_m with L == L_i)] / |B_m|

	The half-credit for ties is what makes this exact for `length`, where
	mode_copy_number takes only ~77 distinct values and exact ties between a
	focal locus and background loci are the norm rather than the exception.

	Mean(u) over a group relates to Mann-Whitney U by sum_i u(i) = U / |B_m|,
	so 2*mean(u) - 1 is exactly Cliff's delta -- per motif, and a
	|G_m|-weighted average of the per-motif deltas once pooled.

	`leave_one_out` is for scoring the background against itself, which the
	permutation null needs. Every background locus is in its own pool and
	would otherwise score a guaranteed half-credit tie against itself; that
	biases its u toward 0.5 by ~1/|B_m| relative to a focal locus, which is
	never in the pool. The permutation compares the two sets directly, so a
	bias present in only one of them is exactly the kind that would show up
	as effect. Dropping the self-comparison removes it.
	"""
	u = np.full(len(focal_labels), np.nan)
	for motif in np.unique(focal_canon):
		bg = np.sort(bg_labels[bg_canon == motif])
		n_bg = len(bg) - 1 if leave_one_out else len(bg)
		if n_bg < 1:
			continue
		rows = np.nonzero(focal_canon == motif)[0]
		labels = focal_labels[rows]
		n_less = np.searchsorted(bg, labels, side="left")
		n_le = np.searchsorted(bg, labels, side="right")
		n_eq = n_le - n_less
		if leave_one_out:
			n_eq = n_eq - 1  # the locus's own tie with itself
		u[rows] = (n_less + 0.5 * n_eq) / n_bg
	return u


def length_matched_u(focal_cn, focal_het, focal_canon,
                     bg_cn, bg_het, bg_canon, n_match, rng,
                     leave_one_out=False):
	"""Heterozygosity percentile against the n_match same-motif background
	loci with the nearest copy number.

	Holding length fixed as well as motif is what separates "this group is
	more heterozygous" from "this group is longer, and longer tracts are
	mechanically more heterozygous".

	Distance ties are overwhelming -- copy number is quarter-integral, so a
	focal locus typically has far more than n_match background loci at
	distance 0 -- and taking the first n_match in file order would match
	systematically by genome position. Each motif's background is therefore
	shuffled once (seeded), and a STABLE argsort on distance keeps that
	shuffle as the tie order, so ties are broken uniformly at random but
	reproducibly.

	Loci sharing a motif and a copy number share a matched set, so the work
	is done once per distinct copy-number value rather than once per locus.

	`leave_one_out` scores the background against itself for the permutation
	null; see motif_matched_u. Here a background locus is at distance 0 from
	itself and so is always selected into its own matched set, which is
	why the loop draws n_match + 1 neighbors and drops the locus itself.
	"""
	if leave_one_out and focal_het is not bg_het:
		raise ValueError(
			"leave_one_out scores the background against itself; pass the "
			"same arrays as both focal and background"
		)

	u = np.full(len(focal_het), np.nan)
	# One extra candidate when leaving the locus out, so that a full n_match
	# neighbors remain after its own entry is removed.
	n_draw = n_match + 1 if leave_one_out else n_match
	for motif in np.unique(focal_canon):
		in_motif = bg_canon == motif
		if int(in_motif.sum()) < n_match:
			continue
		bg_rows = np.nonzero(in_motif)[0]
		shuffle = rng.permutation(len(bg_rows))
		bg_rows = bg_rows[shuffle]
		cn_bg, het_bg = bg_cn[bg_rows], bg_het[bg_rows]

		rows = np.nonzero(focal_canon == motif)[0]
		for cn_value in np.unique(focal_cn[rows]):
			nearest = np.argsort(np.abs(cn_bg - cn_value), kind="stable")[:n_draw]
			same_cn = rows[focal_cn[rows] == cn_value]

			if not leave_one_out:
				matched = np.sort(het_bg[nearest])
				het = focal_het[same_cn]
				n_less = np.searchsorted(matched, het, side="left")
				n_le = np.searchsorted(matched, het, side="right")
				u[same_cn] = (n_less + 0.5 * (n_le - n_less)) / len(matched)
				continue

			# Scoring the background against itself: drop each locus from
			# its own matched set. The candidate set is shared by every locus
			# with this motif and copy number, so it is one set minus (at
			# most) one member each time. A locus need not be among its own
			# n_draw nearest -- more than n_draw background loci can sit at
			# distance 0 -- so when it is absent the extra candidate is
			# dropped instead, and the trim happens in NEIGHBOR order, before
			# sorting by heterozygosity. Trimming after that sort would
			# discard the most heterozygous neighbor every time and push
			# every background u upward.
			het_candidates = het_bg[nearest]
			picked = bg_rows[nearest]
			for row in same_cn:
				matched = np.sort(het_candidates[picked != row][:n_match])
				het = focal_het[row]
				n_less = np.searchsorted(matched, het, side="left")
				n_le = np.searchsorted(matched, het, side="right")
				# len(matched) is n_match except for a motif whose background
				# is exactly n_match loci, where removing the locus itself
				# leaves one fewer. u stays a percentile either way.
				u[row] = (n_less + 0.5 * (n_le - n_less)) / len(matched)
	return u


# ===========================================================================
# Statistics
# ===========================================================================

def cliffs_delta(x, y):
	"""Cliff's delta (P(X>Y) - P(X<Y)) via the Mann-Whitney U identity
	delta = 2*U/(n_x*n_y) - 1, as in nearby_repeat_elements.py and
	str_anno_overlap_analysis.py. U gives tied pairs half credit, so this is
	exact under ties. Returns (delta, two-sided p).

	Thresholds (Romano et al. 2006): |delta| < 0.147 negligible, < 0.33
	small, < 0.474 medium, >= 0.474 large.
	"""
	u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
	delta = 2 * u_stat / (len(x) * len(y)) - 1
	return delta, p_value


def _bootstrap_mean(values, n_boot, chunk, rng):
	"""Bootstrap distribution of mean(values), resampling loci with
	replacement. Chunked so the index matrix stays small for big groups."""
	n = len(values)
	out = np.empty(n_boot)
	for lo in range(0, n_boot, chunk):
		size = min(chunk, n_boot - lo)
		idx = rng.integers(0, n, size=(size, n))
		out[lo:lo + size] = values[idx].mean(axis=1)
	return out


def _percentile_ci(draws, ci_level):
	alpha = 1 - ci_level
	return (float(np.nanpercentile(draws, 100 * (alpha / 2))),
	        float(np.nanpercentile(draws, 100 * (1 - alpha / 2))))


def _stratified_perm_u_means(pooled_u, n_focal, n_perm, chunk, rng):
	"""Permutation null for the pooled mean u-score.

	Group membership is reshuffled WITHIN each canonical motif: in each
	stratum a uniformly random subset of the pooled (focal + background)
	loci, of the same size as the real focal set, is relabeled focal, and the
	pooled mean is recomputed. Stratifying the null the same way the
	statistic is stratified is the point -- an unstratified shuffle would mix
	motifs and let composition masquerade as effect.

	The u-scores themselves are not recomputed per draw. They are fixed
	numbers attached to loci, so reshuffling the labels is already an exact
	test of focal/background exchangeability within motif, and it is the one
	formulation that works unchanged for both scores: the length-matched
	score has no rank identity to exploit, since each locus is ranked against
	its own copy-number neighborhood rather than against a common pool.

	The g smallest of len(pool) uniform draws pick the random size-g subset;
	argpartition does that without a full sort.
	"""
	total_focal = sum(n_focal)
	out = np.empty(n_perm)
	for lo in range(0, n_perm, chunk):
		size = min(chunk, n_perm - lo)
		acc = np.zeros(size)
		for u_pool, g in zip(pooled_u, n_focal):
			if g == 0:
				continue
			draws = rng.random((size, len(u_pool)))
			picked = np.argpartition(draws, g - 1, axis=1)[:, :g]
			acc += u_pool[picked].sum(axis=1)
		out[lo:lo + size] = acc / total_focal
	return out


def vs_background_test(u_focal, canon_focal, u_bg, canon_bg,
                       n_perm, n_boot, ci_level, chunk, rng):
	"""Motif-stratified effect of a focal group against background.

	effect = 2 * mean(u) - 1, which for the motif-matched score is exactly
	the motif-stratified Cliff's delta of the group against background. p
	comes from the within-motif permutation above and the CI from a
	percentile bootstrap over focal loci -- the background is held fixed
	there, being the reference scale rather than a sample under comparison.

	The null is centered on its own mean rather than on 0 so that the test
	stays honest for the length-matched score, where the background's own
	u-scores need not average to exactly 0.5.
	"""
	keep = np.isfinite(u_focal)
	u_focal, canon_focal = u_focal[keep], canon_focal[keep]
	keep_bg = np.isfinite(u_bg)
	u_bg, canon_bg = u_bg[keep_bg], canon_bg[keep_bg]

	n = len(u_focal)
	if n == 0:
		return {"n": 0, "n_background": len(u_bg), "effect": np.nan,
		        "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
		        "null_center": np.nan}

	effect = 2 * float(u_focal.mean()) - 1

	pooled_u, n_focal = [], []
	for motif in np.unique(canon_focal):
		focal_m = u_focal[canon_focal == motif]
		bg_m = u_bg[canon_bg == motif]
		if len(focal_m) == 0 or len(bg_m) == 0:
			continue
		pooled_u.append(np.concatenate([focal_m, bg_m]))
		n_focal.append(len(focal_m))

	null_effect = 2 * _stratified_perm_u_means(
		pooled_u, n_focal, n_perm, chunk, rng
	) - 1
	center = float(null_effect.mean())
	# +1 in both parts: the observed arrangement is itself one of the equally
	# likely ones under the null, so p is never reported as exactly 0.
	n_extreme = int(np.sum(np.abs(null_effect - center) >= abs(effect - center)))
	p_value = (1 + n_extreme) / (n_perm + 1)

	boot = 2 * _bootstrap_mean(u_focal, n_boot, chunk, rng) - 1
	ci_lo, ci_hi = _percentile_ci(boot, ci_level)
	return {"n": n, "n_background": len(u_bg), "effect": effect,
	        "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": float(p_value),
	        "null_center": center}


def _perm_split_deltas(pooled_u, n_t, grand_total, n_t_total, n_o_total,
                       n_perm, chunk, rng):
	"""Permutation null for Delta = delta_T - delta_O.

	Within each canonical motif, which adjacent loci are T and which are O is
	reshuffled, holding that motif's T and O counts fixed. Shuffling within
	motif is what makes this a test of the poly(A) split specifically: a
	motif that simply has a stronger adjacency effect cannot leak into Delta,
	because it contributes to both arms in the same proportion under the null
	as in the data.

	Only the T sum has to be drawn -- O is the complement, so its sum is the
	motif totals minus the T sum. A motif with no T loci contributes a
	constant to O and nothing to T, so it is skipped entirely; one with no O
	loci contributes a constant to T.
	"""
	out = np.empty(n_perm)
	for lo in range(0, n_perm, chunk):
		size = min(chunk, n_perm - lo)
		acc_t = np.zeros(size)
		for u_pool, t in zip(pooled_u, n_t):
			if t == 0:
				continue
			if t == len(u_pool):
				acc_t += u_pool.sum()
				continue
			draws = rng.random((size, len(u_pool)))
			picked = np.argpartition(draws, t - 1, axis=1)[:, :t]
			acc_t += u_pool[picked].sum(axis=1)
		# delta = 2*mean(u) - 1 for each arm, so the -1 cancels in the
		# difference and only the means are needed.
		out[lo:lo + size] = 2 * (
			acc_t / n_t_total - (grand_total - acc_t) / n_o_total
		)
	return out


def split_effect_test(u_t, canon_t, u_o, canon_o, compute_share,
                      n_perm, n_boot, ci_level, chunk, rng):
	"""Comparison 10: a poly(A) group T against the rest of the TE effect O.

	T and O are disjoint and together are every adjacent locus, so this asks
	how much of the near-TE signal is specifically poly(A)-downstream rather
	than near-a-TE-at-all. Both arms are scored against the same background
	on the same 2*mean(u) - 1 scale as everything else, which is what makes
	the difference between them readable.

	Returns Delta = delta_T - delta_O (the effect, with a permutation p and a
	bootstrap CI) and, when `compute_share`, S = the fraction of the total
	adjacency excess carried by T.

	S is exact rather than approximate because each delta is a mean of
	per-locus (2u - 1) and the two arms partition the adjacent set: the total
	excess is n_T*delta_T + n_O*delta_O, and T's part of it is n_T*delta_T.
	It is undefined when that total is <= 0 (no excess to apportion), and is
	not computed for the length-matched score at all -- that score's
	background sits near 0.513 rather than 0.5, an offset that largely
	cancels in the difference Delta but distorts a ratio.
	"""
	u_t, canon_t = u_t[np.isfinite(u_t)], canon_t[np.isfinite(u_t)]
	u_o, canon_o = u_o[np.isfinite(u_o)], canon_o[np.isfinite(u_o)]
	n_t_total, n_o_total = len(u_t), len(u_o)

	empty = {"n": n_t_total, "n_other": n_o_total, "effect": np.nan,
	         "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
	         "null_center": np.nan, "delta_t": np.nan, "delta_o": np.nan,
	         "delta_t_ci_lo": np.nan, "delta_t_ci_hi": np.nan,
	         "delta_o_ci_lo": np.nan, "delta_o_ci_hi": np.nan,
	         "share": np.nan, "share_ci_lo": np.nan, "share_ci_hi": np.nan}
	if n_t_total == 0 or n_o_total == 0:
		return empty

	delta_t = 2 * float(u_t.mean()) - 1
	delta_o = 2 * float(u_o.mean()) - 1
	effect = delta_t - delta_o

	def share_of(n_t, d_t, n_o, d_o):
		total = n_t * d_t + n_o * d_o
		return (n_t * d_t) / total if total > 0 else np.nan

	share = share_of(n_t_total, delta_t, n_o_total, delta_o) if compute_share else np.nan

	# --- permutation p for Delta ---
	pooled_u, n_t = [], []
	grand_total = 0.0
	for motif in np.unique(np.concatenate([canon_t, canon_o])):
		in_t = u_t[canon_t == motif]
		in_o = u_o[canon_o == motif]
		pool = np.concatenate([in_t, in_o])
		pooled_u.append(pool)
		n_t.append(len(in_t))
		grand_total += float(pool.sum())

	null = _perm_split_deltas(pooled_u, n_t, grand_total, n_t_total,
	                          n_o_total, n_perm, chunk, rng)
	center = float(null.mean())
	n_extreme = int(np.sum(np.abs(null - center) >= abs(effect - center)))
	p_value = (1 + n_extreme) / (n_perm + 1)

	# --- bootstrap CIs for Delta and S, resampling each arm separately ---
	boot_delta = np.empty(n_boot)
	boot_t, boot_o = np.empty(n_boot), np.empty(n_boot)
	boot_share = np.full(n_boot, np.nan)
	for lo in range(0, n_boot, chunk):
		size = min(chunk, n_boot - lo)
		mt = u_t[rng.integers(0, n_t_total, size=(size, n_t_total))].mean(axis=1)
		mo = u_o[rng.integers(0, n_o_total, size=(size, n_o_total))].mean(axis=1)
		d_t, d_o = 2 * mt - 1, 2 * mo - 1
		boot_t[lo:lo + size], boot_o[lo:lo + size] = d_t, d_o
		boot_delta[lo:lo + size] = d_t - d_o
		if compute_share:
			total = n_t_total * d_t + n_o_total * d_o
			boot_share[lo:lo + size] = np.where(
				total > 0, (n_t_total * d_t) / np.where(total > 0, total, 1),
				np.nan,
			)
	ci_lo, ci_hi = _percentile_ci(boot_delta, ci_level)
	t_lo, t_hi = _percentile_ci(boot_t, ci_level)
	o_lo, o_hi = _percentile_ci(boot_o, ci_level)
	share_lo, share_hi = ((np.nan, np.nan) if not compute_share
	                      else _percentile_ci(boot_share, ci_level))

	return {"n": n_t_total, "n_other": n_o_total, "effect": effect,
	        "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": float(p_value),
	        "null_center": center, "delta_t": delta_t, "delta_o": delta_o,
	        "delta_t_ci_lo": t_lo, "delta_t_ci_hi": t_hi,
	        "delta_o_ci_lo": o_lo, "delta_o_ci_hi": o_hi,
	        "share": share, "share_ci_lo": share_lo, "share_ci_hi": share_hi}


def between_group_test(u_a, u_b, n_boot, ci_level, chunk, rng):
	"""Cliff's delta between two focal groups' u-scores.

	Both groups are already motif-conditional percentiles against the same
	background, so comparing them directly asks whether one group sits higher
	in its motif's background distribution than the other does in its own --
	with composition already divided out on both sides.
	"""
	u_a, u_b = u_a[np.isfinite(u_a)], u_b[np.isfinite(u_b)]
	if len(u_a) == 0 or len(u_b) == 0:
		return {"n": len(u_a), "n_other": len(u_b), "effect": np.nan,
		        "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan}

	effect, p_value = cliffs_delta(u_a, u_b)

	boot = np.empty(n_boot)
	for lo in range(0, n_boot, chunk):
		size = min(chunk, n_boot - lo)
		idx_a = rng.integers(0, len(u_a), size=(size, len(u_a)))
		idx_b = rng.integers(0, len(u_b), size=(size, len(u_b)))
		for k in range(size):
			boot[lo + k] = cliffs_delta(u_a[idx_a[k]], u_b[idx_b[k]])[0]
	ci_lo, ci_hi = _percentile_ci(boot, ci_level)
	return {"n": len(u_a), "n_other": len(u_b), "effect": float(effect),
	        "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": float(p_value)}


def proportion_test(n_success, n_total, ci_level):
	"""Two-sided binomial test of a proportion against 0.5, with the exact
	(Clopper-Pearson) CI from the same test."""
	if n_total == 0:
		return {"n": 0, "effect": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
		        "p_value": np.nan}
	result = binomtest(int(n_success), int(n_total), 0.5, alternative="two-sided")
	ci = result.proportion_ci(confidence_level=ci_level)
	return {"n": int(n_total), "n_success": int(n_success),
	        "effect": n_success / n_total,
	        "ci_lo": float(ci.low), "ci_hi": float(ci.high),
	        "p_value": float(result.pvalue)}


def correction_label(correction, alpha):
	"""Human-readable name for the correction in force. Copied from
	str_anno_overlap_analysis.py so the two scripts' figures and reports
	label the same setting the same way."""
	method_names = {"bonferroni": "Bonferroni", "fdr_bh": "BH-FDR"}
	method = method_names.get(correction, "uncorrected") if correction else "uncorrected"
	return f"{method}, alpha={alpha:g}"


def correct_p_values(results, correction, alpha):
	"""Apply `correction` to the finite p-values in `results`, in place.

	Sets p_value_corrected and significant on every row. With correction
	None the raw p is carried through unchanged and compared directly to
	alpha, matching str_anno_overlap_analysis.py's convention.

	Rows with a NaN p (a comparison that could not be run -- an empty group,
	a constant covariate) are not tests and are left out of the correction
	rather than counted in m, which would penalize every other test for a
	comparison that never happened.
	"""
	finite = [r for r in results if np.isfinite(r.get("p_value", np.nan))]
	for r in results:
		r["p_value_corrected"], r["significant"] = np.nan, False
	if not finite:
		return results

	p_raw = [r["p_value"] for r in finite]
	if correction:
		reject, p_adj, _, _ = multipletests(p_raw, alpha=alpha, method=correction)
	else:
		p_adj = p_raw
		reject = [p <= alpha for p in p_raw]

	for r, p, sig in zip(finite, p_adj, reject):
		r["p_value_corrected"], r["significant"] = float(p), bool(sig)
	return results


def check_permutation_floor(results, n_perm, correction, alpha, family):
	"""Warn when the permutation grid is too coarse for the correction.

	A permutation p-value cannot go below 1/(n_perm + 1), however strong the
	effect: that is the resolution of the grid, not a property of the data.
	Bonferroni multiplies it by the number of tests, so a family of m tests
	needs n_perm + 1 > m / alpha before a permutation test can clear the
	threshold AT ALL. Below that every vs-background comparison comes back
	non-significant no matter how large the effect -- a silent failure that
	looks exactly like a null result, which is why it is checked rather than
	left to be noticed.

	BH-FDR is laxer (the smallest p is compared against alpha/m only at the
	bottom rank), but the same floor applies to it in the worst case, so the
	warning is raised for any correction.
	"""
	perm_rows = [r for r in results
	             if "permutation" in r["test"] and np.isfinite(r["p_value"])]
	if not perm_rows or not correction:
		return
	n_tests = sum(np.isfinite(r["p_value"]) for r in results)
	floor = 1.0 / (n_perm + 1)
	if floor * n_tests > alpha:
		print(
			f"\n  WARNING: with N_PERM={n_perm:,} a permutation p cannot fall "
			f"below {floor:.2e}, and\n           "
			f"{correction_label(correction, alpha)} over this {family} family "
			f"of {n_tests} tests needs\n           it below "
			f"{alpha / n_tests:.2e}. The {len(perm_rows)} permutation "
			f"comparisons cannot reach\n           significance at any "
			f"effect size. Raise N_PERM to at least "
			f"{int(np.ceil(n_tests / alpha)):,}."
		)


# ===========================================================================
# Strand-logic check
# ===========================================================================

def print_strand_check(df, ref, n_per_strand, flank):
	"""Print the sense-oriented reference around P-down loci, one block per
	TE strand, for manual review.

	Read left to right, each line is: the element's last `flank` bp, the gap,
	then the STR -- all on the ELEMENT's strand, which is the strand its
	poly(A) tail was written on. A correct downstream call on a poly(A)
	element should show the tail: A-runs at the end of the element, running
	into an A-rich tract.

	For a + strand element the sense strand is the reference's, so the three
	pieces are read straight off it. For a - strand element the element lies
	to the STR's genomic RIGHT, its 3' end is at genoStart, and sense order
	is the reverse complement of genomic order -- so each piece is reverse
	complemented and they are emitted in the opposite order. Getting that
	backwards is the failure mode this check exists to catch.
	"""
	print("\n" + "=" * 78)
	print("STRAND CHECK -- sense-oriented reference at P-down loci")
	print("=" * 78)
	print("  format: <element 3' end> <gap> <STR>, all on the element's strand")

	for strand in ("+", "-"):
		sel = df[(df["group"] == G_P_DOWN) & (df["te_strand"] == strand)]
		print(f"\n--- TE strand {strand} "
		      f"({'element to the STR left' if strand == '+' else 'element to the STR right'}, "
		      f"{len(sel):,} loci) ---")
		if sel.empty:
			print("  none")
			continue

		for _, row in sel.head(n_per_strand).iterrows():
			contig = ref[row["chrom"]]
			s0, e0 = int(row["str_start"]), int(row["str_end"])
			te_s, te_e = int(row["te_start"]), int(row["te_end"])

			if strand == "+":
				tail = contig[max(te_e - flank, 0):te_e].upper()
				gap_seq = contig[te_e:s0].upper()
				str_seq = contig[s0:e0].upper()
			else:
				tail = rev_comp(contig[te_s:te_s + flank].upper())
				gap_seq = rev_comp(contig[e0:te_s].upper())
				str_seq = rev_comp(contig[s0:e0].upper())

			print(f"  {row['chrom']}:{s0}-{e0}  {row['te_rep_name']} "
			      f"({row['te_class']}/{row['te_family']})  gap={int(row['gap'])}")
			print(f"    {tail} {gap_seq or '-'} {str_seq}")
			print(f"    plus motif {row['plus_motif']}  sense motif "
			      f"{row['sense_motif']}  sense A% {row['sense_a_frac']:.0%}  "
			      f"canonical {row['canon_motif']}")


# ===========================================================================
# Comparisons
# ===========================================================================

SCORE_MOTIF = "motif-matched"
SCORE_MATCHED = "length-matched"


def group_mask(df, name):
	"""Boolean mask for a group. The four assigned groups are columns of the
	frame; the unions and the tail-excluded set are derived here so that
	nothing is double-assigned."""
	group = df["group"].to_numpy()
	if name == G_ALL_DOWN:
		return np.isin(group, [G_P_DOWN, G_N_DOWN])
	if name == G_ALL_UP:
		return np.isin(group, [G_P_UP, G_N_UP])
	if name == G_DOWN_NO_TAIL:
		return np.isin(group, [G_P_DOWN, G_N_DOWN]) & ~df["is_tail"].to_numpy()
	adjacent = np.isin(group, ASSIGNED_GROUPS)
	if name == G_ALL_ADJ:
		return adjacent
	if name == G_TAILS:
		return df["is_tail"].to_numpy()
	if name == G_NOT_P_DOWN:
		return adjacent & (group != G_P_DOWN)
	if name == G_NOT_TAIL:
		return adjacent & ~df["is_tail"].to_numpy()
	return group == name


def _vs_bg(df, name, u_col, rng, bg_mask=None):
	"""vs_background_test for a named group, pulling its u-scores and the
	background's from the frame."""
	focal = group_mask(df, name)
	if bg_mask is None:
		bg_mask = group_mask(df, G_BACKGROUND)
	canon = df["canon_motif"].to_numpy()
	u = df[u_col].to_numpy()
	return vs_background_test(
		u[focal], canon[focal], u[bg_mask], canon[bg_mask],
		N_PERM, N_BOOT, CI_LEVEL, DRAW_CHUNK, rng,
	)


def _between(df, name_a, name_b, u_col, rng):
	u = df[u_col].to_numpy()
	return between_group_test(
		u[group_mask(df, name_a)], u[group_mask(df, name_b)],
		N_BOOT, CI_LEVEL, DRAW_CHUNK, rng,
	)


def _record(comparison, question, score, target, test, effect_name,
            stats, **extra):
	"""One row of the results table."""
	return {"comparison": comparison, "question": question, "score": score,
	        "target": target, "test": test, "effect_name": effect_name,
	        **stats, **extra}


def run_primary_comparisons(df, score, target, u_col, numbers, rng):
	"""Comparisons 1-8 for one (score, target), restricted to `numbers`.

	Comparison 9 of the spec is not a separate block: it is this same set run
	with score=SCORE_MATCHED, target="heterozygosity" and numbers={2,4,6,8},
	so the length-matched results line up row-for-row with the ones they are
	meant to be read against.
	"""
	results = []
	u = df[u_col].to_numpy()
	higher_a = df["higher_a_orientation"].to_numpy()
	informative = df["ori_informative"].to_numpy()

	def add(number, question, test, effect_name, stats, **extra):
		results.append(_record(number, question, score, target, test,
		                       effect_name, stats, **extra))

	# --- 1-3: each position, against background -------------------------
	for number, groups in [
		("1", (G_ALL_ADJ, G_ALL_DOWN, G_ALL_UP)),
		("2", (G_P_DOWN, G_P_UP)),
		("3", (G_N_DOWN, G_N_UP)),
	]:
		if number not in numbers:
			continue
		for name in groups:
			add(number, f"{name} vs {G_BACKGROUND}", "within-motif permutation",
			    "2*mean(u) - 1", _vs_bg(df, name, u_col, rng), group=name)

	# --- 4: poly(A) vs other TEs, at the same position -------------------
	if "4" in numbers:
		for a, b in [(G_P_DOWN, G_N_DOWN), (G_P_UP, G_N_UP)]:
			add("4", f"{a} vs {b}", "Mann-Whitney on u", "Cliff's delta",
			    _between(df, a, b, u_col, rng), group=a, reference=b)

	# --- 5: downstream vs upstream, within a TE type ---------------------
	if "5" in numbers:
		for a, b in [(G_P_DOWN, G_P_UP), (G_N_DOWN, G_N_UP)]:
			add("5", f"{a} vs {b}", "Mann-Whitney on u", "Cliff's delta",
			    _between(df, a, b, u_col, rng), group=a, reference=b)

	# --- 6: sense-strand A content, within a group -----------------------
	# The two orientations of one motif have identical base composition, so
	# comparing them holds everything fixed except which strand the A's are
	# on -- which is exactly the tail hypothesis's claim. P up and N down are
	# the controls: neither has a tail to find, so an effect there is a sign
	# that A-rich motifs simply behave differently, not that tails do.
	# (A rank correlation of u on sense A% across all A levels used to run
	# here too. It answered a blurrier question -- A content and tail status
	# are confounded across motifs -- and the orientation contrast below is
	# the version that actually isolates strand, so only that one is kept.
	# plot_a_content still shows the full A%-vs-u relationship.)
	if "6" in numbers:
		for name in (G_P_DOWN, G_N_DOWN, G_P_UP):
			eligible = group_mask(df, name) & informative
			add("6", f"higher- vs lower-A% orientation, within {name}",
			    "Mann-Whitney on u", "Cliff's delta",
			    between_group_test(u[eligible & higher_a],
			                       u[eligible & ~higher_a],
			                       N_BOOT, CI_LEVEL, DRAW_CHUNK, rng),
			    group=name)

	# --- 7: orientation count asymmetry ----------------------------------
	# Counts only, and the single most direct read on the strand logic: if
	# the A-rich orientation is not over-represented downstream of poly(A)
	# elements, there is nothing here to interpret. Does not depend on the
	# labels, so it is run once per score rather than per target.
	if "7" in numbers:
		for name in (G_P_DOWN, G_N_DOWN, G_P_UP):
			eligible = group_mask(df, name) & informative
			add("7", f"fraction in the higher-A% orientation, within {name}",
			    "binomial vs 0.5", "fraction",
			    proportion_test(int((eligible & higher_a).sum()),
			                    int(eligible.sum()), CI_LEVEL),
			    group=name)

	# --- 10: poly(A) group vs the rest of the TE effect ------------------
	# Comparisons 1-8 each measure a group against background; none of them
	# says how much BIGGER the poly(A)-downstream effect is than the rest of
	# the near-TE effect, or how much of that effect it accounts for. This
	# does both, and does it on disjoint arms so the group is never partly
	# compared with itself.
	if "10" in numbers:
		canon = df["canon_motif"].to_numpy()
		for version, t_name, o_name in COMPARISON_10_SPLITS:
			t_mask, o_mask = group_mask(df, t_name), group_mask(df, o_name)
			results.append(_record(
				version, f"{t_name} vs {o_name}", score, target,
				"within-motif permutation on the T/O split",
				"delta_T - delta_O",
				split_effect_test(
					u[t_mask], canon[t_mask], u[o_mask], canon[o_mask],
					# The share is a ratio, so it is only meaningful on a
					# score whose background sits at 0.5 -- see
					# split_effect_test.
					compute_share=(score == SCORE_MOTIF),
					n_perm=N_PERM, n_boot=N_BOOT, ci_level=CI_LEVEL,
					chunk=DRAW_CHUNK, rng=rng,
				),
				group=t_name, reference=o_name,
			))

	# --- 8: downstream with putative tails removed -----------------------
	if "8" in numbers:
		add("8", f"{G_DOWN_NO_TAIL} vs {G_BACKGROUND}",
		    "within-motif permutation", "2*mean(u) - 1",
		    _vs_bg(df, G_DOWN_NO_TAIL, u_col, rng), group=G_DOWN_NO_TAIL)

	return results


def run_secondary_comparisons(df, u_cols, rng):
	"""Comparisons 2 and 3 per canonical motif, for motifs with at least
	MIN_N focal loci in that group.

	The pooled primaries already hold motif fixed, so this is a breakdown,
	not a replication: it shows whether the pooled effect is carried by a few
	motifs (which would point at composition after all) or is general. It
	gets its own correction family for that reason.
	"""
	results = []
	canon = df["canon_motif"].to_numpy()
	bg_mask = group_mask(df, G_BACKGROUND)

	for target, label_col in TARGETS.items():
		u = df[u_cols[(SCORE_MOTIF, target)]].to_numpy()
		for name in ASSIGNED_GROUPS:
			focal_mask = group_mask(df, name)
			for motif in np.unique(canon[focal_mask]):
				sel = focal_mask & (canon == motif)
				if int(sel.sum()) < MIN_N:
					continue
				bg_sel = bg_mask & (canon == motif)
				stats = vs_background_test(
					u[sel], canon[sel], u[bg_sel], canon[bg_sel],
					N_PERM, N_BOOT, CI_LEVEL, DRAW_CHUNK, rng,
				)
				results.append(_record(
					"2" if name in (G_P_DOWN, G_P_UP) else "3",
					f"{name} vs {G_BACKGROUND} [{motif}]", SCORE_MOTIF,
					target, "within-motif permutation", "2*mean(u) - 1",
					stats, group=name, motif=motif,
				))
	return results


# ===========================================================================
# Report
# ===========================================================================

def _json_safe(value):
	"""Recursively replace non-finite floats with None.

	json.dump writes NaN as the bare token `NaN`, which Python reads back but
	strict JSON parsers reject. nearby_repeat_elements.py sidesteps this by
	going through pandas' to_json (which emits null); this does the same for
	plain dicts. A NaN here means a comparison could not be run, and null
	says that in a form anything can read.
	"""
	if isinstance(value, dict):
		return {k: _json_safe(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(v) for v in value]
	if isinstance(value, (float, np.floating)):
		return None if not np.isfinite(value) else float(value)
	if isinstance(value, (np.integer,)):
		return int(value)
	if isinstance(value, (np.bool_,)):
		return bool(value)
	return value


def _fmt_p(p):
	if not np.isfinite(p):
		return "     n/a"
	if p == 0:
		return "  <1e-308"  # underflowed, not actually zero
	return f"{p:8.2e}" if p < 1e-3 else f"{p:8.4f}"


def _comparison_sort_key(label):
	"""Sort "1" < "6" < "8" < "10a". Keying on the first character
	alone would file "10a" next to "1"."""
	digits = "".join(ch for ch in label if ch.isdigit())
	return (int(digits) if digits else 99, label)


def _print_split_rows(rows):
	"""Comparison 10 rows: both arms' effects, their difference, and the
	share. Too many quantities for the one-line format the other comparisons
	use, so each gets its own two lines."""
	print(f"  {rows[0]['question']}")
	for r in rows:
		if not np.isfinite(r["effect"]):
			print(f"    {r['score']:<14} {r['target']:<16} SKIPPED "
			      f"(n_T={r['n']}, n_O={r['n_other']})")
			continue
		print(
			f"    {r['score']:<14} {r['target']:<16} "
			f"n_T={r['n']:>6,}  n_O={r['n_other']:>6,}   "
			f"delta_T={r['delta_t']:+.4f}  delta_O={r['delta_o']:+.4f}"
		)
		share = (f"   share={r['share']:.3f} "
		         f"[{r['share_ci_lo']:.3f}, {r['share_ci_hi']:.3f}]"
		         if np.isfinite(r.get("share", np.nan)) else
		         "   share: n/a (length-matched score)")
		print(
			f"    {'':<14} {'':<16} "
			f"Delta={r['effect']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
			f"  p={_fmt_p(r['p_value'])} adj={_fmt_p(r['p_value_corrected'])}"
			f"{'*' if r['significant'] else ' '}{share}"
		)


def print_results(results, title, alpha):
	"""One block per comparison number, in spec order."""
	print("\n" + "=" * 78)
	print(title)
	print("=" * 78)
	print(f"  * = significant ({correction_label(CORRECTION, alpha)}) within "
	      f"this family of {sum(np.isfinite(r['p_value']) for r in results)} "
	      f"tests; adj = the corrected p")

	for number in sorted({r["comparison"] for r in results},
	                     key=_comparison_sort_key):
		rows = [r for r in results if r["comparison"] == number]
		print(f"\n--- comparison {number}: {rows[0]['test']} ---")
		if number.startswith("10"):
			_print_split_rows(rows)
			continue
		for r in rows:
			if not np.isfinite(r["effect"]):
				print(f"  {r['question']:<48} {r['target']:<16} "
				      f"SKIPPED (n={r['n']})")
				continue
			extra = ""
			other = r.get("n_background", r.get("n_other"))
			if other is not None:
				extra = f" vs n={other:,}"
			# The length-matched score's background does not average to
			# exactly 0.5 -- copy number is right-skewed, so the rare
			# high-copy loci sit near the top of their own neighbourhood
			# while the dense low-copy ones sit mid-pack. The permutation
			# already tests against this baseline rather than against 0;
			# it is printed so the effect is not read as if 0 were the null.
			baseline = r.get("null_center", np.nan)
			if np.isfinite(baseline) and abs(baseline) >= 0.005:
				extra += f" [null {baseline:+.3f}]"
			print(
				f"  {r['question']:<48} {r['target']:<16} "
				f"n={r['n']:>6,}{extra:<15}  "
				f"{r['effect_name']}={r['effect']:+.4f} "
				f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
				f"p={_fmt_p(r['p_value'])} "
				f"adj={_fmt_p(r['p_value_corrected'])}"
				f"{'*' if r['significant'] else ' '}"
			)


def print_secondary(results, alpha):
	"""Per-motif breakdown, grouped by target and group, motifs ordered by n."""
	print("\n" + "=" * 78)
	print("SECONDARY -- comparisons 2 and 3 per canonical motif")
	print("=" * 78)
	print(f"  separate correction family of "
	      f"{sum(np.isfinite(r['p_value']) for r in results)} tests "
	      f"({correction_label(CORRECTION, alpha)}); motifs with >= {MIN_N} "
	      f"focal loci only")

	if not results:
		print("\n  no motif reached the minimum in any group")
		return

	for target in TARGETS:
		for name in ASSIGNED_GROUPS:
			rows = [r for r in results
			        if r["target"] == target and r["group"] == name]
			if not rows:
				continue
			rows.sort(key=lambda r: -r["n"])
			print(f"\n--- {name} vs {G_BACKGROUND}, {target} ---")
			for r in rows:
				print(
					f"  {r['motif']:<8} n={r['n']:>6,}  bg={r['n_background']:>6,}  "
					f"2*mean(u)-1={r['effect']:+.4f} "
					f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
					f"adj={_fmt_p(r['p_value_corrected'])}"
					f"{'*' if r['significant'] else ' '}"
				)
			effects = np.array([r["effect"] for r in rows])
			n_sig = sum(r["significant"] for r in rows)
			print(f"  -> {len(rows)} motifs, median {np.median(effects):+.4f}, "
			      f"range [{effects.min():+.4f}, {effects.max():+.4f}], "
			      f"{n_sig} significant")


def print_group_counts(df):
	"""n per group, per position, and per canonical motif."""
	print("\n--- loci per group ---")
	group = df["group"].to_numpy()
	for name in [G_BACKGROUND] + ASSIGNED_GROUPS:
		n = int((group == name).sum())
		print(f"  {name:<21} {n:>7,} ({n / len(df):7.3%})")
	for name in (G_ALL_ADJ, G_ALL_DOWN, G_ALL_UP, G_DOWN_NO_TAIL,
	             G_TAILS, G_NOT_P_DOWN, G_NOT_TAIL):
		n = int(group_mask(df, name).sum())
		print(f"  {name:<21} {n:>7,} ({n / len(df):7.3%})   (derived)")

	print("\n--- adjacent loci by TE class/family (the single adjacent row) ---")
	adjacent = df[np.isin(group, ASSIGNED_GROUPS)]
	by_family = adjacent.groupby(
		["te_is_polya", "te_class", "te_family", "position"]
	).size().unstack(fill_value=0)
	for col in ("downstream", "upstream"):
		if col not in by_family:
			by_family[col] = 0
	by_family = by_family[["downstream", "upstream"]]
	by_family["total"] = by_family.sum(axis=1)
	by_family = by_family.sort_values("total", ascending=False)
	print(f"  {'type':<4} {'class':<12} {'family':<14} "
	      f"{'down':>7} {'up':>7} {'total':>8}")
	for (is_polya, te_class, family), row in by_family.iterrows():
		print(f"  {'P' if is_polya else 'N':<4} {te_class:<12} {family:<14} "
		      f"{row['downstream']:>7,} {row['upstream']:>7,} {row['total']:>8,}")

	print("\n--- loci per canonical motif ---")
	print(f"  {'motif':<8} {'A%':>5} {G_BACKGROUND:>8} {G_P_DOWN:>8} "
	      f"{G_P_UP:>8} {G_N_DOWN:>8} {G_N_UP:>8}")
	counts = df.groupby(["canon_motif", "group"]).size().unstack(fill_value=0)
	for name in [G_BACKGROUND] + ASSIGNED_GROUPS:
		if name not in counts:
			counts[name] = 0
	counts = counts.sort_values(G_BACKGROUND, ascending=False)
	for motif, row in counts.iterrows():
		print(f"  {motif:<8} {a_fraction(motif):>5.0%} "
		      f"{row[G_BACKGROUND]:>8,} {row[G_P_DOWN]:>8,} {row[G_P_UP]:>8,} "
		      f"{row[G_N_DOWN]:>8,} {row[G_N_UP]:>8,}")


# ===========================================================================
# Main
# ===========================================================================

def main():
	rng = np.random.default_rng(SEED)

	print("=" * 78)
	print(f"Poly(A)-tail STR analysis  [MOTIF_LEN={MOTIF_LEN}, D={D} bp]")
	print("=" * 78)

	# --- load ------------------------------------------------------------
	str_df = load_str_data(STR_DATA_PATH)
	print(f"\nLoaded {len(str_df):,} loci from {STR_DATA_PATH}")
	# HipSTR writes a few loci as "MOTIF1/MOTIF2"; take every alternative.
	motif_lens = {
		len(part) for m in str_df["motif"] if isinstance(m, str)
		for part in m.split("/")
	}
	if motif_lens - {MOTIF_LEN}:
		raise ValueError(
			f"{STR_DATA_PATH} holds motif length(s) {sorted(motif_lens)}, "
			f"but MOTIF_LEN is {MOTIF_LEN} -- the two must agree"
		)

	chroms = set(str_df["chrom"].unique())
	te_df, n_non_te = load_te_bed(RMSK_BED_PATH, chroms)
	print(f"\nLoaded {len(te_df):,} TE rows from {RMSK_BED_PATH} "
	      f"on the {len(chroms)} chromosome(s) with STRs")
	print(f"  ignored {n_non_te:,} non-TE rows (repClass not in "
	      f"{sorted(TE_CLASSES)})")
	n_polya = int(te_df['is_polya'].sum())
	print(f"  poly(A)-type (P): {n_polya:,} rows "
	      f"({n_polya / len(te_df):.1%}); other TEs (N): "
	      f"{len(te_df) - n_polya:,}")

	ref = Fasta(REF_GENOME_PATH, as_raw=True)

	# --- motifs ----------------------------------------------------------
	print("\nReading plus-strand motifs from the reference...")
	str_df["plus_motif"] = read_plus_strand_motifs(str_df, ref, MOTIF_LEN)
	bad = [m for m in str_df["plus_motif"] if set(m) - set("ACGT")]
	if bad:
		raise ValueError(f"{len(bad)} loci have non-ACGT reference motifs, "
		                 f"e.g. {bad[:5]}")
	canon_cache = {m: canonical_motif(m) for m in set(str_df["plus_motif"])}
	str_df["canon_motif"] = [canon_cache[m] for m in str_df["plus_motif"]]
	print(f"  {str_df['canon_motif'].nunique()} canonical motifs")

	# Cross-check against the dataset's own motif column: it is rotation-
	# canonicalized, so it should be *a rotation of* what was just read, and
	# a disagreement would mean the coordinates are off by a phase.
	agree = sum(
		isinstance(m, str)
		and bool(set(part.upper() for part in m.split("/"))
		         & set(_rotations(p)))
		for m, p in zip(str_df["motif"], str_df["plus_motif"])
	)
	n_named = int(str_df["motif"].apply(lambda m: isinstance(m, str)).sum())
	print(f"  rotation agreement with the dataset's motif column: "
	      f"{agree:,}/{n_named:,}" + (f" ({len(str_df) - n_named:,} loci have "
	      f"no motif value)" if n_named < len(str_df) else ""))
	if agree != n_named:
		raise ValueError(
			f"{n_named - agree} loci disagree with the dataset's motif "
			f"column -- the coordinates or the reference are out of register"
		)

	# --- adjacency and groups --------------------------------------------
	print("\nAnnotating TE adjacency...")
	str_df = pd.concat([str_df, annotate_te_adjacency(str_df, te_df, D)], axis=1)
	str_df["group"] = assign_groups(str_df)

	n_overlap = int(str_df["te_overlap"].sum())
	n_multi = int(((str_df["n_adjacent"] > 1) & ~str_df["te_overlap"]).sum())
	print(f"\n--- exclusions ---")
	print(f"  overlapping a TE row          {n_overlap:>7,}  "
	      f"(expected 0: create_STR_data_files.py already dropped these)")
	print(f"  adjacent to >1 TE row         {n_multi:>7,}")
	print(f"  usable                        {int((str_df['group'] != '').sum()):>7,}"
	      f" / {len(str_df):,}")
	if n_overlap:
		print("  WARNING: non-zero overlap count -- rmsk_stranded.bed.gz and "
		      "the bed used to build the dataset are not in register.")

	str_df = str_df[str_df["group"] != ""].reset_index(drop=True)

	# --- sense orientation -------------------------------------------------
	# The sense motif is the plus-strand motif read on the ELEMENT's strand;
	# A-richness is only meaningful there, because that is the strand the
	# tail was written on. Background loci have no element, so these are NaN
	# for them and only the group masks ever read them.
	minus = str_df["te_strand"].to_numpy() == "-"
	str_df["sense_motif"] = [
		rev_comp(m) if flip else m
		for m, flip in zip(str_df["plus_motif"], minus)
	]
	# Background loci have no element and so no sense strand; the empty
	# string marks that, and every consumer below tests for it explicitly.
	str_df.loc[str_df["group"] == G_BACKGROUND, "sense_motif"] = ""
	sense = str_df["sense_motif"].to_numpy()
	str_df["sense_a_frac"] = [a_fraction(m) if m else np.nan for m in sense]
	# A motif with equal A and T counts reads the same both ways, so it
	# cannot say which orientation it is in and is excluded from 6 and 7.
	str_df["ori_informative"] = [
		bool(m) and m.count("A") != m.count("T") for m in sense
	]
	str_df["higher_a_orientation"] = [
		bool(m) and m.count("A") > m.count("T") for m in sense
	]
	str_df["is_tail"] = (
		(str_df["group"] == G_P_DOWN)
		& (str_df["sense_a_frac"] > TAIL_MIN_A)
	)
	n_tail = int(str_df["is_tail"].sum())
	n_p_down = int((str_df["group"] == G_P_DOWN).sum())
	print(f"\n  putative tails (P down with sense A% > {TAIL_MIN_A:.0%}): "
	      f"{n_tail:,} / {n_p_down:,} P-down loci "
	      f"({n_tail / max(n_p_down, 1):.1%})")

	# --- strand check, before any statistics ------------------------------
	if RUN_STRAND_CHECK:
		print_strand_check(str_df, ref, N_STRAND_CHECK, STRAND_CHECK_FLANK)

	# --- background sufficiency -------------------------------------------
	# Every score ranks a locus against its motif's background, and the
	# length-matched score needs N_MATCH of them. Motifs that cannot supply
	# that lose their FOCAL loci (their background loci stay, as part of the
	# background of nothing). Dropping them here rather than per comparison
	# keeps the focal set identical across all of them, which is what makes
	# comparison 9 a like-for-like repeat of 2, 4, 6 and 8.
	bg_counts = (
		str_df[str_df["group"] == G_BACKGROUND]["canon_motif"].value_counts()
	)
	ok_motifs = set(bg_counts[bg_counts >= N_MATCH].index)
	thin = (
		np.isin(str_df["group"].to_numpy(), ASSIGNED_GROUPS)
		& ~str_df["canon_motif"].isin(ok_motifs).to_numpy()
	)
	thin_motifs = sorted(set(str_df["canon_motif"][thin]))
	print(f"\n--- background sufficiency (>= {N_MATCH} background loci) ---")
	print(f"  dropped {int(thin.sum()):,} focal loci across "
	      f"{len(thin_motifs)} canonical motif(s)"
	      + (f": {', '.join(thin_motifs)}" if thin_motifs else ""))
	str_df = str_df[~thin].reset_index(drop=True)

	print_group_counts(str_df)

	# --- scores -----------------------------------------------------------
	print("\nComputing u-scores...")
	is_bg = (str_df["group"] == G_BACKGROUND).to_numpy()
	canon = str_df["canon_motif"].to_numpy()
	bg_canon = canon[is_bg]

	u_cols = {}
	for target, label_col in TARGETS.items():
		labels = str_df[label_col].to_numpy(dtype=float)
		col = f"u_{target}"
		values = np.full(len(str_df), np.nan)
		values[~is_bg] = motif_matched_u(
			labels[~is_bg], canon[~is_bg], labels[is_bg], bg_canon
		)
		values[is_bg] = motif_matched_u(
			labels[is_bg], bg_canon, labels[is_bg], bg_canon,
			leave_one_out=True,
		)
		str_df[col] = values
		u_cols[(SCORE_MOTIF, target)] = col
		print(f"  {SCORE_MOTIF:<14} {target:<16} "
		      f"mean(u) focal={np.nanmean(values[~is_bg]):.4f}  "
		      f"background={np.nanmean(values[is_bg]):.4f} (should be ~0.5)")

	cn = str_df[TARGETS["length"]].to_numpy(dtype=float)
	het = str_df[TARGETS["heterozygosity"]].to_numpy(dtype=float)
	col = "u_heterozygosity_matched"
	values = np.full(len(str_df), np.nan)
	values[~is_bg] = length_matched_u(
		cn[~is_bg], het[~is_bg], canon[~is_bg],
		cn[is_bg], het[is_bg], bg_canon, N_MATCH, rng,
	)
	bg_cn, bg_het = cn[is_bg], het[is_bg]
	values[is_bg] = length_matched_u(
		bg_cn, bg_het, bg_canon, bg_cn, bg_het, bg_canon, N_MATCH, rng,
		leave_one_out=True,
	)
	str_df[col] = values
	u_cols[(SCORE_MATCHED, "heterozygosity")] = col
	print(f"  {SCORE_MATCHED:<14} {'heterozygosity':<16} "
	      f"mean(u) focal={np.nanmean(values[~is_bg]):.4f}  "
	      f"background={np.nanmean(values[is_bg]):.4f} (should be ~0.5)")

	# --- primary comparisons ----------------------------------------------
	print(f"\nRunning primary comparisons "
	      f"({N_PERM:,} permutations / {N_BOOT:,} bootstrap draws each)...")
	passes = [
		(SCORE_MOTIF, "length", {"1", "2", "3", "4", "5", "6", "8", "10"}),
		(SCORE_MOTIF, "heterozygosity",
		 {"1", "2", "3", "4", "5", "6", "8", "10"}),
		# Comparison 7 is a count of orientations and never touches a label,
		# so it is run once rather than once per target.
		(SCORE_MOTIF, "-", {"7"}),
		# Comparison 9: the length confound.
		(SCORE_MATCHED, "heterozygosity", {"2", "4", "6", "8", "10"}),
	]
	primary = []
	for score, target, numbers in tqdm(passes, desc="  comparisons"):
		# Comparison 7 counts orientations and never reads a label, so its
		# pass has no target of its own; any u column satisfies the call.
		u_col = (u_cols[(score, target)] if target != "-"
		         else u_cols[(SCORE_MOTIF, "length")])
		primary += run_primary_comparisons(
			str_df, score, target, u_col, numbers, rng,
		)
	correct_p_values(primary, CORRECTION, CORRECTION_ALPHA)
	check_permutation_floor(primary, N_PERM, CORRECTION, CORRECTION_ALPHA,
	                        "primary")

	motif_rows = [r for r in primary if r["score"] == SCORE_MOTIF]
	matched_rows = [r for r in primary if r["score"] == SCORE_MATCHED]
	print_results(motif_rows,
	              "PRIMARY -- motif-matched score (comparisons 1-8)",
	              CORRECTION_ALPHA)
	print_results(matched_rows,
	              "PRIMARY -- comparison 9: length-matched heterozygosity "
	              "(2, 4, 6, 8 repeated)", CORRECTION_ALPHA)
	print(f"  ({correction_label(CORRECTION, CORRECTION_ALPHA)} is over both "
	      f"blocks together -- one family per run.)")

	# --- secondary --------------------------------------------------------
	print("\nRunning secondary per-motif comparisons...")
	secondary = run_secondary_comparisons(str_df, u_cols, rng)
	correct_p_values(secondary, CORRECTION, CORRECTION_ALPHA)
	check_permutation_floor(secondary, N_PERM, CORRECTION, CORRECTION_ALPHA,
	                        "secondary")
	print_secondary(secondary, CORRECTION_ALPHA)

	print_interpretation(primary)

	# --- outputs ----------------------------------------------------------
	if SAVE_RESULTS_PATH:
		os.makedirs(os.path.dirname(SAVE_RESULTS_PATH) or ".", exist_ok=True)
		with open(SAVE_RESULTS_PATH, "w") as f:
			json.dump(_json_safe({
				"config": {
					"motif_len": MOTIF_LEN, "d": D, "tail_min_a": TAIL_MIN_A,
					"n_match": N_MATCH, "min_n": MIN_N, "n_perm": N_PERM,
					"n_boot": N_BOOT, "ci_level": CI_LEVEL,
					"correction": CORRECTION,
					"correction_alpha": CORRECTION_ALPHA, "seed": SEED,
					"str_data_path": STR_DATA_PATH,
					"rmsk_bed_path": RMSK_BED_PATH,
				},
				"counts": {
					"n_overlap_excluded": n_overlap,
					"n_multi_adjacent_excluded": n_multi,
					"n_thin_motif_excluded": int(thin.sum()),
					"thin_motifs": thin_motifs,
					"n_putative_tails": n_tail,
					"by_group": {
						name: int(group_mask(str_df, name).sum())
						for name in [G_BACKGROUND] + ASSIGNED_GROUPS
						+ [G_ALL_ADJ, G_ALL_DOWN, G_ALL_UP, G_DOWN_NO_TAIL,
						   G_TAILS, G_NOT_P_DOWN, G_NOT_TAIL]
					},
				},
				"primary": primary,
				"secondary": secondary,
			}), f, indent=2)
		print(f"\nWrote {SAVE_RESULTS_PATH}")

	plot_primary_motif_matched(primary)
	plot_primary_length_matched(primary)
	plot_split_effects(primary)
	plot_orientation_fraction(primary)
	plot_a_content(str_df, u_cols)
	plot_secondary(secondary)


# ===========================================================================
# Reading the result
# ===========================================================================

def _find(results, comparison, group, target, score=SCORE_MOTIF):
	for r in results:
		if (r["comparison"] == comparison and r.get("group") == group
				and r["target"] == target and r["score"] == score):
			return r
	return None


def print_interpretation(primary):
	"""The three checks from the analysis plan, read off the results.

	These are not extra tests -- every number here is already in the tables
	above. They are the order in which the tables should be read, printed so
	that a run that failed its strand check cannot be mistaken for a run with
	a weak effect.
	"""
	print("\n" + "=" * 78)
	print("READING THE RESULT")
	print("=" * 78)

	print("\n1. Strand logic (comparison 7). The A-rich orientation should be "
	      "strongly\n   over-represented downstream of poly(A) elements. If it "
	      "is not well above\n   0.5, the strand logic is wrong and nothing "
	      "below means anything.")
	for name in (G_P_DOWN, G_N_DOWN, G_P_UP):
		r = _find(primary, "7", name, "-")
		if r is None or not np.isfinite(r["effect"]):
			continue
		role = "<- the test" if name == G_P_DOWN else "   (control)"
		print(f"   {role} {name:<8} {r['effect']:.3f} "
		      f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  n={r['n']:,}  "
		      f"adj={_fmt_p(r['p_value_corrected'])}")
	p_down = _find(primary, "7", G_P_DOWN, "-")
	if p_down is not None and np.isfinite(p_down["effect"]):
		if p_down["effect"] < 0.55:
			print("   VERDICT: NOT above 0.5 by any margin. Check the strand "
			      "logic (and the\n            strand check printed above) "
			      "before interpreting anything else.")
		else:
			print("   VERDICT: as expected under the tail hypothesis.")

	print("\n2. Reproduction (comparison 1). All down vs background should "
	      "match the\n   direction of the original TE-adjacency result "
	      "(nearby_repeat_elements.py\n   found adjacent loci LONGER and more "
	      "heterozygous, delta ~ +0.78 at the\n   boundary for str_len=4).")
	for target in TARGETS:
		for name in (G_ALL_DOWN, G_ALL_UP):
			r = _find(primary, "1", name, target)
			if r is None or not np.isfinite(r["effect"]):
				continue
			print(f"   {name:<10} {target:<16} {r['effect']:+.4f} "
			      f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  n={r['n']:,}")

	print("\n3. Tail exclusion (comparison 8 vs 1), and how much of the whole "
	      "near-TE\n   effect the tails actually carry (comparison 10b's "
	      "share S). S is the honest\n   number here: the drop from 1 to 8 "
	      "is a ratio of means over different sets\n   of loci, which "
	      "understates the share.")
	for target in TARGETS:
		full = _find(primary, "1", G_ALL_DOWN, target, SCORE_MOTIF)
		cut = _find(primary, "8", G_DOWN_NO_TAIL, target, SCORE_MOTIF)
		if full is None or cut is None:
			continue
		if np.isfinite(full["effect"]) and np.isfinite(cut["effect"]):
			print(f"   {target:<16} All down {full['effect']:+.4f} "
			      f"(n={full['n']:,}) -> no tails {cut['effect']:+.4f} "
			      f"(n={cut['n']:,})")
	print()
	for score in (SCORE_MOTIF, SCORE_MATCHED):
		for target in TARGETS:
			r = _find(primary, "10b", G_TAILS, target, score)
			if r is None or not np.isfinite(r["effect"]):
				continue
			share = (f"   tails carry S={r['share']:.3f} "
			         f"[{r['share_ci_lo']:.3f}, {r['share_ci_hi']:.3f}] of the "
			         f"total adjacency excess"
			         if np.isfinite(r.get("share", np.nan)) else
			         "   (no share on the length-matched score)")
			print(f"   {score:<14} {target:<16} "
			      f"delta_T={r['delta_t']:+.4f} vs delta_O={r['delta_o']:+.4f}"
			      f"  Delta={r['effect']:+.4f}{share}")

	print("\n4. Whole-effect accounting (comparison 10). How much bigger is "
	      "the poly(A)\n   effect than the rest of the near-TE effect, on "
	      "disjoint arms?")
	for version, t_name, _ in COMPARISON_10_SPLITS:
		for target in TARGETS:
			r = _find(primary, version, t_name, target, SCORE_MOTIF)
			if r is None or not np.isfinite(r["effect"]):
				continue
			share = (f"  S={r['share']:.3f} "
			         f"[{r['share_ci_lo']:.3f}, {r['share_ci_hi']:.3f}]"
			         if np.isfinite(r.get("share", np.nan)) else "")
			print(f"   {version} {t_name:<16} {target:<16} "
			      f"T={r['n']:>6,} O={r['n_other']:>6,}  "
			      f"Delta={r['effect']:+.4f} "
			      f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
			      f"  adj={_fmt_p(r['p_value_corrected'])}"
			      f"{'*' if r['significant'] else ' '}{share}")


# ===========================================================================
# Plots
# ===========================================================================

def _show_or_save(fig, name, save_dir):
	"""Mirrors _show_or_save in nearby_repeat_elements.py."""
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, f"{name}.png")
		fig.savefig(path, dpi=150, bbox_inches="tight")
		plt.close(fig)
		print(f"  saved {path}")
	else:
		plt.show()


def _forest_series(ax, rows, y, color, label):
	"""One series of effect +/- CI. Filled = significant, open = not.

	Fill carries significance and colour carries the series, so a panel can
	show two series without the two encodings colliding.
	"""
	effects = np.array([r["effect"] for r in rows], dtype=float)
	lo = np.array([r["ci_lo"] for r in rows], dtype=float)
	hi = np.array([r["ci_hi"] for r in rows], dtype=float)
	sig = np.array([bool(r["significant"]) for r in rows])

	ax.hlines(y, lo, hi, color=color, lw=2, alpha=0.85)
	ax.scatter(effects[sig], y[sig], color=color, s=34, zorder=3, label=label)
	ax.scatter(effects[~sig], y[~sig], facecolors="white", edgecolors=color,
	           s=34, linewidths=1.4, zorder=3,
	           label=None if sig.any() else label)


def _n_label(r):
	"""The n(s) for a comparison row.

	A bare "n=" is ambiguous for any two-sample row, and which second group
	it would mean differs by comparison: the background for 1/2/3/8, the
	other group for 4/5/6, the other ARM for 10 -- where neither n is the
	background, since background is the scale both arms are scored against
	rather than a group in the comparison. Both are always shown, in the
	order the row's question names them, so "A vs B  (n=.. vs n=..)" reads
	straight across.
	"""
	other = r.get("n_background", r.get("n_other"))
	if other is None:
		return f"n={r['n']:,}"
	return f"n={r['n']:,} vs n={other:,}"


def _effect_xlabel():
	return ("effect   (vs background: 2*mean(u) - 1;   "
	        "between groups: Cliff's delta)")


def plot_primary_motif_matched(primary):
	"""Figure 1 -- comparisons 1-6 and 8 on the motif-matched score, both
	targets on ONE axes.

	Copy number and heterozygosity are plotted against the same comparison
	rows rather than in side-by-side panels because the interesting thing
	about them is that they agree: every row has the two markers nearly on
	top of each other, which says the poly(A) signal is one phenomenon
	showing up in both labels, not two separate findings. Side-by-side panels
	make that comparison something the reader has to do by eye across a gap.
	Comparison 7 is a count, not an effect, so it gets its own figure.
	"""
	by_key = {}
	for r in primary:
		if (r["score"] != SCORE_MOTIF or r["target"] not in TARGETS
				or not np.isfinite(r["effect"])):
			continue
		by_key.setdefault((r["comparison"], r["question"]), {})[r["target"]] = r
	if not by_key:
		return

	keys = sorted(by_key, reverse=True)
	y = np.arange(len(keys))
	colors = {"length": "tab:blue", "heterozygosity": "tab:red"}

	fig, ax = plt.subplots(figsize=(10, 0.36 * len(keys) + 2.0))
	ax.axvline(0, color="black", lw=0.8, zorder=0)
	for i, (target, color) in enumerate(colors.items()):
		rows = [by_key[k].get(target) for k in keys]
		present = [j for j, r in enumerate(rows) if r is not None]
		if not present:
			continue
		# y counts upward, so the first target in `colors` needs the POSITIVE
		# offset to sit on top of the pair: length above, heterozygosity below.
		offset = (0.5 - i) * 0.34
		_forest_series(ax, [rows[j] for j in present],
		               y[present] + offset, color, target)

	# Labels come from whichever target is present for that row.
	ax.set_yticks(y)
	ax.set_yticklabels(
		[f"{c}. {q}  ({_n_label(next(iter(by_key[(c, q)].values())))})"
		 for c, q in keys],
		fontsize=8,
	)
	ax.set_ylim(-0.7, len(keys) - 0.3)
	ax.grid(axis="x", alpha=0.3)
	ax.set_xlim(-1, 1)
	ax.set_xlabel(_effect_xlabel(), fontsize=9)
	ax.legend(fontsize=9, loc="lower right")
	ax.set_title(
		f"1. Poly(A)-tail analysis: motif-matched score  "
		f"[MOTIF_LEN={MOTIF_LEN}, D={D} bp]\n"
		f"filled = significant ({correction_label(CORRECTION, CORRECTION_ALPHA)}), "
		f"open = not",
		fontsize=11,
	)
	fig.tight_layout()
	_show_or_save(fig, "te_downstream_1_motif_matched", SAVE_DIR)


def plot_primary_length_matched(primary):
	"""Figure 2 -- comparison 9, heterozygosity on the length-matched score.

	The motif-matched heterozygosity values from figure 1 are drawn behind in
	grey, on the same rows. Without them the panel is unreadable: an effect
	of +0.12 only means something next to the +0.53 it started at, and the
	gap between the two series IS the result -- how much of the
	heterozygosity signal was length all along.
	"""
	matched, reference = {}, {}
	for r in primary:
		if r["target"] != "heterozygosity" or not np.isfinite(r["effect"]):
			continue
		key = (r["comparison"], r["question"])
		if r["score"] == SCORE_MATCHED:
			matched[key] = r
		elif r["score"] == SCORE_MOTIF:
			reference[key] = r
	if not matched:
		return

	keys = sorted(matched, reverse=True)
	y = np.arange(len(keys))

	fig, ax = plt.subplots(figsize=(10, 0.46 * len(keys) + 2.0))
	ax.axvline(0, color="black", lw=0.8, zorder=0)

	ref_rows = [(j, reference[k]) for j, k in enumerate(keys) if k in reference]
	if ref_rows:
		_forest_series(ax, [r for _, r in ref_rows],
		               y[[j for j, _ in ref_rows]] + 0.17,
		               "tab:gray", "motif-matched (fig. 1)")
	_forest_series(ax, [matched[k] for k in keys], y - 0.17,
	               "tab:red", "length-matched")

	ax.set_yticks(y)
	ax.set_yticklabels([f"{c}. {q}  ({_n_label(matched[(c, q)])})"
	                    for c, q in keys], fontsize=8)
	ax.set_ylim(-0.7, len(keys) - 0.3)
	ax.grid(axis="x", alpha=0.3)
	ax.set_xlim(-1, 1)
	ax.set_xlabel(_effect_xlabel(), fontsize=9)
	ax.legend(fontsize=9, loc="lower right")
	ax.set_title(
		f"2. Comparison 9: heterozygosity with length held fixed  "
		f"[MOTIF_LEN={MOTIF_LEN}, N_MATCH={N_MATCH}]\n"
		f"shrinkage from grey to red is the part of the signal that was "
		f"copy number",
		fontsize=11,
	)
	fig.tight_layout()
	_show_or_save(fig, "te_downstream_2_length_matched", SAVE_DIR)


def plot_split_effects(primary):
	"""Figure 3 -- comparison 10: each arm against background, on one axes.

	One row per (version, score), each showing the poly(A) arm T and the
	rest-of-the-TE-effect arm O with their bootstrap CIs. The two arms are
	disjoint and together are every adjacent locus, so the GAP between them
	is Delta -- drawn as the bar joining them, annotated with its value, and
	solid only when Delta is significant. That makes the figure one claim
	("the poly(A) arm sits this much higher than everything else near a TE")
	rather than three panels the reader has to reconcile.

	The share S is deliberately not plotted. It needs its own explanation to
	be read correctly -- in particular its no-difference point is each row's
	share of LOCI, not 0.5 -- and a panel cannot carry that. It is still
	computed, and still reported in the comparison 10 text block and in
	READING THE RESULT.
	"""
	rows = []
	for version, t_name, o_name in COMPARISON_10_SPLITS:
		for score, target in [(SCORE_MOTIF, "length"),
		                      (SCORE_MOTIF, "heterozygosity"),
		                      (SCORE_MATCHED, "heterozygosity")]:
			r = _find(primary, version, t_name, target, score)
			if r is not None and np.isfinite(r["effect"]):
				rows.append((version, score, target, r))
	if not rows:
		return

	rows = rows[::-1]  # drawn bottom-up; keep spec order top-down
	y = np.arange(len(rows))

	fig, ax = plt.subplots(figsize=(11, 0.78 * len(rows) + 2.4))
	ax.axvline(0, color="black", lw=0.8, zorder=0)

	off = 0.17
	for i, (_, _, _, r) in enumerate(rows):
		# Delta first, underneath: the bar joining the two arms IS the
		# difference, so it is drawn rather than stated.
		ax.plot([r["delta_o"], r["delta_t"]], [y[i], y[i]],
		        color="tab:purple", lw=2.6 if r["significant"] else 1.4,
		        ls="-" if r["significant"] else ":", alpha=0.75, zorder=2)
		for key, lo_k, hi_k, dy, color in [
			("delta_t", "delta_t_ci_lo", "delta_t_ci_hi", off, "tab:red"),
			("delta_o", "delta_o_ci_lo", "delta_o_ci_hi", -off, "tab:gray"),
		]:
			ax.hlines(y[i] + dy, r[lo_k], r[hi_k], color=color, lw=2.2,
			          zorder=3)
			ax.scatter(r[key], y[i] + dy, color=color, s=46, zorder=4)
		ax.annotate(
			f"$\\Delta$={r['effect']:+.3f}{'*' if r['significant'] else ''}",
			((r["delta_o"] + r["delta_t"]) / 2, y[i] + 0.40),
			ha="center", fontsize=8.5, color="tab:purple",
		)

	# Data-driven limits: everything sits well inside [-1, 1], and the fixed
	# axis would squeeze the arms together and hide the CIs.
	lo = min(0.0, min(r["delta_o_ci_lo"] for *_, r in rows)) - 0.06
	hi = max(r["delta_t_ci_hi"] for *_, r in rows) + 0.10
	ax.set_xlim(lo, hi)

	ax.set_yticks(y)
	ax.set_yticklabels(
		[f"{v}  {sc}\n{tg}\n{_n_label(r)}" for v, sc, tg, r in rows],
		fontsize=8,
	)
	ax.set_ylim(-0.75, len(rows) - 0.25)
	ax.grid(axis="x", alpha=0.3)
	ax.set_xlabel("2*mean(u) - 1 against background", fontsize=9)

	handles = [
		Line2D([], [], color="tab:red", marker="o", lw=2.2, ls="-",
		       label="poly(A) arm (T)"),
		Line2D([], [], color="tab:gray", marker="o", lw=2.2, ls="-",
		       label="rest of the TE effect (O)"),
		Line2D([], [], color="tab:purple", lw=2.6,
		       label=f"$\\Delta$ = T - O (solid/* = significant, "
		             f"{correction_label(CORRECTION, CORRECTION_ALPHA)})"),
	]
	ax.legend(handles=handles, fontsize=8, loc="lower right")

	t_a, t_b = COMPARISON_10_SPLITS[0][1], COMPARISON_10_SPLITS[1][1]
	ax.set_title(
		f"3. Comparison 10: the poly(A) effect against the rest of the TE "
		f"effect  [MOTIF_LEN={MOTIF_LEN}, D={D} bp]\n"
		f"10a T = {t_a};  10b T = {t_b};  each T vs all other adjacent loci "
		f"(disjoint arms, bars are {int(CI_LEVEL * 100)}% bootstrap CIs)",
		fontsize=11,
	)
	fig.tight_layout()
	_show_or_save(fig, "te_downstream_3_split_effects", SAVE_DIR)


def plot_orientation_fraction(primary):
	"""Figure 3 -- comparison 7, the orientation count asymmetry.

	Plotted as the raw fraction against a line at 0.5 rather than rescaled to
	an effect: it is a proportion, it has its own natural null, and on its
	own axes there is no reason to disguise it as one. This is also the
	strand-logic check -- P down far to the right of 0.5 is what says the
	orientation code is right, so it is worth being able to point at one
	figure for it.
	"""
	found = {r.get("group"): r for r in primary
	         if r["comparison"] == "7" and np.isfinite(r["effect"])}
	# The test first, then its two controls -- the same order
	# print_interpretation uses, and the order the figure argues in. Drawn
	# bottom-up, so the list is reversed to put P down on top.
	order = [G_P_DOWN, G_N_DOWN, G_P_UP]
	rows = [found[g] for g in reversed(order) if g in found]
	rows += [r for g, r in found.items() if g not in order]
	if not rows:
		return
	y = np.arange(len(rows))

	colors = {G_P_DOWN: "tab:red", G_N_DOWN: "tab:blue", G_P_UP: "tab:orange"}
	fig, ax = plt.subplots(figsize=(9, 0.8 * len(rows) + 2.2))
	ax.axvline(0.5, color="black", lw=1.0, ls="--", zorder=0)

	for i, r in enumerate(rows):
		color = colors.get(r.get("group"), "tab:gray")
		ax.hlines(y[i], r["ci_lo"], r["ci_hi"], color=color, lw=2.5)
		ax.scatter(r["effect"], y[i],
		           color=color if r["significant"] else "white",
		           edgecolors=color, linewidths=1.4, s=70, zorder=3)
		ax.annotate(f"{r['effect']:.3f}", (r["effect"], y[i]),
		            textcoords="offset points", xytext=(0, 11),
		            ha="center", fontsize=9, color=color)

	ax.set_yticks(y)
	ax.set_yticklabels(
		[f"{r['group']}  (n={r['n']:,})"
		 + ("  <- the test" if r.get("group") == G_P_DOWN else "  (control)")
		 for r in rows],
		fontsize=9,
	)
	ax.set_ylim(-0.7, len(rows) - 0.3)
	ax.set_xlim(0, 1)
	ax.set_xlabel("fraction of loci whose sense motif is the A-rich "
	              "orientation   (0.5 = no asymmetry)", fontsize=9)
	ax.grid(axis="x", alpha=0.3)
	ax.set_title(
		f"4. Comparison 7: sense-strand orientation asymmetry  "
		f"[MOTIF_LEN={MOTIF_LEN}, D={D} bp]\n"
		f"motifs with unequal A and T counts only; "
		f"filled = significant ({correction_label(CORRECTION, CORRECTION_ALPHA)})",
		fontsize=11,
	)
	fig.tight_layout()
	_show_or_save(fig, "te_downstream_4_orientation", SAVE_DIR)


def plot_a_content(df, u_cols):
	"""Mean u against sense-strand A%, per group.

	Descriptive context for comparison 6 rather than a plot of it: comparison
	6 contrasts the two ORIENTATIONS of a motif, which holds composition
	fixed, while this shows the whole A%-vs-u relationship across motifs,
	where A content and tail status are confounded. It is kept because it is
	the clearest single picture of the effect -- the tail hypothesis predicts
	a rising line for P down and flat lines for the two controls, and that is
	what it shows -- but the test to quote is comparison 6.

	Point area is proportional to n, because the extreme A% bins can be thin.
	"""
	specs = [(SCORE_MOTIF, "length"), (SCORE_MOTIF, "heterozygosity"),
	         (SCORE_MATCHED, "heterozygosity")]
	fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4.5),
	                         sharey=True)
	axes = np.atleast_1d(axes)

	colors = {G_P_DOWN: "tab:red", G_N_DOWN: "tab:blue", G_P_UP: "tab:orange"}
	for ax, (score, target) in zip(axes, specs):
		u = df[u_cols[(score, target)]].to_numpy()
		for name, color in colors.items():
			mask = group_mask(df, name)
			frac = df["sense_a_frac"].to_numpy()
			levels = np.unique(frac[mask & np.isfinite(frac)])
			means, sizes = [], []
			for level in levels:
				sel = mask & (frac == level) & np.isfinite(u)
				means.append(np.mean(u[sel]) if sel.any() else np.nan)
				sizes.append(int(sel.sum()))
			sizes = np.array(sizes, dtype=float)
			ax.plot(levels, means, "-", color=color, alpha=0.6, lw=1)
			ax.scatter(levels, means, color=color, label=f"{name}",
			           s=12 + 110 * sizes / max(sizes.max(), 1), alpha=0.8)
		ax.axhline(0.5, color="black", lw=0.8, ls="--")
		ax.set_title(f"{score}\n{target}", fontsize=10)
		ax.set_xlabel("sense-strand A fraction of the motif")
		ax.grid(alpha=0.3)
	axes[0].set_ylabel("mean u (0.5 = background median)")
	axes[0].legend(fontsize=8)
	fig.suptitle(
		f"5. Comparison 6: label percentile vs sense-strand A content  "
		f"[MOTIF_LEN={MOTIF_LEN}]  (point area ~ n)",
		fontsize=11,
	)
	fig.tight_layout()
	_show_or_save(fig, "te_downstream_5_a_content", SAVE_DIR)


def plot_secondary(secondary):
	"""Per-motif effects for comparisons 2 and 3, one panel per target.

	The pooled primaries already control for motif; this shows whether they
	are carried by one or two motifs or hold broadly. Motifs are ordered by
	A content so that any tail signal reads left to right.
	"""
	if not secondary:
		return
	fig, axes = plt.subplots(1, len(TARGETS), figsize=(7 * len(TARGETS), 5),
	                         sharey=True)
	axes = np.atleast_1d(axes)
	colors = {G_P_DOWN: "tab:red", G_P_UP: "tab:orange",
	          G_N_DOWN: "tab:blue", G_N_UP: "tab:cyan"}

	motifs = sorted({r["motif"] for r in secondary},
	                key=lambda m: (a_fraction(m), m))
	x_of = {m: i for i, m in enumerate(motifs)}

	for ax, target in zip(axes, TARGETS):
		for name, color in colors.items():
			rows = [r for r in secondary
			        if r["target"] == target and r["group"] == name
			        and np.isfinite(r["effect"])]
			if not rows:
				continue
			x = np.array([x_of[r["motif"]] for r in rows], dtype=float)
			offset = (list(colors).index(name) - 1.5) * 0.16
			effects = np.array([r["effect"] for r in rows])
			lo = np.array([r["ci_lo"] for r in rows])
			hi = np.array([r["ci_hi"] for r in rows])
			ax.vlines(x + offset, lo, hi, color=color, lw=1.5, alpha=0.7)
			ax.scatter(x + offset, effects, color=color, s=22, label=name,
			           edgecolors=["black" if r["significant"] else "none"
			                       for r in rows],
			           linewidths=0.8, zorder=3)
		ax.axhline(0, color="black", lw=0.8)
		ax.set_xticks(range(len(motifs)))
		ax.set_xticklabels(
			[f"{m}\n{a_fraction(m):.0%}A" for m in motifs], fontsize=7
		)
		ax.set_title(target, fontsize=10)
		ax.grid(axis="y", alpha=0.3)
	axes[0].set_ylabel("2*mean(u) - 1  (vs same-motif background)")
	axes[0].legend(fontsize=8)
	fig.suptitle(
		f"6. Secondary: comparisons 2 and 3 per canonical motif  "
		f"[MOTIF_LEN={MOTIF_LEN}, >= {MIN_N} focal loci]  "
		f"black edge = significant "
		f"({correction_label(CORRECTION, CORRECTION_ALPHA)})",
		fontsize=11,
	)
	fig.tight_layout()
	_show_or_save(fig, "te_downstream_6_by_motif", SAVE_DIR)


if __name__ == "__main__":
	main()
