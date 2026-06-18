""" Handles loading STR data and converting to k-mer count representations
for linear models.

Requires a TSV file (the Caduceus standard) with columns:
- ID: Locus identifier (used to pair reverse-complement samples downstream)
- chrom: Chromosome name (e.g., 'chr1')
- str_start: Start index of the STR
- str_end: End index of the STR (first base after the STR)
- motif: Repeat unit sequence (may be blank; the motif is re-derived from the
  reference, so the column is not relied upon)
- ref_copy_number: Reference copy number (used to derive the motif length)
- rev_comp: Boolean indicating if sequence is reverse complement
- split: One of 'train', 'val', or 'test' indicating data split
- One column per regression target (e.g. 'mode_copy_number', 'heterozygosity');
  these are carried through in 'metadata' and transformed by the trainer.

As well as reference genome FASTA file to extract sequences. (.fa and .fai,
see data/reference_genome/download.sh)

Function create_kmer_data() creates a dataset of k-mer count vectors for each
STR for the specified n_flanking_bp and additionally has a one-hot encoding of
the STR motif. Both orientations of each locus are emitted as independent
samples, and every sample (k-mers and motif) reflects its own strand, mirroring
the Caduceus pipeline; the two per-locus predictions are averaged downstream.
K-mer counts can optionally be stratified by flank side and/or by distance from
the STR.
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import product as itertools_product

from pyfaidx import Fasta
from tqdm import tqdm


_VALID_BASES = set("ACGT")
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def reverse_complement(seq):
	"""Return the reverse complement of an (uppercase) DNA string."""
	return seq.translate(_COMPLEMENT)[::-1]


def rotation_canonicalize(seq):
	"""Canonicalize an STR motif over rotation only (NOT reverse complement).

	Returns the lexicographically smallest rotation of the motif, so the phase
	of the STR boundary doesn't matter (e.g. CA -> AC, TG -> GT) while strand is
	preserved (AC stays distinct from GT, A from T). Each sample is assigned the
	motif on its own strand (rev_comp samples pass the reverse-complemented
	motif), mirroring how Caduceus feeds the real strand sequence.
	"""
	return min(seq[i:] + seq[:i] for i in range(len(seq)))


def all_kmers(k):
	"""Return sorted list of all 4^k DNA k-mers."""
	return sorted(
		"".join(bases) for bases in itertools_product("ACGT", repeat=k)
	)


def is_primitive(seq):
	"""True if *seq* is not a repetition of a shorter unit (e.g. 'AA' is not).

	Used to restrict the motif one-hot vocabulary to true period-`motif_len`
	repeat units, excluding homopolymeric dinucleotides like AA/CC/GG/TT.
	"""
	n = len(seq)
	for d in range(1, n):
		if n % d == 0 and seq == seq[:d] * (n // d):
			return False
	return True


def count_kmers(seq, k):
	"""Count k-mers in *seq*, skipping any k-mer with non-ACGT bases.

	Returns:
		Counter mapping k-mer string -> count.
	"""
	counts = Counter()
	for i in range(len(seq) - k + 1):
		kmer = seq[i : i + k]
		if _VALID_BASES.issuperset(kmer):
			counts[kmer] += 1
	return counts


def infer_motif_len(str_df):
	"""Infer the repeat-unit length from the first row of a STR DataFrame.

	Uses (str_end - str_start) / ref_copy_number, rounded to the nearest
	integer.  All rows in a given data file share the same motif length.
	"""
	row = str_df.iloc[0]
	motif_len = round(
		(row["str_end"] - row["str_start"]) / row["ref_copy_number"]
	)
	assert motif_len in (1, 2), (
		f"Unexpected motif length {motif_len} inferred from first row"
	)
	return motif_len


def build_distance_edges(distance_bins, n_flanking_bp):
	"""Return bin edges [0, e1, ..., n_flanking_bp] from a distance_bins config.

	`distance_bins` is a list of upper edges (bp from the STR). None means a
	single bin spanning the whole flank. The final edge n_flanking_bp is
	appended automatically if not already present.
	"""
	if not distance_bins:
		return [0, n_flanking_bp]

	edges = [int(e) for e in distance_bins]
	if any(b <= a for a, b in zip(edges, edges[1:])):
		raise ValueError(f"distance_bins must be strictly increasing: {edges}")
	if edges[0] <= 0 or edges[-1] > n_flanking_bp:
		raise ValueError(
			f"distance_bins {edges} must be in (0, n_flanking_bp={n_flanking_bp}]."
		)
	if edges[-1] != n_flanking_bp:
		edges.append(n_flanking_bp)
	return [0] + edges


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def create_kmer_data(
	data_path,
	ref_path,
	n_flanking_bp,
	k_values=(3, 4, 5),
	flank_mode="separate",
	distance_bins=None,
):
	"""Create k-mer count dataset for linear baseline models.

	For each sample, extracts left and right flanking sequences from the
	reference genome and counts k-mers. Reverse-complement samples are handled
	the same way as in the HyenaDNA/Caduceus pipelines: the genomic right flank
	is reverse-complemented to become the model's left flank and vice versa.
	Both orientations of each locus appear as independent samples (augmentation);
	the trainer post-hoc averages them per locus at evaluation time.

	K-mers are NOT canonicalized — a k-mer and its reverse complement are
	separate features. The STR motif is rotation-canonicalized only (phase-
	invariant) and assigned per strand, so a forward sample and its RC twin get
	different motif categories (e.g. motif_G vs motif_C), reflecting the actual
	strand rather than folding the two together.

	Flank handling (`flank_mode`):
		- "separate": left and right flanks are distinct feature groups.
		- "pooled":   left and right k-mer counts are summed into one group.

	Distance binning (`distance_bins`): when set, k-mers are counted separately
	in distance bins measured from the STR (e.g. [1000, 5000] -> the closest
	1 kb vs 1-5 kb out). The STR-adjacent base is at the end of the left flank
	and the start of the right flank.

	Features (in order):
		- For each k in k_values, for each side, for each distance bin:
			4^k counts (one per possible k-mer, sorted lexicographically)
		- One-hot encoding of the canonical STR motif

	Args:
		data_path: Path to TSV file with STR data.
		ref_path: Path to reference genome FASTA file.
		n_flanking_bp: Number of flanking base pairs on each side.
		k_values: Sequence of k values for k-mer counting.
		flank_mode: "separate" (left/right distinct) or "pooled" (summed).
		distance_bins: Optional list of bp upper-edges for distance stratification.

	Returns:
		Tuple of (splits, feature_names) where:
			splits: dict with keys 'train', 'val', 'test', each a dict with:
			  'X': np.ndarray of shape (n_samples, n_features)
			  'metadata': pd.DataFrame (columns from input TSV)
			feature_names: list of str of length n_features
	"""

	k_values = tuple(k_values)
	if flank_mode not in ("separate", "pooled"):
		raise ValueError(f"flank_mode must be 'separate' or 'pooled', got {flank_mode!r}")

	# ------------------------------------------------------------------
	# Load data and reference genome
	# ------------------------------------------------------------------

	str_df = pd.read_csv(data_path, sep="\t")
	ref_genome = Fasta(ref_path, sequence_always_upper=True)

	motif_len = infer_motif_len(str_df)

	# Distance bins and the side labels we materialize.
	edges = build_distance_edges(distance_bins, n_flanking_bp)
	bins = list(zip(edges, edges[1:]))  # [(lo, hi), ...]
	n_bins = len(bins)
	sides = ["left", "right"] if flank_mode == "separate" else ["flank"]
	multi_bin = n_bins > 1

	# ------------------------------------------------------------------
	# Build feature schema and (k, side, bin) -> column-offset lookup
	# ------------------------------------------------------------------

	kmer_vocab = {k: all_kmers(k) for k in k_values}
	feature_names = []
	# col_offset[(k, side, bin_idx)] -> {kmer: column index}
	col_offset = {}
	offset = 0
	for k in k_values:
		for side in sides:
			for b in range(n_bins):
				col_offset[(k, side, b)] = {
					km: offset + j for j, km in enumerate(kmer_vocab[k])
				}
				bin_tag = f"_bin{b}" if multi_bin else ""
				for km in kmer_vocab[k]:
					feature_names.append(f"{side}{bin_tag}_{k}mer_{km}")
				offset += 4 ** k

	# Motif one-hot features (rotation-only canonical set; strand preserved).
	# Restrict to primitive units so str2 yields {AC,AG,AT,CG,CT,GT} (no
	# homopolymers) and str1 yields {A,C,G,T}.
	canonical_motifs = sorted(set(
		rotation_canonicalize(km) for km in all_kmers(motif_len) if is_primitive(km)
	))
	motif_to_idx = {m: i for i, m in enumerate(canonical_motifs)}
	motif_feature_offset = len(feature_names)
	for m in canonical_motifs:
		feature_names.append(f"motif_{m}")

	n_features = len(feature_names)
	n_samples = len(str_df)

	print(f"Building k-mer dataset:")
	print(f"  data_path:      {data_path}")
	print(f"  n_flanking_bp:  {n_flanking_bp}")
	print(f"  k_values:       {k_values}")
	print(f"  flank_mode:     {flank_mode}")
	print(f"  distance_bins:  {bins}")
	print(f"  motif_len:      {motif_len}  ({canonical_motifs})")
	print(f"  n_features:     {n_features}")
	print(f"  n_samples:      {n_samples}")

	# ------------------------------------------------------------------
	# Build feature matrix
	# ------------------------------------------------------------------

	X = np.zeros((n_samples, n_features), dtype=np.float32)
	L = n_flanking_bp

	for i in tqdm(range(n_samples), desc="Counting k-mers"):
		row = str_df.iloc[i]
		chrom = row["chrom"]
		start = int(row["str_start"])
		end = int(row["str_end"])

		# Extract raw genomic flanks (pyfaidx Sequence objects)
		left_seq_obj = ref_genome[chrom][start - n_flanking_bp : start]
		right_seq_obj = ref_genome[chrom][end : end + n_flanking_bp]

		# Reverse complement: swap flanks and complement (same convention as the
		# HyenaDNA/Caduceus pipelines). After this, the STR-adjacent base is at
		# the end of left_seq and the start of right_seq for both orientations.
		if row["rev_comp"]:
			left_seq = right_seq_obj.reverse.complement.seq
			right_seq = left_seq_obj.reverse.complement.seq
		else:
			left_seq = left_seq_obj.seq
			right_seq = right_seq_obj.seq

		# Slice each flank into distance bins (measured from the STR) and count.
		for b, (lo, hi) in enumerate(bins):
			left_bin = left_seq[L - hi : L - lo]
			right_bin = right_seq[lo:hi]

			for k in k_values:
				if flank_mode == "separate":
					for side, seg in (("left", left_bin), ("right", right_bin)):
						cmap = col_offset[(k, side, b)]
						for kmer, cnt in count_kmers(seg, k).items():
							X[i, cmap[kmer]] = cnt
				else:  # pooled: sum left + right counts into one group
					cmap = col_offset[(k, "flank", b)]
					pooled = count_kmers(left_bin, k)
					pooled.update(count_kmers(right_bin, k))
					for kmer, cnt in pooled.items():
						X[i, cmap[kmer]] = cnt

		# Motif one-hot, on this sample's strand: take the forward-strand unit and
		# reverse-complement it for rev_comp samples, then rotation-canonicalize.
		motif_bases = ref_genome[chrom][start : start + motif_len].seq
		if row["rev_comp"]:
			motif_bases = reverse_complement(motif_bases)
		canonical = rotation_canonicalize(motif_bases)
		X[i, motif_feature_offset + motif_to_idx[canonical]] = 1.0

	# ------------------------------------------------------------------
	# Split into train / val / test
	# ------------------------------------------------------------------

	splits = {}
	for split_name in ("train", "val", "test"):
		mask = (str_df["split"] == split_name).values
		n_split = mask.sum()
		splits[split_name] = {
			"X": X[mask],
			"metadata": str_df.loc[mask].reset_index(drop=True),
		}
		print(f"  {split_name}: {n_split} samples")

	return splits, feature_names
