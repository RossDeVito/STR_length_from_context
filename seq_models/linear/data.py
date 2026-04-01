""" Handles loading STR data and converting to k-mer count representations
for linear models. 

Requires a TSV file with columns:
- chrom: Chromosome name (e.g., 'chr1')
- str_start: Start index of the STR
- str_end: End index of the STR (first base after the STR)
- copy_number: Copy number of the STR in reference genome
- rev_comp: Boolean indicating if sequence is reverse complement
- split: One of 'train', 'val', or 'test' indicating data split

As well as reference genome FASTA file to extract sequences. (.fa and .fai,
see data/STR_data/reference_genome/download.sh)

Function create_kmer_data() creates a dataset of k-mer count vectors for
each STR for the specified n_flanking_bp and additionally has a one-hot
encoding of the STR motif.
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import product as itertools_product

from pyfaidx import Fasta
from tqdm import tqdm


# Canonical motif groupings (same as scripts/eval_preds/get_baseline_performance.py)
STR_CANONICALIZATION = {
	"A": "A", "C": "C", "G": "G", "T": "T",
	"AC": "AC", "CA": "AC", "AG": "AG", "GA": "AG",
	"AT": "AT", "TA": "AT", "CG": "CG", "GC": "CG",
	"CT": "CT", "TC": "CT", "GT": "GT", "TG": "GT",
}

_VALID_BASES = set("ACGT")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def all_kmers(k):
	"""Return sorted list of all 4^k DNA k-mers."""
	return sorted(
		"".join(bases) for bases in itertools_product("ACGT", repeat=k)
	)


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

	Uses (str_end - str_start) / copy_number, rounded to the nearest
	integer.  All rows in a given data file share the same motif length.
	"""
	row = str_df.iloc[0]
	motif_len = round((row["str_end"] - row["str_start"]) / row["copy_number"])
	assert motif_len in (1, 2), (
		f"Unexpected motif length {motif_len} inferred from first row"
	)
	return motif_len


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def create_kmer_data(
	data_path,
	ref_path,
	n_flanking_bp,
	k_values=(3, 4, 5),
):
	"""Create k-mer count dataset for linear baseline models.

	For each sample, extracts left and right flanking sequences from
	the reference genome and counts k-mers separately for each flank.
	Reverse-complement samples are handled the same way as in the
	HyenaDNA pipeline: the genomic right flank is reverse-complemented
	to become the model's left flank and vice versa.

	K-mers are NOT canonicalized — a k-mer and its reverse complement
	are separate features.  Since both forward and RC orientations of
	each locus appear as independent samples, the model sees both
	strand contexts and can learn motif–k-mer strand relationships
	directly.

	The STR motif identity IS canonicalized (e.g. CA -> AC) so that
	forward and RC samples of the same locus share a single motif
	category.

	Features (in order):
		- For each k in k_values, for each flank (left, right):
			4^k counts (one per possible k-mer, sorted lexicographically)
		- One-hot encoding of the canonical STR motif

	Args:
		data_path: Path to TSV file with STR data.
		ref_path: Path to reference genome FASTA file.
		n_flanking_bp: Number of flanking base pairs on each side.
		k_values: Sequence of k values for k-mer counting.

	Returns:
		Tuple of (splits, feature_names) where:
			splits: dict with keys 'train', 'val', 'test', each a dict
				containing:
				  'X': np.ndarray of shape (n_samples, n_features)
				  'y': np.ndarray of shape (n_samples,) — log1p(copy_number)
				  'metadata': pd.DataFrame (columns from input TSV)
			feature_names: list of str of length n_features
	"""

	k_values = tuple(k_values)

	# ------------------------------------------------------------------
	# Load data and reference genome
	# ------------------------------------------------------------------

	str_df = pd.read_csv(data_path, sep="\t")
	ref_genome = Fasta(ref_path, sequence_always_upper=True)

	motif_len = infer_motif_len(str_df)

	# ------------------------------------------------------------------
	# Build feature schema
	# ------------------------------------------------------------------

	# K-mer features: left_{k}mer_{ACGT...}, right_{k}mer_{ACGT...}
	kmer_vocab = {}  # k -> sorted list of k-mer strings
	feature_names = []

	for k in k_values:
		kmers = all_kmers(k)
		kmer_vocab[k] = kmers
		for flank in ("left", "right"):
			for kmer in kmers:
				feature_names.append(f"{flank}_{k}mer_{kmer}")

	# Motif one-hot features
	canonical_motifs = sorted(set(
		STR_CANONICALIZATION[m]
		for m in STR_CANONICALIZATION
		if len(m) == motif_len
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
	print(f"  motif_len:      {motif_len}")
	print(f"  n_features:     {n_features}")
	print(f"  n_samples:      {n_samples}")

	# ------------------------------------------------------------------
	# Build feature matrix and label vector
	# ------------------------------------------------------------------

	# Pre-compute k-mer -> column index lookup per (k, flank)
	# so the inner loop just does dict lookups
	kmer_col_offset = {}
	offset = 0
	for k in k_values:
		n_kmers = 4 ** k
		kmer_col_offset[(k, "left")] = {
			km: offset + j for j, km in enumerate(kmer_vocab[k])
		}
		offset += n_kmers
		kmer_col_offset[(k, "right")] = {
			km: offset + j for j, km in enumerate(kmer_vocab[k])
		}
		offset += n_kmers

	X = np.zeros((n_samples, n_features), dtype=np.float32)
	y = np.zeros(n_samples, dtype=np.float32)

	for i in tqdm(range(n_samples), desc="Counting k-mers"):
		row = str_df.iloc[i]
		chrom = row["chrom"]
		start = int(row["str_start"])
		end = int(row["str_end"])

		# Extract raw genomic flanks (pyfaidx Sequence objects)
		left_seq_obj = ref_genome[chrom][start - n_flanking_bp : start]
		right_seq_obj = ref_genome[chrom][end : end + n_flanking_bp]

		# Reverse complement: swap flanks and complement (same as
		# HyenaDNA data.py __getitem__)
		if row["rev_comp"]:
			left_seq = right_seq_obj.reverse.complement.seq
			right_seq = left_seq_obj.reverse.complement.seq
		else:
			left_seq = left_seq_obj.seq
			right_seq = right_seq_obj.seq

		# Count k-mers per flank
		for k in k_values:
			for flank, seq in (("left", left_seq), ("right", right_seq)):
				col_map = kmer_col_offset[(k, flank)]
				for kmer, count in count_kmers(seq, k).items():
					X[i, col_map[kmer]] = count

		# Motif one-hot (canonicalized)
		motif_bases = ref_genome[chrom][start : start + motif_len].seq
		canonical = STR_CANONICALIZATION[motif_bases]
		X[i, motif_feature_offset + motif_to_idx[canonical]] = 1.0

		# Label: log1p(copy_number) to match HyenaDNA target
		y[i] = np.log1p(row["copy_number"])

	# ------------------------------------------------------------------
	# Split into train / val / test
	# ------------------------------------------------------------------

	splits = {}
	for split_name in ("train", "val", "test"):
		mask = (str_df["split"] == split_name).values
		n_split = mask.sum()
		splits[split_name] = {
			"X": X[mask],
			"y": y[mask],
			"metadata": str_df.loc[mask].reset_index(drop=True),
		}
		print(f"  {split_name}: {n_split} samples")

	return splits, feature_names