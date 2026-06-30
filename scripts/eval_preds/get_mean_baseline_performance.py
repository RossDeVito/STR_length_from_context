""" Get baseline performance metrics.

Baseline models:
	- Mean
	- Mean by STR motif

Targets and data follow the Caduceus standard (see seq_models/caduceus): the
multi-objective targets are `length` (mode_copy_number) and `variation`
(heterozygosity). Each locus appears twice in the data (forward and reverse
complement); Caduceus predict.py evaluates per-locus (one row per ID after
RC-averaging), so we dedup to one row per locus here too.
"""

import os
import json

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import scipy.stats as stats

from pyfaidx import Fasta

from eval_multiple_preds import get_metrics


# Regression targets, mirroring DEFAULT_TARGETS in seq_models/caduceus/model.py
# (hardcoded here to avoid importing the heavy caduceus/torch stack).
TARGETS = {
	"length": "mode_copy_number",
	"variation": "heterozygosity",
}

REF_GENOME_FILE = "../../data/reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa"

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(seq):
	"""Return the reverse complement of an (uppercase) DNA string."""
	return seq.translate(_COMPLEMENT)[::-1]


def canonicalize_motif(seq):
	"""Canonicalize an STR motif so it is invariant to rotation and strand.

	Returns the lexicographically smallest string over all rotations of the
	motif and all rotations of its reverse complement. Strand is not meaningful
	(the Caduceus pipeline trains with rev_comp augmentation), so e.g. AC, CA,
	GT and TG all collapse to "AC", and A/T -> "A", C/G -> "C".
	"""
	candidates = []
	for s in (seq, reverse_complement(seq)):
		for i in range(len(s)):
			candidates.append(s[i:] + s[:i])
	return min(candidates)


def load_and_add_motif(data_file, str_len):

	data_df = pd.read_csv(
		data_file,
		sep="\t"
	)

	# One row per locus (the two rev_comp orientations share motif + targets).
	data_df = data_df.drop_duplicates("ID").reset_index(drop=True)

	ref_genome = Fasta(REF_GENOME_FILE, sequence_always_upper=True)

	for idx, row in data_df.iterrows():
		chrom = row["chrom"]
		start = row["str_start"]
		end = row["str_end"]

		str_seq = ref_genome[chrom][start:end]
		data_df.at[idx, "motif"] = canonicalize_motif(str_seq[:str_len].seq)

	return data_df


if __name__ == '__main__':

	# Options
	data_dir = "../../data/STR_data/HipSTR_labeled_STRs"
	data_fname = "str_len_{}_n_flanking_10000.tsv"

	# Load data (one row per locus) and add canonical motif column
	str_data = {
		str_len: load_and_add_motif(
			os.path.join(data_dir, data_fname.format(str_len)),
			str_len=str_len
		)
		for str_len in [1, 2, 4]
	}

	# Compute baselines per STR length and per target.
	baseline_perf = {1: dict(), 2: dict(), 4: dict()}

	for str_len, data_df in str_data.items():
		train_df = data_df[data_df['split'] == 'train']
		test_df = data_df[data_df['split'] == 'test']

		for out_name, col in TARGETS.items():
			# Overall mean and per-motif means from the training set.
			overall_mean = train_df[col].mean()
			motif_means = train_df.groupby('motif')[col].mean().to_dict()

			overall_pred = np.full(len(test_df), overall_mean)
			motif_pred = test_df['motif'].map(motif_means)

			baseline_perf[str_len][out_name] = {
				'overall_mean': get_metrics(
					pred=overall_pred,
					true=test_df[col]
				),
				'motif_mean': get_metrics(
					pred=motif_pred,
					true=test_df[col]
				),
			}

	print("Baseline Performance:")
	for str_len in [1, 2, 4]:
		print(f"STR Length {str_len}:")
		for out_name in TARGETS:
			print(f"  {out_name}:")
			for baseline_type in ['overall_mean', 'motif_mean']:
				print(f"    {baseline_type}:")
				for metric_name, metric_value in baseline_perf[str_len][out_name][baseline_type].items():
					print(f"      {metric_name}: {metric_value:.4f}")

	# Save baseline performance to pretty JSON
	os.makedirs("predictions/baseline", exist_ok=True)
	with open("predictions/baseline/mean_performance.json", "w") as f:
		json.dump(baseline_perf, f, indent=4)