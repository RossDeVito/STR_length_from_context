""" Get baseline performance metrics.

Baseline models:
	- Mean
	- Mean by STR motif
"""

import os
import json

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import scipy.stats as stats

from pyfaidx import Fasta

from eval_multiple_preds import get_metrics


STR_CANONICALIZATION = {
	"A": "A",
	"C": "C",
	"G": "G",
	"T": "T",
	"AC": "AC",
	"CA": "AC",
	"AG": "AG",
	"GA": "AG",
	"AT": "AT",
	"TA": "AT",
	"CG": "CG",
	"GC": "CG",
	"CT": "CT",
	"TC": "CT",
	"GT": "GT",
	"TG": "GT",
}

REF_GENOME_FILE = "../../data/STR_data/reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa"



def load_and_add_motif(data_file, str_len):

	data_df = pd.read_csv(
		data_file,
		sep="\t"
	)

	ref_genome = Fasta(REF_GENOME_FILE, sequence_always_upper=True)

	for idx, row in data_df.iterrows():
		chrom = row["chrom"]
		start = row["str_start"]
		end = row["str_end"]

		str_seq = ref_genome[chrom][start:end]
		try:
			motif = STR_CANONICALIZATION[str_seq[:str_len].seq]
		except KeyError:
			print(f"Warning: Unrecognized STR motif {str_seq[:str_len].seq} at {chrom}:{start}-{end}.")
			print(f"STR sequence: {str_seq}")
			print(f"Index: {idx}")
			raise KeyError(f"Unrecognized STR motif {str_seq[:str_len].seq} at {chrom}:{start}-{end}.")

		data_df.at[idx, "motif"] = motif

	return data_df


if __name__ == '__main__':

	# Options
	data_dir = "../../data/STR_data/filtered_labeled_strs"
	data_fname = "str_len_{}_max_cn_perc_0.975_n_flanking_10000_filtered.tsv"

	# Load data and add motif column
	str1_data = load_and_add_motif(
		os.path.join(data_dir, data_fname.format(1)),
		str_len=1
	)
	str2_data = load_and_add_motif(
		os.path.join(data_dir, data_fname.format(2)),
		str_len=2
	)

	# Get overall means from training set
	str1_data['mean_copy_number'] = str1_data[
		str1_data['split'] == 'train'
	]['copy_number'].mean()

	str2_data['mean_copy_number'] = str2_data[
		str2_data['split'] == 'train'
	]['copy_number'].mean()

	# Get motif-specific means from training set
	str1_motif_means = str1_data[
		str1_data['split'] == 'train'
	].groupby('motif')['copy_number'].mean().to_dict()
	str2_motif_means = str2_data[
		str2_data['split'] == 'train'
	].groupby('motif')['copy_number'].mean().to_dict()

	str1_data['motif_mean_copy_number'] = str1_data['motif'].map(
		str1_motif_means
	)
	str2_data['motif_mean_copy_number'] = str2_data['motif'].map(
		str2_motif_means
	)

	# Compute metrics
	baseline_perf = {
		1: dict(),
		2: dict()
	}

	str1_test = str1_data[str1_data['split'] == 'test']
	str2_test = str2_data[str2_data['split'] == 'test']

	baseline_perf[1]['overall_mean'] = get_metrics(
		pred=str1_test['mean_copy_number'],
		true=str1_test['copy_number']
	)
	baseline_perf[2]['overall_mean'] = get_metrics(
		pred=str2_test['mean_copy_number'],
		true=str2_test['copy_number']
	)

	baseline_perf[1]['motif_mean'] = get_metrics(
		pred=str1_test['motif_mean_copy_number'],
		true=str1_test['copy_number']
	)
	baseline_perf[2]['motif_mean'] = get_metrics(
		pred=str2_test['motif_mean_copy_number'],
		true=str2_test['copy_number']
	)

	print("Baseline Performance:")
	for str_len in [1, 2]:
		print(f"STR Length {str_len}:")
		for baseline_type in ['overall_mean', 'motif_mean']:
			print(f"  {baseline_type}:")
			for metric_name, metric_value in baseline_perf[str_len][baseline_type].items():
				print(f"    {metric_name}: {metric_value:.4f}")

	# Save baseline performance to pretty JSON
	os.makedirs("predictions/baseline", exist_ok=True)
	with open("predictions/baseline/mean_performance.json", "w") as f:
		json.dump(baseline_perf, f, indent=4)