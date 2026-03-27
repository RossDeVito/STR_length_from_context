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
	"N": "N",
	"NN": "NN",
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
			print(f"Warning: Unrecognized STR motif {str_seq[:str_len].seq} at {chrom}:{start}-{end}. Setting motif to 'N'.")
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
