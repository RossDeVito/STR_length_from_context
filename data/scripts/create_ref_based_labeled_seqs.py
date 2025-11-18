"""Create data files with reference-based labeled sequences.

Filters to just perfect repeats.

Will also create 8:1:1 train/val/test splits and create reverse complements
for each STR that are in the same split as the original sequence.

Options:

	str_len: Length of the STR to consider (e.g., 2 for dinucleotide repeats)
	n_flanking: Number of flanking bases to include on each side of the STR
	copy_num_percentile_max: Maximum copy number percentile to include
		(e.g., 0.99 to include STRs up to the 99th percentile of copy number)
	
"""

import os

import numpy as np
import pandas as pd
import tqdm

from pyfaidx import Fasta


if __name__ == "__main__":

	# Options
	str_len = 1
	n_flanking = 512	# int(np.ceil(window_size / 2).item())
	copy_num_percentile_max = 0.975

	# Paths
	script_dir = os.path.abspath(os.path.dirname(__file__))

	ref_genome_path = os.path.join(
		script_dir,
		"..",
		"STR_data",
		"reference_genome",
		"GRCh38_full_analysis_set_plus_decoy_hla.fa"
	)
	hipstr_ref_path = os.path.join(
		script_dir,
		"..",
		"STR_data",
		"HipSTR-reference",
		"hg38.hipstr_reference.bed"
	)
	output_dir = os.path.join(
		script_dir,
		"..",
		"STR_data",
		"reference_based_labeled_seqs"
	)


	# Load HipSTR reference
	str_df = pd.read_csv(
		hipstr_ref_path,
		sep="\t",
		header=None,
		names=[
			"chrom", "start", "end", "motif_len",
			"ref_repeats", "name", "motif"
		]
	)

	# Filter to STRs with specified motif length and within
	# copy number percentile
	str_df = str_df[str_df["motif_len"] == str_len]
	copy_num_threshold = str_df["ref_repeats"].quantile(copy_num_percentile_max)
	str_df = str_df[str_df["ref_repeats"] <= copy_num_threshold]
	print(
		f"Number of STRs with motif length {str_len} "
		f"and up to {copy_num_percentile_max} percentile "
		f"({copy_num_threshold} repeats): {len(str_df)}"
	)


	# Find perfect repeat STRs in reference genome
	perfect_STRs = []
	ref_genome = Fasta(ref_genome_path, sequence_always_upper=True)

	imperfect_count = 0
	out_of_bounds_count = 0

	# Have chrom 14 and X map to 'test', 13 and 3 to 'val', rest to 'train'
	chrom_to_split = {
		"chr14": "test",
		"chrX": "test",
		"chr13": "val",
		"chr3": "val"
	}

	for _, row in tqdm.tqdm(str_df.iterrows(), total=len(str_df)):

		str_seq = ref_genome[row["chrom"]][row["start"] - 1 : row["end"]]
		
		# Check if perfect repeat
		base_motif = str_seq[:str_len].seq
		perfect_motif_str = (
			base_motif * int(row["ref_repeats"]) + 
			base_motif[: int(row["ref_repeats"] % 1 * str_len)]
		)

		if str_seq != perfect_motif_str:
			imperfect_count += 1
			continue

		# Get flanking sequences
		if (
			row["start"] - 1 - n_flanking < 0 
			or row["end"] + n_flanking > len(ref_genome[row["chrom"]])
		):
			out_of_bounds_count += 1
			continue
			
		pre_seq = ref_genome[row["chrom"]][
			row["start"] - 1 - n_flanking : row["start"] - 1
		]
		post_seq = ref_genome[row["chrom"]][
			row["end"] : row["end"] + n_flanking
		]

		# Add sequence and reverse complement to list
		perfect_STRs.append({
			"HipSTR_name": row["name"],
			"chrom": row["chrom"],
			"rev_comp": False,
			"copy_number": row["ref_repeats"],
			"pre_seq": pre_seq.seq,
			"str_seq": str_seq.seq,
			"post_seq": post_seq.seq,
			"split": chrom_to_split.get(row["chrom"], "train"),
		})

		perfect_STRs.append({
			"HipSTR_name": row["name"],
			"chrom": row["chrom"],
			"rev_comp": True,
			"copy_number": row["ref_repeats"],
			"pre_seq": post_seq.reverse.complement.seq,
			"str_seq": str_seq.reverse.complement.seq,
			"post_seq": pre_seq.reverse.complement.seq,
			"split": chrom_to_split.get(row["chrom"], "train"),
		})

	print(f"Number of imperfect STRs excluded: {imperfect_count}")
	print(f"Number of out-of-bounds STRs excluded: {out_of_bounds_count}")
	print(f"Number of perfect STRs included: {len(perfect_STRs) / 2}")

	# Create dataframe and train/val/test splits
	perfect_str_df = pd.DataFrame(perfect_STRs)

	# Get split counts and fraction of total
	print(perfect_str_df["split"].value_counts())
	print(
		perfect_str_df["split"].value_counts() / perfect_str_df.shape[0]
	)

	# Save to output files
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	perfect_str_df.to_csv(
		os.path.join(
			output_dir,
			f"str_len_{str_len}_max_cnp_perc_{copy_num_percentile_max}_n_flanking_{n_flanking}.tsv"
		),
		sep="\t",
		index=False
	)