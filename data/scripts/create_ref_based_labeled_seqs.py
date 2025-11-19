"""Create data files with reference-based labeled sequences.

Filters to just perfect repeats.

Will also create 8:1:1 train/val/test splits and create reverse complements
for each STR that are in the same split as the original sequence.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tqdm

from pyfaidx import Fasta


def get_args():
	parser = argparse.ArgumentParser(
		description="Create data files with reference-based labeled sequences."
	)
	
	parser.add_argument(
		"-s", 
		"--str-len", 
		type=int, 
		help="Length of the STR to consider (e.g., 2 for dinucleotide repeats)."
	)
	
	parser.add_argument(
		"-f",
		"--n-flanking", 
		type=int, 
		help="Number of flanking bases to include on each side of the STR."
	)
	
	parser.add_argument(
		"-p",
		"--copy-num-percentile-max", 
		type=float, 
		help="Maximum copy number percentile to include (e.g., 0.99)."
	)
	
	parser.add_argument(
		"-d",
		"--distance-from-edge",
		type=int,
		default=0,
		help="Minimum distance from chromosome edge to consider STR."
	)

	return parser.parse_args()


if __name__ == "__main__":

	# Parse Command Line Arguments
	args = get_args()
	
	str_len = args.str_len
	n_flanking = args.n_flanking
	copy_num_percentile_max = args.copy_num_percentile_max

	print(f"Running with settings:")
	print(f"\tSTR length: {str_len}")
	print(f"\tNumber of flanking bases: {n_flanking}")
	print(f"\tMaximum copy number percentile: {copy_num_percentile_max}")
	print(f"\tDistance from edge: {args.distance_from_edge}")

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
	
	# Handle case where dataframe might be empty after filtering by len
	if str_df.empty:
		print(f"No STRs found with motif length {str_len}")
		exit()

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
	
	buffer_distance = max(n_flanking, args.distance_from_edge)

	# Flag to validate that pre + str + post seqs match reference once
	first_check_done = False

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
			row["start"] - 1 - buffer_distance < 0 
			or row["end"] + buffer_distance > len(ref_genome[row["chrom"]])
		):
			out_of_bounds_count += 1
			continue
			
		pre_seq = ref_genome[row["chrom"]][
			row["start"] - 1 - n_flanking : row["start"] - 1
		]
		post_seq = ref_genome[row["chrom"]][
			row["end"] : row["end"] + n_flanking
		]

		# One time integrity check
		if not first_check_done:
			reconstructed_seq = pre_seq.seq + str_seq.seq + post_seq.seq
			
			# Fetch the contiguous block from reference
			ground_truth_seq = ref_genome[row["chrom"]][
				row["start"] - 1 - n_flanking : row["end"] + n_flanking
			].seq

			# Compare
			if reconstructed_seq != ground_truth_seq:
				print(f"ERROR: Mismatch on {row['name']}")
				print(f"Reconstructed: {reconstructed_seq}")
				print(f"Ground Truth:  {ground_truth_seq}")
				raise ValueError("Sequence reconstruction integrity check failed.")
			
			print(f"Integrity check passed on {row['name']} (Pre + STR + Post matches Reference).")
			first_check_done = True

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
	
	if perfect_str_df.empty:
		print("No perfect STRs found matching criteria. Exiting without saving.")
		exit()

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
			f"str_len_{str_len}_max_cnp_perc_{copy_num_percentile_max}_n_flanking_{n_flanking}_buffer_bp_{buffer_distance}.tsv"
		),
		sep="\t",
		index=False
	)