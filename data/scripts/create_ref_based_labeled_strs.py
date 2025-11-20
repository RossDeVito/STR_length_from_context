"""Create data files with indices of reference genome STRs.

Filters to just perfect repeats.

Checks that STR is at least n_flanking bases from edge of chromosome
so that flanking sequence can be extracted on both sides of at most that
length.

Adds two rows for each STR: the original sequence and its reverse complement.

Will also create 8:1:1 train/val/test splits and create reverse complements
for each STR that are in the same split as the original sequence.

Output columns:
- HipSTR_name: Name of the STR in HipSTR reference
- chrom: Chromosome of the STR
- rev_comp: Boolean indicating if sequence is reverse complement
- copy_number: Copy number of the STR in reference genome
- str_start: Start position of the STR in reference genome with 0-based
	indexing to match pyfaidx
- str_end: End position of the STR in reference genome with 0-based
	indexing to match pyfaidx. Note that this is the first base *after* the STR.
- split: One of 'train', 'val', or 'test' indicating data split

CLI arguments:
- str_len: Length of the STR to consider (e.g., 2 for dinucleotide repeats)
- n_flanking: Number of flanking bases that must exist on each side of the STR
- copy_num_percentile_max: Maximum copy number percentile to include (e.g.,
	0.99) for given str_len
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
		"reference_based_labeled_strs"
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

	# Filter by copy number percentile
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

	# Flag that str and flanking sequence loading check still needs to be done
	first_check_done = False

	for _, row in tqdm.tqdm(str_df.iterrows(), total=len(str_df)):

		start_idx = row["start"] - 1  # Convert to 0-based indexing
		end_idx = row["end"]  # End is 0-based exclusive

		str_seq = ref_genome[row["chrom"]][start_idx:end_idx]
		
		# Check if perfect repeat
		base_motif = str_seq[:str_len].seq
		perfect_motif_str = (
			base_motif * int(row["ref_repeats"] + 1)
		)[: end_idx - start_idx]

		if str_seq.seq != perfect_motif_str:
			imperfect_count += 1
			continue

		# Confirm flanking sequence of max length n_flanking exists
		if (
			row["start"] - 1 - n_flanking < 0 
			or row["end"] + n_flanking > len(ref_genome[row["chrom"]])
		):
			out_of_bounds_count += 1
			continue

		# One time integrity check using 100 flanking bases
		if not first_check_done:
			n_flanking_check = n_flanking

			pre_seq = ref_genome[row["chrom"]][
				start_idx - n_flanking_check : start_idx
			]
			assert len(pre_seq) == n_flanking_check

			post_seq = ref_genome[row["chrom"]][
				end_idx : end_idx + n_flanking_check
			]
			assert len(post_seq) == n_flanking_check

			# Reconstruct full sequence
			reconstructed_seq = pre_seq.seq + str_seq.seq + post_seq.seq
			
			# Fetch the contiguous block from reference
			ground_truth_seq = ref_genome[row["chrom"]][
				start_idx - n_flanking_check : end_idx + n_flanking_check
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
			"str_start": start_idx,
			"str_end": end_idx,
			"split": chrom_to_split.get(row["chrom"], "train"),
		})

		perfect_STRs.append({
			"HipSTR_name": row["name"],
			"chrom": row["chrom"],
			"rev_comp": True,
			"copy_number": row["ref_repeats"],
			"str_start": start_idx,
			"str_end": end_idx,
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
			f"str_len_{str_len}_max_cn_perc_{copy_num_percentile_max}_n_flanking_{n_flanking}.tsv"
		),
		sep="\t",
		index=False
	)