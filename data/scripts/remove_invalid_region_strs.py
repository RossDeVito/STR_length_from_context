"""Remove STRs in invalid regions from dataset.

Filters output of create_ref_based_labeled_strs.py to remove STRs located
in mobile elements or segmental duplications (where there may be data 
leakage issues) and ENCODE blacklist regions (which may be low quality).
Assumes the exclusion region files are in the format resulting from
data/invalid_regions/download.sh.

Output columns are the same as input:
- HipSTR_name: Name of the STR in HipSTR reference
- chrom: Chromosome of the STR
- rev_comp: Boolean indicating if sequence is reverse complement
- copy_number: Copy number of the STR in reference genome
- str_start: Start position of the STR in reference genome with 0-based
	indexing to match pyfaidx
- str_end: End position of the STR in reference genome with 0-based
	indexing to match pyfaidx. Note that this is the first base *after* the STR.
- split: One of 'train', 'val', or 'test' indicating data split

Outputs to output dir with same filename as input file with suffix
"_filtered".

CLI args:
	str_file: Path to input TSV file with labeled STR sequences.
	mobile_elements_bed: Path to BED file with mobile element regions.
		Default: "../invalid_regions/mobile_elements.bed.gz".
	seg_dups_bed: Path to BED file with segmental duplication regions.
		Default: "../invalid_regions/seg_dups.bed.gz".
	blacklist_bed: Path to BED file with ENCODE blacklist regions.
		Default: "../invalid_regions/blacklist.bed.gz".
	output_dir: Directory to save filtered STR file. Default:
		"../STR_data/filtered_labeled_strs/"
"""

import os
import argparse
import numpy as np
import pandas as pd
import tqdm


def get_args():
	parser = argparse.ArgumentParser(
		description="Remove STRs in invalid regions from dataset."
	)
	
	parser.add_argument(
		"--str_file",
		type=str,
		required=True,
		help="Path to input TSV file with labeled STR sequences."
	)
	
	parser.add_argument(
		"--mobile_elements_bed",
		type=str,
		default="../invalid_regions/mobile_elements.bed.gz",
		help="Path to BED file with mobile element regions. "
	)

	parser.add_argument(
		"--seg_dups_bed",
		type=str,
		default="../invalid_regions/seg_dups.bed.gz",
		help="Path to BED file with segmental duplication regions. "
	)
	
	parser.add_argument(
		"--blacklist_bed",
		type=str,
		default="../invalid_regions/blacklist.bed.gz",
		help="Path to BED file with ENCODE blacklist regions. "
	)
	
	parser.add_argument(
		"--output_dir",
		type=str,
		default="../STR_data/filtered_labeled_strs/",
		help=(
			"Directory to save filtered STR file. "
			"Default: ../STR_data/filtered_labeled_strs/"
		)
	)
	
	return parser.parse_args()


class GenomicIntersector:
	"""Efficiently checks overlaps between genomic intervals."""
	
	def __init__(self, region_df, name="region"):
		"""
		region_df must have columns: [chrom, start, end]
		"""
		self.name = name
		self.chrom_intervals = {}
		
		# Group by chrom, sort, and merge overlapping intervals so that
		# binary search to work correctly
		for chrom, group in region_df.groupby("chrom"):
			starts = group["start"].values
			ends = group["end"].values
			
			# Sort by start
			sort_idx = np.argsort(starts)
			starts = starts[sort_idx]
			ends = ends[sort_idx]
			
			# Merge overlapping intervals
			if len(starts) > 0:
				merged_starts = []
				merged_ends = []
				
				curr_start = starts[0]
				curr_end = ends[0]
				
				for i in range(1, len(starts)):
					if starts[i] < curr_end:
						# Overlap or adjacent, extend current end
						curr_end = max(curr_end, ends[i])
					else:
						# Gap found, save current and start new
						merged_starts.append(curr_start)
						merged_ends.append(curr_end)
						curr_start = starts[i]
						curr_end = ends[i]
				
				merged_starts.append(curr_start)
				merged_ends.append(curr_end)
				
				self.chrom_intervals[chrom] = (
					np.array(merged_starts), 
					np.array(merged_ends)
				)

	def check_overlaps(self, query_df):
		"""
		Returns a boolean array: True if row in query_df overlaps any region.
		"""
		result = np.zeros(len(query_df), dtype=bool)
		
		# Iterate over chromosomes present in the query
		for chrom, group in query_df.groupby("chrom"):
			if chrom not in self.chrom_intervals:
				continue
				
			ref_starts, ref_ends = self.chrom_intervals[chrom]
			
			q_starts = group["str_start"].values
			q_ends = group["str_end"].values
			
			# Find insertion points of query starts into reference ends
			# searchsorted returns the index where q_start would fit in ref_ends to keep order
			# side='right': strictly greater
			idx = np.searchsorted(ref_ends, q_starts, side='right')
			
			# Check valid indices:
			# An overlap exists if: ref_start[idx] < q_end
			# We must be careful with indices out of bounds (idx == len(ref_starts))
			valid_mask = idx < len(ref_starts)
			
			# For the valid indices, check the start condition
			# We only check those where idx is valid to avoid IndexError
			overlap_mask = np.zeros(len(q_starts), dtype=bool)
			overlap_mask[valid_mask] = ref_starts[idx[valid_mask]] < q_ends[valid_mask]
			
			# Assign back to original indices
			result[group.index] = overlap_mask
			
		return result


if __name__ == "__main__":

	args = get_args()

	# Load files
	str_df = pd.read_csv(args.str_file, sep="\t")
	mobile_df = pd.read_csv(
		args.mobile_elements_bed,
		sep="\t",
		header=None,
		names=["chrom", "start", "end", "class"],
		compression="gzip"
	)
	seg_dups_df = pd.read_csv(
		args.seg_dups_bed,
		sep="\t",
		header=None,
		names=["chrom", "start", "end"],
		compression="gzip"
	)
	blacklist_df = pd.read_csv(
		args.blacklist_bed,
		sep="\t",
		header=None,
		names=["chrom", "start", "end"],
		compression="gzip"
	)

	# Build intersectors
	print("Building interval trees for fast lookup...")
	
	# Simple ones
	black_searcher = GenomicIntersector(blacklist_df, "Blacklist")
	seg_searcher = GenomicIntersector(seg_dups_df, "SegDups")
	
	mobile_classes = mobile_df["class"].unique()
	mobile_searchers = {}
	for m_class in mobile_classes:
		# Create a searcher for just this class (e.g., just SINEs)
		subset = mobile_df[mobile_df["class"] == m_class]
		mobile_searchers[m_class] = GenomicIntersector(subset[[ "chrom", "start", "end" ]], m_class)

	# 3. Check Overlaps
	print("Checking overlaps...")
	
	# Track overlap columns
	str_df["overlap_blacklist"] = black_searcher.check_overlaps(str_df)
	str_df["overlap_segdup"] = seg_searcher.check_overlaps(str_df)
	
	for m_class, searcher in mobile_searchers.items():
		col_name = f"overlap_{m_class}"
		str_df[col_name] = searcher.check_overlaps(str_df)

	# 4. Calculate Stats
	total_strs = len(str_df)
	print(f"\nTotal STRs: {total_strs}")
	print("-" * 45)
	
	# Define widths
	label_w = 25  # Wide enough for "Overlapping Blacklist:"
	count_w = 10  # Wide enough for "1235380"
	
	def print_stat(label, mask):
		count = mask.sum()
		pct = mask.mean()
		# 1. Create the label with the colon attached
		text = f"{label}:"
		
		# 2. Print using fixed widths
		# {text:<{label_w}} : Left-align the text within 25 spaces
		# {count:>{count_w}}: Right-align the count within 10 spaces
		print(f"{text:<{label_w}} {count:>{count_w}} ({pct:>6.1%})")

	# Manual prints
	print_stat("Overlapping Blacklist", str_df['overlap_blacklist'])
	print_stat("Overlapping SegDups", str_df['overlap_segdup'])
	
	# Loop prints
	for m_class in sorted(mobile_searchers.keys()):
		print_stat(f"Overlapping {m_class}", str_df[f"overlap_{m_class}"])

	# 5. Combined Filter
	filter_cols = ["overlap_blacklist", "overlap_segdup"] + [f"overlap_{c}" for c in mobile_searchers.keys()]
	str_df["is_excluded"] = str_df[filter_cols].any(axis=1)
	
	n_excluded = str_df["is_excluded"].sum()
	n_kept = total_strs - n_excluded
	
	print("-" * 45)
	# Reuse the same logic for the summary lines
	print(f"{'Total Excluded:':<{label_w}} {n_excluded:>{count_w}} ({n_excluded/total_strs:>6.1%})")
	print(f"{'Total Kept:':<{label_w}} {n_kept:>{count_w}} ({n_kept/total_strs:>6.1%})")

	# 6. Save Output
	# Drop the temporary columns before saving
	cols_to_drop = filter_cols + ["is_excluded"]
	clean_df = str_df[~str_df["is_excluded"]].drop(columns=cols_to_drop)
	removed_df = str_df[str_df["is_excluded"]]
	
	# Construct output filename
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
		
	base_name = os.path.basename(args.str_file)
	out_name = base_name.replace(".tsv", "_filtered.tsv")
	out_path = os.path.join(args.output_dir, out_name)
	
	clean_df.to_csv(out_path, sep="\t", index=False)
	print(f"\nSaved filtered dataset to: {out_path}")

	# Save removed STRs
	removed_name = base_name.replace(".tsv", "_removed.tsv")
	removed_path = os.path.join(args.output_dir, removed_name)
	removed_df.to_csv(removed_path, sep="\t", index=False)
	print(f"Saved removed STRs to: {removed_path}")