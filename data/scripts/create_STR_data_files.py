"""Create and filter labeled STR dataset from EnsembleTR population data.

Combines and replaces the previous create_ref_based_labeled_strs.py and
remove_invalid_region_strs.py scripts, adapted to use EnsembleTR allele
frequency data instead of HipSTR reference copy numbers.

Pipeline:
	1. Load EnsembleTR repeat_info and afreq_het tables.
	2. Merge on locus ID.
	3. Filter to specified motif length.
	4. Compute pooled pan-human labels (equivalent to pooling all samples
	   across populations and then computing stats): heterozygosity, and
	   user-selected subset of {mean_copy_number, median_copy_number,
	   mode_copy_number}.
	5. Drop loci below a minimum total call count.
	6. Drop loci missing calls in any population, to ensure pooled labels
	   reflect genuinely pan-human diversity without pop-specific
	   missingness confounds.
	7. Verify perfect repeat in reference genome; drop imperfect.
	8. Check sufficient flanking sequence; drop out-of-bounds.
	9. Filter STRs overlapping mobile elements, segmental duplications,
	   and ENCODE blacklist regions.
	10. Assign chromosome-based train/val(chr13)/test(chr14) splits.
	11. Duplicate each locus with a reverse complement entry in the same
	    split.

Output columns:
	- ID: EnsembleTR locus identifier (chr:start-end)
	- chrom, str_start, str_end (0-based half-open), rev_comp, split
	- num_called_total: pooled sample count across all populations
	- heterozygosity: variability label
	- {mean_copy_number, median_copy_number, mode_copy_number}: copy number
	labels (subset per args)

CLI arguments:
	-s, --str-len: Motif length (e.g., 2 for dinucleotide). Required.
	-f, --n-flanking: Minimum flanking bases required on each side of
		the STR. Required.
	--copy-number-stats: Comma-separated subset of {mean,median,mode}
		copy number statistics to include as labels. At least one
		required. Default: "median".
	--min-num-called: Minimum total samples called (summed across
		populations) for a locus to be kept. Guards against noisy labels
		from low-N. Default 2000.
	--repeat-info: Path to EnsembleTR repeat_info TSV.
		Default: "../STR_data/EnsembleTR/repeat_info.tsv".
	--afreq-het: Path to EnsembleTR afreq_het TSV.
		Default: "../STR_data/EnsembleTR/afreq_het.tsv".
	--ref-genome: Path to reference genome FASTA.
		Default: "../STR_data/reference_genome/
		GRCh38_full_analysis_set_plus_decoy_hla.fa".
	--mobile-elements-bed: Path to BED file with mobile element regions.
		Default: "../invalid_regions/mobile_elements.bed.gz".
	--seg-dups-bed: Path to BED file with segmental duplication regions.
		Default: "../invalid_regions/seg_dups.bed.gz".
	--blacklist-bed: Path to BED file with ENCODE blacklist regions.
		Default: "../invalid_regions/blacklist.bed.gz".
	--output-dir: Directory to save filtered STR file.
		Default: "../STR_data/ensembletr_labeled_strs/".

Example usage:
	python create_labeled_and_filtered_strs.py \\
		--str-len 2 \\
		--n-flanking 10000 \\
		--copy-number-stats median
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tqdm

from pyfaidx import Fasta


POPULATIONS = ['AFR', 'AMR', 'EAS', 'SAS', 'EUR', 'H3Africa']


# ======================================================================
# CLI
# ======================================================================

def get_args():
	parser = argparse.ArgumentParser(
		description="Create labeled and filtered STR dataset from EnsembleTR."
	)

	parser.add_argument(
		"-s", "--str-len",
		type=int, required=True,
		help="Motif length (e.g., 2 for dinucleotide)."
	)
	parser.add_argument(
		"-f", "--n-flanking",
		type=int, required=True,
		help="Minimum flanking bases required on each side of the STR."
	)
	parser.add_argument(
		"--copy-number-stats",
		type=str, default="median",
		help=(
			"Comma-separated subset of {mean,median,mode} copy number "
			"statistics to include as labels. At least one required."
		)
	)
	parser.add_argument(
		"--min-num-called",
		type=int, default=2000,
		help=(
			"Minimum total samples called (summed across populations) for "
			"a locus to be kept. Guards against noisy labels from low-N. "
			"Default 2000."
		)
	)

	parser.add_argument(
		"--repeat-info",
		type=str,
		default="../STR_data/EnsembleTR/repeat_info.tsv",
	)
	parser.add_argument(
		"--afreq-het",
		type=str,
		default="../STR_data/EnsembleTR/afreq_het.tsv",
	)
	parser.add_argument(
		"--ref-genome",
		type=str,
		default="../STR_data/reference_genome/"
		        "GRCh38_full_analysis_set_plus_decoy_hla.fa",
	)

	parser.add_argument(
		"--mobile-elements-bed",
		type=str, default="../invalid_regions/mobile_elements.bed.gz",
	)
	parser.add_argument(
		"--seg-dups-bed",
		type=str, default="../invalid_regions/seg_dups.bed.gz",
	)
	parser.add_argument(
		"--blacklist-bed",
		type=str, default="../invalid_regions/blacklist.bed.gz",
	)

	parser.add_argument(
		"--output-dir",
		type=str, default="../STR_data/ensembletr_labeled_strs/",
	)

	args = parser.parse_args()

	requested = [s.strip() for s in args.copy_number_stats.split(",") if s.strip()]
	allowed = {"mean", "median", "mode"}
	if not requested:
		parser.error(
			"--copy-number-stats must include at least one of mean/median/mode."
		)
	bad = [s for s in requested if s not in allowed]
	if bad:
		parser.error(
			f"Invalid --copy-number-stats entries: {bad}. Allowed: {allowed}."
		)
	args.copy_number_stats = requested

	return args


# ======================================================================
# Label computation
# ======================================================================

def parse_afreq_dict(s):
	"""Return dict of {allele_float: freq_float} or None if empty/missing."""
	if pd.isna(s) or s == "" or s == "{}":
		return None
	d = json.loads(s)
	if len(d) == 0:
		return None
	return {float(k): float(v) for k, v in d.items()}


def pool_afreq(row, populations):
	"""Pool across populations weighted by numcalled.

	Equivalent to pooling all samples and then computing frequencies.
	Returns (alleles, freqs, total_called). If no pop has calls, returns
	(None, None, 0).
	"""
	pooled = {}
	total_called = 0

	for pop in populations:
		n = row[f'numcalled_{pop}']
		if pd.isna(n) or n == 0:
			continue
		freqs_dict = parse_afreq_dict(row[f'afreq_{pop}'])
		if freqs_dict is None:
			continue
		for allele, freq in freqs_dict.items():
			pooled[allele] = pooled.get(allele, 0.0) + freq * n
		total_called += int(n)

	if total_called == 0 or not pooled:
		return None, None, 0

	alleles = np.array(list(pooled.keys()), dtype=float)
	freqs = np.array(list(pooled.values()), dtype=float) / total_called
	return alleles, freqs, total_called


def heterozygosity(freqs):
	if freqs is None:
		return np.nan
	return float(1.0 - np.sum(freqs ** 2))


def mean_copy_number(alleles, freqs):
	if alleles is None:
		return np.nan
	return float(np.sum(alleles * freqs))


def median_copy_number(alleles, freqs):
	if alleles is None:
		return np.nan
	order = np.argsort(alleles)
	sorted_alleles = alleles[order]
	sorted_freqs = freqs[order]
	cum = np.cumsum(sorted_freqs)
	idx = min(int(np.searchsorted(cum, 0.5)), len(sorted_alleles) - 1)
	return float(sorted_alleles[idx])


def mode_copy_number(alleles, freqs):
	if alleles is None:
		return np.nan
	max_freq = freqs.max()
	tied = alleles[freqs == max_freq]
	return float(tied.min())


# ======================================================================
# Invalid region filtering
# ======================================================================

class GenomicIntersector:
	"""Efficient overlap checks between genomic intervals."""

	def __init__(self, region_df, name="region"):
		self.name = name
		self.chrom_intervals = {}

		for chrom, group in region_df.groupby("chrom"):
			starts = group["start"].values
			ends = group["end"].values

			sort_idx = np.argsort(starts)
			starts = starts[sort_idx]
			ends = ends[sort_idx]

			if len(starts) == 0:
				continue

			merged_starts = []
			merged_ends = []
			curr_start, curr_end = starts[0], ends[0]
			for i in range(1, len(starts)):
				if starts[i] < curr_end:
					curr_end = max(curr_end, ends[i])
				else:
					merged_starts.append(curr_start)
					merged_ends.append(curr_end)
					curr_start, curr_end = starts[i], ends[i]
			merged_starts.append(curr_start)
			merged_ends.append(curr_end)

			self.chrom_intervals[chrom] = (
				np.array(merged_starts), np.array(merged_ends)
			)

	def check_overlaps(self, query_df):
		result = np.zeros(len(query_df), dtype=bool)
		for chrom, group in query_df.groupby("chrom"):
			if chrom not in self.chrom_intervals:
				continue
			ref_starts, ref_ends = self.chrom_intervals[chrom]
			q_starts = group["str_start"].values
			q_ends = group["str_end"].values

			idx = np.searchsorted(ref_ends, q_starts, side='right')
			valid_mask = idx < len(ref_starts)
			overlap_mask = np.zeros(len(q_starts), dtype=bool)
			overlap_mask[valid_mask] = (
				ref_starts[idx[valid_mask]] < q_ends[valid_mask]
			)
			result[group.index] = overlap_mask
		return result


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

	args = get_args()

	print(f"Settings:")
	print(f"  STR motif length:   {args.str_len}")
	print(f"  Flanking required:  {args.n_flanking}")
	print(f"  Copy number stats:  {args.copy_number_stats}")
	print(f"  Min num_called:     {args.min_num_called}")

	# ------------------------------------------------------------------
	# Load and merge EnsembleTR tables
	# ------------------------------------------------------------------
	print("\nLoading EnsembleTR tables...")
	info_df = pd.read_csv(args.repeat_info, sep="\t")
	print(f"  repeat_info: {len(info_df)} loci")
	calls_df = pd.read_csv(args.afreq_het, sep="\t")
	print(f"  afreq_het:   {len(calls_df)} loci")

	call_cols = ['ID']
	for pop in POPULATIONS:
		call_cols += [f'afreq_{pop}', f'het_{pop}', f'numcalled_{pop}']

	str_df = info_df.merge(calls_df[call_cols], on='ID', how='inner')
	print(f"  merged:      {len(str_df)} loci")

	# ------------------------------------------------------------------
	# Filter to requested motif length
	# ------------------------------------------------------------------
	str_df['motif_len'] = str_df['Motif'].str.len()
	str_df = str_df[str_df['motif_len'] == args.str_len].reset_index(drop=True)
	print(f"\nAfter motif length == {args.str_len} filter: {len(str_df)}")
	if str_df.empty:
		print("No loci remaining. Exiting.")
		exit()

	# ------------------------------------------------------------------
	# Require calls in all populations
	# ------------------------------------------------------------------
	before = len(str_df)
	all_pops_called = pd.Series(True, index=str_df.index)
	for pop in POPULATIONS:
		all_pops_called &= (str_df[f'numcalled_{pop}'] > 0)
	str_df = str_df[all_pops_called].reset_index(drop=True)
	print(f"\nAfter requiring calls in all {len(POPULATIONS)} pops: "
	      f"{len(str_df)} (dropped {before - len(str_df)})")
	if str_df.empty:
		print("No loci remaining. Exiting.")
		exit()

	# ------------------------------------------------------------------
	# Compute pooled labels
	# ------------------------------------------------------------------
	print("\nComputing pooled labels...")
	pooled = [pool_afreq(row, POPULATIONS) for _, row in tqdm.tqdm(
		str_df.iterrows(), total=len(str_df)
	)]
	alleles_list = [p[0] for p in pooled]
	freqs_list = [p[1] for p in pooled]
	total_called = [p[2] for p in pooled]

	str_df['num_called_total'] = total_called
	str_df['heterozygosity'] = [heterozygosity(f) for f in freqs_list]
	if 'mean' in args.copy_number_stats:
		str_df['mean_copy_number'] = [
			mean_copy_number(a, f) for a, f in zip(alleles_list, freqs_list)
		]
	if 'median' in args.copy_number_stats:
		str_df['median_copy_number'] = [
			median_copy_number(a, f) for a, f in zip(alleles_list, freqs_list)
		]
	if 'mode' in args.copy_number_stats:
		str_df['mode_copy_number'] = [
			mode_copy_number(a, f) for a, f in zip(alleles_list, freqs_list)
		]

	# ------------------------------------------------------------------
	# Minimum total samples called
	# ------------------------------------------------------------------
	print(f"\nnum_called_total distribution:")
	print(f"  min:    {int(str_df['num_called_total'].min())}")
	print(f"  median: {int(str_df['num_called_total'].median())}")
	print(f"  max:    {int(str_df['num_called_total'].max())}")
	
	before = len(str_df)
	str_df = str_df[
		str_df['num_called_total'] >= args.min_num_called
	].reset_index(drop=True)
	print(f"\nAfter min_num_called >= {args.min_num_called}: "
	      f"{len(str_df)} (dropped {before - len(str_df)})")
	if str_df.empty:
		print("No loci remaining. Exiting.")
		exit()

# ------------------------------------------------------------------
	# Verify perfect repeat in reference genome + flanking bounds
	# ------------------------------------------------------------------
	print("\nChecking perfect repeat and flanking bounds in reference...")
	ref_genome = Fasta(args.ref_genome, sequence_always_upper=True)

	kept_rows = []
	internal_imperfect_count = 0
	boundary_truncation_count = 0
	out_of_bounds_count = 0
	first_check_done = False

	CONTEXT = 15  # bp of flanking context to show for diagnostics
	MAX_DIAG_PRINTS = 10

	for _, row in tqdm.tqdm(str_df.iterrows(), total=len(str_df)):
		chrom = row["Chrom"]
		start_idx = int(row["Start"]) - 1
		end_idx = int(row["End"])

		str_seq = ref_genome[chrom][start_idx:end_idx]
		str_region_len = end_idx - start_idx

		motif = row["Motif"]
		k = len(motif)

		# Derive base motif from the reference sequence itself, which
		# automatically captures whatever rotation/strand the locus is in.
		base_motif = str_seq[:k].seq
		expected = (base_motif * (str_region_len // k + 1))[:str_region_len]

		chrom_len = len(ref_genome[chrom])
		pre_start = max(0, start_idx - CONTEXT)
		post_end = min(chrom_len, end_idx + CONTEXT)

		# --- Check 1: internal perfection ---
		if str_seq.seq != expected:
			internal_imperfect_count += 1
			if internal_imperfect_count <= MAX_DIAG_PRINTS:
				pre_ctx = ref_genome[chrom][pre_start:start_idx].seq
				post_ctx = ref_genome[chrom][end_idx:post_end].seq
				print(f"\n  INTERNAL IMPERFECT example "
				      f"{internal_imperfect_count}: {row['ID']}")
				print(f"    Motif col: {motif}")
				print(f"    base_motif from ref (first {k}): {base_motif}")
				print(f"    coords: {start_idx}-{end_idx}  "
				      f"(len {str_region_len})")
				print(f"    median copy number: {row['median_copy_number']:.2f}"
				      if 'median_copy_number' in row.index
				      else f"    median copy number: (not computed)")
				print(f"    heterozygosity:     {row['heterozygosity']:.3f}")
				print(f"    pre-flank:  ...{pre_ctx}")
				print(f"    str_seq:       {str_seq.seq}")
				print(f"    expected:      {expected}")
				print(f"    post-flank: {post_ctx}...")
			continue

		# --- Check 2: boundaries don't truncate a longer tract ---
		# The bases immediately before and after the STR should NOT
		# continue the repeat pattern; if they do, the annotated boundary
		# is cutting off part of a longer perfect tract.
		boundary_extends = False
		extend_side = None
		extend_base = None
		extend_expected = None

		if end_idx < chrom_len:
			next_base = ref_genome[chrom][end_idx:end_idx + 1].seq
			expected_next = base_motif[str_region_len % k]
			if next_base == expected_next:
				boundary_extends = True
				extend_side = "right"
				extend_base = next_base
				extend_expected = expected_next

		if not boundary_extends and start_idx > 0:
			prev_base = ref_genome[chrom][start_idx - 1:start_idx].seq
			expected_prev = base_motif[-1]
			if prev_base == expected_prev:
				boundary_extends = True
				extend_side = "left"
				extend_base = prev_base
				extend_expected = expected_prev

		if boundary_extends:
			boundary_truncation_count += 1
			if boundary_truncation_count <= MAX_DIAG_PRINTS:
				pre_ctx = ref_genome[chrom][pre_start:start_idx].seq
				post_ctx = ref_genome[chrom][end_idx:post_end].seq
				print(f"\n  BOUNDARY TRUNCATION example "
				      f"{boundary_truncation_count}: {row['ID']}  "
				      f"({extend_side})")
				print(f"    Motif col: {motif}")
				print(f"    base_motif from ref (first {k}): {base_motif}")
				print(f"    coords: {start_idx}-{end_idx}  "
				      f"(len {str_region_len})")
				print(f"    repeat continues {extend_side}ward: "
				      f"base {extend_base!r} matches expected "
				      f"{extend_expected!r}")
				print(f"    median copy number: {row['median_copy_number']:.2f}"
				      if 'median_copy_number' in row.index
				      else f"    median copy number: (not computed)")
				print(f"    heterozygosity:     {row['heterozygosity']:.3f}")
				print(f"    pre-flank:  ...{pre_ctx}")
				print(f"    str_seq:       {str_seq.seq}")
				print(f"    post-flank: {post_ctx}...")
			continue

		# --- Check 3: flanking sequence exists ---
		if start_idx - args.n_flanking < 0 or end_idx + args.n_flanking > chrom_len:
			out_of_bounds_count += 1
			continue

		# --- One-time integrity check ---
		if not first_check_done:
			def revcomp(s):
				return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
			valid_representations = set()
			for m in (motif, revcomp(motif)):
				for i in range(k):
					valid_representations.add(m[i:] + m[:i])
			if base_motif not in valid_representations:
				raise ValueError(
					f"Base motif {base_motif!r} from reference at {row['ID']} "
					f"is not a rotation/RC of Motif column {motif!r}."
				)

			pre = ref_genome[chrom][start_idx - args.n_flanking:start_idx]
			post = ref_genome[chrom][end_idx:end_idx + args.n_flanking]
			gt = ref_genome[chrom][
				start_idx - args.n_flanking:end_idx + args.n_flanking
			]
			reconstructed = pre.seq + str_seq.seq + post.seq
			if reconstructed != gt.seq:
				raise ValueError(
					f"Sequence reconstruction failed at {row['ID']}."
				)
			print(f"  integrity check passed on {row['ID']} "
			      f"(base motif from ref: {base_motif}, Motif col: {motif})")
			first_check_done = True

		kept_rows.append({
			"ID": row["ID"],
			"chrom": chrom,
			"str_start": start_idx,
			"str_end": end_idx,
			"motif": motif,
			"num_called_total": row["num_called_total"],
			"heterozygosity": row["heterozygosity"],
			**{
				f"{stat}_copy_number": row[f"{stat}_copy_number"]
				for stat in args.copy_number_stats
			},
		})

	print(f"\n  internal imperfect:  {internal_imperfect_count}")
	print(f"  boundary truncation: {boundary_truncation_count}")
	print(f"  out-of-bounds:       {out_of_bounds_count}")
	print(f"  kept:                {len(kept_rows)}")

	if not kept_rows:
		print("No STRs remaining. Exiting.")
		exit()

	str_df = pd.DataFrame(kept_rows)

	# ------------------------------------------------------------------
	# Filter invalid regions
	# ------------------------------------------------------------------
	print("\nLoading invalid region BEDs...")
	mobile_df = pd.read_csv(
		args.mobile_elements_bed, sep="\t", header=None,
		names=["chrom", "start", "end", "class"], compression="gzip",
	)
	seg_df = pd.read_csv(
		args.seg_dups_bed, sep="\t", header=None,
		names=["chrom", "start", "end"], compression="gzip",
	)
	black_df = pd.read_csv(
		args.blacklist_bed, sep="\t", header=None,
		names=["chrom", "start", "end"], compression="gzip",
	)

	print("Building interval trees...")
	black_searcher = GenomicIntersector(black_df, "Blacklist")
	seg_searcher = GenomicIntersector(seg_df, "SegDups")
	mobile_searchers = {
		m_class: GenomicIntersector(
			mobile_df[mobile_df["class"] == m_class][["chrom", "start", "end"]],
			m_class
		)
		for m_class in mobile_df["class"].unique()
	}

	print("Checking overlaps...")
	str_df["overlap_blacklist"] = black_searcher.check_overlaps(str_df)
	str_df["overlap_segdup"] = seg_searcher.check_overlaps(str_df)
	for m_class, searcher in mobile_searchers.items():
		str_df[f"overlap_{m_class}"] = searcher.check_overlaps(str_df)

	total = len(str_df)
	label_w = 25
	count_w = 10

	def print_stat(label, mask):
		count = mask.sum()
		pct = mask.mean()
		text = f"{label}:"
		print(f"{text:<{label_w}} {count:>{count_w}} ({pct:>6.1%})")

	print(f"\nTotal STRs: {total}")
	print("-" * 45)
	print_stat("Overlapping Blacklist", str_df["overlap_blacklist"])
	print_stat("Overlapping SegDups", str_df["overlap_segdup"])
	for m_class in sorted(mobile_searchers.keys()):
		print_stat(f"Overlapping {m_class}", str_df[f"overlap_{m_class}"])

	filter_cols = (
		["overlap_blacklist", "overlap_segdup"]
		+ [f"overlap_{c}" for c in mobile_searchers.keys()]
	)
	str_df["is_excluded"] = str_df[filter_cols].any(axis=1)
	n_excluded = str_df["is_excluded"].sum()
	n_kept = total - n_excluded

	print("-" * 45)
	print(f"{'Total Excluded:':<{label_w}} {n_excluded:>{count_w}} "
	      f"({n_excluded/total:>6.1%})")
	print(f"{'Total Kept:':<{label_w}} {n_kept:>{count_w}} "
	      f"({n_kept/total:>6.1%})")

	str_df = str_df[~str_df["is_excluded"]].drop(
		columns=filter_cols + ["is_excluded"]
	).reset_index(drop=True)

	# ------------------------------------------------------------------
	# Chromosome-based splits
	# ------------------------------------------------------------------
	chrom_to_split = {
		"chr14": "test",
		"chr13": "val",
	}
	str_df["split"] = str_df["chrom"].map(
		lambda c: chrom_to_split.get(c, "train")
	)

	print(f"\nSplit counts:")
	print(str_df["split"].value_counts())
	print(f"\nSplit fractions:")
	print(str_df["split"].value_counts() / len(str_df))

	# ------------------------------------------------------------------
	# Add reverse-complement entries
	# ------------------------------------------------------------------
	str_df["rev_comp"] = False
	rc_df = str_df.copy()
	rc_df["rev_comp"] = True
	final_df = pd.concat([str_df, rc_df], ignore_index=True)

	# ------------------------------------------------------------------
	# 10. Save
	# ------------------------------------------------------------------
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	stats_str = "-".join(args.copy_number_stats)
	out_name = (
		f"str_len_{args.str_len}"
		f"_n_flanking_{args.n_flanking}"
		f"_stats_{stats_str}.tsv"
	)
	out_path = os.path.join(args.output_dir, out_name)
	final_df.to_csv(out_path, sep="\t", index=False)

	print(f"\nSaved {len(final_df)} rows to {out_path}")