"""Create labeled, filtered STR datasets for ML models from HipSTR/statSTR data
with splits aligned to Caduceus/HyenaDNA/Basenji pretraining splits.

Population-level labels (heterozygosity, mode copy number, ...) are computed
by statSTR and merged with the HipSTR reference, so this script reads the
precomputed labels directly rather than computing them from allele
frequencies.

Input: the merged statSTR + HipSTR-reference table (1000G_HipSTR_stats.tsv)
with columns:
	chrom  start  end_ref  end_call  hipstr_name  motif  motif_len
	ref_copy_number  het  mode  numcalled

HipSTR reference / statSTR coordinates are 1-based fully-closed: the repeat
tract spans 1-based positions [start, end_ref] inclusive, so tract length =
end_ref - start + 1 = ref_copy_number * motif_len. We subtract 1 from start
for 0-based pyfaidx extraction; str_start/str_end in the OUTPUT are 0-based
half-open. We use `end_ref` (the canonical
STR boundary) -- NOT `end_call`, which includes HipSTR's data-adaptive
flanking extension and would break the perfect-repeat check.

Pipeline:
	1. Load merged statSTR/HipSTR table.
	2. Filter to specified motif length (motif_len column).
	3. Drop loci below a minimum call count (numcalled).
	4. Verify perfect repeat in reference genome; drop imperfect.
	5. Check boundaries don't truncate a longer tract; drop truncated.
	6. Check sufficient flanking sequence; drop out-of-bounds.
	7. Filter STRs overlapping mobile elements, segmental duplications,
	   and ENCODE blacklist regions.
	8. Assign chromosome-based train/val/test splits or using segment based
		pretraining splits (loci overlapping no pretraining interval default
		to test).
	9. Duplicate each locus with a reverse-complement entry in the same split.

Labels carried through:
	het       -> heterozygosity
	mode      -> mode_copy_number
	numcalled -> num_called_total
	ref_copy_number is always carried through as-is.

Output columns:
	- ID: HipSTR locus identifier (hipstr_name, e.g. Human_STR_3)
	- chrom, str_start, str_end (0-based half-open), rev_comp, split
	- motif, ref_copy_number
	- <carried label columns above>

CLI arguments:
	-s, --str-len: Motif length (e.g., 2 for dinucleotide). Required.
	-f, --n-flanking: Minimum flanking bases required on each side. Required.
	--min-num-called: Minimum samples called for a locus to be kept.
		Default 2000.
	--hipstr-stats: Path to merged statSTR/HipSTR-reference TSV.
		Default: "../STR_data/HipSTR_data/1000G_HipSTR_stats.tsv".
	--ref-genome: Path to reference genome FASTA.
		Default: "../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa".
	--mobile-elements-bed / --seg-dups-bed / --blacklist-bed: invalid-region BEDs.
	--output-dir: Directory to save filtered STR file.
		Default: "../STR_data/HipSTR_labeled_STRs/".

Example usage:
	python create_STR_data_files.py \\
		--str-len 2 \\
		--n-flanking 10000
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tqdm

from pyfaidx import Fasta


# Map statSTR / reference stat column names -> output label column names.
LABEL_RENAME = {
	"het": "heterozygosity",
	"mode": "mode_copy_number",
	"numcalled": "num_called_total",
}


# ======================================================================
# CLI
# ======================================================================

def get_args():
	parser = argparse.ArgumentParser(
		description="Create labeled and filtered STR dataset from HipSTR/statSTR."
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
		"--min-num-called",
		type=int, default=2000,
		help=(
			"Minimum samples called (numcalled) for a locus to be kept. "
			"Guards against noisy labels from low-N. Default 2000."
		)
	)

	parser.add_argument(
		"--hipstr-stats",
		type=str,
		default="../STR_data/HipSTR_data/1000G_HipSTR_stats.tsv",
	)
	parser.add_argument(
		"--ref-genome",
		type=str,
		default="../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa",
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
		type=str, default="../STR_data/HipSTR_labeled_STRs/",
	)

	parser.add_argument(
		"--split-mode",
		type=str, default="chromosome",
		choices=["chromosome", "pretraining"],
		help=(
			"How to assign train/val/test. 'chromosome': chr13=val, "
			"chr14=test, rest=train. 'pretraining': assign by which "
			"Caduceus/HyenaDNA pretraining-split interval each locus's "
			"flanking window overlaps (priority train > val > test; loci "
			"whose window overlaps no interval default to test)."
		)
	)
	parser.add_argument(
		"--pretraining-split-bed",
		type=str, default="../reference_genome/sequences_human.bed",
		help=(
			"BED of pretraining splits (chrom, start, end, split) with split "
			"in {train, valid, test}. Required for --split-mode pretraining."
		)
	)

	return parser.parse_args()


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

	def check_overlaps(self, query_df, start_col="str_start", end_col="str_end"):
		result = np.zeros(len(query_df), dtype=bool)
		for chrom, group in query_df.groupby("chrom"):
			if chrom not in self.chrom_intervals:
				continue
			ref_starts, ref_ends = self.chrom_intervals[chrom]
			q_starts = group[start_col].values
			q_ends = group[end_col].values

			idx = np.searchsorted(ref_ends, q_starts, side='right')
			valid_mask = idx < len(ref_starts)
			overlap_mask = np.zeros(len(q_starts), dtype=bool)
			overlap_mask[valid_mask] = (
				ref_starts[idx[valid_mask]] < q_ends[valid_mask]
			)
			result[group.index] = overlap_mask
		return result


# ======================================================================
# Helpers
# ======================================================================

def revcomp(s):
	return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def fmt_labels(row, label_cols):
	parts = []
	for lc in label_cols:
		val = row[lc]
		if isinstance(val, (float, np.floating)):
			parts.append(f"{lc}={val:.3f}")
		else:
			parts.append(f"{lc}={val}")
	return ", ".join(parts)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

	args = get_args()

	print(f"Settings:")
	print(f"  STR motif length:   {args.str_len}")
	print(f"  Flanking required:  {args.n_flanking}")
	print(f"  Min num_called:     {args.min_num_called}")

	# ------------------------------------------------------------------
	# Load merged statSTR / HipSTR-reference table
	# ------------------------------------------------------------------
	print("\nLoading HipSTR/statSTR table...")
	str_df = pd.read_csv(args.hipstr_stats, sep="\t", dtype={"chrom": str})
	print(f"  loaded: {len(str_df)} loci")

	required = {"chrom", "start", "end_ref", "motif", "motif_len",
	            "ref_copy_number", "numcalled"}
	missing = required - set(str_df.columns)
	if missing:
		sys.exit(f"ERROR: input missing required columns: {missing}\n"
		         f"Got: {list(str_df.columns)}")

	# Carry through whichever label columns are present, renamed to output names.
	present = {c: LABEL_RENAME[c] for c in LABEL_RENAME if c in str_df.columns}
	str_df = str_df.rename(columns=present)
	label_cols = list(present.values())
	print(f"  label columns carried: {label_cols}")
	if "num_called_total" not in label_cols:
		sys.exit("ERROR: 'numcalled' column required for the min-call filter.")

	# ------------------------------------------------------------------
	# Filter to requested motif length
	# ------------------------------------------------------------------
	str_df = str_df[str_df["motif_len"] == args.str_len].reset_index(drop=True)
	print(f"\nAfter motif_len == {args.str_len} filter: {len(str_df)}")
	if str_df.empty:
		print("No loci remaining. Exiting.")
		sys.exit()

	# ------------------------------------------------------------------
	# Minimum samples called
	# ------------------------------------------------------------------
	print(f"\nnum_called_total distribution:")
	print(f"  min:    {int(str_df['num_called_total'].min())}")
	print(f"  median: {int(str_df['num_called_total'].median())}")
	print(f"  max:    {int(str_df['num_called_total'].max())}")

	before = len(str_df)
	str_df = str_df[
		str_df["num_called_total"] >= args.min_num_called
	].reset_index(drop=True)
	print(f"\nAfter num_called_total >= {args.min_num_called}: "
	      f"{len(str_df)} (dropped {before - len(str_df)})")
	if str_df.empty:
		print("No loci remaining. Exiting.")
		sys.exit()

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
		chrom = row["chrom"]
		start_idx = int(row["start"]) - 1  # HipSTR coords are 1-based; -1 -> 0-based
		end_idx = int(row["end_ref"])      # 1-based inclusive end == 0-based exclusive

		str_seq = ref_genome[chrom][start_idx:end_idx]
		str_region_len = end_idx - start_idx

		motif = row["motif"]
		k = int(row["motif_len"])          # key off motif_len, not len(motif)

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
				      f"{internal_imperfect_count}: {row['hipstr_name']}")
				print(f"    motif col: {motif}")
				print(f"    base_motif from ref (first {k}): {base_motif}")
				print(f"    coords: {start_idx}-{end_idx}  "
				      f"(len {str_region_len})")
				print(f"    labels: {fmt_labels(row, label_cols)}")
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
				      f"{boundary_truncation_count}: {row['hipstr_name']}  "
				      f"({extend_side})")
				print(f"    motif col: {motif}")
				print(f"    base_motif from ref (first {k}): {base_motif}")
				print(f"    coords: {start_idx}-{end_idx}  "
				      f"(len {str_region_len})")
				print(f"    repeat continues {extend_side}ward: "
				      f"base {extend_base!r} matches expected "
				      f"{extend_expected!r}")
				print(f"    labels: {fmt_labels(row, label_cols)}")
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
			valid_representations = set()
			for variant in str(motif).split("/"):  # handle multi-motif strings
				if len(variant) != k:
					continue
				for m in (variant, revcomp(variant)):
					for i in range(k):
						valid_representations.add(m[i:] + m[:i])
			if valid_representations and base_motif not in valid_representations:
				raise ValueError(
					f"Base motif {base_motif!r} from reference at "
					f"{row['hipstr_name']} is not a rotation/RC of motif "
					f"column {motif!r}."
				)

			pre = ref_genome[chrom][start_idx - args.n_flanking:start_idx]
			post = ref_genome[chrom][end_idx:end_idx + args.n_flanking]
			gt = ref_genome[chrom][
				start_idx - args.n_flanking:end_idx + args.n_flanking
			]
			reconstructed = pre.seq + str_seq.seq + post.seq
			if reconstructed != gt.seq:
				raise ValueError(
					f"Sequence reconstruction failed at {row['hipstr_name']}."
				)
			print(f"  integrity check passed on {row['hipstr_name']} "
			      f"(base motif from ref: {base_motif}, motif col: {motif})")
			first_check_done = True

		kept_rows.append({
			"ID": row["hipstr_name"],
			"chrom": chrom,
			"str_start": start_idx,
			"str_end": end_idx,
			"motif": motif,
			"ref_copy_number": row["ref_copy_number"],
			**{lc: row[lc] for lc in label_cols},
		})

	print(f"\n  internal imperfect:  {internal_imperfect_count}")
	print(f"  boundary truncation: {boundary_truncation_count}")
	print(f"  out-of-bounds:       {out_of_bounds_count}")
	print(f"  kept:                {len(kept_rows)}")

	if not kept_rows:
		print("No STRs remaining. Exiting.")
		sys.exit()

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
	# Splits
	# ------------------------------------------------------------------
	if args.split_mode == "chromosome":
		# Whole-chromosome: chr13=val, chr14=test, rest=train.
		chrom_to_split = {
			"chr14": "test",
			"chr13": "val",
		}
		str_df["split"] = str_df["chrom"].map(
			lambda c: chrom_to_split.get(c, "train")
		)
	else:
		# Pretraining-aligned: assign each locus by which pretraining-split
		# interval its FLANKING WINDOW (STR +/- n_flanking) overlaps, with
		# priority train > val > test. This guarantees a 'test' locus's
		# window never overlapped a pretraining train (or val) interval, i.e.
		# its flanks were never seen during Caduceus pretraining. Loci whose
		# window overlaps no interval at all are labeled 'none'.
		print("\nLoading pretraining split BED...")
		pre_df = pd.read_csv(
			args.pretraining_split_bed, sep="\t", header=None,
			names=["chrom", "start", "end", "split"],
			dtype={"chrom": str},
		)
		pre_df["split"] = pre_df["split"].replace({"valid": "val"})
		print("  pretraining-split interval counts:")
		print(pre_df["split"].value_counts().to_string())

		# Flanking window the model actually sees (in-bounds guaranteed by
		# the earlier flanking check).
		str_df["window_start"] = str_df["str_start"] - args.n_flanking
		str_df["window_end"] = str_df["str_end"] + args.n_flanking

		split_searchers = {
			s: GenomicIntersector(
				pre_df[pre_df["split"] == s][["chrom", "start", "end"]], s
			)
			for s in ["train", "val", "test"]
		}
		ov = {
			s: split_searchers[s].check_overlaps(
				str_df, start_col="window_start", end_col="window_end"
			)
			for s in ["train", "val", "test"]
		}

		# Priority train > val > test (assign least-strict first, then
		# overwrite with stricter). Loci whose window overlaps no
		# pretraining-split interval default to 'test' -- their flanks were
		# never seen during pretraining, so they are safe to evaluate on.
		split = np.full(len(str_df), "test", dtype=object)
		split[ov["test"]] = "test"
		split[ov["val"]] = "val"
		split[ov["train"]] = "train"
		str_df["split"] = split

		str_df = str_df.drop(columns=["window_start", "window_end"])

	print(f"\nSplit counts:")
	print(str_df["split"].value_counts().to_string())
	print(f"\nSplit fractions:")
	print((str_df["split"].value_counts() / len(str_df)).to_string())

	# ------------------------------------------------------------------
	# Add reverse-complement entries
	# ------------------------------------------------------------------
	str_df["rev_comp"] = False
	rc_df = str_df.copy()
	rc_df["rev_comp"] = True
	final_df = pd.concat([str_df, rc_df], ignore_index=True)

	# ------------------------------------------------------------------
	# Save
	# ------------------------------------------------------------------
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	out_name = (
		f"str_len_{args.str_len}"
		f"_n_flanking_{args.n_flanking}.tsv"
	)
	out_path = os.path.join(args.output_dir, out_name)
	final_df.to_csv(out_path, sep="\t", index=False)

	print(f"\nSaved {len(final_df)} rows to {out_path}")