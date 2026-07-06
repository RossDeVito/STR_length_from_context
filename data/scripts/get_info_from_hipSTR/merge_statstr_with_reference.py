"""Join combined statSTR output with the HipSTR reference BED.

Inputs:
  --statstr-file PATH  Combined statSTR .tab file (header line + data rows).
                       First three columns must be chrom / start / end.
  --ref-bed PATH       HipSTR reference BED, no header, columns:
                         chrom  start  end  motif_len  ref_copy_number
                         hipstr_name  motif
  --out-file PATH      Output TSV (with header).

Join key: (chrom, start). Expected to be 1-to-1. The script asserts
1-to-1-ness on both sides and reports unmatched rows.
"""

import argparse
import sys

import pandas as pd


REF_COLS = [
	"chrom", "start", "end",
	"motif_len", "ref_copy_number",
	"hipstr_name", "motif",
]
KEY = ["chrom", "start"]


def parse_args():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--statstr-file", required=True)
	p.add_argument("--ref-bed", required=True)
	p.add_argument("--out-file", required=True)
	return p.parse_args()


def main():
	args = parse_args()

	# --- Load ---
	print(f"Loading statSTR: {args.statstr_file}")
	stats = pd.read_csv(args.statstr_file, sep="\t")
	for c in KEY:
		if c not in stats.columns:
			sys.exit(f"ERROR: statSTR file missing column {c!r}. "
			         f"Got: {list(stats.columns)}")
	print(f"  rows: {len(stats):,}")

	print(f"Loading reference BED: {args.ref_bed}")
	ref = pd.read_csv(
		args.ref_bed,
		sep="\t",
		header=None,
		names=REF_COLS,
		dtype={"chrom": str},
	)
	print(f"  rows: {len(ref):,}")

	# --- Restrict to autosomes ---
	# statSTR was run on chr1..chr22 only. The HipSTR reference also contains
	# chrX/chrY (and some loci on chrY are duplicated on the (chrom,start,end)
	# key), so drop non-autosomes from both sides for a clean 1-to-1 join.
	autosomes = {f"chr{i}" for i in range(1, 23)}
	n_stats_before, n_ref_before = len(stats), len(ref)
	stats = stats[stats["chrom"].isin(autosomes)].reset_index(drop=True)
	ref = ref[ref["chrom"].isin(autosomes)].reset_index(drop=True)
	print(f"After autosome filter:")
	print(f"  statSTR rows: {n_stats_before:,} -> {len(stats):,}")
	print(f"  reference rows: {n_ref_before:,} -> {len(ref):,}")

	# --- Sanity: no duplicate keys on either side (uses original chrom,start) ---
	for name, df in [("statSTR file", stats), ("reference BED", ref)]:
		dup_mask = df.duplicated(subset=KEY, keep=False)
		n_dup = int(dup_mask.sum())
		if n_dup:
			print(f"\nERROR: {n_dup} rows with duplicate {tuple(KEY)} "
			      f"in {name}:", file=sys.stderr)
			dup_rows = df.loc[dup_mask].sort_values(KEY)
			print(dup_rows.to_string(index=False), file=sys.stderr)
			sys.exit(1)

	# --- Rename `end` on each side so both survive the merge ---
	# Both files are 0-based half-open. `start` matches exactly between
	# them, but `end` does not: HipSTR data-adaptively extends the VCF REF
	# allele to encompass observed polymorphic flanking bases, so the
	# call-side `end` is typically a few bp larger than the reference STR
	# `end`, by a locus-dependent amount.
	# Refs:
	#   Ziaei Jam et al. 2023 (EnsembleTR paper), Nat Commun, Methods:
	#     "HipSTR in some cases adjusts the coordinates of an STR region
	#      to encompass polymorphic flanking regions around the repeat."
	#   TRTools mergeSTR docs note that mergeSTR strips HipSTR flanking
	#   basepairs from alleles before merging for this exact reason.
	# Not a build issue (both sides are hg38) -- coordinates would be
	# shifted by orders of magnitude more if it were.
	stats = stats.rename(columns={"end": "end_call"})
	ref = ref.rename(columns={"end": "end_ref"})

	# --- Diagnostic: show what the keys look like on each side ---
	print("\nstatSTR head:")
	print(stats[["chrom", "start", "end_call"]].head(3).to_string(index=False))
	print("reference head:")
	print(ref[["chrom", "start", "end_ref"]].head(3).to_string(index=False))

	# --- Inner merge on (chrom, start) ---
	merged = stats.merge(ref, on=KEY, how="inner", validate="one_to_one")
	print(f"\nMerged rows: {len(merged):,}")

	# --- Coverage diagnostics ---
	n_stats_unmatched = len(stats) - len(merged)
	n_ref_unmatched = len(ref) - len(merged)
	print(f"statSTR rows with NO ref match : {n_stats_unmatched:,}")
	print(f"Reference rows with NO statSTR : {n_ref_unmatched:,}")

	if n_stats_unmatched > 0:
		# This is unexpected -- statSTR was run on the same VCFs that came
		# from the HipSTR reference, so every called locus should be in ref.
		print("WARNING: some statSTR rows did not match the reference on "
		      "(chrom, start); this should not happen for HipSTR calls.")

		# Dump unmatched statSTR rows to a sibling file for troubleshooting.
		stat_idx = pd.MultiIndex.from_frame(stats[["chrom", "start"]])
		matched_idx = pd.MultiIndex.from_frame(merged[["chrom", "start"]])
		unmatched = stats[~stat_idx.isin(matched_idx)].reset_index(drop=True)

		out_path = args.out_file
		stem, dot, ext = out_path.rpartition(".")
		unmatched_path = (
			f"{stem}.unmatched_statstr.{ext}" if dot
			else f"{out_path}.unmatched_statstr"
		)
		unmatched.to_csv(unmatched_path, sep="\t", index=False)
		print(f"Wrote {len(unmatched):,} unmatched statSTR rows to: {unmatched_path}")
	else:
		print("OK: all statSTR rows matched a reference locus on (chrom, start).")

	# --- Column order ---
	stat_cols = [c for c in stats.columns
	             if c not in {"chrom", "start", "end_call"}]
	out_cols = (
		["chrom", "start", "end_ref", "end_call",
		 "hipstr_name", "motif", "motif_len", "ref_copy_number"]
		+ stat_cols
	)
	merged = merged[out_cols]

	# Strip statSTR's "-1" group suffix from stat column names
	# (e.g. "het-1" -> "het"). Only affects columns that actually end in "-1".
	rename_map = {c: c[:-2] for c in stat_cols if c.endswith("-1")}
	if rename_map:
		merged = merged.rename(columns=rename_map)
		print(f"Renamed columns: {rename_map}")

	merged.to_csv(args.out_file, sep="\t", index=False)
	print(f"\nWrote: {args.out_file}")


if __name__ == "__main__":
	main()