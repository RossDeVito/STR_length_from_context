"""Join combined statSTR output with the HipSTR reference BED.

Inputs:
  --statstr-file PATH  Combined statSTR .tab file (header line + data rows).
                       First three columns must be chrom / start / end.
  --ref-bed PATH       HipSTR reference BED, no header, columns:
                         chrom  start  end  motif_len  ref_copy_number
                         hipstr_name  motif
  --out-file PATH      Output TSV (with header).

Join key: (chrom, start, end). Expected to be 1-to-1. The script asserts
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
KEY = ["chrom", "start", "end"]


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

	# --- Sanity: no duplicate keys on either side ---
	for name, df in [("statSTR file", stats), ("reference BED", ref)]:
		dup_mask = df.duplicated(subset=KEY, keep=False)
		n_dup = int(dup_mask.sum())
		if n_dup:
			print(f"\nERROR: {n_dup} rows with duplicate (chrom,start,end) "
			      f"in {name}:", file=sys.stderr)
			# Show all involved rows, sorted by the key for readability.
			dup_rows = df.loc[dup_mask].sort_values(KEY)
			print(dup_rows.to_string(index=False), file=sys.stderr)
			sys.exit(1)

	# --- Inner merge ---
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
		print("WARNING: some statSTR rows did not match the reference; "
		      "this should not happen for HipSTR calls.")

	# --- Column order: locate columns then statSTR stat columns ---
	stat_cols = [c for c in stats.columns if c not in KEY]
	out_cols = KEY + ["hipstr_name", "motif", "motif_len", "ref_copy_number"] + stat_cols
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