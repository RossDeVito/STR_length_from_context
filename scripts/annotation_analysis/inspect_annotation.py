""" Prelim sanity-check tool: open one annotation from annotation_config.yaml
(bed or bigwig) and print what fields exist plus example values, before
writing real per-locus extraction logic against it.

For a bed entry with a `join_path` (e.g. ChromHMM_state / ChromHMM_group),
prints the full merged field set (raw bed columns + joined metadata columns)
regardless of which single column that entry's `value_column` points to, so
you can see the whole picture behind either name.

Configuration is via the editable constants below (no argparse), mirroring
plot_labels_by_chromhmm.py.
"""

import sys

import annotation_utils as au


# ===========================================================================
# Configuration
# ===========================================================================

# ANNOTATION_TYPE = "phyloP"  # key into annotation_config.yaml
ANNOTATION_TYPE = "ChromHMM_state"

N_EXAMPLES = 10  # example rows / intervals / unique values to print per section
MAX_ROWS = 100_000  # cap on bed rows read for the peek (join tables read in full)
REGION = None  # bigwig only; "chrom:start-end" to sample from, or None to auto-pick


# ===========================================================================
# Describing a bed / joined DataFrame
# ===========================================================================

def describe_dataframe(df, n_examples, max_rows=None):
	print(f"Shape: {len(df):,} rows x {len(df.columns)} columns", end="")
	if max_rows is not None and len(df) >= max_rows:
		print(f"  (sampled first {max_rows:,} rows -- may be incomplete)")
	else:
		print()

	print(f"Columns: {', '.join(df.columns)}")
	for col in df.columns:
		n_unique = df[col].nunique(dropna=True)
		examples = df[col].dropna().unique()[:n_examples]
		print(f"  {col:20s} {str(df[col].dtype):10s} {n_unique:,} unique  e.g. {list(examples)}")

	print(f"\nExample rows:\n{df.head(n_examples).to_string(index=False)}")


# ===========================================================================
# Describing a bigwig
# ===========================================================================

def parse_region(region_str):
	chrom, coords = region_str.split(":")
	start, end = coords.split("-")
	return chrom, int(start), int(end)


def find_example_region(bw, window=2_000_000):
	"""bw.intervals() can legitimately return None/empty near chrom starts
	(telomere N-gaps have no stored bigWig intervals). Scan chroms in file
	order, growing the query window, until one yields data."""
	for chrom, chrom_len in bw.chroms().items():
		w = min(window, chrom_len)
		while w <= chrom_len:
			intervals = bw.intervals(chrom, 0, w)
			if intervals:
				return chrom, 0, w, intervals
			if w == chrom_len:
				break
			w = min(w * 4, chrom_len)
	return None


def describe_bigwig(bw, n_examples, region):
	chroms = bw.chroms()
	print(f"Chromosomes ({len(chroms)}):")
	for chrom, length in list(chroms.items())[:n_examples]:
		print(f"  {chrom}: {length:,} bp")
	if len(chroms) > n_examples:
		print(f"  ... and {len(chroms) - n_examples} more")

	header = bw.header()
	n_bases = header["nBasesCovered"]
	mean = header["sumData"] / n_bases if n_bases else float("nan")
	variance = header["sumSquared"] / n_bases - mean ** 2 if n_bases else float("nan")
	print(
		f"\nHeader stats:\n"
		f"  nBasesCovered: {n_bases:,}\n"
		f"  min: {header['minVal']:.4g}   max: {header['maxVal']:.4g}\n"
		f"  mean: {mean:.4g}   std: {variance ** 0.5:.4g}"
	)

	if region:
		chrom, start, end = parse_region(region)
		intervals = bw.intervals(chrom, start, end)
		if not intervals:
			print(f"\nNo intervals found in {region}")
			return
	else:
		found = find_example_region(bw)
		if found is None:
			print("\nNo example intervals found in the first few chromosomes; "
				"set REGION explicitly.")
			return
		chrom, start, end, intervals = found

	print(f"\nExample (start, end, value) intervals from {chrom}:{start}-{end}:")
	for interval in intervals[:n_examples]:
		print(f"  {interval}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

	try:
		entry = au.get_entry(ANNOTATION_TYPE)
	except KeyError as e:
		print(e.args[0])
		sys.exit(1)

	print("=" * 70)
	print(f"Annotation type: {ANNOTATION_TYPE}")
	print("=" * 70)
	print(f"File type:    {entry['file_type']}" + (" (+ join)" if "join_path" in entry else ""))
	print(f"Value type:   {entry['value_type']}")
	print(f"Genome build: {entry.get('genome_build', 'n/a')}")
	print(f"Description:  {entry['description'].strip()}")
	print(f"DOI:          {entry.get('doi', 'n/a')}")
	print(f"Path:         {au.resolve_path(entry['path'])}")

	if entry["file_type"] == "bigwig":
		print("\n" + "-" * 70)
		bw = au.load_bigwig(entry)
		try:
			describe_bigwig(bw, N_EXAMPLES, REGION)
		finally:
			bw.close()

	elif entry["file_type"] == "bed":
		print("\n" + "-" * 70)
		print("Fields (all fields, including any from a joined metadata file):")
		print("-" * 70)
		df = au.load_annotation(ANNOTATION_TYPE, max_rows=MAX_ROWS)
		describe_dataframe(df, N_EXAMPLES, max_rows=MAX_ROWS)

		value_col = entry["value_column"]
		print("\n" + "-" * 70)
		print(f"Value field for {ANNOTATION_TYPE!r}: '{value_col}'"
			+ (" (from joined file)" if "join_path" in entry else " (from primary file)"))
		print("-" * 70)
		n_unique = df[value_col].nunique(dropna=True)
		print(f"{n_unique:,} unique values, e.g. {list(df[value_col].dropna().unique()[:N_EXAMPLES])}")

	else:
		raise ValueError(f"Unknown file_type {entry['file_type']!r}")
