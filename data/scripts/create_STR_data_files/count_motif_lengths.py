"""Print counts of all motif lengths in 1000G_HipSTR_stats.tsv."""

import os

import pandas as pd

# Path to stats TSV (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_TSV = os.path.join(
	SCRIPT_DIR, "..", "..", "STR_data", "HipSTR_data", "1000G_HipSTR_stats.tsv"
)


if __name__ == "__main__":
	df = pd.read_csv(STATS_TSV, sep="\t", usecols=["motif_len"])

	counts = df["motif_len"].value_counts().sort_index()

	print(f"Total STRs: {len(df)}\n")
	print(f"{'motif_len':>10}  {'count':>10}")
	for motif_len, count in counts.items():
		print(f"{motif_len:>10}  {count:>10}")
