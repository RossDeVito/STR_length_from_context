"""Plot mode copy number and heterozygosity distributions for filtered
labeled STR datasets (motif lengths 1 and 2).

Loads two TSV files produced by create_labeled_and_filtered_strs.py and
generates overlayed KDE/histogram plots comparing the two motif lengths.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================================
# Paths (edit these)
# ======================================================================

STR_LEN_1_PATH = (
	"../STR_data/EnsembleTR_labeled_STRs/"
	"str_len_1_n_flanking_10000_stats_mean-median-mode.tsv"
)
STR_LEN_2_PATH = (
	"../STR_data/EnsembleTR_labeled_STRs/"
	"str_len_2_n_flanking_10000_stats_mean-median-mode.tsv"
)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

	# ------------------------------------------------------------------
	# Load data
	# ------------------------------------------------------------------
	print("Loading data...")
	df1 = pd.read_csv(STR_LEN_1_PATH, sep="\t")
	df2 = pd.read_csv(STR_LEN_2_PATH, sep="\t")

	# Drop RC duplicates so each locus is counted once
	df1 = df1[~df1["rev_comp"]].reset_index(drop=True)
	df2 = df2[~df2["rev_comp"]].reset_index(drop=True)

	df1["motif_len"] = 1
	df2["motif_len"] = 2
	combined = pd.concat([df1, df2], ignore_index=True)
	combined["motif_len"] = combined["motif_len"].astype(str)

	print(f"  Motif length 1: {len(df1)} loci")
	print(f"  Motif length 2: {len(df2)} loci")

	# ------------------------------------------------------------------
	# Heterozygosity distribution
	# ------------------------------------------------------------------
	print("\nPlotting heterozygosity...")
	fig, ax = plt.subplots(figsize=(8, 5))
	sns.histplot(
		data=combined,
		x="heterozygosity",
		hue="motif_len",
		stat="proportion",
		common_norm=False,
		element="step",
		fill=False,
		bins=60,
		ax=ax,
	)
	ax.set_xlabel("Heterozygosity (1 - Σp²)")
	ax.set_ylabel("Proportion of loci")
	ax.set_title("Heterozygosity by motif length")
	plt.tight_layout()
	plt.show()

	# ------------------------------------------------------------------
	# Mode copy number distribution
	# ------------------------------------------------------------------
	print("\nPlotting mode copy number...")
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))

	for ax, df, label in zip(axes, [df1, df2], ["Motif length 1", "Motif length 2"]):
		sns.histplot(
			df["mode_copy_number"],
			stat="proportion",
			element="bars",
			fill=True,
			bins=40,
			ax=ax,
			alpha=0.6,
		)
		ax.set_xlabel("Mode copy number")
		ax.set_ylabel("Proportion of loci")
		ax.set_title(f"{label}  (n = {len(df):,})")

		q = df["mode_copy_number"].quantile([0.5, 0.95, 0.99])
		stats_text = (
			f"min:    {df['mode_copy_number'].min():.1f}\n"
			f"median: {q[0.5]:.1f}\n"
			f"95%ile: {q[0.95]:.1f}\n"
			f"99%ile: {q[0.99]:.1f}\n"
			f"max:    {df['mode_copy_number'].max():.1f}"
		)
		ax.text(
			0.95, 0.95, stats_text,
			transform=ax.transAxes,
			ha="right", va="top",
			fontsize=9,
			family="monospace",
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
		)

	plt.tight_layout()
	plt.show()

	# ------------------------------------------------------------------
	# Joint view: heterozygosity vs mode copy number
	# ------------------------------------------------------------------
	print("\nPlotting joint heterozygosity vs mode copy number...")
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))

	for ax, df, label in zip(axes, [df1, df2], ["Motif length 1", "Motif length 2"]):
		ax.hexbin(
			df["mode_copy_number"],
			df["heterozygosity"],
			gridsize=40,
			bins="log",
			cmap="viridis",
			mincnt=1,
		)
		ax.set_xlabel("Mode copy number")
		ax.set_ylabel("Heterozygosity")
		ax.set_title(f"{label}  (n = {len(df):,})")

	plt.tight_layout()
	plt.show()

	# ------------------------------------------------------------------
	# Mode vs median copy number
	# ------------------------------------------------------------------
	print("\nPlotting mode vs median copy number...")
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))

	for ax, df, label in zip(axes, [df1, df2], ["Motif length 1", "Motif length 2"]):
		ax.hexbin(
			df["mode_copy_number"],
			df["median_copy_number"],
			gridsize=40,
			bins="log",
			cmap="viridis",
			mincnt=1,
		)
		# y = x reference line
		lo = min(df["mode_copy_number"].min(), df["median_copy_number"].min())
		hi = max(df["mode_copy_number"].max(), df["median_copy_number"].max())
		ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, alpha=0.7,
		        label="y = x")
		ax.set_aspect("equal", adjustable="box")
		ax.set_xlim(lo, hi)
		ax.set_ylim(lo, hi)
		ax.set_xlabel("Mode copy number")
		ax.set_ylabel("Median copy number")
		ax.set_title(f"{label}  (n = {len(df):,})")

		# Fraction that disagree
		disagree = (df["mode_copy_number"] != df["median_copy_number"]).mean()
		ax.text(
			0.05, 0.95,
			f"disagree: {disagree:.1%}",
			transform=ax.transAxes,
			ha="left", va="top",
			fontsize=9,
			family="monospace",
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
		)
		ax.legend(loc="lower right", fontsize=9)

	plt.tight_layout()
	plt.show()