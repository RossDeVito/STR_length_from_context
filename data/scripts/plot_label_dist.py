"""Plot mode copy number and heterozygosity distributions for filtered
labeled STR datasets across a configurable set of motif lengths.

Loads one TSV file per motif length (produced by create_STR_data_files.py;
HipSTR/statSTR labeled STRs) and generates overlaid/faceted histogram plots
comparing the motif lengths.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# ======================================================================
# Config (edit these)
# ======================================================================

# Motif lengths to include in the plots.
MOTIF_LENGTHS = [1, 2, 4]

# Directory holding the per-length TSV files and the filename template
# used to build each path. {length} is replaced by the motif length.
STR_DATA_DIR = "../STR_data/HipSTR_labeled_STRs/"
FILENAME_TEMPLATE = "str_len_{length}_n_flanking_10000.tsv"


def path_for_length(length):
	return STR_DATA_DIR + FILENAME_TEMPLATE.format(length=length)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

	# ------------------------------------------------------------------
	# Load data
	# ------------------------------------------------------------------
	print("Loading data...")
	dfs = {}
	for length in MOTIF_LENGTHS:
		df = pd.read_csv(path_for_length(length), sep="\t")
		# Drop RC duplicates so each locus is counted once
		df = df[~df["rev_comp"]].reset_index(drop=True)
		df["motif_len"] = length
		dfs[length] = df
		print(f"  Motif length {length}: {len(df)} loci")

	labels = {length: f"Motif length {length}" for length in MOTIF_LENGTHS}

	combined = pd.concat(dfs.values(), ignore_index=True)
	combined["motif_len"] = combined["motif_len"].astype(str)
	# Keep a consistent hue ordering across plots
	hue_order = [str(length) for length in MOTIF_LENGTHS]

	# ------------------------------------------------------------------
	# Correlations: heterozygosity vs mode / reference copy number
	# ------------------------------------------------------------------
	print("\nCorrelations with heterozygosity:")
	for length in MOTIF_LENGTHS:
		df = dfs[length]
		print(f"  {labels[length]} (n = {len(df):,}):")
		for col, name in [
			("mode_copy_number", "mode copy number"),
			("ref_copy_number", "reference copy number"),
		]:
			pear_r, pear_p = pearsonr(df[col], df["heterozygosity"])
			spear_r, spear_p = spearmanr(df[col], df["heterozygosity"])
			print(f"    vs {name}:")
			print(f"      Pearson  r = {pear_r:+.3f}  (p = {pear_p:.2e})")
			print(f"      Spearman r = {spear_r:+.3f}  (p = {spear_p:.2e})")

	# Transformed targets as used for training: arcsin(sqrt(het)) vs
	# log(mode copy number + 1). Spearman is unchanged by these monotonic
	# transforms; Pearson can differ.
	print("\nCorrelations between arcsin(sqrt(het)) and log(mode copy number + 1):")
	for length in MOTIF_LENGTHS:
		df = dfs[length]
		het_t = np.arcsin(np.sqrt(df["heterozygosity"]))
		log_mode = np.log1p(df["mode_copy_number"])
		pear_r, pear_p = pearsonr(log_mode, het_t)
		spear_r, spear_p = spearmanr(log_mode, het_t)
		print(f"  {labels[length]} (n = {len(df):,}):")
		print(f"    Pearson  r = {pear_r:+.3f}  (p = {pear_p:.2e})")
		print(f"    Spearman r = {spear_r:+.3f}  (p = {spear_p:.2e})")

	print("\nCorrelations between mode and reference copy number:")
	for length in MOTIF_LENGTHS:
		df = dfs[length]
		pear_r, pear_p = pearsonr(df["mode_copy_number"], df["ref_copy_number"])
		spear_r, spear_p = spearmanr(df["mode_copy_number"], df["ref_copy_number"])
		print(f"  {labels[length]} (n = {len(df):,}):")
		print(f"    Pearson  r = {pear_r:+.3f}  (p = {pear_p:.2e})")
		print(f"    Spearman r = {spear_r:+.3f}  (p = {spear_p:.2e})")

	# ------------------------------------------------------------------
	# Heterozygosity distribution
	# ------------------------------------------------------------------
	print("\nPlotting heterozygosity...")
	fig, ax = plt.subplots(figsize=(8, 5))
	sns.histplot(
		data=combined,
		x="heterozygosity",
		hue="motif_len",
		hue_order=hue_order,
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
	# Heterozygosity distribution with arcsin(sqrt(x)) transform
	# (matches the target transform used for model training)
	# ------------------------------------------------------------------
	print("\nPlotting arcsin(sqrt(heterozygosity))...")
	fig, ax = plt.subplots(figsize=(8, 5))
	combined["het_transformed"] = np.arcsin(np.sqrt(combined["heterozygosity"]))
	sns.histplot(
		data=combined,
		x="het_transformed",
		hue="motif_len",
		hue_order=hue_order,
		stat="proportion",
		common_norm=False,
		element="step",
		fill=False,
		bins=60,
		ax=ax,
	)
	ax.set_xlabel("arcsin(sqrt(heterozygosity))")
	ax.set_ylabel("Proportion of loci")
	ax.set_title("Transformed heterozygosity by motif length")
	plt.tight_layout()
	plt.show()

	# ------------------------------------------------------------------
	# Mode copy number distribution
	# ------------------------------------------------------------------
	print("\nPlotting mode copy number...")
	n = len(MOTIF_LENGTHS)
	fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
	axes = axes[0]

	for ax, length in zip(axes, MOTIF_LENGTHS):
		df = dfs[length]
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
		ax.set_title(f"{labels[length]}  (n = {len(df):,})")

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
	# Mode copy number distribution, log(x + 1) transformed
	# (matches the target transform used for model training)
	# ------------------------------------------------------------------
	print("\nPlotting log(mode copy number + 1)...")
	fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
	axes = axes[0]

	for ax, length in zip(axes, MOTIF_LENGTHS):
		df = dfs[length]
		log_mode = np.log1p(df["mode_copy_number"])
		sns.histplot(
			log_mode,
			stat="proportion",
			element="bars",
			fill=True,
			bins=40,
			ax=ax,
			alpha=0.6,
		)
		ax.set_xlabel("log(mode copy number + 1)")
		ax.set_ylabel("Proportion of loci")
		ax.set_title(f"{labels[length]}  (n = {len(df):,})")

		q = log_mode.quantile([0.5, 0.95, 0.99])
		stats_text = (
			f"min:    {log_mode.min():.2f}\n"
			f"median: {q[0.5]:.2f}\n"
			f"95%ile: {q[0.95]:.2f}\n"
			f"99%ile: {q[0.99]:.2f}\n"
			f"max:    {log_mode.max():.2f}"
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
	fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
	axes = axes[0]

	for ax, length in zip(axes, MOTIF_LENGTHS):
		df = dfs[length]
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
		ax.set_title(f"{labels[length]}  (n = {len(df):,})")

	plt.tight_layout()
	plt.show()
