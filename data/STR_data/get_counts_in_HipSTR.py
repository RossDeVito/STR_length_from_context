"""Get counts of STRs with different motif lengths in HipSTR reference."""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

	hipstr_ref_path = os.path.join(
		"HipSTR-reference",
		"hg38.hipstr_reference.bed"
	)

	str_df = pd.read_csv(
		hipstr_ref_path,
		sep="\t",
		header=None,
		names=[
			"chrom", "start", "end", "motif_len",
			"ref_repeats", "name", "motif"
		]
	)

	# Get counts of STRs by motif length
	print(str_df["motif_len"].value_counts().sort_index())

	# print summary statistics of repeat counts for STRs with motif
	# lengths 1 and 2. Then plot the distributions.
	for motif_len in [1, 2]:
		subset_df = str_df[str_df["motif_len"] == motif_len]
		print(
			f"Motif length {motif_len} - "
			f"Mean repeats: {subset_df['ref_repeats'].mean():.2f}, "
			f"Median repeats: {subset_df['ref_repeats'].median():.2f}, "
			f"Min repeats: {subset_df['ref_repeats'].min()}, "
			f"Max repeats: {subset_df['ref_repeats'].max()}, "
			f"STD repeats: {subset_df['ref_repeats'].std():.2f}"
		)

		plt.figure(figsize=(8, 6))
		sns.histplot(
			subset_df["ref_repeats"],
			bins=200,
			kde=True,
		)
		plt.title(f"Distribution of Reference Repeat Counts (Motif Length {motif_len})")
		plt.xlabel("Reference Repeat Counts")
		plt.ylabel("Number of STRs")
		plt.grid(axis="y")

		# Add vertical lines for quntiles
		quantiles = [.95, .975, .99, .999]
		for q in quantiles:
			q_value = subset_df["ref_repeats"].quantile(q)

			print(
				f"Motif length {motif_len} - "
				f"{q*100}th Percentile Repeat Count: {q_value}"
			)

			plt.axvline(
				x=q_value,
				color="red",
				linestyle="--",
				label=f"{q*100}th Percentile: {q_value:.2f}"
			)
			plt.text(
				x=q_value,
				y=plt.ylim()[1]*0.9,
				s=f"{q*100}th Percentile: {q_value:.2f}",
				rotation=90,
				color="red",
				ha="right",
				va="top"
			)

		plt.tight_layout()
		plt.show()

