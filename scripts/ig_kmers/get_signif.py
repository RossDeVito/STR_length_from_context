""" Correct p-values and get significant kmers from enrichment results. 

Saves corrected results and diagnostic plots into source results dir.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


def estimate_effect_size_threshold(effect_sizes, percentile=99.0):
	"""Estimate effect size threshold from absolute effect size distribution.

	Args:
		effect_sizes: array of effect sizes.
		percentile: percentile of |effect_size| above which k-mers are
			considered to have large effects. Default: 99.

	Returns:
		threshold: absolute effect size threshold.
	"""
	return np.percentile(np.abs(effect_sizes), percentile)


def correct_and_report(df, label, fdr_alpha=0.01, es_percentile=99.0):
	"""Apply FDR/Bonferroni correction, effect size thresholding, and print summary.

	Args:
		df: DataFrame with enrichment results.
		label: Label for printing.
		fdr_alpha: FDR threshold for significance. Default: 0.01.
		es_percentile: Percentile for effect size threshold. Default: 99.

	Returns:
		Corrected DataFrame with added columns.
	"""
	if len(df) == 0:
		print(f"\n{label}: No results to correct.")
		return df

	df = df.copy()
	df["p_adj_fdr_bh"] = multipletests(
		df["p_value"], method="fdr_bh"
	)[1]
	df["p_adj_bonferroni"] = multipletests(
		df["p_value"], method="bonferroni"
	)[1]

	# Effect size
	df["effect_size"] = df["raw_ig_observed"] - df["raw_ig_expected"]

	# Effect size percentile rank (within this correction scope)
	from scipy.stats import percentileofscore
	es_values = df["effect_size"].values
	df["effect_size_percentile"] = [
		percentileofscore(es_values, v, kind="rank")
		for v in es_values
	]

	# Effect size threshold
	es_threshold = estimate_effect_size_threshold(
		es_values, es_percentile
	)

	# Significance masks
	fdr_sig = df["p_adj_fdr_bh"] < fdr_alpha
	es_sig = df["effect_size"].abs() > es_threshold
	dual_sig = fdr_sig & es_sig

	df["sig_fdr"] = fdr_sig
	df["sig_effect_size"] = es_sig
	df["sig_dual"] = dual_sig

	print(f"\n{label}:")
	print(f"  Total kmers tested: {len(df)}")
	print(f"  Effect size threshold ({es_percentile}th percentile): "
	      f"{es_threshold:.2e}")

	for alpha in [0.05, 0.01, 0.001]:
		n_fdr = (df["p_adj_fdr_bh"] < alpha).sum()
		n_bonf = (df["p_adj_bonferroni"] < alpha).sum()
		n_dual = ((df["p_adj_fdr_bh"] < alpha) & es_sig).sum()
		print(f"  alpha={alpha}: FDR={n_fdr}, Bonferroni={n_bonf}, "
		      f"FDR+ES={n_dual}")

	return df


def plot_diagnostics(df, label, save_dir, fdr_alpha=0.01):
	"""Plot p-value histogram, effect size distribution, and volcano plot."""
	fig, axes = plt.subplots(1, 3, figsize=(18, 5))

	effect_sizes = df["effect_size"].values

	# Get the threshold already computed in correct_and_report
	es_sig = df["sig_effect_size"]
	es_threshold = df.loc[es_sig, "effect_size"].abs().min() if es_sig.any() \
		else df["effect_size"].abs().max()

	# --- P-value histogram ---
	ax = axes[0]
	ax.hist(df["p_value"], bins=50, edgecolor="black", linewidth=0.5)
	ax.set_xlabel("Uncorrected p-value")
	ax.set_ylabel("Count")
	ax.set_title(f"P-value distribution\n{label}")
	ax.axhline(
		y=len(df) / 50, color="red", linestyle="--", alpha=0.7,
		label="Expected under null"
	)
	ax.legend()

	# --- Effect size distribution ---
	ax = axes[1]
	ax.hist(effect_sizes, bins=50, edgecolor="black", linewidth=0.5)
	ax.axvline(0, color="red", linestyle="--", alpha=0.7)
	ax.axvline(es_threshold, color="orange", linestyle="--", alpha=0.7,
	           label=f"Threshold: ±{es_threshold:.2e}")
	ax.axvline(-es_threshold, color="orange", linestyle="--", alpha=0.7)
	ax.set_xlabel("Effect size (observed - expected raw IG)")
	ax.set_ylabel("Count")
	ax.set_title(f"Effect size distribution\n{label}")
	ax.legend(fontsize=8)

	# --- Volcano plot ---
	ax = axes[2]
	neg_log_p = -np.log10(df["p_adj_fdr_bh"].clip(lower=1e-300))

	fdr_sig = df["p_adj_fdr_bh"] < fdr_alpha
	dual_mask = df["sig_dual"]
	pval_only = fdr_sig & ~df["sig_effect_size"]

	# Gray: not FDR significant
	ax.scatter(
		effect_sizes[~fdr_sig], neg_log_p[~fdr_sig],
		s=15, alpha=0.4, c="gray", edgecolors="none",
	)

	# Salmon: FDR significant but small effect
	if pval_only.any():
		ax.scatter(
			effect_sizes[pval_only], neg_log_p[pval_only],
			s=15, alpha=0.5, c="salmon", edgecolors="none",
			label=f"FDR only (n={pval_only.sum()})"
		)

	# Red: dual threshold
	if dual_mask.any():
		ax.scatter(
			effect_sizes[dual_mask], neg_log_p[dual_mask],
			s=20, alpha=0.8, c="red", edgecolors="none",
			label=f"FDR + effect (n={dual_mask.sum()})"
		)

	ax.axhline(-np.log10(fdr_alpha), color="blue", linestyle="--", alpha=0.5)
	ax.axvline(es_threshold, color="orange", linestyle="--", alpha=0.5)
	ax.axvline(-es_threshold, color="orange", linestyle="--", alpha=0.5)

	ax.set_xlabel("Effect size (observed - expected raw IG)")
	ax.set_ylabel("-log10(adjusted p-value)")
	ax.set_title(f"Volcano plot\n{label}")
	ax.legend(fontsize=8)

	plt.tight_layout()

	safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
	save_path = os.path.join(save_dir, f"diagnostics_{safe_label}.png")
	fig.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"  Saved diagnostic plots to {save_path}")


if __name__ == "__main__":

	# Paths
	# enrichment_dir = "output/enrichment/str2/dev/dev_bymotif_rank_abs"
	enrichment_dir = "output/enrichment/str2/dev/dev6_bymotif_rank_abs"
	os.path.join(enrichment_dir, "enrichment_results_corrected.tsv")

	# Load enrichment results
	enrichment_path = os.path.join(enrichment_dir, "enrichment_results.tsv")
	enrichment_df = pd.read_csv(enrichment_path, sep="\t")

	print("Kmer enrichment results:")
	print(f"Total rows: {len(enrichment_df)}")
	print(f"Motif classes: {sorted(enrichment_df['motif_class'].unique())}")

	# Split into "all" (aggregate) and per-motif results
	all_df = enrichment_df[enrichment_df["motif_class"] == "all"]
	motif_df = enrichment_df[enrichment_df["motif_class"] != "all"]

	# Correct separately: "all" on its own, per-motif together
	all_corrected = correct_and_report(all_df, "Aggregate (all motifs)")
	motif_corrected = correct_and_report(motif_df, "Per-motif (corrected together)")

	# Diagnostic plots
	plot_diagnostics(all_corrected, "Aggregate", enrichment_dir)
	plot_diagnostics(motif_corrected, "Per-motif combined", enrichment_dir)
	for motif in sorted(motif_corrected["motif_class"].unique()):
		motif_subset = motif_corrected[
			motif_corrected["motif_class"] == motif
		]
		plot_diagnostics(motif_subset, f"Motif {motif}", enrichment_dir)

	# Recombine and filter to dual-significant only
	corrected_df = pd.concat(
		[all_corrected, motif_corrected], ignore_index=True
	)
	sig_df = corrected_df[corrected_df["sig_dual"]].copy()

	# Save
	sig_df.to_csv(
		os.path.join(enrichment_dir, "enrichment_results_corrected_sig.tsv"),
		sep="\t",
		index=False,
		float_format="%.6g"
	)
	corrected_df.to_csv(
		os.path.join(enrichment_dir, "enrichment_results_corrected.tsv"),
		sep="\t",
		index=False,
		float_format="%.6g"
		)
	print("\nSaved {:} dual-significant k-mers to {:}".format(
		len(sig_df),
		os.path.join(enrichment_dir, "enrichment_results_corrected_sig.tsv")
	))
