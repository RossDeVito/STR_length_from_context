"""Correlate local GC content -- and its upstream/downstream gradient --
around each STR's flanks with its length/heterozygosity labels.

Unlike the other scripts in this directory, this one does not read a
precomputed annotation file (no annotation_config.yaml / annotation_utils.py
involved) -- GC content is recomputed directly from the reference FASTA for
exactly the loci and window sizes needed here.

For each window size n in WINDOW_SIZES, two per-locus scalars are derived
from the same n-bp upstream flank ([str_start - n, str_start)) and n-bp
downstream flank ([str_end, str_end + n)) -- the same flank convention used
in seq_models/caduceus/data.py and data/scripts/create_GC_content_anno_file.py:

  GC content : combined GC fraction over both flanks together (up to 2n bp),
               i.e. (GC bases upstream + GC bases downstream) / (valid bases
               upstream + valid bases downstream). N bases are excluded from
               both numerator and denominator.
  GC gradient: abs(GC_upstream - GC_downstream), the two flanks' fractions
               computed separately, then differenced.

Each is Spearman-correlated (rho + two-sided p-value) against every label in
TARGETS -- the same {length: mode_copy_number, heterozygosity: heterozygosity}
targets used elsewhere in this directory (e.g. str_anno_overlap_analysis.py).
scipy.stats.spearmanr ranks with ties averaged, which matters here: coarse
windows (e.g. n=5, only 6 possible GC fractions) produce heavily-tied values,
and the labels themselves are tied across many loci.

An optional percentile bootstrap (resampling loci with replacement and
recomputing rho from scratch each draw, since which values tie changes with
the resample) gives a CI on rho, at CI_LEVEL over N_BOOTSTRAP resamples.

Every (window size, metric, target) combination run in one execution is one
test in the same family, so raw p-values are Bonferroni-corrected across all
of them (statsmodels.stats.multitest.multipletests, mirroring
CLIFFS_DELTA_CORRECTION in str_anno_overlap_analysis.py); the plot marks each
point significant/not at SIGNIFICANCE_ALPHA on the corrected value.

Configuration is via the editable constants below (no argparse), mirroring
the rest of this directory.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from pyfaidx import Fasta
from tqdm import tqdm


# ===========================================================================
# Configuration
# ===========================================================================

STR_LEN = 4
N_FLANKING = 10000  # must match the STR tsv's filename suffix

STR_DATA_PATH = f"../../data/STR_data/HipSTR_labeled_STRs/str_len_{STR_LEN}_n_flanking_{N_FLANKING}.tsv"
REF_PATH = "../../data/reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa"

# n-bp flank window(s), on EACH side of the STR (upstream and downstream) --
# i.e. GC content for a given n is computed over up to 2*n bp per locus.
WINDOW_SIZES = [5, 10, 25, 50, 100, 250, 500, 1000, 5000]

# Targets, mirroring str_anno_overlap_analysis.py / get_mean_baseline_performance.py.
TARGETS = {
	"length": "mode_copy_number",
	"heterozygosity": "heterozygosity",
}

# Percentile bootstrap CI (over loci) for each Spearman rho.
RUN_BOOTSTRAP = True
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

SEED = 147

# Significance threshold applied to the Bonferroni-corrected p-value (across
# every window_size x metric x target test run this execution) for the
# plot's significant/not-significant marker.
SIGNIFICANCE_ALPHA = 0.01

# Output. SAVE_PATH=None -> plt.show(); SAVE_RESULTS_PATH=None -> no json.
SAVE_PATH = None
SAVE_RESULTS_PATH = None


# ===========================================================================
# Loading
# ===========================================================================

def load_str_data(path):
	"""Load HipSTR-labeled STRs, keeping one row per locus (drop the
	reverse-complement duplicate; it shares the same position/labels)."""
	df = pd.read_csv(path, sep="\t")
	df = df[~df["rev_comp"]].reset_index(drop=True)
	return df


# ===========================================================================
# GC content from the reference
# ===========================================================================

def _gc_counts(seq):
	"""(n_gc, n_valid) for an uppercase sequence string. N/ambiguous bases
	are excluded from both counts, matching create_GC_content_anno_file.py."""
	arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
	n_gc = int(np.count_nonzero((arr == ord("G")) | (arr == ord("C"))))
	n_valid = int(np.count_nonzero(arr != ord("N")))
	return n_gc, n_valid


def _flank_seq(ref_genome, chrom, start, end, chrom_len):
	"""Reference sequence over [start, end), clipped to the chromosome.

	pyfaidx applies Python-style negative indexing to out-of-range
	coordinates rather than clipping (e.g. rec[-5:10] reads near the
	chromosome's OTHER end), so a flank near a chromosome start must be
	clipped here first or it would silently read the wrong sequence.
	"""
	lo, hi = max(start, 0), min(end, chrom_len)
	if lo >= hi:
		return ""
	return str(ref_genome[chrom][lo:hi])


def compute_gc_features(str_df, ref_path, window_sizes):
	"""Returns {window_size: DataFrame(gc_content, gc_gradient)}, one row per
	locus (aligned with str_df's index). Only the single largest window is
	ever fetched from the reference per locus per side; smaller windows are
	sliced from it (the bases nearest the STR), rather than re-fetching per
	window size.

	A locus/window with zero valid bases on a side gets NaN in whichever
	output(s) need that side -- only possible right at a chromosome end.
	"""
	ref_genome = Fasta(ref_path, sequence_always_upper=True)
	chrom_len_cache = {}
	max_n = max(window_sizes)

	chrom_arr = str_df["chrom"].to_numpy()
	start_arr = str_df["str_start"].to_numpy()
	end_arr = str_df["str_end"].to_numpy()
	n_loci = len(str_df)

	gc_content = {n: np.full(n_loci, np.nan) for n in window_sizes}
	gc_gradient = {n: np.full(n_loci, np.nan) for n in window_sizes}

	for i in tqdm(range(n_loci), desc="  GC content from reference"):
		chrom = chrom_arr[i]
		if chrom not in chrom_len_cache:
			chrom_len_cache[chrom] = len(ref_genome[chrom])
		chrom_len = chrom_len_cache[chrom]

		start, end = int(start_arr[i]), int(end_arr[i])
		# [start - max_n, start): the LAST n chars are nearest the STR.
		up_max = _flank_seq(ref_genome, chrom, start - max_n, start, chrom_len)
		# [end, end + max_n): the FIRST n chars are nearest the STR.
		down_max = _flank_seq(ref_genome, chrom, end, end + max_n, chrom_len)

		for n in window_sizes:
			gc_up, valid_up = _gc_counts(up_max[-n:])
			gc_down, valid_down = _gc_counts(down_max[:n])

			if valid_up + valid_down > 0:
				gc_content[n][i] = (gc_up + gc_down) / (valid_up + valid_down)
			if valid_up > 0 and valid_down > 0:
				gc_gradient[n][i] = abs(gc_up / valid_up - gc_down / valid_down)

	return {
		n: pd.DataFrame({"gc_content": gc_content[n], "gc_gradient": gc_gradient[n]})
		for n in window_sizes
	}


# ===========================================================================
# Statistics
# ===========================================================================

def spearman_with_ci(x, y, run_bootstrap, n_boot, ci_level, rng):
	"""Spearman rho + two-sided p-value on the paired, finite entries of x
	and y, plus an optional percentile bootstrap CI over loci.

	Each bootstrap draw resamples loci with replacement and recomputes rho
	from scratch (not just re-indexing precomputed ranks): which values tie
	changes with the resample, and scipy's tie-averaged ranking needs to see
	that to stay correct.
	"""
	mask = np.isfinite(x) & np.isfinite(y)
	x, y = x[mask], y[mask]
	n = len(x)

	rho, p_value = spearmanr(x, y)

	ci_lo, ci_hi = np.nan, np.nan
	if run_bootstrap and n >= 3:
		boot_idx = rng.integers(0, n, size=(n_boot, n))
		boot_rho = np.array([spearmanr(x[idx], y[idx])[0] for idx in boot_idx])
		alpha = 1 - ci_level
		ci_lo = float(np.nanpercentile(boot_rho, 100 * (alpha / 2)))
		ci_hi = float(np.nanpercentile(boot_rho, 100 * (1 - alpha / 2)))

	return {
		"n": n, "rho": float(rho), "p_value": float(p_value),
		"ci_lo": ci_lo, "ci_hi": ci_hi,
	}


# ===========================================================================
# Main
# ===========================================================================

def main():
	rng = np.random.default_rng(SEED)

	print("=" * 70)
	print(f"GC content & gradient vs STR labels  [str_len={STR_LEN}]")
	print("=" * 70)

	str_df = load_str_data(STR_DATA_PATH)
	print(f"\nLoaded {len(str_df)} loci from {STR_DATA_PATH}")

	print(f"\nComputing GC content over window sizes {WINDOW_SIZES} bp "
	      f"(each side) from {REF_PATH}...")
	features_by_window = compute_gc_features(str_df, REF_PATH, WINDOW_SIZES)

	tasks = [
		(n, metric, target_name, label_col)
		for n in WINDOW_SIZES
		for metric in ("gc_content", "gc_gradient")
		for target_name, label_col in TARGETS.items()
	]

	if RUN_BOOTSTRAP:
		print(f"\nBootstrapping each rho ({N_BOOTSTRAP} resamples, "
		      f"{int(CI_LEVEL * 100)}% CI)...")

	results = []
	for n, metric, target_name, label_col in tqdm(tasks, desc="  rho + bootstrap CI"):
		values = features_by_window[n][metric].to_numpy()
		labels = str_df[label_col].to_numpy()
		stat = spearman_with_ci(
			values, labels, RUN_BOOTSTRAP, N_BOOTSTRAP, CI_LEVEL, rng
		)
		stat.update({"window_size": n, "metric": metric, "target": target_name})
		results.append(stat)

	results_df = pd.DataFrame(results)

	# --- Bonferroni correction across every test run this execution ---------

	n_tests = len(results_df)
	_, p_bonf, _, _ = multipletests(
		results_df["p_value"].to_numpy(), alpha=SIGNIFICANCE_ALPHA, method="bonferroni"
	)
	results_df["p_value_bonferroni"] = p_bonf
	results_df["significant"] = p_bonf < SIGNIFICANCE_ALPHA

	# --- report -------------------------------------------------------------

	print("\n" + "=" * 70)
	print(f"RESULTS  (Bonferroni over {n_tests} tests, alpha={SIGNIFICANCE_ALPHA})")
	print("=" * 70)
	for _, row in results_df.iterrows():
		ci_str = (
			f"  {int(CI_LEVEL * 100)}% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]"
			if RUN_BOOTSTRAP else ""
		)
		sig_mark = "*" if row["significant"] else " "
		print(
			f"window={int(row['window_size']):>5}bp  {row['metric']:<12} "
			f"vs {row['target']:<15} n={int(row['n']):<7} "
			f"rho={row['rho']:+.4f}  p={row['p_value']:.3g}  "
			f"p_bonf={row['p_value_bonferroni']:.3g}{sig_mark}{ci_str}"
		)

	if SAVE_RESULTS_PATH:
		os.makedirs(os.path.dirname(os.path.abspath(SAVE_RESULTS_PATH)), exist_ok=True)
		with open(SAVE_RESULTS_PATH, "w") as f:
			json.dump(
				{
					"str_len": STR_LEN,
					"n_flanking": N_FLANKING,
					"window_sizes": WINDOW_SIZES,
					"targets": TARGETS,
					"run_bootstrap": RUN_BOOTSTRAP,
					"n_bootstrap": N_BOOTSTRAP,
					"ci_level": CI_LEVEL,
					"seed": SEED,
					"n_tests": n_tests,
					"significance_alpha": SIGNIFICANCE_ALPHA,
					"results": results_df.to_dict(orient="records"),
				},
				f, indent=2,
			)
		print(f"\nWrote {SAVE_RESULTS_PATH}")

	plot(results_df)


# ===========================================================================
# Plot
# ===========================================================================

def plot(results_df):
	"""One panel per target: Spearman rho vs. window size, one line for GC
	content and one for the GC gradient, with bootstrap CI error bars.
	Points are marked filled circle if significant (Bonferroni-corrected p <
	SIGNIFICANCE_ALPHA) or open circle if not."""
	targets = list(TARGETS.keys())
	fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 5),
	                          squeeze=False)
	axes = axes[0]

	colors = {"gc_content": "#3170ad", "gc_gradient": "#e8443a"}
	labels = {
		"gc_content": "GC content (both flanks)",
		"gc_gradient": "|GC upstream - GC downstream|",
	}

	for ax, target_name in zip(axes, targets):
		sub = results_df[results_df["target"] == target_name]
		for metric in ("gc_content", "gc_gradient"):
			m = sub[sub["metric"] == metric].sort_values("window_size")
			yerr = None
			if RUN_BOOTSTRAP:
				yerr = np.abs(np.vstack(
					[m["rho"] - m["ci_lo"], m["ci_hi"] - m["rho"]]
				))
			# marker=""/"None" here: markers are drawn separately below, per
			# point, so significant/not can use different marker styles.
			ax.errorbar(
				m["window_size"], m["rho"], yerr=yerr, marker="None", capsize=3,
				color=colors[metric], label=labels[metric], zorder=1,
			)

			sig = m[m["significant"]]
			not_sig = m[~m["significant"]]
			ax.scatter(sig["window_size"], sig["rho"], marker="o",
			           color=colors[metric], s=45, zorder=3)
			ax.scatter(not_sig["window_size"], not_sig["rho"], marker="o",
			           facecolors="none", edgecolors=colors[metric], s=45,
			           linewidths=1.5, zorder=3)

		ax.axhline(0, color="gray", lw=1, ls=":")
		ax.set_xscale("log")
		ax.set_xlabel("Window size (bp, each side)")
		ax.set_ylabel("Spearman rho")
		ax.set_title(f"vs {TARGETS[target_name].replace('_', ' ')}")

		sig_handles = [
			Line2D([0], [0], marker="o", linestyle="None", color="gray",
			       markersize=7,
			       label=f"significant (Bonferroni, α={SIGNIFICANCE_ALPHA})"),
			Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none",
			       markeredgecolor="gray", markersize=7, label="not significant"),
		]
		handles, _ = ax.get_legend_handles_labels()
		ax.legend(handles=handles + sig_handles, fontsize=8)

	fig.suptitle(
		f"GC content & gradient vs STR labels  [str_len={STR_LEN}"
		+ (f", {int(CI_LEVEL * 100)}% bootstrap CI, {N_BOOTSTRAP} resamples"
		   if RUN_BOOTSTRAP else "") + "]"
	)
	plt.tight_layout()

	if SAVE_PATH:
		os.makedirs(os.path.dirname(os.path.abspath(SAVE_PATH)), exist_ok=True)
		fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
		print(f"Saved plot to {SAVE_PATH}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
