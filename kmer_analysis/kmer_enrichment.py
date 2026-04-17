"""Stratified permutation test for k-mer enrichment in IG attributions.

Loads k-mer data produced by kmer_extract.py and runs a distance-stratified
permutation test to identify canonical k-mers whose IG attribution scores
are significantly different from expected given their distance from the STR.

At each distance, the null hypothesis is that k-mer identity does not
predict IG score — any k-mer at that distance is equally likely to receive
any score. The test permutes scores across k-mer labels within each
distance stratum, preserving the positional attribution gradient.

Test statistic: grand mean of a k-mer's scores (or ranks) across all
distances and occurrences. The null distribution is constructed by
repeating the stratified permutation many times and recomputing the grand
mean each time. P-values are computed following Phipson & Smyth (2010)
to ensure valid p-values under multiple testing correction.

The test can be run on:
  - Signed IG scores (two-sided: detects k-mers with higher or lower
    attribution than expected)
  - Absolute IG scores (one-sided higher: detects k-mers the model
    attends to regardless of direction)
  - Rank-transformed scores (more robust to outliers) or raw scores

Results can optionally be stratified by STR motif class and/or computed
within distance windows in addition to the global (all-distance) test.

Args:
    --config: Path to YAML config file.
    --output_dir: Directory to save outputs.

Config keys:
    desc (str): Output subdirectory name.
    input_path (str): Path to parquet file from kmer_extract.py.
    n_permutations (int): Number of random permutations (e.g. 10000).
    seed (int, optional): Random seed. Default: 42.
    use_ranks (bool, optional): Rank-transform scores within each
        distance before testing. Default: true.
    use_absolute (bool, optional): Use |IG| instead of signed IG.
        Default: false.
    by_motif (bool, optional): Run tests separately for each STR motif
        class. Default: false.
    min_kmer_count (int, optional): Minimum total occurrences for a k-mer
        to be tested. Default: 30.
    distance_windows (list of [start, end], optional): Distance windows
        for windowed analysis. Each window is inclusive on both ends.
        If omitted, runs a single global test across all distances.
        If provided, runs one test per window (no global test).
        Intended for separate FDR correction from the global run.
        Example: [[1, 25], [26, 50], [51, 100], [101, 200]]
"""

import argparse
import datetime
import os
import json

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable, **kwargs):
		return iterable


# ---------------------------------------------------------------------------
# Permutation test core
# ---------------------------------------------------------------------------

def run_stratified_permutation_test(
	scores_by_distance,
	kmer_ids_by_distance,
	n_kmers,
	n_permutations,
	rng,
	use_ranks=True,
	use_absolute=False,
):
	"""Run distance-stratified permutation test for all k-mers.

	Args:
		scores_by_distance: dict mapping distance -> 1D array of IG scores.
		kmer_ids_by_distance: dict mapping distance -> 1D array of integer
			k-mer IDs (same length as corresponding scores array).
		n_kmers: Total number of unique k-mer IDs.
		n_permutations: Number of random permutations.
		rng: numpy random Generator.
		use_ranks: If True, rank-transform scores within each distance.
		use_absolute: If True, take absolute value of scores before
			testing (and before ranking if use_ranks is also True).

	Returns:
		dict with keys:
			observed_means: (n_kmers,) array of observed grand means.
			null_means: (n_permutations, n_kmers) array of null grand means.
			kmer_counts: (n_kmers,) array of total occurrences per k-mer.
	"""
	distances = sorted(scores_by_distance.keys())

	# Preprocess scores
	processed = {}
	for d in distances:
		s = scores_by_distance[d].copy()
		if use_absolute:
			s = np.abs(s)
		if use_ranks:
			s = rankdata(s).astype(np.float32)
		processed[d] = s

	# Compute observed grand means using bincount
	kmer_sums = np.zeros(n_kmers, dtype=np.float64)
	kmer_counts = np.zeros(n_kmers, dtype=np.float64)

	for d in distances:
		s = processed[d]
		ids = kmer_ids_by_distance[d]
		kmer_sums += np.bincount(ids, weights=s, minlength=n_kmers)
		kmer_counts += np.bincount(ids, minlength=n_kmers)

	# Avoid division by zero for k-mers with no occurrences
	safe_counts = np.where(kmer_counts > 0, kmer_counts, 1.0)
	observed_means = kmer_sums / safe_counts

	# Permutation null
	null_means = np.zeros((n_permutations, n_kmers), dtype=np.float32)

	for p in tqdm(range(n_permutations), desc="Permutations", leave=False):
		perm_sums = np.zeros(n_kmers, dtype=np.float64)

		for d in distances:
			shuffled = rng.permutation(processed[d])
			ids = kmer_ids_by_distance[d]
			perm_sums += np.bincount(
				ids, weights=shuffled, minlength=n_kmers
			)

		null_means[p] = perm_sums / safe_counts

	return {
		"observed_means": observed_means,
		"null_means": null_means,
		"kmer_counts": kmer_counts,
	}


def compute_pvalues(observed_means, null_means, kmer_counts,
                    min_count, use_absolute):
	"""Compute p-values from permutation results.

	For absolute IG: one-sided test (higher than expected).
	For signed IG: two-sided test with direction reported separately.

	P-values follow Phipson & Smyth (2010): p = (b + 1) / (m + 1).

	Args:
		observed_means: (n_kmers,) observed grand means.
		null_means: (n_permutations, n_kmers) null grand means.
		kmer_counts: (n_kmers,) total occurrences per k-mer.
		min_count: Minimum count to test a k-mer.
		use_absolute: Whether absolute IG was used.

	Returns:
		DataFrame with per-k-mer results.
	"""
	n_permutations = null_means.shape[0]
	n_kmers = len(observed_means)

	results = []

	for ki in range(n_kmers):
		count = int(kmer_counts[ki])
		if count < min_count:
			continue

		obs = observed_means[ki]
		null = null_means[:, ki]
		null_mean = float(np.mean(null))

		if use_absolute:
			# One-sided: is this k-mer's |IG| higher than expected?
			b = int(np.sum(null >= obs))
			p_value = (b + 1) / (n_permutations + 1)
			direction = "higher"
		else:
			# Two-sided: is this k-mer's IG different from expected?
			null_center = null_mean
			obs_dev = abs(obs - null_center)
			null_devs = np.abs(null - null_center)
			b = int(np.sum(null_devs >= obs_dev))
			p_value = (b + 1) / (n_permutations + 1)
			direction = "higher" if obs > null_center else "lower"

		results.append({
			"kmer_id": ki,
			"n_occurrences": count,
			"observed_mean": float(obs),
			"null_mean": null_mean,
			"p_value": p_value,
			"effect_direction": direction,
		})

	return pd.DataFrame(results)


def compute_raw_ig_effect_sizes(scores_by_distance, kmer_ids_by_distance,
                                n_kmers):
	"""Compute raw signed IG observed and expected means per k-mer.

	The observed mean is the k-mer's actual mean signed IG across all
	occurrences. The expected mean is the positionally-expected value
	under the null — the weighted average of per-distance pool means,
	weighted by the k-mer's occurrence count at each distance. The
	difference (observed - expected) is an interpretable effect size
	in IG units, controlling for position.

	No permutations needed — the expected value is deterministic.

	Args:
		scores_by_distance: dict mapping distance -> 1D array of raw
			signed IG scores.
		kmer_ids_by_distance: dict mapping distance -> 1D array of
			integer k-mer IDs.
		n_kmers: Total number of unique k-mer IDs.

	Returns:
		dict with keys:
			raw_observed: (n_kmers,) observed mean signed IG per k-mer.
			raw_expected: (n_kmers,) positionally-expected mean signed IG.
	"""
	obs_sums = np.zeros(n_kmers, dtype=np.float64)
	exp_sums = np.zeros(n_kmers, dtype=np.float64)
	counts = np.zeros(n_kmers, dtype=np.float64)

	for d in scores_by_distance:
		s = scores_by_distance[d]
		ids = kmer_ids_by_distance[d]
		pool_mean = float(np.mean(s))

		obs_sums += np.bincount(ids, weights=s, minlength=n_kmers)
		per_kmer_counts = np.bincount(ids, minlength=n_kmers).astype(
			np.float64
		)
		exp_sums += per_kmer_counts * pool_mean
		counts += per_kmer_counts

	safe_counts = np.where(counts > 0, counts, 1.0)

	return {
		"raw_observed": obs_sums / safe_counts,
		"raw_expected": exp_sums / safe_counts,
	}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

	__spec__ = None

	parser = argparse.ArgumentParser(
		description="Stratified permutation test for k-mer enrichment."
	)
	parser.add_argument(
		"--config", type=str, required=True,
		help="Path to the configuration YAML file.",
	)
	parser.add_argument(
		"--output_dir", type=str, default=".",
		help="Directory to save outputs.",
	)
	args = parser.parse_args()

	# ------------------------------------------------------------------
	# Load config
	# ------------------------------------------------------------------
	print(f"Loading config from {args.config}")
	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	input_path = config["input_path"]
	n_permutations = config["n_permutations"]
	seed = config.get("seed", 42)
	use_ranks = config.get("use_ranks", True)
	use_absolute = config.get("use_absolute", False)
	by_motif = config.get("by_motif", False)
	min_kmer_count = config.get("min_kmer_count", 30)
	distance_windows = config.get("distance_windows", None)

	# ------------------------------------------------------------------
	# Set up output directory
	# ------------------------------------------------------------------
	run_dir = os.path.join(args.output_dir, config["desc"])
	os.makedirs(run_dir, exist_ok=True)

	config_save_path = os.path.join(run_dir, "config.yaml")
	with open(config_save_path, "w") as f:
		yaml.dump(config, f)
	print(f"Outputs will be saved to: {run_dir}")

	# ------------------------------------------------------------------
	# Load k-mer data
	# ------------------------------------------------------------------
	print(f"Loading k-mer data from {input_path}")
	df = pd.read_parquet(input_path, engine="pyarrow")

	n_rows = len(df)
	unique_kmers = sorted(df["kmer"].unique())
	n_unique_kmers = len(unique_kmers)
	kmer_to_id = {km: i for i, km in enumerate(unique_kmers)}

	print(f"  {n_rows} rows, {n_unique_kmers} unique k-mers")
	print(f"  Distance range: {df['distance'].min()} - {df['distance'].max()}")
	print(f"  Motif classes: {sorted(df['motif_class'].unique())}")

	# Map k-mer strings to integer IDs for efficient bincount
	df["kmer_id"] = df["kmer"].map(kmer_to_id).astype(np.int32)

	# ------------------------------------------------------------------
	# Build analysis scopes
	# ------------------------------------------------------------------
	# Each scope is a (label, DataFrame subset) pair.
	# If by_motif, run separately per motif class. Always include "all".
	scopes = [("all", df)]

	if by_motif:
		for motif in sorted(df["motif_class"].unique()):
			scopes.append((motif, df[df["motif_class"] == motif]))

	# Each scope is crossed with distance windows.
	# Either global (all distances) OR windowed, not both.
	if distance_windows is not None:
		windows = []
		for w in distance_windows:
			windows.append((f"{w[0]}-{w[1]}", w))
	else:
		windows = [("global", None)]

	print(f"\nAnalysis plan:")
	print(f"  Scopes: {[s[0] for s in scopes]}")
	print(f"  Windows: {[w[0] for w in windows]}")
	print(f"  Permutations: {n_permutations}")
	print(f"  Use ranks: {use_ranks}")
	print(f"  Use absolute: {use_absolute}")
	print(f"  Min k-mer count: {min_kmer_count}")

	# ------------------------------------------------------------------
	# Run permutation tests
	# ------------------------------------------------------------------
	rng = np.random.default_rng(seed)
	all_results = []

	for scope_label, scope_df in scopes:
		for window_label, window_range in windows:

			# Filter to distance window
			if window_range is not None:
				wdf = scope_df[
					(scope_df["distance"] >= window_range[0])
					& (scope_df["distance"] <= window_range[1])
				]
			else:
				wdf = scope_df

			if len(wdf) == 0:
				print(f"\n  [{scope_label} / {window_label}] "
				      f"No data, skipping.")
				continue

			# Build per-distance arrays
			scores_by_dist = {}
			kmer_ids_by_dist = {}

			for d, group in wdf.groupby("distance"):
				scores_by_dist[d] = group["ig_score"].values
				kmer_ids_by_dist[d] = group["kmer_id"].values

			n_distances = len(scores_by_dist)
			n_obs = len(wdf)

			print(f"\n  [{scope_label} / {window_label}] "
			      f"{n_obs} observations across {n_distances} distances")

			# Run test
			result = run_stratified_permutation_test(
				scores_by_distance=scores_by_dist,
				kmer_ids_by_distance=kmer_ids_by_dist,
				n_kmers=n_unique_kmers,
				n_permutations=n_permutations,
				rng=rng,
				use_ranks=use_ranks,
				use_absolute=use_absolute,
			)

			# Compute p-values
			res_df = compute_pvalues(
				observed_means=result["observed_means"],
				null_means=result["null_means"],
				kmer_counts=result["kmer_counts"],
				min_count=min_kmer_count,
				use_absolute=use_absolute,
			)

			if len(res_df) == 0:
				print(f"    No k-mers met min_count threshold.")
				continue

			# Compute raw IG effect sizes (always from untransformed scores)
			raw_ig = compute_raw_ig_effect_sizes(
				scores_by_distance=scores_by_dist,
				kmer_ids_by_distance=kmer_ids_by_dist,
				n_kmers=n_unique_kmers,
			)

			# Merge raw IG effect sizes into results by kmer_id
			res_df["raw_ig_observed"] = res_df["kmer_id"].map(
				lambda ki: float(raw_ig["raw_observed"][ki])
			)
			res_df["raw_ig_expected"] = res_df["kmer_id"].map(
				lambda ki: float(raw_ig["raw_expected"][ki])
			)

			# Map k-mer IDs back to strings
			id_to_kmer = {v: k for k, v in kmer_to_id.items()}
			res_df["kmer"] = res_df["kmer_id"].map(id_to_kmer)

			# Add scope and window labels
			res_df["motif_class"] = scope_label
			res_df["window"] = window_label

			# Sort by p-value
			res_df = res_df.sort_values("p_value")

			n_tested = len(res_df)
			n_sig_01 = int((res_df["p_value"] < 0.01).sum())
			n_sig_05 = int((res_df["p_value"] < 0.05).sum())
			print(f"    Tested {n_tested} k-mers. "
			      f"p<0.01: {n_sig_01}, p<0.05: {n_sig_05} "
			      f"(uncorrected)")

			all_results.append(res_df)

	# ------------------------------------------------------------------
	# Combine and save results
	# ------------------------------------------------------------------
	if len(all_results) == 0:
		print("\nNo results produced. Check data and thresholds.")
	else:
		combined = pd.concat(all_results, ignore_index=True)

		# Select and order output columns
		output_cols = [
			"kmer", "motif_class", "window", "n_occurrences",
			"observed_mean", "null_mean", "p_value",
			"effect_direction",
			"raw_ig_observed", "raw_ig_expected",
		]
		combined = combined[output_cols]

		results_path = os.path.join(run_dir, "enrichment_results.tsv")
		combined.to_csv(results_path, sep="\t", index=False,
		                float_format="%.6g")
		print(f"\nResults saved to {results_path}")
		print(f"Total result rows: {len(combined)}")

	# ------------------------------------------------------------------
	# Save run metadata
	# ------------------------------------------------------------------
	run_meta = {
		"config": config,
		"n_unique_kmers": n_unique_kmers,
		"kmer_id_mapping": kmer_to_id,
		"scopes_run": [s[0] for s in scopes],
		"windows_run": [w[0] for w in windows],
		"timestamp": datetime.datetime.now().isoformat(),
	}

	meta_out_path = os.path.join(run_dir, "meta.json")
	with open(meta_out_path, "w") as f:
		json.dump(run_meta, f, indent=4)
	print(f"Metadata saved to {meta_out_path}")

	print("\n--- Done ---")