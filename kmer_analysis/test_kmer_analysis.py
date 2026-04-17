"""Tests for kmer_extract.py and kmer_enrichment.py.

Run with: pytest test_kmer_analysis.py -v
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

# Import functions from the scripts.
# Adjust the import path to match where the scripts live in the repo.
from kmer_extract import (
	reverse_complement,
	canonical_kmer,
	extract_motif_from_sequence,
	STR_CANONICALIZATION,
)
from kmer_enrichment import (
	run_stratified_permutation_test,
	compute_pvalues,
)


# ===========================================================================
# kmer_extract.py — unit tests
# ===========================================================================

class TestReverseComplement:

	def test_basic(self):
		assert reverse_complement("ACGT") == "ACGT"

	def test_single_base(self):
		assert reverse_complement("A") == "T"
		assert reverse_complement("C") == "G"
		assert reverse_complement("G") == "C"
		assert reverse_complement("T") == "A"

	def test_palindrome(self):
		assert reverse_complement("AATT") == "AATT"
		assert reverse_complement("GCGC") == "GCGC"

	def test_asymmetric(self):
		assert reverse_complement("AAAA") == "TTTT"
		assert reverse_complement("GATTACA") == "TGTAATC"

	def test_empty(self):
		assert reverse_complement("") == ""


class TestCanonicalKmer:

	def test_already_canonical(self):
		# "AAAA" < "TTTT", so AAAA is canonical
		assert canonical_kmer("AAAA") == "AAAA"

	def test_rc_is_canonical(self):
		# "TTTT" > "AAAA", so canonical is AAAA
		assert canonical_kmer("TTTT") == "AAAA"

	def test_palindrome(self):
		# ACGT is its own RC
		assert canonical_kmer("ACGT") == "ACGT"

	def test_pair_consistency(self):
		# A k-mer and its RC should give the same canonical form
		assert canonical_kmer("AGCT") == canonical_kmer("AGCT")
		kmer = "GATTAC"
		rc = reverse_complement(kmer)
		assert canonical_kmer(kmer) == canonical_kmer(rc)

	def test_lexicographic_order(self):
		# "ACG" < "CGT" (RC of ACG), so ACG is canonical
		assert canonical_kmer("ACG") == "ACG"
		assert canonical_kmer("CGT") == "ACG"


class TestExtractMotifFromSequence:

	def test_homopolymer_a(self):
		# AAAAAA... in STR region
		seq = "X" * 10 + "N" * 100 + "AAAAAA" + "X" * 20 + "AAAAAA" + "N" * 100
		lf_end = 110
		n_str_bp = 6
		motif = extract_motif_from_sequence(seq, lf_end, n_str_bp)
		assert motif == "A"

	def test_homopolymer_g(self):
		# G canonicalizes to C
		seq = "X" * 10 + "N" * 100 + "GGGGGG" + "X" * 20 + "GGGGGG" + "N" * 100
		motif = extract_motif_from_sequence(seq, 110, 6)
		assert motif == "C"

	def test_dinucleotide_ac(self):
		seq = "X" * 10 + "N" * 100 + "ACACAC" + "X" * 20 + "ACACAC" + "N" * 100
		motif = extract_motif_from_sequence(seq, 110, 6)
		assert motif == "AC"

	def test_dinucleotide_tg(self):
		# TG canonicalizes to AC
		seq = "X" * 10 + "N" * 100 + "TGTGTG" + "X" * 20 + "TGTGTG" + "N" * 100
		motif = extract_motif_from_sequence(seq, 110, 6)
		assert motif == "AC"

	def test_with_prompt_characters(self):
		# STR region has X mixed in
		seq = "X" * 10 + "N" * 100 + "AXA" + "X" * 20 + "AXA" + "N" * 100
		motif = extract_motif_from_sequence(seq, 110, 3)
		assert motif == "A"


class TestSTRCanonicalization:

	def test_all_homopolymers_have_entries(self):
		for base in "ACGT":
			assert base in STR_CANONICALIZATION

	def test_complement_pairs(self):
		assert STR_CANONICALIZATION["A"] == STR_CANONICALIZATION["T"]
		assert STR_CANONICALIZATION["C"] == STR_CANONICALIZATION["G"]

	def test_dinucleotide_rc_pairs(self):
		# AC/TG/CA/GT should all canonicalize to the same thing
		vals = {STR_CANONICALIZATION[k] for k in ["AC", "TG", "CA", "GT"]}
		assert len(vals) == 1

		# AG/TC/CT/GA should all canonicalize to the same thing
		vals = {STR_CANONICALIZATION[k] for k in ["AG", "TC", "CT", "GA"]}
		assert len(vals) == 1


# ===========================================================================
# kmer_extract.py — integration test with synthetic data
# ===========================================================================

def make_synthetic_npz(tmpdir, n_loci=50, n_flank=100, n_str_bp=4,
                       n_prefix=8, n_str_prompt=10):
	"""Create a synthetic attribution NPZ and meta.json for testing.

	Returns the path to the NPZ file.
	"""
	seq_len = n_prefix + n_flank + n_str_bp + n_str_prompt + n_str_bp + n_flank
	n_samples = n_loci * 2  # forward + RC

	rng = np.random.default_rng(42)

	# Build synthetic sequences
	bases = list("ACGT")
	sequences = []
	hipstr_names = []
	rev_comp_flags = []

	for i in range(n_loci):
		name = f"STR_{i:04d}"

		# Random flanking sequence
		left_flank = "".join(rng.choice(bases, n_flank))
		right_flank = "".join(rng.choice(bases, n_flank))
		str_bases = "AC" * (n_str_bp // 2)  # dinucleotide AC repeat

		fwd_seq = (
			"X" * n_prefix
			+ left_flank
			+ str_bases
			+ "X" * n_str_prompt
			+ str_bases
			+ right_flank
		)
		assert len(fwd_seq) == seq_len

		# RC sequence: reverse complement of flanks, reversed layout
		rc_right = reverse_complement(left_flank)
		rc_left = reverse_complement(right_flank)
		rc_str = reverse_complement(str_bases)

		rc_seq = (
			"X" * n_prefix
			+ rc_left
			+ rc_str
			+ "X" * n_str_prompt
			+ rc_str
			+ rc_right
		)
		assert len(rc_seq) == seq_len

		# Forward
		sequences.append(fwd_seq)
		hipstr_names.append(name)
		rev_comp_flags.append(False)

		# RC
		sequences.append(rc_seq)
		hipstr_names.append(name)
		rev_comp_flags.append(True)

	# Synthetic attributions: higher near STR boundary, decaying outward
	attributions = rng.normal(0, 0.01, size=(n_samples, seq_len)).astype(
		np.float32
	)

	# Add signal near STR boundary
	lf_end = n_prefix + n_flank
	rf_start = n_prefix + n_flank + n_str_bp + n_str_prompt + n_str_bp
	for i in range(n_samples):
		for j in range(min(20, n_flank)):
			attributions[i, lf_end - 1 - j] += 0.1 / (j + 1)
			attributions[i, rf_start + j] += 0.1 / (j + 1)

	# Predictions
	raw_predictions = rng.normal(2.0, 0.5, n_samples).astype(np.float32)
	raw_baseline_predictions = np.full(n_samples, 1.5, dtype=np.float32)
	predictions = raw_predictions.copy()
	baseline_predictions = raw_baseline_predictions.copy()
	labels = rng.normal(10.0, 3.0, n_samples).astype(np.float32)
	convergence_deltas = rng.normal(0, 0.001, n_samples).astype(np.float32)

	raw_pred_diffs = np.abs(raw_predictions - raw_baseline_predictions)
	safe_diffs = np.where(raw_pred_diffs > 1e-6, raw_pred_diffs, 1.0)
	relative_deltas = np.abs(convergence_deltas) / safe_diffs

	position_labels = (
		["prefix_prompt"] * n_prefix
		+ ["left_flank"] * n_flank
		+ ["left_str"] * n_str_bp
		+ ["str_prompt"] * n_str_prompt
		+ ["right_str"] * n_str_bp
		+ ["right_flank"] * n_flank
	)

	npz_path = os.path.join(tmpdir, "attributions.npz")
	np.savez_compressed(
		npz_path,
		attributions=attributions,
		input_ids=np.zeros((n_samples, seq_len), dtype=np.int32),
		sequences=np.array(sequences),
		predictions=predictions,
		baseline_predictions=baseline_predictions,
		raw_predictions=raw_predictions,
		raw_baseline_predictions=raw_baseline_predictions,
		labels=labels,
		convergence_deltas=convergence_deltas,
		relative_convergence_deltas=relative_deltas,
		position_labels=np.array(position_labels),
		hipstr_names=np.array(hipstr_names),
		rev_comp=np.array(rev_comp_flags),
	)

	meta = {
		"sequence_layout": {
			"n_prefix_prompt": n_prefix,
			"n_flanking_bp": n_flank,
			"n_str_bp": n_str_bp,
			"n_str_prompt": n_str_prompt,
			"seq_len": seq_len,
		}
	}
	meta_path = os.path.join(tmpdir, "meta.json")
	with open(meta_path, "w") as f:
		json.dump(meta, f)

	return npz_path


try:
	import pyarrow  # noqa: F401
	HAS_PYARROW = True
except ImportError:
	HAS_PYARROW = False


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestKmerExtractIntegration:
	"""Integration test: run kmer_extract.py end-to-end on synthetic data."""

	def test_end_to_end(self, tmp_path):
		"""Verify extraction produces valid parquet with expected structure."""
		npz_path = make_synthetic_npz(str(tmp_path), n_loci=20, n_flank=50)

		config = {
			"desc": "test_run",
			"attr_npz_path": npz_path,
			"k": 5,
			"max_distance": 30,
			"pred_diff_threshold": 0.0,
			"rel_delta_threshold": None,
		}
		config_path = os.path.join(str(tmp_path), "config.yaml")
		with open(config_path, "w") as f:
			yaml.dump(config, f)

		output_dir = os.path.join(str(tmp_path), "output")

		# Run the script as a subprocess to match real usage
		import subprocess
		result = subprocess.run(
			["python", os.path.join(os.path.dirname(__file__), "kmer_extract.py"),
			 "--config", config_path,
			 "--output_dir", output_dir],
			capture_output=True, text=True,
		)
		assert result.returncode == 0, (
			f"kmer_extract.py failed:\nSTDOUT:\n{result.stdout}\n"
			f"STDERR:\n{result.stderr}"
		)

		# Check output exists
		run_dir = os.path.join(output_dir, "test_run")
		parquet_path = os.path.join(run_dir, "kmer_data.parquet")
		assert os.path.exists(parquet_path)

		# Load and validate
		df = pd.read_parquet(parquet_path)
		assert set(df.columns) == {"distance", "ig_score", "kmer",
		                           "motif_class"}
		assert len(df) > 0
		assert df["distance"].min() >= 1
		assert df["distance"].max() <= 30
		assert df["ig_score"].dtype == np.float32

		# All k-mers should be length 5
		kmer_lengths = df["kmer"].str.len().unique()
		assert list(kmer_lengths) == [5]

		# All k-mers should be canonical (kmer <= its RC)
		for kmer in df["kmer"].unique():
			rc = reverse_complement(kmer)
			assert kmer <= rc, f"{kmer} is not canonical (RC={rc})"

		# Should have both flanks contributing (2 entries per locus per
		# distance)
		n_loci = 20
		max_d = 30
		expected_rows = n_loci * 2 * max_d  # 2 flanks
		assert len(df) == expected_rows

		# Motif class should be AC (synthetic data uses AC repeat)
		assert set(df["motif_class"].unique()) == {"AC"}

		# Meta should exist
		meta_path = os.path.join(run_dir, "meta.json")
		assert os.path.exists(meta_path)

	def test_pred_diff_filter(self, tmp_path):
		"""Verify pred_diff filter reduces locus count."""
		npz_path = make_synthetic_npz(str(tmp_path), n_loci=20, n_flank=50)

		# Run with strict filter
		config = {
			"desc": "test_filtered",
			"attr_npz_path": npz_path,
			"k": 4,
			"max_distance": 10,
			"pred_diff_threshold": 10.0,  # Very strict — should filter most
		}
		config_path = os.path.join(str(tmp_path), "config.yaml")
		with open(config_path, "w") as f:
			yaml.dump(config, f)

		output_dir = os.path.join(str(tmp_path), "output")

		import subprocess
		result = subprocess.run(
			["python", os.path.join(os.path.dirname(__file__), "kmer_extract.py"),
			 "--config", config_path,
			 "--output_dir", output_dir],
			capture_output=True, text=True,
		)

		# Might fail because all samples filtered — that's expected
		# Either it produces fewer rows or raises ValueError
		run_dir = os.path.join(output_dir, "test_filtered")
		parquet_path = os.path.join(run_dir, "kmer_data.parquet")

		if result.returncode == 0 and os.path.exists(parquet_path):
			df = pd.read_parquet(parquet_path)
			# Should have fewer rows than unfiltered
			assert len(df) < 20 * 2 * 10
		else:
			# Expected: all samples filtered out
			assert "No samples passed filters" in result.stderr or \
       			"No locus pairs passed filters" in result.stderr


# ===========================================================================
# kmer_enrichment.py — unit tests
# ===========================================================================

class TestRunStratifiedPermutationTest:

	def test_basic_output_shape(self):
		"""Verify output arrays have correct shapes."""
		rng = np.random.default_rng(42)
		n_obs_per_dist = 100
		n_kmers = 10
		n_perms = 50

		scores_by_dist = {
			1: rng.normal(0, 1, n_obs_per_dist).astype(np.float32),
			2: rng.normal(0, 1, n_obs_per_dist).astype(np.float32),
		}
		kmer_ids_by_dist = {
			1: rng.integers(0, n_kmers, n_obs_per_dist),
			2: rng.integers(0, n_kmers, n_obs_per_dist),
		}

		result = run_stratified_permutation_test(
			scores_by_dist, kmer_ids_by_dist,
			n_kmers=n_kmers, n_permutations=n_perms, rng=rng,
			use_ranks=False, use_absolute=False,
		)

		assert result["observed_means"].shape == (n_kmers,)
		assert result["null_means"].shape == (n_perms, n_kmers)
		assert result["kmer_counts"].shape == (n_kmers,)

	def test_null_centered_on_observed_under_no_signal(self):
		"""Under null (random labels), observed should be near null mean."""
		rng = np.random.default_rng(123)
		n_obs = 500
		n_kmers = 5
		n_perms = 200

		# All scores drawn from same distribution — no k-mer effect
		scores = rng.normal(0, 1, n_obs).astype(np.float32)
		kmer_ids = rng.integers(0, n_kmers, n_obs)

		result = run_stratified_permutation_test(
			scores_by_distance={1: scores},
			kmer_ids_by_distance={1: kmer_ids},
			n_kmers=n_kmers, n_permutations=n_perms, rng=rng,
			use_ranks=False, use_absolute=False,
		)

		# Each k-mer's observed mean should be within the null range
		for ki in range(n_kmers):
			if result["kmer_counts"][ki] > 0:
				null_min = result["null_means"][:, ki].min()
				null_max = result["null_means"][:, ki].max()
				obs = result["observed_means"][ki]
				# Not a strict test — just check it's in a reasonable range
				null_range = null_max - null_min
				assert obs > null_min - null_range
				assert obs < null_max + null_range

	def test_ranks_mode(self):
		"""Verify rank mode produces rank-scale values."""
		rng = np.random.default_rng(42)
		n_obs = 50
		n_kmers = 3

		scores = rng.normal(0, 1, n_obs).astype(np.float32)
		kmer_ids = rng.integers(0, n_kmers, n_obs)

		result = run_stratified_permutation_test(
			scores_by_distance={1: scores},
			kmer_ids_by_distance={1: kmer_ids},
			n_kmers=n_kmers, n_permutations=10, rng=rng,
			use_ranks=True, use_absolute=False,
		)

		# With ranks, values should be in [1, n_obs] range
		for ki in range(n_kmers):
			if result["kmer_counts"][ki] > 0:
				assert result["observed_means"][ki] >= 1.0
				assert result["observed_means"][ki] <= n_obs

	def test_absolute_mode(self):
		"""Verify absolute mode produces non-negative observed means."""
		rng = np.random.default_rng(42)
		n_obs = 50
		n_kmers = 3

		# Include negative scores
		scores = rng.normal(0, 1, n_obs).astype(np.float32)
		kmer_ids = rng.integers(0, n_kmers, n_obs)

		result = run_stratified_permutation_test(
			scores_by_distance={1: scores},
			kmer_ids_by_distance={1: kmer_ids},
			n_kmers=n_kmers, n_permutations=10, rng=rng,
			use_ranks=False, use_absolute=True,
		)

		for ki in range(n_kmers):
			if result["kmer_counts"][ki] > 0:
				assert result["observed_means"][ki] >= 0


class TestComputePvalues:

	def test_extreme_kmer_detected(self):
		"""A k-mer with observed mean far above null should get low p."""
		n_kmers = 3
		n_perms = 1000

		observed = np.array([0.0, 0.0, 5.0])  # k-mer 2 is extreme
		null = np.random.default_rng(42).normal(
			0, 0.5, size=(n_perms, n_kmers)
		).astype(np.float32)
		counts = np.array([100.0, 100.0, 100.0])

		res = compute_pvalues(observed, null, counts,
		                      min_count=10, use_absolute=False)

		# k-mer 2 should have very low p-value
		kmer2_row = res[res["kmer_id"] == 2].iloc[0]
		assert kmer2_row["p_value"] < 0.01
		assert kmer2_row["effect_direction"] == "higher"

	def test_null_kmer_not_significant(self):
		"""A k-mer at the null center should have high p-value."""
		n_kmers = 2
		n_perms = 1000

		rng = np.random.default_rng(42)
		null = rng.normal(0, 1, size=(n_perms, n_kmers)).astype(np.float32)
		observed = np.array([0.0, 0.0])  # right at center
		counts = np.array([100.0, 100.0])

		res = compute_pvalues(observed, null, counts,
		                      min_count=10, use_absolute=False)

		for _, row in res.iterrows():
			assert row["p_value"] > 0.1

	def test_min_count_filter(self):
		"""K-mers below min_count should be excluded."""
		observed = np.array([1.0, 1.0])
		null = np.zeros((100, 2), dtype=np.float32)
		counts = np.array([5.0, 50.0])

		res = compute_pvalues(observed, null, counts,
		                      min_count=30, use_absolute=False)

		# Only k-mer 1 (count=50) should be tested
		assert len(res) == 1
		assert res.iloc[0]["kmer_id"] == 1

	def test_pvalue_never_zero(self):
		"""P-values should never be exactly zero (Phipson & Smyth)."""
		observed = np.array([100.0])  # Extremely far from null
		null = np.zeros((1000, 1), dtype=np.float32)
		counts = np.array([50.0])

		res = compute_pvalues(observed, null, counts,
		                      min_count=10, use_absolute=False)

		assert res.iloc[0]["p_value"] > 0

	def test_pvalue_upper_bound(self):
		"""P-values should not exceed 1."""
		observed = np.array([0.0])
		null = np.zeros((100, 1), dtype=np.float32)
		counts = np.array([50.0])

		res = compute_pvalues(observed, null, counts,
		                      min_count=10, use_absolute=False)

		assert res.iloc[0]["p_value"] <= 1.0

	def test_absolute_mode_one_sided(self):
		"""In absolute mode, a k-mer with high |IG| should be detected."""
		observed = np.array([5.0])
		rng = np.random.default_rng(42)
		null = rng.uniform(0, 1, size=(1000, 1)).astype(np.float32)
		counts = np.array([100.0])

		res = compute_pvalues(observed, null, counts,
		                      min_count=10, use_absolute=True)

		assert res.iloc[0]["p_value"] < 0.01
		assert res.iloc[0]["effect_direction"] == "higher"


# ===========================================================================
# kmer_enrichment.py — signal detection test
# ===========================================================================

class TestSignalDetection:

	def test_planted_signal_detected(self):
		"""Plant a k-mer with consistently high scores; verify detection."""
		rng = np.random.default_rng(42)
		n_obs_per_dist = 200
		n_distances = 10
		n_kmers = 20
		enriched_kmer = 7  # This k-mer will have elevated scores

		scores_by_dist = {}
		kmer_ids_by_dist = {}

		for d in range(1, n_distances + 1):
			# Base scores: normal noise
			scores = rng.normal(0, 1, n_obs_per_dist).astype(np.float32)
			kmer_ids = rng.integers(0, n_kmers, n_obs_per_dist)

			# Plant signal: wherever kmer 7 appears, add a positive shift
			signal_mask = kmer_ids == enriched_kmer
			scores[signal_mask] += 2.0

			scores_by_dist[d] = scores
			kmer_ids_by_dist[d] = kmer_ids

		result = run_stratified_permutation_test(
			scores_by_distance=scores_by_dist,
			kmer_ids_by_distance=kmer_ids_by_dist,
			n_kmers=n_kmers, n_permutations=500, rng=rng,
			use_ranks=False, use_absolute=False,
		)

		res_df = compute_pvalues(
			result["observed_means"], result["null_means"],
			result["kmer_counts"], min_count=5, use_absolute=False,
		)

		# The enriched k-mer should be significant
		enriched_row = res_df[res_df["kmer_id"] == enriched_kmer]
		assert len(enriched_row) == 1
		assert enriched_row.iloc[0]["p_value"] < 0.01
		assert enriched_row.iloc[0]["effect_direction"] == "higher"

	def test_planted_signal_detected_with_ranks(self):
		"""Same test but with rank-based scoring."""
		rng = np.random.default_rng(42)
		n_obs_per_dist = 200
		n_distances = 10
		n_kmers = 20
		enriched_kmer = 7

		scores_by_dist = {}
		kmer_ids_by_dist = {}

		for d in range(1, n_distances + 1):
			scores = rng.normal(0, 1, n_obs_per_dist).astype(np.float32)
			kmer_ids = rng.integers(0, n_kmers, n_obs_per_dist)
			signal_mask = kmer_ids == enriched_kmer
			scores[signal_mask] += 2.0

			scores_by_dist[d] = scores
			kmer_ids_by_dist[d] = kmer_ids

		result = run_stratified_permutation_test(
			scores_by_distance=scores_by_dist,
			kmer_ids_by_distance=kmer_ids_by_dist,
			n_kmers=n_kmers, n_permutations=500, rng=rng,
			use_ranks=True, use_absolute=False,
		)

		res_df = compute_pvalues(
			result["observed_means"], result["null_means"],
			result["kmer_counts"], min_count=5, use_absolute=False,
		)

		enriched_row = res_df[res_df["kmer_id"] == enriched_kmer]
		assert len(enriched_row) == 1
		assert enriched_row.iloc[0]["p_value"] < 0.01

	def test_no_false_positives_under_null(self):
		"""Under pure null (no signal), most k-mers should not be significant."""
		rng = np.random.default_rng(99)
		n_obs_per_dist = 200
		n_distances = 5
		n_kmers = 20

		scores_by_dist = {}
		kmer_ids_by_dist = {}

		for d in range(1, n_distances + 1):
			scores_by_dist[d] = rng.normal(
				0, 1, n_obs_per_dist
			).astype(np.float32)
			kmer_ids_by_dist[d] = rng.integers(0, n_kmers, n_obs_per_dist)

		result = run_stratified_permutation_test(
			scores_by_distance=scores_by_dist,
			kmer_ids_by_distance=kmer_ids_by_dist,
			n_kmers=n_kmers, n_permutations=500, rng=rng,
			use_ranks=True, use_absolute=False,
		)

		res_df = compute_pvalues(
			result["observed_means"], result["null_means"],
			result["kmer_counts"], min_count=5, use_absolute=False,
		)

		# At most ~5% should have p < 0.05 under null
		n_sig = (res_df["p_value"] < 0.05).sum()
		n_tested = len(res_df)
		# Allow some slack: up to 20% false positive rate
		# (permutation tests can be slightly liberal with few permutations)
		assert n_sig / max(n_tested, 1) < 0.20, (
			f"{n_sig}/{n_tested} significant at p<0.05 under null"
		)