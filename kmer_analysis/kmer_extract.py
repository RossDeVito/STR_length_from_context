"""Extract k-mer IG attribution data for downstream enrichment analysis.

Loads IG attribution results, pairs forward and reverse complement samples,
normalizes attributions, averages forward and RC attributions in genomic
coordinates, extracts canonical k-mers at each flanking position, and
outputs a parquet file for downstream permutation testing.

Normalization and the forward/RC combination are the shared ones in
seq_models/ig_attribution_norm.py; norm_scheme selects between "l1_mass"
(each orientation rescaled to unit absolute attribution mass across the
flanks) and "pred_diff" (each orientation divided by the ABSOLUTE
prediction shift). Scores stay signed through the average. This used to
divide by the SIGNED prediction difference, which flipped the sign of
every sample predicted below its baseline, so a "positive" k-mer score
meant "in the direction this locus moved from baseline" rather than
"pushes the prediction up" -- and the two meanings were mixed together in
one output column.

Each row in the output represents one k-mer occurrence at one locus,
with its distance from the STR, normalized IG score (mean over k positions),
canonical k-mer identity, and STR motif class.

Distance is defined as the number of bases between the k-mer's closest
base and the STR boundary. A k-mer at distance 1 is directly adjacent
to the STR and extends k-1 bases further away. Both left and right flanks
are pooled by absolute distance.

Args:
    --config: Path to YAML config file.
    --output_dir: Directory to save outputs.

Config keys:
    desc (str): Output subdirectory name.
    attr_npz_path (str): Path to IG attribution NPZ file.
    k (int): K-mer length (e.g. 4, 5, 6).
    max_distance (int, optional): Maximum distance from STR boundary to
        include. Limits output size and is recommended for large flanks.
        Default: use full flank length minus k + 1.
    pred_diff_threshold (float, optional): Minimum |raw_pred - raw_baseline|
        to include a sample. Samples below this have uninformative flanking
        sequence. Default: 0.0 (no filter).
    rel_delta_threshold (float, optional): Maximum relative convergence
        delta to include a sample. Default: None (no filter).
    norm_scheme (str, optional): Per-orientation attribution
        normalization, "l1_mass" or "pred_diff". Default: "l1_mass".
        See seq_models/ig_attribution_norm.py.
"""

import argparse
import datetime
import os
import json

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import seq_models.ig_attribution_norm as ign


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq):
	"""Return the reverse complement of a DNA sequence."""
	return seq.translate(COMPLEMENT)[::-1]


def canonical_kmer(kmer):
	"""Return the lexicographically smaller of a k-mer and its RC."""
	rc = reverse_complement(kmer)
	return min(kmer, rc)


# STR motif canonicalization (matches get_mean_baseline_performance.py)
def canonicalize_motif(motif):
	"""Canonicalize an STR motif so it is invariant to rotation and strand.

	Returns the lexicographically smallest string over all rotations of the
	motif and all rotations of its reverse complement (e.g. "AC", "CA",
	"GT", "TG" all collapse to "AC").
	"""
	candidates = []
	for s in (motif, reverse_complement(motif)):
		for i in range(len(s)):
			candidates.append(s[i:] + s[:i])
	return min(candidates)


def detect_repeat_unit(seq, max_period=6):
	"""Detect the smallest period (<= max_period) that tiles `seq`."""
	for period in range(1, min(max_period, len(seq)) + 1):
		unit = seq[:period]
		if all(seq[i] == unit[i % period] for i in range(len(seq))):
			return unit
	return seq[:max_period]


def extract_motif_from_sequence(sequence, lf_end, n_str_bp):
	"""Extract and canonicalize the STR motif from a forward sequence.

	Reads the first few non-prompt bases of the left STR region to
	determine the repeat unit.

	Args:
		sequence: Full sequence string from the npz.
		lf_end: Array index where left flank ends / left STR begins.
		n_str_bp: Number of STR bases stored on each side.

	Returns:
		Canonicalized motif string (e.g. "A", "AC", "AAT").
	"""
	str_region = sequence[lf_end:lf_end + n_str_bp]
	str_clean = str_region.replace("X", "").replace("x", "")

	if len(str_clean) == 0:
		return "unknown"

	repeat_unit = detect_repeat_unit(str_clean)

	return canonicalize_motif(repeat_unit)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

	__spec__ = None

	parser = argparse.ArgumentParser(
		description="Extract k-mer IG attribution data for enrichment analysis."
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

	attr_npz_path = config["attr_npz_path"]
	k = config["k"]
	max_distance = config.get("max_distance", None)
	pred_diff_threshold = config.get("pred_diff_threshold", 0.0)
	rel_delta_threshold = config.get("rel_delta_threshold", None)
	norm_scheme = ign.check_scheme(
		config.get("norm_scheme", ign.DEFAULT_NORM_SCHEME)
	)

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
	# Load attribution data
	# ------------------------------------------------------------------
	print(f"Loading attributions from {attr_npz_path}")
	attr_dir = os.path.dirname(attr_npz_path)
	data = np.load(attr_npz_path, allow_pickle=True)

	meta_path = os.path.join(attr_dir, "meta.json")
	with open(meta_path, "r") as f:
		meta = json.load(f)

	layout = meta["sequence_layout"]

	attributions = data["attributions"]
	sequences = data["sequences"]
	raw_predictions = data["raw_predictions"]
	raw_baseline_predictions = data["raw_baseline_predictions"]
	relative_deltas = data["relative_convergence_deltas"]
	rev_comp = data["rev_comp"]
	hipstr_names = data["hipstr_names"]

	# ------------------------------------------------------------------
	# Sequence layout
	# ------------------------------------------------------------------
	flanks = ign.FlankLayout.from_layout(layout)

	n_prefix = flanks.n_prefix
	n_left_flank = flanks.n_left_flank
	n_str_bp = flanks.n_left_str
	n_str_prompt = flanks.n_str_prompt
	n_right_flank = flanks.n_right_flank

	lf_start = flanks.lf_start
	lf_end = flanks.lf_end
	rf_start = flanks.rf_start
	rf_end = flanks.rf_end

	print(f"Sequence layout: prefix={n_prefix}, "
	      f"left_flank={n_left_flank}, str_bp={n_str_bp}, "
	      f"str_prompt={n_str_prompt}, right_flank={n_right_flank}")
	print(f"K-mer length: {k}")

	# ------------------------------------------------------------------
	# Compute valid distance range
	# ------------------------------------------------------------------
	max_valid_distance = min(n_left_flank, n_right_flank) - k + 1
	if max_distance is not None:
		max_distance = min(max_distance, max_valid_distance)
	else:
		max_distance = max_valid_distance

	if max_distance < 1:
		raise ValueError(
			f"No valid distances: flank length {n_left_flank}, k={k}. "
			f"Need flank >= k."
		)

	print(f"Distance range: 1 to {max_distance}")

	# ------------------------------------------------------------------
	# Filter samples
	# ------------------------------------------------------------------
	raw_pred_diffs = raw_predictions - raw_baseline_predictions

	samp_filter = ign.filter_samples(
		raw_pred_diffs, relative_deltas,
		pred_diff_threshold, rel_delta_threshold,
	)
	mask = samp_filter.mask
	n_kept = samp_filter.n_kept

	print(f"\nFiltering: {n_kept} samples kept, {samp_filter.n_filtered} "
	      f"filtered (of {samp_filter.n_total} total)")
	print(f"  Excluded by |pred_diff| <= {pred_diff_threshold}: "
	      f"{samp_filter.n_excluded_pred_diff}")
	if rel_delta_threshold is not None:
		print(f"  Excluded by rel_delta > {rel_delta_threshold}: "
		      f"{samp_filter.n_excluded_rel_delta}")

	if n_kept == 0:
		raise ValueError("No samples passed filters. Adjust thresholds.")

	# ------------------------------------------------------------------
	# Pair forward and RC samples by locus
	# ------------------------------------------------------------------
	fwd_indices = np.where(~rev_comp)[0]
	rc_indices = np.where(rev_comp)[0]

	assert len(fwd_indices) == len(rc_indices), (
		f"Unequal forward ({len(fwd_indices)}) and RC ({len(rc_indices)}) "
		f"sample counts."
	)

	# Build locus pairs: match by HipSTR name
	fwd_lookup = {}
	for idx in fwd_indices:
		name = str(hipstr_names[idx])
		fwd_lookup[name] = idx

	rc_lookup = {}
	for idx in rc_indices:
		name = str(hipstr_names[idx])
		rc_lookup[name] = idx

	# Pair and filter: both strands must pass
	locus_names = []
	kept_fwd_idx = []
	kept_rc_idx = []

	for name in fwd_lookup:
		if name not in rc_lookup:
			continue
		fi = fwd_lookup[name]
		ri = rc_lookup[name]
		if mask[fi] and mask[ri]:
			locus_names.append(name)
			kept_fwd_idx.append(fi)
			kept_rc_idx.append(ri)

	kept_fwd_idx = np.array(kept_fwd_idx)
	kept_rc_idx = np.array(kept_rc_idx)
	n_loci = len(locus_names)

	print(f"\nPaired loci (both strands pass filters): {n_loci}")

	if n_loci == 0:
		raise ValueError("No locus pairs passed filters.")

	# ------------------------------------------------------------------
	# Normalize attributions (signed, shared scheme) and average forward
	# with flipped-RC in genomic coordinates
	# ------------------------------------------------------------------
	print(f"Normalization scheme: {norm_scheme}")

	fwd_norm = ign.normalize_signed(
		attributions[kept_fwd_idx], raw_pred_diffs[kept_fwd_idx],
		flanks, norm_scheme,
	)
	rc_norm = ign.normalize_signed(
		attributions[kept_rc_idx], raw_pred_diffs[kept_rc_idx],
		flanks, norm_scheme,
	)

	avg = ign.combine_fwd_rc(fwd_norm, rc_norm, flanks)
	avg_left = avg[:, lf_start:lf_end]
	avg_right = avg[:, rf_start:rf_end]

	print(f"Forward/RC averaging complete. "
	      f"Left flank shape: {avg_left.shape}, "
	      f"Right flank shape: {avg_right.shape}")

	# ------------------------------------------------------------------
	# Extract STR motif class for each locus (from forward sequence)
	# ------------------------------------------------------------------
	motif_classes = []
	for fi in kept_fwd_idx:
		seq = str(sequences[fi])
		motif = extract_motif_from_sequence(seq, lf_end, n_str_bp)
		motif_classes.append(motif)

	motif_counts = {}
	for m in motif_classes:
		motif_counts[m] = motif_counts.get(m, 0) + 1
	print(f"\nSTR motif class distribution:")
	for m, c in sorted(motif_counts.items()):
		print(f"  {m}: {c} loci")

	# ------------------------------------------------------------------
	# Extract k-mers and build output table
	# ------------------------------------------------------------------
	print(f"\nExtracting {k}-mers at distances 1-{max_distance} "
	      f"from both flanks...")

	rows = []

	for locus_i in tqdm(range(n_loci), desc="Processing loci"):
		fwd_seq = str(sequences[kept_fwd_idx[locus_i]])
		motif = motif_classes[locus_i]

		for d in range(1, max_distance + 1):

			# --- Left flank ---
			# K-mer closest base at distance d, extending away from STR.
			# In array coords within left flank (0 = distal, n-1 = proximal):
			#   proximal position = n_left_flank - d
			#   distal position   = n_left_flank - d - k + 1
			lf_prox = n_left_flank - d
			lf_dist = n_left_flank - d - k + 1

			# Sequence from full forward string
			seq_start = lf_start + lf_dist
			seq_end = lf_start + lf_prox + 1
			kmer_seq = fwd_seq[seq_start:seq_end]

			# IG score: mean over k positions in averaged left flank
			ig_score = float(avg_left[locus_i, lf_dist:lf_prox + 1].mean())

			canon = canonical_kmer(kmer_seq)
			rows.append((d, ig_score, canon, motif))

			# --- Right flank ---
			# K-mer closest base at distance d, extending away from STR.
			# In array coords within right flank (0 = proximal, n-1 = distal):
			#   proximal position = d - 1
			#   distal position   = d - 1 + k - 1 = d + k - 2
			rf_prox = d - 1
			rf_dist_end = d + k - 2

			# Sequence from full forward string
			seq_start = rf_start + rf_prox
			seq_end = rf_start + rf_dist_end + 1
			kmer_seq = fwd_seq[seq_start:seq_end]

			# IG score: mean over k positions in averaged right flank
			ig_score = float(avg_right[locus_i, rf_prox:rf_dist_end + 1].mean())

			canon = canonical_kmer(kmer_seq)
			rows.append((d, ig_score, canon, motif))

	# ------------------------------------------------------------------
	# Build DataFrame and save
	# ------------------------------------------------------------------
	print(f"\nBuilding DataFrame ({len(rows)} rows)...")
	df = pd.DataFrame(rows, columns=["distance", "ig_score", "kmer",
	                                  "motif_class"])

	# Convert kmer and motif_class to categorical for space efficiency
	df["kmer"] = df["kmer"].astype("category")
	df["motif_class"] = df["motif_class"].astype("category")
	df["distance"] = df["distance"].astype(np.int16)
	df["ig_score"] = df["ig_score"].astype(np.float32)

	n_unique_kmers = df["kmer"].nunique()
	print(f"Unique canonical {k}-mers: {n_unique_kmers}")
	print(f"Distance range: {df['distance'].min()} - {df['distance'].max()}")

	parquet_path = os.path.join(run_dir, "kmer_data.parquet")
	df.to_parquet(parquet_path, index=False, engine="pyarrow")
	print(f"Saved to {parquet_path}")

	# ------------------------------------------------------------------
	# Save metadata
	# ------------------------------------------------------------------
	run_meta = {
		"config": config,
		"source_meta_path": meta_path,
		"sequence_layout": layout,
		"n_loci": n_loci,
		"n_samples_total": samp_filter.n_total,
		"n_samples_filtered": samp_filter.n_filtered,
		"norm_scheme": norm_scheme,
		"n_rows": len(rows),
		"n_unique_kmers": n_unique_kmers,
		"max_distance_used": max_distance,
		"motif_class_counts": motif_counts,
		"timestamp": datetime.datetime.now().isoformat(),
	}

	meta_out_path = os.path.join(run_dir, "meta.json")
	with open(meta_out_path, "w") as f:
		json.dump(run_meta, f, indent=4)
	print(f"Metadata saved to {meta_out_path}")

	print("\n--- Done ---")