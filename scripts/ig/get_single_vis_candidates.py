"""Find good candidate loci for single-example IG attribution visualization.

Loads IG attribution results, pairs forward and reverse complement samples
by HipSTR_name, and ranks loci by a composite score combining:
  1. Prediction accuracy  — |prediction - label| in copy-number space (lower is better)
  2. Convergence delta    — relative convergence delta (lower is better)
  3. Prediction difference — |F(input) - F(baseline)| (higher is better;
                             indicates informative flanking sequence)

Per-locus aggregation takes the *worse* of the forward/RC pair for each
criterion (max error, max delta, min pred_diff), since both strands will
be visualized.
"""

import numpy as np
import json
import os

# ============================================================================
# Configuration
# ============================================================================

ATTR_DIR = "output/str2/v1/2000_p128_300steps_rt"

# Hard filters: loci failing these are excluded entirely.
# Set to None to disable a filter.
MIN_PRED_DIFF = 0.05       # minimum abs prediction difference (both strands)
MAX_REL_DELTA = None        # maximum relative convergence delta (both strands)
MAX_PRED_ERROR = None       # maximum abs prediction error (both strands)

# Composite score weights (will be normalized to sum to 1).
# Each metric is converted to a percentile rank among loci that pass
# hard filters, then combined as: score = sum(weight_i * rank_i).
# Higher composite score = better candidate.
W_PRED_ERROR = 1.0    # weight for prediction accuracy (low error)
W_REL_DELTA = 1.0     # weight for convergence quality (low delta)
W_PRED_DIFF = 5.0     # weight for prediction difference (high diff)

# Number of top candidates to display
N_TOP = 20

# ============================================================================
# Load data
# ============================================================================

data = np.load(os.path.join(ATTR_DIR, "attributions.npz"), allow_pickle=True)

with open(os.path.join(ATTR_DIR, "meta.json"), "r") as f:
	meta = json.load(f)

raw_predictions = data["raw_predictions"]
raw_baseline_predictions = data["raw_baseline_predictions"]
relative_deltas = data["relative_convergence_deltas"]
labels = data["labels"]
rev_comp = data["rev_comp"]
hipstr_names = data["hipstr_names"]

n_samples = len(labels)

# ============================================================================
# Per-sample metrics
# ============================================================================

# Labels are in copy-number space; predictions are in log1p space.
# Compute error in copy-number space.
pred_errors = np.abs(np.expm1(raw_predictions) - labels)

raw_pred_diffs = raw_predictions - raw_baseline_predictions
abs_pred_diffs = np.abs(raw_pred_diffs)

print(f"Loaded {n_samples} samples")
print(f"  Forward: {np.sum(~rev_comp)}, RC: {np.sum(rev_comp)}")
print(f"  Unique loci: {len(np.unique(hipstr_names))}")

# ============================================================================
# Group by locus: pair forward and RC
# ============================================================================

unique_names = np.unique(hipstr_names)
n_loci = len(unique_names)

# Build lookup: hipstr_name -> {fwd_idx, rc_idx}
locus_pairs = {}
for i in range(n_samples):
	name = str(hipstr_names[i])
	if name not in locus_pairs:
		locus_pairs[name] = {"fwd": None, "rc": None}
	if rev_comp[i]:
		locus_pairs[name]["rc"] = i
	else:
		locus_pairs[name]["fwd"] = i

# Validate: every locus must have exactly one forward and one RC
incomplete = [name for name, pair in locus_pairs.items()
              if pair["fwd"] is None or pair["rc"] is None]
if incomplete:
	print(f"\nWARNING: {len(incomplete)} loci missing a strand:")
	for name in incomplete[:5]:
		pair = locus_pairs[name]
		print(f"  {name}: fwd={'present' if pair['fwd'] is not None else 'MISSING'}, "
		      f"rc={'present' if pair['rc'] is not None else 'MISSING'}")
	if len(incomplete) > 5:
		print(f"  ... and {len(incomplete) - 5} more")

# Keep only complete pairs
complete_names = [name for name, pair in locus_pairs.items()
                  if pair["fwd"] is not None and pair["rc"] is not None]
print(f"\nComplete locus pairs: {len(complete_names)}")

# ============================================================================
# Per-locus aggregated metrics (worst of fwd/RC)
# ============================================================================

locus_metrics = []

for name in complete_names:
	fi = locus_pairs[name]["fwd"]
	ri = locus_pairs[name]["rc"]

	locus_pred_error = max(pred_errors[fi], pred_errors[ri])
	locus_rel_delta = max(relative_deltas[fi], relative_deltas[ri])
	locus_pred_diff = min(abs_pred_diffs[fi], abs_pred_diffs[ri])

	locus_metrics.append({
		"name": name,
		"fwd_idx": fi,
		"rc_idx": ri,
		"pred_error": locus_pred_error,
		"rel_delta": locus_rel_delta,
		"pred_diff": locus_pred_diff,
		# Keep per-strand values for the summary table
		"fwd_pred_error": pred_errors[fi],
		"rc_pred_error": pred_errors[ri],
		"fwd_rel_delta": relative_deltas[fi],
		"rc_rel_delta": relative_deltas[ri],
		"fwd_pred_diff": abs_pred_diffs[fi],
		"rc_pred_diff": abs_pred_diffs[ri],
		"label": labels[fi],
		"fwd_prediction": raw_predictions[fi],
		"rc_prediction": raw_predictions[ri],
	})

# ============================================================================
# Apply hard filters
# ============================================================================

filtered = locus_metrics

if MIN_PRED_DIFF is not None:
	before = len(filtered)
	filtered = [m for m in filtered if m["pred_diff"] > MIN_PRED_DIFF]
	print(f"Filter pred_diff > {MIN_PRED_DIFF}: {before} -> {len(filtered)}")

if MAX_REL_DELTA is not None:
	before = len(filtered)
	filtered = [m for m in filtered if m["rel_delta"] <= MAX_REL_DELTA]
	print(f"Filter rel_delta <= {MAX_REL_DELTA}: {before} -> {len(filtered)}")

if MAX_PRED_ERROR is not None:
	before = len(filtered)
	filtered = [m for m in filtered if m["pred_error"] <= MAX_PRED_ERROR]
	print(f"Filter pred_error <= {MAX_PRED_ERROR}: {before} -> {len(filtered)}")

n_passed = len(filtered)
print(f"\nLoci passing all filters: {n_passed}")

if n_passed == 0:
	raise ValueError("No loci passed filters. Adjust thresholds.")

# ============================================================================
# Composite ranking via percentile ranks
# ============================================================================

# Extract metric arrays for ranking
arr_pred_error = np.array([m["pred_error"] for m in filtered])
arr_rel_delta = np.array([m["rel_delta"] for m in filtered])
arr_pred_diff = np.array([m["pred_diff"] for m in filtered])


def percentile_ranks(values, higher_is_better=True):
	"""Convert values to percentile ranks in [0, 1]. Higher rank = better."""
	n = len(values)
	if n == 1:
		return np.array([1.0])
	order = np.argsort(values)
	ranks = np.empty(n)
	ranks[order] = np.arange(n)
	# Normalize to [0, 1]
	ranks /= (n - 1)
	if not higher_is_better:
		ranks = 1.0 - ranks
	return ranks


rank_pred_error = percentile_ranks(arr_pred_error, higher_is_better=False)
rank_rel_delta = percentile_ranks(arr_rel_delta, higher_is_better=False)
rank_pred_diff = percentile_ranks(arr_pred_diff, higher_is_better=True)

# Normalize weights
total_w = W_PRED_ERROR + W_REL_DELTA + W_PRED_DIFF
w_err = W_PRED_ERROR / total_w
w_delta = W_REL_DELTA / total_w
w_diff = W_PRED_DIFF / total_w

composite = (w_err * rank_pred_error +
             w_delta * rank_rel_delta +
             w_diff * rank_pred_diff)

# Sort by composite score descending
sort_order = np.argsort(-composite)

# ============================================================================
# Display top candidates
# ============================================================================

n_show = min(N_TOP, n_passed)
print(f"\n{'='*100}")
print(f"Top {n_show} candidate loci (weights: error={w_err:.2f}, "
      f"delta={w_delta:.2f}, diff={w_diff:.2f})")
print(f"{'='*100}")

header = (f"{'Rank':>4}  {'Score':>5}  {'HipSTR Name':<30}  "
          f"{'Label':>7}  {'PredErr':>8}  {'RelDelta':>9}  {'PredDiff':>9}  "
          f"{'FwdIdx':>6}  {'RcIdx':>6}")
print(header)
print("-" * len(header))

for rank, idx in enumerate(sort_order[:n_show], 1):
	m = filtered[idx]
	print(f"{rank:>4}  {composite[idx]:>5.3f}  {m['name']:<30}  "
	      f"{m['label']:>7.2f}  {m['pred_error']:>8.3f}  "
	      f"{m['rel_delta']:>9.4f}  {m['pred_diff']:>9.4f}  "
	      f"{m['fwd_idx']:>6}  {m['rc_idx']:>6}")

# ============================================================================
# Store ranked results for part 2 (interactive selection)
# ============================================================================

ranked_loci = [filtered[idx] for idx in sort_order]
ranked_scores = composite[sort_order]

print(f"\nResults available as `ranked_loci` (list of dicts) and "
      f"`ranked_scores` (array)")
print(f"Access top locus: ranked_loci[0]['name'], indices: "
      f"ranked_loci[0]['fwd_idx'], ranked_loci[0]['rc_idx']")