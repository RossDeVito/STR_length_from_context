import numpy as np

data = np.load(
	"output/str2/dev/2000_dev_1000steps_rt/attributions.npz",
	allow_pickle=True
)

# Access arrays by key
attributions = data["attributions"]                          # (n_samples, seq_len)
input_ids = data["input_ids"]                                # (n_samples, seq_len)
sequences = data["sequences"]                                # (n_samples,) strings
predictions = data["predictions"]                            # (n_samples,)
baseline_predictions = data["baseline_predictions"]          # (n_samples,)
raw_predictions = data["raw_predictions"]                    # (n_samples,)
raw_baseline_predictions = data["raw_baseline_predictions"]  # (n_samples,)
labels = data["labels"]                                      # (n_samples,)
convergence_deltas = data["convergence_deltas"]              # (n_samples,)
relative_deltas = data["relative_convergence_deltas"]        # (n_samples,)
position_labels = data["position_labels"]                    # (seq_len,) strings
hipstr_names = data["hipstr_names"]                          # (n_samples,) strings
rev_comp = data["rev_comp"]                                  # (n_samples,) bools

# Raw prediction difference (in model output space, log space if log_transform)
raw_pred_diffs = np.abs(raw_predictions - raw_baseline_predictions)
abs_deltas = np.abs(convergence_deltas)

# ---------------------------------------------------------------------------
# Per-sample detail
# ---------------------------------------------------------------------------
# print("Per-sample breakdown:")
# print(f"{'idx':>4}  {'hipstr_name':>20}  {'raw_pred_diff':>13}  "
#       f"{'abs_delta':>10}  {'rel_delta':>10}  {'label':>6}")
# print("-" * 80)

# for i in range(len(convergence_deltas)):
# 	print(f"{i:4d}  {str(hipstr_names[i]):>20}  {raw_pred_diffs[i]:13.4f}  "
# 	      f"{abs_deltas[i]:10.4f}  {relative_deltas[i]:10.4f}  "
# 	      f"{labels[i]:6.1f}")

# ---------------------------------------------------------------------------
# Filter: samples where flanking sequence actually affects prediction
# ---------------------------------------------------------------------------
threshold = 0.05  # in raw (log) space
informative = raw_pred_diffs > threshold
uninformative = ~informative

n_informative = np.sum(informative)
n_uninformative = np.sum(uninformative)

print(f"\n{'='*80}")
print(f"Filtering with raw_pred_diff threshold = {threshold}")
print(f"  Informative (flanking matters):       {n_informative} / {len(labels)}")
print(f"  Uninformative (flanking ~no effect):  {n_uninformative} / {len(labels)}")

# ---------------------------------------------------------------------------
# Stats for informative samples
# ---------------------------------------------------------------------------
if n_informative > 0:
	inf_abs = abs_deltas[informative]
	inf_rel = relative_deltas[informative]
	inf_diffs = raw_pred_diffs[informative]

	print(f"\n--- Informative samples (raw_pred_diff > {threshold}) ---")
	print(f"  Raw pred diff  -- "
	      f"mean: {np.mean(inf_diffs):.4f}, "
	      f"median: {np.median(inf_diffs):.4f}, "
	      f"range: [{np.min(inf_diffs):.4f}, {np.max(inf_diffs):.4f}]")
	print(f"  Absolute delta -- "
	      f"mean: {np.mean(inf_abs):.4f}, "
	      f"median: {np.median(inf_abs):.4f}, "
	      f"max: {np.max(inf_abs):.4f}")
	print(f"  Relative delta -- "
	      f"mean: {np.mean(inf_rel):.4f}, "
	      f"median: {np.median(inf_rel):.4f}, "
	      f"max: {np.max(inf_rel):.4f}")
	print(f"  Samples > 5% relative delta: "
	      f"{np.mean(inf_rel > 0.05) * 100:.1f}%")
	print(f"  Samples > 1% relative delta: "
	      f"{np.mean(inf_rel > 0.01) * 100:.1f}%")

# ---------------------------------------------------------------------------
# Stats for uninformative samples
# ---------------------------------------------------------------------------
if n_uninformative > 0:
	uninf_abs = abs_deltas[uninformative]
	uninf_rel = relative_deltas[uninformative]
	uninf_diffs = raw_pred_diffs[uninformative]

	print(f"\n--- Uninformative samples (raw_pred_diff <= {threshold}) ---")
	print(f"  Raw pred diff  -- "
	      f"mean: {np.mean(uninf_diffs):.4f}, "
	      f"median: {np.median(uninf_diffs):.4f}, "
	      f"range: [{np.min(uninf_diffs):.4f}, {np.max(uninf_diffs):.4f}]")
	print(f"  Absolute delta -- "
	      f"mean: {np.mean(uninf_abs):.4f}, "
	      f"median: {np.median(uninf_abs):.4f}, "
	      f"max: {np.max(uninf_abs):.4f}")
	print(f"  Relative delta -- "
	      f"mean: {np.mean(uninf_rel):.4f}, "
	      f"median: {np.median(uninf_rel):.4f}, "
	      f"max: {np.max(uninf_rel):.4f}")
	print(f"  (High relative deltas here are expected -- "
	      f"small denominator, not poor convergence)")