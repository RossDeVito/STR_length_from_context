import numpy as np

data = np.load(
	"output/str2/v1/2000_dev_100steps_rt/attributions.npz",
	allow_pickle=True
)

# Access arrays by key
attributions = data["attributions"]          # (n_samples, seq_len)
input_ids = data["input_ids"]               # (n_samples, seq_len)
sequences = data["sequences"]               # (n_samples,) strings
predictions = data["predictions"]           # (n_samples,)
baseline_predictions = data["baseline_predictions"]  # (n_samples,)
labels = data["labels"]                     # (n_samples,)
deltas = data["convergence_deltas"]         # (n_samples,)
position_labels = data["position_labels"]   # (seq_len,) strings