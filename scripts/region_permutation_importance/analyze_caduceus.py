""" Analyze Caduceus region permutation importance results.

Consumes the per-task, native-space NPZ written by
seq_models/caduceus/region_permutation_importance.py and produces three views,
per target (length and/or heterozygosity):

  View 1 - Per-orientation (model-position) bias
    Aggregates by the model's INPUT position (left vs right flank) across all
    orientation rows. Shows whether the model treats its left input differently
    from its right input (an up-vs-downstream POSITIONAL bias). This is the same
    framing as the HyenaDNA analyzer.

  View 2 - RC-averaged genomic
    Pairs each locus's forward and reverse-complement rows by ``id`` and averages
    their importances after aligning to genomic coordinates (a flank-label swap at
    region granularity), removing the orientation artifact. Gives the true
    biological upstream/downstream signal. A length-vs-heterozygosity overlay
    compares the spatial decay of the two targets (merged by ``id`` across their
    source runs).

  View 3 - Long-range reach + clustering
    From the RC-averaged per-locus distance profiles (a "time series" of
    importance vs distance), computes per-locus reach metrics and clusters the
    profile shapes (agglomerative with dendrogram cuts, KMeans, optional DBSCAN)
    to surface loci whose importance reaches farther from the STR.

Configuration is via the editable constants below (no argparse), mirroring the
original permutation_analysis.py. Each target points at a source run via
RUN_DIRS; the two targets may come from one multi-task run or two separate
(possibly single-task) model runs.

Predictions in the NPZ are already in native units (meta predictions_space ==
"native"); no log/transform inversion is applied here.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sk_metrics
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import trange


# ===========================================================================
# Configuration (edit me)
# ===========================================================================

# task -> source run dir. One run can supply multiple tasks (multi-task model),
# or point each task at a different (possibly single-task) model's run.
RUN_DIRS = {
	"length":    "output/caduceus_v0/str4/str4_f5000_fiv_v4",
	"variation": "output/caduceus_v0/str4/str4_f5000_fiv_v4",
}

# Where to write analysis output. None -> {shared run dir}/analysis when all tasks
# share one run; otherwise a path is REQUIRED (raises if the runs differ).
OUT_DIR = None

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
SEED = 42

# Clustering
N_CLUSTERS = 3
CLUSTER_METHOD = "both"      # "agglomerative" | "kmeans" | "dbscan" | "both"
AGG_DIST_THRESHOLD = None    # optional ward-distance cut for agglomerative; None
                             # -> use N_CLUSTERS (maxclust). Set a float to cut by
                             # cophenetic distance instead.
DBSCAN_EPS = 0.5             # only used when DBSCAN is requested
DBSCAN_MIN_SAMPLES = 10
N_EXAMPLES_PER_CLUSTER = 5   # representative loci to plot per cluster

# Reach metrics
FAR_THRESHOLD_BP = 1000      # "beyond" distance for frac_beyond
REACH_QUANTILE = 0.90        # cumulative-importance fraction for reach distance

FLANK_COLORS = {"left": "#3170ad", "right": "#e8443a"}
FLANK_LABELS = {"left": "Upstream", "right": "Downstream"}


# ===========================================================================
# Loading
# ===========================================================================

def load_run(run_dir):
	"""Load the NPZ and meta.json for a single run."""
	data = np.load(
		os.path.join(run_dir, "permutation_results.npz"), allow_pickle=True
	)
	with open(os.path.join(run_dir, "meta.json"), "r") as f:
		meta = json.load(f)
	return data, meta


def resolve_sources(run_dirs):
	"""Build {task: (data, meta, run_dir)}, loading each distinct run once.

	Validates that each task exists in its source run's task_names.
	"""
	cache = {}
	sources = {}
	for task, run_dir in run_dirs.items():
		if run_dir not in cache:
			cache[run_dir] = load_run(run_dir)
		data, meta = cache[run_dir]
		run_tasks = [str(t) for t in meta["task_names"]]
		if task not in run_tasks:
			raise ValueError(
				f"Task {task!r} not found in run {run_dir} "
				f"(available: {run_tasks})."
			)
		sources[task] = (data, meta, run_dir)
	return sources


def get_task_arrays(data, task):
	"""Return (baseline (N,), permuted (R,S,N), true (N,)) for a task."""
	return (
		data[f"baseline_predictions_{task}"],
		data[f"permuted_predictions_{task}"],
		data[f"true_labels_{task}"],
	)


# ===========================================================================
# Shared helpers
# ===========================================================================

def get_metrics(pred, true):
	"""Regression metrics in native space (matches the project's eval suite)."""
	return {
		"MSE": sk_metrics.mean_squared_error(true, pred),
		"MAE": sk_metrics.mean_absolute_error(true, pred),
		"MAPE": sk_metrics.mean_absolute_percentage_error(true, pred),
		"R2": sk_metrics.r2_score(true, pred),
		"Pearson_r": stats.pearsonr(true, pred).statistic,
		"Spearman_r": stats.spearmanr(true, pred).statistic,
	}


def abs_pct_change_per_locus(baseline, permuted):
	"""Per-region, per-locus mean |percent change| (mean over perms, then abs).

	Args:
		baseline: (N,) native predictions.
		permuted: (R, S, N) native predictions.

	Returns:
		(R, N) array of |mean-over-perms percent change|.
	"""
	pct = (permuted - baseline[None, None, :]) / baseline[None, None, :] * 100.0
	return np.abs(pct.mean(axis=1))


def region_index_maps(data):
	"""Map (flank, distance-bin) -> row index in the R region axis.

	Returns:
		Tuple (left_rows, right_rows, dist_bp_mid) where left_rows[d] / right_rows[d]
		are region-axis indices ordered by region_idx (0 = STR-adjacent), and
		dist_bp_mid[d] is the midpoint distance (bp) of bin d.
	"""
	flanks = np.array([str(x) for x in data["region_flanks"]])
	idxs = np.array(data["region_idxs"]).astype(int)
	d_start = np.array(data["region_distance_bp_starts"]).astype(float)
	d_end = np.array(data["region_distance_bp_ends"]).astype(float)

	def rows_for(flank):
		sel = np.where(flanks == flank)[0]
		order = np.argsort(idxs[sel])
		return sel[order]

	left_rows = rows_for("left")
	right_rows = rows_for("right")
	dist_mid = (d_start[left_rows] + d_end[left_rows]) / 2.0
	return left_rows, right_rows, dist_mid


# ===========================================================================
# View 1 - Per-orientation (model-position) bias
# ===========================================================================

def per_orientation_local(abs_pct, n_bootstrap, ci_level, rng):
	"""Per-region mean |percent change| over ALL rows, with bootstrap CI."""
	n_regions, n_loci = abs_pct.shape
	point = abs_pct.mean(axis=1)

	alpha = 1 - ci_level
	boot = np.empty((n_bootstrap, n_regions), dtype=np.float32)
	for b in trange(n_bootstrap, desc="  bootstrap (per-orientation local)"):
		idx = rng.integers(0, n_loci, size=n_loci)
		boot[b] = abs_pct[:, idx].mean(axis=1)
	lo = np.percentile(boot, 100 * (alpha / 2), axis=0)
	hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
	return point, lo, hi


def per_orientation_global(baseline, permuted, true):
	"""Per-region metric deltas (positive = region matters). Native space."""
	n_regions, n_perms, _ = permuted.shape
	baseline_metrics = get_metrics(baseline, true)
	names = list(baseline_metrics.keys())
	error_metrics = {"MSE", "MAE", "MAPE"}

	per_perm = {n: np.empty((n_regions, n_perms)) for n in names}
	for r in range(n_regions):
		for s in range(n_perms):
			m = get_metrics(permuted[r, s, :], true)
			for n in names:
				per_perm[n][r, s] = m[n]

	mean_shuffled = {n: per_perm[n].mean(axis=1) for n in names}
	delta = {}
	for n in names:
		if n in error_metrics:
			delta[n] = mean_shuffled[n] - baseline_metrics[n]
		else:
			delta[n] = baseline_metrics[n] - mean_shuffled[n]
	return baseline_metrics, delta


def plot_per_orientation(data, task, local, glob, save_dir):
	"""Two-panel spatial profile (local |pct change| + global delta Spearman)."""
	point, lo, hi = local
	baseline_metrics, delta = glob
	left_rows, right_rows, dist_mid = region_index_maps(data)

	sns.set_theme(style="whitegrid")
	fig, (ax_l, ax_g) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

	for flank, rows in (("left", left_rows), ("right", right_rows)):
		color = FLANK_COLORS[flank]
		ax_l.plot(dist_mid, point[rows], color=color, marker="o", markersize=3,
				  linewidth=1.0, label=FLANK_LABELS[flank])
		ax_l.fill_between(dist_mid, lo[rows], hi[rows], color=color, alpha=0.25)
		ax_g.plot(dist_mid, delta["Spearman_r"][rows], color=color, marker="o",
				  markersize=3, linewidth=1.0, label=FLANK_LABELS[flank])

	ax_l.set_ylabel("Mean |percent change| (%)")
	ax_l.set_title(f"{task} - local effect per region (model-position view, "
				   f"{int(CI_LEVEL * 100)}% bootstrap CI)")
	ax_l.legend(loc="upper right")
	ax_l.set_ylim(bottom=0)

	ax_g.axhline(0, color="gray", lw=0.5, ls="--")
	ax_g.set_ylabel("delta Spearman (baseline - shuffled)")
	ax_g.set_xlabel("Distance from STR (bp)")
	ax_g.set_title(f"{task} - global Spearman drop per region "
				   f"(baseline = {baseline_metrics['Spearman_r']:.4f})")
	ax_g.legend(loc="upper right")

	plt.tight_layout()
	os.makedirs(save_dir, exist_ok=True)
	path = os.path.join(save_dir, "spatial_profiles.png")
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"  saved {path}")


# ===========================================================================
# View 2 - RC-averaged genomic
# ===========================================================================

def build_rc_averaged(data, task):
	"""Pair forward+rc rows by id and average importances in genomic coords.

	Returns:
		Dict with keys:
			ids: (M,) paired locus ids.
			up: (M, D) genomic-upstream |pct change| per distance bin.
			down: (M, D) genomic-downstream per distance bin.
			dist_mid: (D,) distance midpoints (bp).
	"""
	baseline, permuted, _ = get_task_arrays(data, task)
	abs_pct = abs_pct_change_per_locus(baseline, permuted)   # (R, N)

	left_rows, right_rows, dist_mid = region_index_maps(data)
	ids = np.array([str(x) for x in data["ids"]])
	rev_comp = np.array(data["rev_comp"]).astype(bool)

	# Per-row left/right profiles over distance bins: (N, D).
	left_prof = abs_pct[left_rows, :].T
	right_prof = abs_pct[right_rows, :].T

	# Group row indices by id.
	by_id = {}
	for i, locus in enumerate(ids):
		by_id.setdefault(locus, []).append(i)

	paired_ids, up_list, down_list = [], [], []
	n_missing = 0
	for locus, rows in by_id.items():
		fwd = [i for i in rows if not rev_comp[i]]
		rc = [i for i in rows if rev_comp[i]]
		if not fwd or not rc:
			n_missing += 1
			continue
		f, r = fwd[0], rc[0]
		# genomic upstream  = mean(fwd.left,  rc.right)
		# genomic downstream= mean(fwd.right, rc.left)
		up = (left_prof[f] + right_prof[r]) / 2.0
		down = (right_prof[f] + left_prof[r]) / 2.0
		paired_ids.append(locus)
		up_list.append(up)
		down_list.append(down)

	if n_missing:
		print(f"  [{task}] skipped {n_missing} loci missing an orientation.")

	return {
		"ids": np.array(paired_ids),
		"up": np.array(up_list),
		"down": np.array(down_list),
		"dist_mid": dist_mid,
	}


def _profile_mean_ci(profiles, n_bootstrap, ci_level, rng, desc):
	"""Mean profile over loci + bootstrap CI (resampling loci)."""
	n_loci = profiles.shape[0]
	point = profiles.mean(axis=0)
	alpha = 1 - ci_level
	boot = np.empty((n_bootstrap, profiles.shape[1]), dtype=np.float32)
	for b in trange(n_bootstrap, desc=desc):
		idx = rng.integers(0, n_loci, size=n_loci)
		boot[b] = profiles[idx].mean(axis=0)
	lo = np.percentile(boot, 100 * (alpha / 2), axis=0)
	hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
	return point, lo, hi


def plot_genomic_profiles(rc_avg, task, n_bootstrap, ci_level, rng, save_dir):
	"""Genomic upstream/downstream spatial profile (mean + bootstrap CI)."""
	dist_mid = rc_avg["dist_mid"]
	up = _profile_mean_ci(rc_avg["up"], n_bootstrap, ci_level, rng,
						   f"  bootstrap ({task} genomic up)")
	down = _profile_mean_ci(rc_avg["down"], n_bootstrap, ci_level, rng,
							f"  bootstrap ({task} genomic down)")

	sns.set_theme(style="whitegrid")
	fig, ax = plt.subplots(figsize=(12, 5))
	for prof, flank in ((up, "left"), (down, "right")):
		point, lo, hi = prof
		color = FLANK_COLORS[flank]
		ax.plot(dist_mid, point, color=color, marker="o", markersize=3,
				linewidth=1.0, label=FLANK_LABELS[flank])
		ax.fill_between(dist_mid, lo, hi, color=color, alpha=0.25)

	ax.set_xlabel("Distance from STR (bp)")
	ax.set_ylabel("Mean |percent change| (%)")
	ax.set_title(f"{task} - RC-averaged genomic effect "
				 f"(n={rc_avg['ids'].shape[0]} loci)")
	ax.legend(loc="upper right")
	ax.set_ylim(bottom=0)

	plt.tight_layout()
	os.makedirs(save_dir, exist_ok=True)
	path = os.path.join(save_dir, f"genomic_profile_{task}.png")
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"  saved {path}")
	return {"up_mean": up[0], "down_mean": down[0], "dist_mid": dist_mid}


def plot_length_vs_het(profile_summaries, save_dir):
	"""Overlay each task's genomic up/down mean profile, normalized to its own max."""
	if len(profile_summaries) < 2:
		print("  [length-vs-het] need >=2 tasks; skipping overlay.")
		return

	sns.set_theme(style="whitegrid")
	fig, (ax_up, ax_down) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
	cmap = plt.get_cmap("tab10")

	for k, (task, summ) in enumerate(profile_summaries.items()):
		color = cmap(k)
		dist_mid = summ["dist_mid"]
		up = summ["up_mean"] / (summ["up_mean"].max() or 1.0)
		down = summ["down_mean"] / (summ["down_mean"].max() or 1.0)
		ax_up.plot(dist_mid, up, color=color, marker="o", markersize=3,
				   linewidth=1.0, label=task)
		ax_down.plot(dist_mid, down, color=color, marker="o", markersize=3,
					 linewidth=1.0, label=task)

	ax_up.set_title("Upstream (normalized to own max)")
	ax_down.set_title("Downstream (normalized to own max)")
	for ax in (ax_up, ax_down):
		ax.set_xlabel("Distance from STR (bp)")
		ax.legend(loc="upper right")
	ax_up.set_ylabel("Normalized mean |percent change|")

	fig.suptitle("Length vs heterozygosity - spatial importance comparison",
				 fontsize=13, fontweight="bold")
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	os.makedirs(save_dir, exist_ok=True)
	path = os.path.join(save_dir, "length_vs_het_overlay.png")
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"  saved {path}")


# ===========================================================================
# View 3 - Long-range reach + clustering
# ===========================================================================

def compute_reach_metrics(profiles, dist_mid, far_threshold, reach_quantile):
	"""Per-locus reach scalars from |importance|-vs-distance profiles.

	Args:
		profiles: (M, D) per-locus combined |pct change| profiles.
		dist_mid: (D,) distance midpoints (bp).
		far_threshold: distance (bp) for frac_beyond.
		reach_quantile: cumulative-importance fraction for reach distance.

	Returns:
		DataFrame with columns center_of_mass, frac_beyond, reach_distance.
	"""
	w = np.clip(profiles, 0, None)
	total = w.sum(axis=1)
	safe = np.where(total > 0, total, 1.0)

	com = (w * dist_mid[None, :]).sum(axis=1) / safe
	beyond = w[:, dist_mid > far_threshold].sum(axis=1) / safe

	cum = np.cumsum(w, axis=1) / safe[:, None]
	reach_idx = (cum >= reach_quantile).argmax(axis=1)
	reach_dist = dist_mid[reach_idx]

	df = pd.DataFrame({
		"center_of_mass": com,
		"frac_beyond": beyond,
		"reach_distance": reach_dist,
	})
	df.loc[total <= 0, ["center_of_mass", "frac_beyond", "reach_distance"]] = 0.0
	return df


def normalize_profiles(profiles):
	"""Unit-area normalize each profile (so SHAPE, not magnitude, drives clusters).

	Returns the normalized profiles and a boolean mask of usable (non-zero) rows.
	"""
	w = np.clip(profiles, 0, None)
	total = w.sum(axis=1, keepdims=True)
	usable = (total[:, 0] > 0)
	norm = np.zeros_like(w)
	norm[usable] = w[usable] / total[usable]
	return norm, usable


def cluster_kmeans(norm, n_clusters, seed):
	km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
	labels = km.fit_predict(norm)
	return labels


def cluster_agglomerative(norm, n_clusters, dist_threshold):
	"""Ward agglomerative via scipy; return (labels, linkage_matrix)."""
	Z = linkage(norm, method="ward")
	if dist_threshold is not None:
		labels = fcluster(Z, t=dist_threshold, criterion="distance")
	else:
		labels = fcluster(Z, t=n_clusters, criterion="maxclust")
	return labels - 1, Z  # 0-indexed


def cluster_dbscan(norm, eps, min_samples):
	return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(norm)


def suggest_k_silhouette(norm, seed, k_range=range(2, 8)):
	"""Print a silhouette-vs-k sweep to help pick KMeans k."""
	print("  KMeans silhouette sweep:")
	for k in k_range:
		if k >= norm.shape[0]:
			break
		labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(norm)
		score = silhouette_score(norm, labels)
		print(f"    k={k}: silhouette={score:.4f}")


def plot_clusters(profiles, norm, usable, labels, dist_mid, reach_df, ids,
				  task, method, save_dir):
	"""Cluster-mean profiles, sizes, examples, and reach histogram."""
	uniq = sorted(set(labels.tolist()))
	cmap = plt.get_cmap("tab10")
	color_for = {lab: cmap(i % 10) for i, lab in enumerate(uniq)}

	# Rank clusters by mean center-of-mass to tag the long-range cluster.
	com_by_cluster = {
		lab: reach_df["center_of_mass"].values[labels == lab].mean()
		for lab in uniq
	}
	long_range_lab = max(com_by_cluster, key=com_by_cluster.get)

	sns.set_theme(style="whitegrid")
	fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	ax_mean, ax_size, ax_ex, ax_hist = axes.ravel()

	# (a) cluster-mean normalized profiles
	for lab in uniq:
		members = norm[labels == lab]
		if members.shape[0] == 0:
			continue
		mean_prof = members.mean(axis=0)
		tag = " (long-range)" if lab == long_range_lab else ""
		name = "noise" if lab == -1 else f"cluster {lab}"
		ax_mean.plot(dist_mid, mean_prof, color=color_for[lab], marker="o",
					 markersize=3, label=f"{name}{tag} (n={members.shape[0]})")
	ax_mean.set_title(f"{task} [{method}] - cluster-mean shape (unit-area)")
	ax_mean.set_xlabel("Distance from STR (bp)")
	ax_mean.set_ylabel("Normalized importance")
	ax_mean.legend(fontsize=8)

	# (b) cluster sizes
	sizes = [int((labels == lab).sum()) for lab in uniq]
	ax_size.bar([str(l) for l in uniq], sizes,
				color=[color_for[l] for l in uniq])
	ax_size.set_title("Cluster sizes")
	ax_size.set_xlabel("cluster")
	ax_size.set_ylabel("n loci")

	# (c) representative examples from the long-range cluster (raw profiles)
	lr_rows = np.where(labels == long_range_lab)[0]
	centroid = norm[lr_rows].mean(axis=0)
	d = np.linalg.norm(norm[lr_rows] - centroid[None, :], axis=1)
	pick = lr_rows[np.argsort(d)[:N_EXAMPLES_PER_CLUSTER]]
	for i in pick:
		ax_ex.plot(dist_mid, profiles[i], linewidth=1.0, alpha=0.8,
				   label=str(ids[i]))
	ax_ex.set_title(f"Long-range cluster {long_range_lab} - example loci (raw)")
	ax_ex.set_xlabel("Distance from STR (bp)")
	ax_ex.set_ylabel("|percent change| (%)")
	ax_ex.legend(fontsize=7)

	# (d) center-of-mass histogram colored by cluster
	for lab in uniq:
		ax_hist.hist(reach_df["center_of_mass"].values[labels == lab], bins=30,
					 alpha=0.5, color=color_for[lab],
					 label=("noise" if lab == -1 else f"cluster {lab}"))
	ax_hist.set_title("Importance center-of-mass by cluster")
	ax_hist.set_xlabel("center-of-mass distance (bp)")
	ax_hist.set_ylabel("n loci")
	ax_hist.legend(fontsize=8)

	plt.tight_layout()
	os.makedirs(save_dir, exist_ok=True)
	path = os.path.join(save_dir, f"clusters_{method}.png")
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"  saved {path}")
	return long_range_lab


def plot_dendrogram(Z, task, save_dir):
	fig, ax = plt.subplots(figsize=(14, 5))
	dendrogram(Z, no_labels=True, ax=ax,
			   color_threshold=(AGG_DIST_THRESHOLD or None))
	if AGG_DIST_THRESHOLD is not None:
		ax.axhline(AGG_DIST_THRESHOLD, color="red", ls="--",
				   label=f"cut @ {AGG_DIST_THRESHOLD}")
		ax.legend()
	ax.set_title(f"{task} - ward dendrogram of profile shapes")
	ax.set_xlabel("loci")
	ax.set_ylabel("ward distance")
	plt.tight_layout()
	os.makedirs(save_dir, exist_ok=True)
	path = os.path.join(save_dir, "dendrogram_agglomerative.png")
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"  saved {path}")


def run_clustering(rc_avg, task, save_dir):
	"""Reach metrics + clustering on the RC-averaged combined profiles."""
	# Primary "time series": mean of upstream & downstream (distance is symmetric).
	profiles = (rc_avg["up"] + rc_avg["down"]) / 2.0
	dist_mid = rc_avg["dist_mid"]
	ids = rc_avg["ids"]

	reach_df = compute_reach_metrics(
		profiles, dist_mid, FAR_THRESHOLD_BP, REACH_QUANTILE
	)
	reach_df.insert(0, "id", ids)

	norm, usable = normalize_profiles(profiles)
	if usable.sum() < 2:
		print(f"  [{task}] too few usable profiles to cluster; "
			  f"writing reach metrics only.")
		os.makedirs(save_dir, exist_ok=True)
		reach_df.to_csv(os.path.join(save_dir, f"reach_metrics_{task}.tsv"),
						sep="\t", index=False)
		return

	# Cluster only usable (non-zero) rows; map labels back to all loci (-2 = empty).
	norm_u = norm[usable]
	profiles_u = profiles[usable]
	reach_u = reach_df[usable].reset_index(drop=True)
	ids_u = ids[usable]

	methods = (["agglomerative", "kmeans"] if CLUSTER_METHOD == "both"
			   else [CLUSTER_METHOD])

	reach_df["cluster_agglomerative"] = -2
	reach_df["cluster_kmeans"] = -2
	reach_df["cluster_dbscan"] = -2

	for method in methods:
		seed = SEED
		if method == "kmeans":
			suggest_k_silhouette(norm_u, seed)
			labels = cluster_kmeans(norm_u, N_CLUSTERS, seed)
		elif method == "agglomerative":
			labels, Z = cluster_agglomerative(norm_u, N_CLUSTERS, AGG_DIST_THRESHOLD)
			plot_dendrogram(Z, task, save_dir)
		elif method == "dbscan":
			print("  NOTE: DBSCAN often labels smooth importance-decay profiles as "
				  "noise (-1) and is eps-sensitive; treat as exploratory.")
			labels = cluster_dbscan(norm_u, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
		else:
			raise ValueError(f"Unknown CLUSTER_METHOD: {method}")

		lr = plot_clusters(profiles_u, norm_u, usable, labels, dist_mid,
						   reach_u, ids_u, task, method, save_dir)
		reach_df.loc[usable, f"cluster_{method}"] = labels

		# Export the long-range cluster members for this method.
		lr_ids = ids_u[labels == lr]
		lr_df = reach_u[labels == lr].sort_values(
			"center_of_mass", ascending=False
		)
		lr_path = os.path.join(save_dir, f"long_range_loci_{task}_{method}.tsv")
		lr_df.to_csv(lr_path, sep="\t", index=False)
		print(f"  saved {lr_path} ({len(lr_ids)} loci)")

	os.makedirs(save_dir, exist_ok=True)
	reach_df.to_csv(os.path.join(save_dir, f"reach_metrics_{task}.tsv"),
					sep="\t", index=False)


# ===========================================================================
# Main
# ===========================================================================

def resolve_out_dir(run_dirs, out_dir):
	if out_dir is not None:
		return out_dir
	distinct = set(run_dirs.values())
	if len(distinct) == 1:
		return os.path.join(next(iter(distinct)), "analysis")
	raise ValueError(
		"OUT_DIR is None but tasks come from different runs "
		f"({sorted(distinct)}); set OUT_DIR explicitly."
	)


if __name__ == "__main__":

	out_dir = resolve_out_dir(RUN_DIRS, OUT_DIR)
	print(f"Writing analysis to {out_dir}")

	sources = resolve_sources(RUN_DIRS)
	rng = np.random.default_rng(SEED)

	profile_summaries = {}

	for task, (data, meta, run_dir) in sources.items():
		print(f"\n=== Task: {task} (source: {run_dir}) ===")
		baseline, permuted, true = get_task_arrays(data, task)
		R, S, N = permuted.shape
		print(f"  {N} rows, {R} regions, {S} perms/region")

		# --- View 1: per-orientation (model-position) bias ---
		abs_pct = abs_pct_change_per_locus(baseline, permuted)
		local = per_orientation_local(abs_pct, N_BOOTSTRAP, CI_LEVEL, rng)
		glob = per_orientation_global(baseline, permuted, true)
		plot_per_orientation(
			data, task, local, glob,
			os.path.join(out_dir, "per_orientation", task),
		)

		# --- View 2: RC-averaged genomic ---
		rc_avg = build_rc_averaged(data, task)
		summ = plot_genomic_profiles(
			rc_avg, task, N_BOOTSTRAP, CI_LEVEL, rng,
			os.path.join(out_dir, "genomic_rc_avg"),
		)
		profile_summaries[task] = summ

		# --- View 3: long-range reach + clustering ---
		run_clustering(
			rc_avg, task, os.path.join(out_dir, "clustering", task),
		)

	# --- View 2 overlay: length vs heterozygosity ---
	plot_length_vs_het(
		profile_summaries, os.path.join(out_dir, "genomic_rc_avg")
	)

	print("\n--- Done ---")
