""" Interpret linear-baseline Ridge coefficients: rank and plot by feature and
by category (flank side / distance-from-STR bin, or motif).

Works for both feature families the trainer supports (kmer or gc; see
seq_models/linear/train_and_pred.py's `feature_type` config key) -- feature
names are parsed generically by regex, and every downstream ranking/plotting
function only touches the resulting category columns, not family internals.

Consumes `coefficients.tsv` and `config.yaml` written by
seq_models/linear/train_and_pred.py for a single run directory, and produces,
per target (whatever columns are present in coefficients.tsv):

  - coef_ranked_{target}.tsv        full feature table, sorted by |coef|
  - top_overall_{target}.png        top-N features overall, by |coef|
  - top_by_category_{target}.png    top-N features within each category
  - category_importance_{target}.{tsv,png}
                                     mean/sum |coef| per category (e.g.
                                     "Upstream 0-1000bp", "Motif")
  - spatial_profile_{target}.png    mirrored upstream/downstream mean |coef|
                                     vs. distance-from-STR bin (only produced
                                     when the run used flank_mode="separate"
                                     with multiple spatial bins)
  - kmer_size_importance_{target}.{tsv,png}
                                     mean/sum |coef| grouped by k-mer size
                                     (kmer runs only; a no-op for gc runs,
                                     whose features have no k-mer size)

Coefficients are already in standardized-feature space (train_and_pred.py
scales X with StandardScaler before fitting Ridge), so |coef| is comparable
across features/categories without further normalization.

Configuration is via the editable constants below (no argparse), mirroring
scripts/region_permutation_importance/analyze_caduceus.py.
"""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


# ===========================================================================
# Configuration
# ===========================================================================

# RUN_DIR = "../eval_preds/predictions/baseline/lin/str3/str3_f5000_sep"
# RUN_DIR = "../eval_preds/predictions/baseline/lin/str4/str4_f5000_sep_dist"
# RUN_DIR = "../eval_preds/predictions/baseline/lin/str3/str3_f5000_gc_100_pooled"
RUN_DIR = "../eval_preds/predictions/baseline/lin/str4/str4_f5000_gc_100_pooled"

OUT_DIR = None  # None -> {RUN_DIR}/analysis
TOP_N = 30      # top-N features to show in bar plots

# Same convention as scripts/region_permutation_importance/analyze_caduceus.py
FLANK_LABELS = {"left": "Upstream", "right": "Downstream", "flank": "Flank (pooled)"}
FLANK_COLORS = {"left": "#3170ad", "right": "#e8443a", "flank": "#3170ad"}
MOTIF_LABEL = "Motif"
MOTIF_COLOR = "#5a9e5a"

_KMER_RE = re.compile(r"^(left|right|flank)(?:_bin(\d+))?_(\d+)mer_([ACGT]+)$")
_GC_RE = re.compile(r"^(left|right|flank)(?:_bin(\d+))?_gc$")
_MOTIF_RE = re.compile(r"^motif_(.+)$")


# ===========================================================================
# Loading & parsing
# ===========================================================================

def build_distance_edges(distance_bins, n_flanking_bp):
	"""Return bin edges [0, e1, ..., n_flanking_bp] from a distance_bins config.

	`distance_bins` may be None (whole flank), an int (uniform-width bins of
	that size, bp -- a smaller remainder bin is kept at the far end if it
	doesn't evenly divide n_flanking_bp), or a list (explicit bp edges).

	Mirrors seq_models/linear/data.py::build_distance_edges (duplicated here,
	not imported, for self-containment -- same convention as the other
	analysis scripts in this repo, e.g. the duplicated get_metrics helper in
	scripts/eval_preds/*.py).
	"""
	if not distance_bins:
		return [0, n_flanking_bp]

	if isinstance(distance_bins, int):
		n_full = n_flanking_bp // distance_bins
		remainder = n_flanking_bp % distance_bins
		edges = [i * distance_bins for i in range(n_full + 1)]
		if remainder > 0:
			edges.append(n_flanking_bp)
		return edges

	edges = [int(e) for e in distance_bins]
	if edges[-1] != n_flanking_bp:
		edges.append(n_flanking_bp)
	return [0] + edges


def load_run(run_dir):
	"""Load coefficients.tsv and config.yaml for a single linear-baseline run."""
	coef_df = pd.read_csv(os.path.join(run_dir, "coefficients.tsv"), sep="\t")
	with open(os.path.join(run_dir, "config.yaml"), "r") as f:
		config = yaml.safe_load(f)
	return coef_df, config


def parse_feature_name(name, edges):
	"""Parse one feature_name into its structured category fields.

	Args:
		name: feature name, e.g. "left_bin0_3mer_AAA" (kmer), "left_bin0_gc"
			(gc), or "motif_AC" (motif).
		edges: bin edges from build_distance_edges (bin b spans
			[edges[b], edges[b+1])), used to label the bp range of each bin.

	Returns:
		dict with keys: side, bin_idx, bin_lo_bp, bin_hi_bp, k, kmer_or_motif,
		spatial_category.
	"""
	m = _KMER_RE.match(name)
	if m:
		side, bin_idx, k, kmer = m.groups()
		bin_idx = int(bin_idx) if bin_idx is not None else 0
		lo, hi = edges[bin_idx], edges[bin_idx + 1]
		return {
			"side": side,
			"bin_idx": bin_idx,
			"bin_lo_bp": lo,
			"bin_hi_bp": hi,
			"k": int(k),
			"kmer_or_motif": kmer,
			"spatial_category": f"{FLANK_LABELS[side]} {lo}-{hi}bp",
		}
	m = _GC_RE.match(name)
	if m:
		side, bin_idx = m.groups()
		bin_idx = int(bin_idx) if bin_idx is not None else 0
		lo, hi = edges[bin_idx], edges[bin_idx + 1]
		return {
			"side": side,
			"bin_idx": bin_idx,
			"bin_lo_bp": lo,
			"bin_hi_bp": hi,
			"k": None,
			"kmer_or_motif": "gc",
			"spatial_category": f"{FLANK_LABELS[side]} {lo}-{hi}bp",
		}
	m = _MOTIF_RE.match(name)
	if m:
		return {
			"side": None,
			"bin_idx": None,
			"bin_lo_bp": None,
			"bin_hi_bp": None,
			"k": None,
			"kmer_or_motif": m.group(1),
			"spatial_category": MOTIF_LABEL,
		}
	raise ValueError(f"Unrecognized feature name: {name!r}")


def build_feature_meta(feature_names, distance_bins, n_flanking_bp):
	"""Parse every feature name into a metadata DataFrame (same row order)."""
	edges = build_distance_edges(distance_bins, n_flanking_bp)
	rows = [parse_feature_name(name, edges) for name in feature_names]
	meta = pd.DataFrame(rows)
	meta.insert(0, "feature_name", feature_names)
	return meta


# ===========================================================================
# Ranking / aggregation
# ===========================================================================

def rank_features(coef_df, meta, target):
	df = meta.copy()
	df["coef"] = coef_df[target].values
	df["abs_coef"] = df["coef"].abs()
	return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def category_importance(ranked, group_col):
	return (
		ranked.groupby(group_col)["abs_coef"]
		.agg(mean_abs_coef="mean", sum_abs_coef="sum", n_features="count")
		.reset_index()
		.sort_values("mean_abs_coef", ascending=False)
		.reset_index(drop=True)
	)


def kmer_size_importance(ranked):
	sub = ranked[ranked["k"].notna()].copy()
	if sub.empty:
		return sub
	sub["k"] = sub["k"].astype(int)
	return (
		sub.groupby("k")["abs_coef"]
		.agg(mean_abs_coef="mean", sum_abs_coef="sum", n_features="count")
		.reset_index()
		.sort_values("mean_abs_coef", ascending=False)
		.reset_index(drop=True)
	)


# ===========================================================================
# Plotting
# ===========================================================================

def _category_color(cat):
	if cat == MOTIF_LABEL:
		return MOTIF_COLOR
	for side, label in FLANK_LABELS.items():
		if cat.startswith(label):
			return FLANK_COLORS[side]
	return "#888888"


def plot_top_overall(ranked, target, out_path, top_n):
	top = ranked.head(top_n).iloc[::-1]  # ascending, so barh reads top-down
	colors = [_category_color(c) for c in top["spatial_category"]]
	fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(top))))
	ax.barh(top["feature_name"], top["coef"], color=colors)
	ax.axvline(0, color="black", linewidth=0.8)
	ax.set_xlabel("Ridge coefficient (standardized-feature space)")
	ax.set_title(f"Top {top_n} features overall — {target}")
	fig.tight_layout()
	fig.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def plot_top_by_category(ranked, target, out_path, top_n):
	categories = ranked["spatial_category"].unique().tolist()
	categories = sorted(c for c in categories if c != MOTIF_LABEL) + (
		[MOTIF_LABEL] if MOTIF_LABEL in categories else []
	)
	n = len(categories)
	ncols = min(3, n)
	nrows = -(-n // ncols)
	fig, axes = plt.subplots(
		nrows, ncols,
		figsize=(5 * ncols, max(3, 0.25 * top_n) * nrows),
		squeeze=False,
	)
	for i, cat in enumerate(categories):
		ax = axes[i // ncols][i % ncols]
		sub = ranked[ranked["spatial_category"] == cat].head(top_n).iloc[::-1]
		ax.barh(sub["feature_name"], sub["coef"], color=_category_color(cat))
		ax.axvline(0, color="black", linewidth=0.8)
		ax.set_title(cat, fontsize=10)
		ax.tick_params(axis="y", labelsize=7)
	for j in range(n, nrows * ncols):
		axes[j // ncols][j % ncols].axis("off")
	fig.suptitle(f"Top {top_n} features by category — {target}")
	fig.tight_layout()
	fig.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def plot_category_importance(cat_imp, target, out_path):
	fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(cat_imp))))
	colors = [_category_color(c) for c in cat_imp["spatial_category"]]
	ax.barh(cat_imp["spatial_category"], cat_imp["mean_abs_coef"], color=colors)
	ax.invert_yaxis()
	ax.set_xlabel("Mean |Ridge coefficient|")
	ax.set_title(f"Category importance — {target}")
	fig.tight_layout()
	fig.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def plot_spatial_profile(ranked, target, out_path):
	"""Mirrored upstream/downstream mean |coef| vs. distance-from-STR bin.

	Only meaningful when the run used flank_mode="separate" with multiple
	distance_bins; returns False (no plot written) otherwise.
	"""
	sub = ranked[ranked["side"].isin(["left", "right"])]
	if sub.empty or sub["bin_idx"].nunique() <= 1:
		return False

	prof = (
		sub.groupby(["side", "bin_idx", "bin_lo_bp"])["abs_coef"]
		.mean()
		.reset_index()
		.rename(columns={"abs_coef": "mean_abs_coef"})
	)

	fig, ax = plt.subplots(figsize=(7, 4))
	for side in ("left", "right"):
		s = prof[prof["side"] == side].sort_values("bin_idx")
		if s.empty:
			continue
		xs = [-lo if side == "left" else lo for lo in s["bin_lo_bp"]]
		ax.plot(
			xs, s["mean_abs_coef"], marker="o", color=FLANK_COLORS[side],
			label=FLANK_LABELS[side],
		)
	ax.axvline(0, color="black", linewidth=0.8)
	ax.set_xlabel("Distance from STR (bp); negative = upstream")
	ax.set_ylabel("Mean |Ridge coefficient|")
	ax.set_title(f"Spatial importance profile — {target}")
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	return True


def plot_kmer_size_importance(k_imp, target, out_path):
	fig, ax = plt.subplots(figsize=(5, 3.5))
	ax.bar(k_imp["k"].astype(str), k_imp["mean_abs_coef"], color="#7a6fbf")
	ax.set_xlabel("k-mer size")
	ax.set_ylabel("Mean |Ridge coefficient|")
	ax.set_title(f"K-mer size importance — {target}")
	fig.tight_layout()
	fig.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
	sns.set_theme(style="whitegrid")

	coef_df, config = load_run(RUN_DIR)
	feature_names = coef_df["feature_name"].tolist()
	targets = [c for c in coef_df.columns if c != "feature_name"]

	meta = build_feature_meta(
		feature_names,
		distance_bins=config.get("distance_bins", None),
		n_flanking_bp=config["n_flanking_bp"],
	)

	out_dir = OUT_DIR or os.path.join(RUN_DIR, "analysis")
	os.makedirs(out_dir, exist_ok=True)

	print(f"Loaded {len(feature_names)} features from {RUN_DIR}")
	print(f"Targets: {targets}")

	for target in targets:
		print(f"\n=== {target} ===")
		ranked = rank_features(coef_df, meta, target)

		ranked_path = os.path.join(out_dir, f"coef_ranked_{target}.tsv")
		ranked.to_csv(ranked_path, sep="\t", index=False)
		print(f"Saved ranked coefficients to {ranked_path}")

		plot_top_overall(
			ranked, target,
			os.path.join(out_dir, f"top_overall_{target}.png"), TOP_N,
		)
		plot_top_by_category(
			ranked, target,
			os.path.join(out_dir, f"top_by_category_{target}.png"), TOP_N,
		)

		cat_imp = category_importance(ranked, "spatial_category")
		cat_imp.to_csv(
			os.path.join(out_dir, f"category_importance_{target}.tsv"),
			sep="\t", index=False,
		)
		plot_category_importance(
			cat_imp, target,
			os.path.join(out_dir, f"category_importance_{target}.png"),
		)

		if plot_spatial_profile(
			ranked, target,
			os.path.join(out_dir, f"spatial_profile_{target}.png"),
		):
			print("Saved spatial importance profile plot.")

		k_imp = kmer_size_importance(ranked)
		if not k_imp.empty:
			k_imp.to_csv(
				os.path.join(out_dir, f"kmer_size_importance_{target}.tsv"),
				sep="\t", index=False,
			)
			plot_kmer_size_importance(
				k_imp, target,
				os.path.join(out_dir, f"kmer_size_importance_{target}.png"),
			)

		print("Top 5 overall by |coef|:")
		print(
			ranked[["feature_name", "spatial_category", "coef"]]
			.head(5)
			.to_string(index=False)
		)

	print(f"\nAll outputs saved to {out_dir}")
