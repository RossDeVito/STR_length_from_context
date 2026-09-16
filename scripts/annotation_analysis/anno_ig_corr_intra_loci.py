"""Intra-locus correlation between per-bp IG attributions and a continuous
genomic annotation (e.g. phyloP), with a bootstrap CI and a permutation
null.

The question is whether, *within* a locus's flanks, the positions the model
leans on are the positions a continuous annotation calls interesting.

Two nuisance signals have to be removed before that question is answerable:

1. Per-locus attribution scale. Handled by the shared normalization in
   seq_models/ig_attribution_norm.py (NORM_SCHEME, default "l1_mass") plus
   the forward/RC symmetrization, which also halves the prompt-proximity
   artifact. Each locus contributes one signed, RC-averaged profile.

2. Positional bias. Attribution magnitude is a strong function of distance
   from the STR (and of model-input position generally) for reasons that
   have nothing to do with genomics, and phyloP has its own distance-from-STR
   profile. Correlating raw attribution against the annotation within a
   locus would mostly measure the product of those two shared shapes.
   So attributions are RANKED ACROSS LOCI AT EACH POSITION first: the value
   at, say, +10 bp becomes that locus's quantile among all loci' values at
   +10 bp. Every position then has the same marginal distribution
   (uniform), the shared positional profile is gone by construction, and
   what is left per position is "high or low for this position".

Per locus we then take a Spearman correlation between that quantile track
and the annotation track over the same flanking positions. The effect size
is the mean per-locus rho; its CI is a percentile bootstrap over loci.

Significance comes from a permutation null that keeps both marginals
intact: attribution tracks are randomly re-paired with annotation tracks
(a permutation of loci), and the mean rho is recomputed. This destroys the
locus-specific pairing while preserving each track's own autocorrelation
and shape, which a naive positional shuffle would not. Two-sided p-value:

    p = (1 + #{ |rho_bar_perm| >= |rho_bar_obs| }) / (N_PERMUTATIONS + 1)

Speed. The whole permutation scheme reduces to lookups in one precomputed
matrix. Spearman is Pearson-of-ranks, so if every locus's ranks are taken
over the SAME set of positions, then row-standardizing the rank matrices
makes C = Za @ Zb.T the full n_loci x n_loci matrix of "attribution track i
vs annotation track j" correlations. The observed per-locus rhos are its
diagonal, and each permutation is mean(C[i, perm[i]]) -- a gather, not a
correlation. That is why a single common position mask (the intersection of
every kept locus's annotated positions) is used rather than per-locus
masks: it makes the observed and permuted statistics comparable on
identical position sets, and it makes 10,000 permutations cost less than
one pass of the naive loop.

One caveat the design absorbs rather than removes: the across-loci
ranking makes each locus's quantile track depend weakly on the other loci
(a locus's value at a position is its rank among the others there), so the
per-locus rhos are not strictly independent. The bootstrap CI treats them
as if they were, which at n in the hundreds is a negligible
approximation. The p-value does not rely on it at all -- the permutation
null is built from the very same quantile matrix, so whatever dependence
the ranking induces is present in the null and the observed statistic
alike.

IG samples are joined to genomic coordinates via attributions.npz's `ids`
array (the source tsv's `ID` column, shared by a locus's fwd/RC pair), so
STR_DATA_PATH must be the tsv used to train the checkpoint behind ATTR_DIR.
Flanks map to the reference as left = [str_start - n_flanking_bp,
str_start) and right = [str_end, str_end + n_flanking_bp), matching
seq_models/caduceus/data.py.

Configuration is via the editable constants below (no argparse), mirroring
the rest of this directory.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from tqdm import tqdm

import seq_models.ig_attribution_norm as ign

import annotation_utils as au


# ===========================================================================
# Configuration
# ===========================================================================

ATTR_DIR = "../ig/output/caduceus_v0/str3/str3_f5000_fiv_v4"

# Which task's attributions to use (must be one of the checkpoint's active
# targets -- see {ATTR_DIR}/meta.json "task_names").
# TASK = "length"
TASK = "variation"

# tsv used to train the checkpoint behind ATTR_DIR (see its config.yaml
# "data_path"). Must be the exact file so attributions.npz's `ids` can be
# joined to chrom/str_start/str_end.
STR_DATA_PATH = "../../data/STR_data/HipSTR_labeled_STRs/str_len_3_n_flanking_10000.tsv"

# Continuous annotation to correlate against: an exact annotation_config.yaml
# key whose value_type is "continuous".
# ANNOTATION_TYPE = "phyloP"
ANNOTATION_TYPE = "GC_content_5bp"
# ANNOTATION_TYPE = "GC_content_9bp"
# ANNOTATION_TYPE = "GC_content_25bp"

# abs() the per-bp annotation values before correlating. For phyloP this
# turns "conserved vs accelerated" into "under selection either way".
ANNOTATION_ABS = False

# What to correlate on the attribution side:
#   "signed" -- the RC-averaged signed attribution (positive = pushes the
#               prediction up). Asks whether the annotation tracks the
#               DIRECTION of influence.
#   "abs"    -- |RC-averaged attribution|. Asks whether the annotation
#               tracks the MAGNITUDE of influence, regardless of direction.
# The across-loci positional ranking (below) is applied to whichever of
# these is selected, so "abs" ranks magnitudes against magnitudes.
ATTR_VALUE = "signed"  # "abs" | "signed"

# Per-orientation attribution normalization: "l1_mass" or "pred_diff".
# See seq_models/ig_attribution_norm.py. Note that under the across-loci
# per-position ranking this choice only matters through the per-locus
# scalar it divides out -- but that scalar is correlated with locus
# properties under "pred_diff", so "l1_mass" stays the default.
NORM_SCHEME = "l1_mass"

# Sample filters, mirroring scripts/ig/ig_by_pos.py.
PRED_DIFF_THRESHOLD = 0.05
REL_DELTA_THRESHOLD = 0.05

# Drop a locus if less than this fraction of its flanking positions have a
# non-NaN annotation value. Loci near a contig end (whose flank window runs
# off the chromosome) and loci on chromosomes absent from the annotation
# file are caught here.
MIN_LOCUS_COVERAGE = 0.95

# After dropping low-coverage loci, positions are restricted to those
# annotated in EVERY remaining locus (see the module docstring). Error out
# if that intersection retains less than this fraction of flanking
# positions -- at that point the common-mask design is throwing away too
# much and MIN_LOCUS_COVERAGE should be raised instead.
MIN_COMMON_POSITION_FRAC = 0.5

# Bootstrap (over loci) for the CI on the mean per-locus rho.
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

# Permutation null: number of random attribution<->annotation re-pairings.
N_PERMUTATIONS = 10_000

# Permutations are generated in chunks of this many to bound peak memory
# (each chunk is PERM_CHUNK x n_loci int64).
PERM_CHUNK = 1000

# Seed for every random draw in this script (bootstrap + permutations).
SEED = 147

# Guard on the n_loci x n_loci pairwise correlation matrix (float64).
MAX_PAIR_MATRIX_GB = 4.0

# Output. SAVE_PATH=None -> plt.show(); SAVE_RESULTS_PATH=None -> no json.
SAVE_PATH = None
SAVE_RESULTS_PATH = None


# ===========================================================================
# Loading
# ===========================================================================

def load_attributions(attr_dir, task):
	"""Load the npz + meta.json and return the pieces this script needs."""
	data = np.load(os.path.join(attr_dir, "attributions.npz"), allow_pickle=True)
	with open(os.path.join(attr_dir, "meta.json")) as f:
		meta = json.load(f)

	if task not in meta["task_names"]:
		raise KeyError(
			f"TASK {task!r} is not one of the checkpoint's targets "
			f"{meta['task_names']}."
		)
	return data, meta


def load_str_coords(str_data_path, ids):
	"""chrom/str_start/str_end for each locus id, in the order given."""
	tsv = pd.read_csv(str_data_path, sep="\t").drop_duplicates("ID").set_index("ID")

	missing = pd.Index(ids).difference(tsv.index)
	if len(missing):
		raise KeyError(
			f"{len(missing)} attribution locus id(s) are absent from "
			f"{str_data_path!r} (e.g. {missing[:5].tolist()}). Is "
			f"STR_DATA_PATH the tsv this checkpoint was trained on?"
		)
	return tsv.loc[ids, ["chrom", "str_start", "str_end"]].reset_index()


# ===========================================================================
# Annotation tracks over the flanks
# ===========================================================================

def _flank_values(bw, chrom, start, end, chrom_len):
	"""bw.values() over [start, end), NaN-padded where the window runs off
	the contig (pyBigWig raises rather than clipping)."""
	out = np.full(end - start, np.nan, dtype=np.float64)
	lo, hi = max(start, 0), min(end, chrom_len)
	if lo < hi:
		out[lo - start:hi - start] = bw.values(chrom, lo, hi, numpy=True)
	return out


def extract_annotation_tracks(bw, coords_df, n_flank, use_abs):
	"""(n_loci, 2 * n_flank) per-bp annotation values in forward genomic
	order: left flank [str_start - n_flank, str_start) then right flank
	[str_end, str_end + n_flank).

	This is the same coordinate convention the model's flanks are built
	with (seq_models/caduceus/data.py), so column k lines up bp-for-bp
	with the attribution profile's flank column k.
	"""
	chrom_lengths = bw.chroms()

	chrom_arr = coords_df["chrom"].to_numpy()
	start_arr = coords_df["str_start"].to_numpy()
	end_arr = coords_df["str_end"].to_numpy()

	tracks = np.full((len(coords_df), 2 * n_flank), np.nan, dtype=np.float64)
	n_skipped_chrom = 0

	for i in tqdm(range(len(coords_df)), desc="  annotation tracks"):
		chrom = chrom_arr[i]
		if chrom not in chrom_lengths:
			n_skipped_chrom += 1
			continue  # all-NaN row, dropped by the coverage filter

		chrom_len = chrom_lengths[chrom]
		start, end = int(start_arr[i]), int(end_arr[i])

		tracks[i, :n_flank] = _flank_values(
			bw, chrom, start - n_flank, start, chrom_len
		)
		tracks[i, n_flank:] = _flank_values(
			bw, chrom, end, end + n_flank, chrom_len
		)

	if n_skipped_chrom:
		print(f"  {n_skipped_chrom} loci on chromosomes absent from the "
		      f"annotation file (all-NaN)")

	return np.abs(tracks) if use_abs else tracks


# ===========================================================================
# Rank transforms
# ===========================================================================

def positional_quantiles(attr_flanks):
	"""Rank each position's attribution values ACROSS loci, as quantiles.

	attr_flanks is (n_loci, n_positions); the return has the same shape,
	with entry [l, p] = (rank of locus l among all loci at position p) /
	n_loci, ties averaged. Every column is ranked over the same n_loci, so
	the result is comparable across columns -- which is what makes the
	downstream within-locus (across-position) correlation meaningful.

	This is the step that removes the positional bias: after it, each
	position has an identical uniform marginal, so the shared
	distance-from-STR profile cannot contribute to any per-locus
	correlation.
	"""
	n_loci = attr_flanks.shape[0]
	return (rankdata(attr_flanks, axis=0) - 0.5) / n_loci


def row_rank_standardize(matrix):
	"""Row-wise ranks, centered and scaled to unit norm.

	With Za/Zb built this way, Za[i] @ Zb[j] is exactly the Spearman rho
	between row i of one matrix and row j of the other (Spearman is
	Pearson-of-ranks, and this is that Pearson written as a dot product).
	Ties are averaged, so the identity holds under ties too.

	Rows with zero rank variance (a constant track) would divide by zero;
	callers must drop them first.
	"""
	ranks = rankdata(matrix, axis=1)
	centered = ranks - ranks.mean(axis=1, keepdims=True)
	norms = np.linalg.norm(centered, axis=1, keepdims=True)
	return centered / norms


# ===========================================================================
# Statistics
# ===========================================================================

def bootstrap_mean_ci(values, n_boot, ci_level, rng):
	"""Percentile bootstrap CI for the mean, resampling loci."""
	n = len(values)
	boot_idx = rng.integers(0, n, size=(n_boot, n))
	boot_means = values[boot_idx].mean(axis=1)

	alpha = 1 - ci_level
	return (
		float(np.percentile(boot_means, 100 * (alpha / 2))),
		float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
		boot_means,
	)


def permutation_null(pair_corr, n_perm, chunk, rng):
	"""Mean rho under random attribution<->annotation re-pairings.

	pair_corr[i, j] is the Spearman rho between attribution track i and
	annotation track j, so a permutation `perm` is scored by
	mean_i pair_corr[i, perm[i]] -- no correlation is recomputed. A plain
	permutation is used (not a derangement): the expected number of loci
	left paired with themselves is 1 regardless of n, which is negligible
	and biases the null, if at all, toward the observed value.
	"""
	n = pair_corr.shape[0]
	rows = np.arange(n)
	out = np.empty(n_perm, dtype=np.float64)

	done = 0
	with tqdm(total=n_perm, desc="  permutations") as pbar:
		while done < n_perm:
			size = min(chunk, n_perm - done)
			# rng.permuted permutes each row independently.
			perms = rng.permuted(np.tile(rows, (size, 1)), axis=1)
			out[done:done + size] = pair_corr[rows[None, :], perms].mean(axis=1)
			done += size
			pbar.update(size)

	return out


# ===========================================================================
# Main
# ===========================================================================

def main():
	rng = np.random.default_rng(SEED)

	entry = au.get_entry(ANNOTATION_TYPE)
	if entry["value_type"] != "continuous":
		raise ValueError(
			f"ANNOTATION_TYPE {ANNOTATION_TYPE!r} has value_type "
			f"{entry['value_type']!r}; this script handles continuous "
			f"annotations only."
		)
	if ATTR_VALUE not in ("abs", "signed"):
		raise ValueError(f"ATTR_VALUE must be 'abs' or 'signed', got {ATTR_VALUE!r}")

	# --- attributions: filter, normalize, RC-symmetrize -------------------

	data, meta = load_attributions(ATTR_DIR, TASK)
	flanks = ign.FlankLayout.from_meta(meta)

	# Human-readable name for TASK, e.g. "variation" -> "heterozygosity" --
	# meta.json's "targets" maps each task name to the label column it was
	# trained on (see {ATTR_DIR}/meta.json). Falls back to TASK itself if
	# the checkpoint doesn't carry that mapping.
	task_display = meta.get("targets", {}).get(TASK, TASK)

	print("=" * 70)
	print(f"Intra-locus IG x {ANNOTATION_TYPE} correlation  "
	      f"[task={task_display}]")
	print("=" * 70)

	loci = ign.per_locus_attributions(
		data[f"attributions_{TASK}"],
		data[f"raw_predictions_{TASK}"],
		data[f"raw_baseline_predictions_{TASK}"],
		data[f"relative_convergence_deltas_{TASK}"],
		data["rev_comp"], flanks, ids=data["ids"], scheme=NORM_SCHEME,
		pred_diff_threshold=PRED_DIFF_THRESHOLD,
		rel_delta_threshold=REL_DELTA_THRESHOLD,
	)
	samp = loci.sample_filter

	print(f"\nSamples: {samp.n_kept} kept, {samp.n_filtered} filtered "
	      f"(of {samp.n_total})")
	print(f"  excluded by pred_diff <= {PRED_DIFF_THRESHOLD}: "
	      f"{samp.n_excluded_pred_diff}")
	print(f"  excluded by rel_delta > {REL_DELTA_THRESHOLD}: "
	      f"{samp.n_excluded_rel_delta}")
	print(f"Loci with both orientations kept: {loci.n_loci} of "
	      f"{loci.n_loci_total}")
	print(f"Normalization: {ign.scheme_description(NORM_SCHEME)}; "
	      f"attribution value: {ATTR_VALUE}")

	# --- annotation tracks over the same flanking positions ---------------

	coords = load_str_coords(STR_DATA_PATH, loci.locus_ids)

	n_flank = flanks.n_left_flank
	if flanks.n_right_flank != n_flank:
		raise ValueError(
			f"Asymmetric flanks ({n_flank} left, {flanks.n_right_flank} "
			f"right); the forward/RC combination assumes they match."
		)

	print(f"\nReading {ANNOTATION_TYPE} over {2 * n_flank:,} flanking bp "
	      f"for {len(coords)} loci"
	      + (" (abs)" if ANNOTATION_ABS else "") + "...")
	bw = au.load_bigwig(entry)
	try:
		anno = extract_annotation_tracks(bw, coords, n_flank, ANNOTATION_ABS)
	finally:
		bw.close()

	# --- locus coverage filter + common position mask ---------------------

	valid = np.isfinite(anno)
	coverage = valid.mean(axis=1)
	keep_loci = coverage >= MIN_LOCUS_COVERAGE

	n_dropped = int((~keep_loci).sum())
	if n_dropped:
		print(f"  dropped {n_dropped} loci with annotation coverage < "
		      f"{MIN_LOCUS_COVERAGE:.0%}")
	if not keep_loci.any():
		raise ValueError(
			"No loci meet MIN_LOCUS_COVERAGE. Check that the annotation "
			"and the STR tsv use the same genome build."
		)

	loci = loci.select(keep_loci)
	anno = anno[keep_loci]
	coords = coords[keep_loci].reset_index(drop=True)

	common = valid[keep_loci].all(axis=0)
	n_common = int(common.sum())
	frac_common = n_common / (2 * n_flank)
	print(f"  common annotated positions: {n_common:,} of {2 * n_flank:,} "
	      f"({frac_common:.1%})")
	if frac_common < MIN_COMMON_POSITION_FRAC:
		raise ValueError(
			f"Only {frac_common:.1%} of flanking positions are annotated in "
			f"every locus, below MIN_COMMON_POSITION_FRAC="
			f"{MIN_COMMON_POSITION_FRAC:.0%}. Raise MIN_LOCUS_COVERAGE to "
			f"drop the sparse loci instead of the shared positions."
		)

	anno = anno[:, common]

	# Attribution flanks in forward genomic order, matching the annotation
	# columns. avg_signed is already RC-flipped into genomic coordinates.
	signed = loci.avg_signed
	attr_flanks = np.concatenate(
		[signed[:, flanks.lf_start:flanks.lf_end],
		 signed[:, flanks.rf_start:flanks.rf_end]],
		axis=1,
	)
	if ATTR_VALUE == "abs":
		attr_flanks = np.abs(attr_flanks)
	attr_flanks = attr_flanks[:, common]

	# --- drop constant tracks (undefined Spearman) ------------------------

	varying = (
		(anno.max(axis=1) > anno.min(axis=1)) &
		(attr_flanks.max(axis=1) > attr_flanks.min(axis=1))
	)
	if not varying.all():
		print(f"  dropped {int((~varying).sum())} loci with a constant "
		      f"attribution or annotation track")
		loci = loci.select(varying)
		anno = anno[varying]
		attr_flanks = attr_flanks[varying]
		coords = coords[varying].reset_index(drop=True)

	n_loci = attr_flanks.shape[0]
	if n_loci < 3:
		raise ValueError(f"Only {n_loci} loci survive; nothing to test.")

	print(f"\nAnalysis set: {n_loci} loci x {n_common:,} positions")

	# --- rank transforms and the pairwise rho matrix ----------------------

	pair_gb = n_loci * n_loci * 8 / 1e9
	if pair_gb > MAX_PAIR_MATRIX_GB:
		raise MemoryError(
			f"The {n_loci} x {n_loci} pairwise correlation matrix needs "
			f"{pair_gb:.1f} GB, over MAX_PAIR_MATRIX_GB="
			f"{MAX_PAIR_MATRIX_GB}. Raise the cap or subsample loci."
		)

	print("Ranking attributions across loci, per position...")
	attr_q = positional_quantiles(attr_flanks)

	print("Building the pairwise Spearman matrix "
	      f"({n_loci} x {n_loci}, {pair_gb * 1e3:.0f} MB)...")
	za = row_rank_standardize(attr_q)
	zb = row_rank_standardize(anno)
	pair_corr = za @ zb.T

	rho = np.diag(pair_corr).copy()  # observed per-locus rho
	rho_bar = float(rho.mean())

	# --- bootstrap CI and permutation p-value -----------------------------

	print(f"\nBootstrapping the mean over loci ({N_BOOTSTRAP} resamples)...")
	ci_lo, ci_hi, _ = bootstrap_mean_ci(rho, N_BOOTSTRAP, CI_LEVEL, rng)

	print(f"Permutation null ({N_PERMUTATIONS:,} re-pairings)...")
	null = permutation_null(pair_corr, N_PERMUTATIONS, PERM_CHUNK, rng)

	n_extreme = int(np.sum(np.abs(null) >= abs(rho_bar)))
	p_value = (1 + n_extreme) / (N_PERMUTATIONS + 1)

	# --- report -----------------------------------------------------------

	frac_pos = float((rho > 0).mean())
	print("\n" + "=" * 70)
	print("RESULTS")
	print("=" * 70)
	print(f"mean per-locus Spearman rho : {rho_bar:+.4f}")
	print(f"{int(CI_LEVEL * 100)}% bootstrap CI          : "
	      f"[{ci_lo:+.4f}, {ci_hi:+.4f}]")
	print(f"median / IQR                : {np.median(rho):+.4f} / "
	      f"[{np.percentile(rho, 25):+.4f}, {np.percentile(rho, 75):+.4f}]")
	print(f"loci with rho > 0           : {frac_pos:.1%} ({int((rho > 0).sum())} "
	      f"of {n_loci})")
	print(f"permutation null mean / sd  : {null.mean():+.5f} / {null.std():.5f}")
	print(f"|null| >= |observed|        : {n_extreme} of {N_PERMUTATIONS:,}")
	print(f"two-sided permutation p     : {p_value:.3g}"
	      + (f"  (< {1 / (N_PERMUTATIONS + 1):.3g}, the floor for "
	         f"{N_PERMUTATIONS:,} permutations)" if n_extreme == 0 else ""))

	results = {
		"attr_dir": ATTR_DIR,
		"task": TASK,
		"task_display": task_display,
		"annotation_type": ANNOTATION_TYPE,
		"annotation_abs": ANNOTATION_ABS,
		"attr_value": ATTR_VALUE,
		"norm_scheme": NORM_SCHEME,
		"pred_diff_threshold": PRED_DIFF_THRESHOLD,
		"rel_delta_threshold": REL_DELTA_THRESHOLD,
		"seed": SEED,
		"n_loci": n_loci,
		"n_positions": n_common,
		"mean_rho": rho_bar,
		"ci_level": CI_LEVEL,
		"ci_lo": ci_lo,
		"ci_hi": ci_hi,
		"median_rho": float(np.median(rho)),
		"frac_rho_positive": frac_pos,
		"n_bootstrap": N_BOOTSTRAP,
		"n_permutations": N_PERMUTATIONS,
		"null_mean": float(null.mean()),
		"null_sd": float(null.std()),
		"n_null_extreme": n_extreme,
		"p_value": p_value,
	}

	if SAVE_RESULTS_PATH:
		os.makedirs(os.path.dirname(os.path.abspath(SAVE_RESULTS_PATH)),
		            exist_ok=True)
		with open(SAVE_RESULTS_PATH, "w") as f:
			json.dump(results, f, indent=2)
		print(f"\nWrote {SAVE_RESULTS_PATH}")

	plot(rho, rho_bar, ci_lo, ci_hi, null, p_value, n_loci, n_common,
	     task_display)


# ===========================================================================
# Plot
# ===========================================================================

def plot(rho, rho_bar, ci_lo, ci_hi, null, p_value, n_loci, n_positions,
         task_display):
	"""Left: the per-locus rho distribution. Right: the permutation null
	against the observed mean, on its own (much narrower) scale."""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	abs_label = " |attr|" if ATTR_VALUE == "abs" else " signed attr"
	anno_label = f"|{ANNOTATION_TYPE}|" if ANNOTATION_ABS else ANNOTATION_TYPE

	ax1.hist(rho, bins=60, color="#3170ad", alpha=0.75, edgecolor="none")
	ax1.axvline(0, color="gray", lw=1, ls=":")
	ax1.axvspan(ci_lo, ci_hi, color="#e8443a", alpha=0.25,
	            label=f"{int(CI_LEVEL * 100)}% CI of mean")
	ax1.axvline(rho_bar, color="#e8443a", lw=1.5,
	            label=f"mean = {rho_bar:+.4f}")
	ax1.set_xlabel("Per-locus Spearman rho\n(positional-quantile"
	               f"{abs_label} vs {anno_label})")
	ax1.set_ylabel("Loci")
	ax1.set_title(f"[{task_display}] Per-locus correlation\n{n_loci} loci x "
	              f"{n_positions:,} positions")
	ax1.legend(fontsize=8)

	ax2.hist(null, bins=60, color="#999999", alpha=0.8, edgecolor="none",
	         label=f"null ({len(null):,} re-pairings)")
	ax2.axvline(0, color="gray", lw=1, ls=":")
	ax2.axvline(rho_bar, color="#e8443a", lw=1.5,
	            label=f"observed = {rho_bar:+.4f}")
	ax2.set_xlabel("Mean per-locus Spearman rho")
	ax2.set_ylabel("Permutations")
	ax2.set_title(f"[{task_display}] Permutation null\ntwo-sided p = {p_value:.3g}")
	ax2.legend(fontsize=8)
	# The observed mean is usually far outside the null; keep both visible.
	lo = min(null.min(), rho_bar) - 0.05 * (null.max() - null.min() + 1e-9)
	hi = max(null.max(), rho_bar) + 0.05 * (null.max() - null.min() + 1e-9)
	ax2.set_xlim(lo, hi)

	fig.suptitle(
		f"Intra-locus IG x {anno_label} correlation  "
		f"[{ign.scheme_description(NORM_SCHEME)}, "
		f"pred_diff>{PRED_DIFF_THRESHOLD}, "
		f"rel_delta<={REL_DELTA_THRESHOLD}, seed={SEED}]",
		fontsize=10,
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
