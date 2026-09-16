"""Shared per-locus IG attribution normalization and forward/RC combination.

Every downstream IG-by-position plot (scripts/ig/ig_by_pos.py,
ig_by_pos_label_strat.py, ig_by_pos_error_strat.py,
scripts/annotation_analysis/str_anno_overlap_ig_by_pos.py,
scripts/ig/single_vis.py, kmer_analysis/kmer_extract.py) used to carry its
own copy of the same three steps, and they had drifted apart. This module
is the single definition. It lives in the installed seq_models package so
that every one of those directories can import it without a sys.path
hack, next to the integrated_gradients.py runs that produce the arrays it
consumes.

Terminology, per locus l and orientation s in {fwd, rc}:

  a[s, p]  raw signed IG attribution at model-input position p
           (attributions_{TASK} row for that sample).
  d[s]     raw_predictions - raw_baseline_predictions for that sample --
           the signed prediction shift from baseline. IG's completeness
           axiom gives sum_p a[s, p] == d[s].
  F        flank positions (left_flank union right_flank) in model-input
           index space, excluding the prefix prompt, the STR bases, the
           STR prompt/gap tokens and any suffix prompt. Only the flanks
           are ever summed because the IG baseline perturbs flanking
           positions only -- everything else is ~zero by construction.
  m[s]     sum over p in F of |a[s, p]|, the absolute attribution mass.

Sample filters (applied before any normalization): a locus is kept only
if BOTH orientations have |d[s]| > pred_diff_threshold and
relative_convergence_deltas <= rel_delta_threshold.

Step 1 -- per-orientation scaling, selected by `scheme`:

  "l1_mass" (default)
      a_norm[s, p] = a[s, p] / m[s]
    so sum_{p in F} |a_norm[s, p]| == 1 for each orientation. The SIGNED
    sum is then d[s] / m[s], which lies in [-1, 1] and is NOT 1: it is
    the "net direction" fraction and is expected to vary across loci.

  "pred_diff"
      a_norm[s, p] = a[s, p] / |d[s]|
    so sum_p a_norm[s, p] == sign(d[s]) = +-1 for each orientation. Here
    the absolute mass m[s] / |d[s]| >= 1 varies across loci, and that
    per-locus scalar is strongly correlated with locus properties (e.g.
    r = -0.56 against mode copy number on str3_f5000_fiv_v4), so it shows
    up as a flat per-group vertical offset in any stratified plot.

Both schemes keep the SIGN of a[s, p]; neither takes an absolute value at
this step. Note in particular that dividing by the SIGNED d[s] (which
several of these scripts used to do) is wrong: it makes "positive" mean
"in the direction this locus moved from baseline" rather than "pushes the
prediction up", so the result is sign-flipped for below-baseline loci and
the forward/RC average and any downstream sign analysis are corrupted.

The two schemes are mutually exclusive alternatives, not composable
stages: applying "pred_diff" and then L1-normalizing is identical to
"l1_mass" alone, because |d[s]| is constant over p and cancels.

Step 2 -- RC flip and forward/RC averaging. In the RC run the model sees
    prefix | RC(genomic_right_flank) | RC(STR) | RC(genomic_left_flank)
so the RC's left-flank model positions hold the reverse complement of the
genomic RIGHT flank, STR-adjacent end at lf_end-1. Reversing each RC
flank aligns it with the forward flank by genomic distance from the STR.
The average is taken on SIGNED, normalized attributions. Under "l1_mass"
the averaged vector has unit absolute mass only where the two
orientations agree in sign everywhere; in general its mass is <= 1, the
shortfall measuring forward/RC disagreement. That is expected.

Step 3 -- absolute value, taken only where a magnitude is wanted:
    avg_abs[l, p] = |avg_signed[l, p]|
This is abs-of-mean, not the mean-of-abs these scripts used to compute.
abs-of-mean is the magnitude of the attribution of the strand-averaged
predictor, so positions where forward and RC disagree in sign are
correctly down-weighted instead of being summed as if they agreed.
"""

from dataclasses import dataclass

import numpy as np

# Valid values for the `scheme` argument / a caller's NORM_SCHEME config.
NORM_SCHEMES = ("l1_mass", "pred_diff")

# Default scheme for every caller.
DEFAULT_NORM_SCHEME = "l1_mass"


@dataclass(frozen=True)
class FlankLayout:
	"""Model-input index boundaries derived from meta.json sequence_layout.

	The single definition of lf_start/lf_end/rf_start/rf_end, which every
	caller previously recomputed inline.
	"""

	n_prefix: int
	n_left_flank: int
	n_left_str: int
	n_str_prompt: int
	n_right_str: int
	n_right_flank: int
	n_suffix_prompt: int
	seq_len: int

	lf_start: int
	lf_end: int
	rf_start: int
	rf_end: int

	@classmethod
	def from_layout(cls, layout):
		"""Build from meta.json's "sequence_layout" dict."""
		n_prefix = layout["n_prefix_prompt"]
		n_left_flank = layout["n_flanking_bp"]
		n_left_str = layout["n_str_bp"]
		n_str_prompt = layout["n_str_prompt"]
		n_right_str = layout["n_str_bp"]
		n_right_flank = layout["n_flanking_bp"]
		# A trailing suffix_prompt block (if any) sits after right_flank and
		# is never indexed here, so it doesn't affect the boundary math.
		n_suffix_prompt = layout.get("n_suffix_prompt", 0)

		lf_start = n_prefix
		lf_end = n_prefix + n_left_flank
		rf_start = lf_end + n_left_str + n_str_prompt + n_right_str
		rf_end = rf_start + n_right_flank

		return cls(
			n_prefix=n_prefix,
			n_left_flank=n_left_flank,
			n_left_str=n_left_str,
			n_str_prompt=n_str_prompt,
			n_right_str=n_right_str,
			n_right_flank=n_right_flank,
			n_suffix_prompt=n_suffix_prompt,
			seq_len=layout["seq_len"],
			lf_start=lf_start,
			lf_end=lf_end,
			rf_start=rf_start,
			rf_end=rf_end,
		)

	@classmethod
	def from_meta(cls, meta):
		"""Build from a loaded meta.json dict."""
		return cls.from_layout(meta["sequence_layout"])

	@property
	def region_blocks(self):
		"""(name, start, end) blocks in model-input index space."""
		return [
			("prefix_prompt", 0, self.n_prefix),
			("left_flank", self.lf_start, self.lf_end),
			("left_str", self.lf_end, self.lf_end + self.n_left_str),
			("str_prompt", self.lf_end + self.n_left_str,
			 self.lf_end + self.n_left_str + self.n_str_prompt),
			("right_str", self.lf_end + self.n_left_str + self.n_str_prompt,
			 self.rf_start),
			("right_flank", self.rf_start, self.rf_end),
		]

	def flank_mass(self, matrix):
		"""Sum of |values| over both flanks, per row.

		Only the flanks are summed: the IG baseline perturbs flanking
		positions only, so prompt tokens and STR bases are ~zero by
		construction and are never plotted.
		"""
		return (
			np.abs(matrix[:, self.lf_start:self.lf_end]).sum(axis=1) +
			np.abs(matrix[:, self.rf_start:self.rf_end]).sum(axis=1)
		)


@dataclass
class SampleFilter:
	"""Per-sample filter mask plus the counts each script reports."""

	mask: np.ndarray
	n_total: int
	n_kept: int
	n_excluded_pred_diff: int
	n_excluded_rel_delta: int

	@property
	def n_filtered(self):
		return self.n_total - self.n_kept


@dataclass
class LocusAttributions:
	"""Per-locus forward/RC-combined attributions and the scalars normalized out.

	Attributes
	----------
	avg_signed : (n_loci, seq_len)
		Step-2 output: signed, normalized, RC-flipped forward/RC average.
		Non-flank positions are zero.
	avg_abs : (n_loci, seq_len)
		|avg_signed| (step 3).
	locus_ids : (n_loci,) or None
		Locus ids, if `ids` was supplied.
	mass_fwd, mass_rc : (n_loci,)
		Raw absolute attribution mass m[s] over the flanks, before any
		normalization. Normalized out of the profiles but kept for
		reporting.
	pred_diff_fwd, pred_diff_rc : (n_loci,)
		Signed d[s] = raw_predictions - raw_baseline_predictions.
	fwd_idx, rc_idx : (n_loci,)
		Indices into the original per-sample arrays, for pulling any other
		per-sample quantity (labels, predictions) for the kept loci.
	"""

	avg_signed: np.ndarray
	avg_abs: np.ndarray
	locus_ids: np.ndarray
	mass_fwd: np.ndarray
	mass_rc: np.ndarray
	pred_diff_fwd: np.ndarray
	pred_diff_rc: np.ndarray
	fwd_idx: np.ndarray
	rc_idx: np.ndarray
	scheme: str
	sample_filter: SampleFilter
	n_loci_total: int

	@property
	def n_loci(self):
		return len(self.fwd_idx)

	@property
	def mass_ratio(self):
		"""Mean over orientations of m[s] / |d[s]|.

		The per-locus scalar that "pred_diff" leaves in the profiles and
		"l1_mass" divides out: total |attribution| expressed in units of
		the prediction shift, >= 1, larger where positive and negative
		attributions cancel more. This is the quantity the stratified
		scripts print per group.
		"""
		return (
			self.mass_fwd / np.abs(self.pred_diff_fwd) +
			self.mass_rc / np.abs(self.pred_diff_rc)
		) / 2

	def select(self, keep):
		"""Return a copy restricted to the loci selected by a boolean/index array.

		Normalization is per-locus independent, so subsetting after the
		fact is equivalent to having filtered first. Callers use this for
		locus-level filters applied downstream (e.g. dropping label==0
		loci for MAPE, or loci with no annotation assignment).
		"""
		return LocusAttributions(
			avg_signed=self.avg_signed[keep],
			avg_abs=self.avg_abs[keep],
			locus_ids=(None if self.locus_ids is None
			           else self.locus_ids[keep]),
			mass_fwd=self.mass_fwd[keep],
			mass_rc=self.mass_rc[keep],
			pred_diff_fwd=self.pred_diff_fwd[keep],
			pred_diff_rc=self.pred_diff_rc[keep],
			fwd_idx=self.fwd_idx[keep],
			rc_idx=self.rc_idx[keep],
			scheme=self.scheme,
			sample_filter=self.sample_filter,
			n_loci_total=self.n_loci_total,
		)


def check_scheme(scheme):
	"""Raise if `scheme` is not one of NORM_SCHEMES."""
	if scheme not in NORM_SCHEMES:
		raise ValueError(
			f"Unknown normalization scheme {scheme!r}; "
			f"expected one of {NORM_SCHEMES}."
		)
	return scheme


def magnitude_axis_label(scheme):
	"""Y-axis label for a plot of |normalized attribution|."""
	check_scheme(scheme)
	if scheme == "l1_mass":
		return "|Normalized attribution|\n(fraction of total |attribution|)"
	return "|Normalized attribution|\n(fraction of total prediction shift)"


def signed_axis_label(scheme):
	"""Y-axis label for a plot of signed normalized attribution."""
	check_scheme(scheme)
	if scheme == "l1_mass":
		return "Normalized attribution\n(fraction of total |attribution|)"
	return "Normalized attribution\n(fraction of total prediction shift)"


def scheme_description(scheme):
	"""Short phrase for plot titles / console output."""
	check_scheme(scheme)
	if scheme == "l1_mass":
		return "L1 mass normalized (per orientation)"
	return "prediction-shift normalized (per orientation)"


def filter_samples(raw_pred_diffs, relative_deltas,
                    pred_diff_threshold, rel_delta_threshold):
	"""Per-sample filter shared by every caller.

	Drops samples whose flanking sequence barely moves the prediction
	(|d| <= pred_diff_threshold; their attributions are noise and their
	relative convergence deltas are inflated by the small denominator)
	and samples whose IG approximation did not converge
	(relative_convergence_deltas > rel_delta_threshold).

	Either threshold may be 0 / None to disable that filter.
	"""
	n_samples = len(raw_pred_diffs)
	mask = np.ones(n_samples, dtype=bool)

	n_excluded_pred_diff = 0
	n_excluded_rel_delta = 0

	if pred_diff_threshold and pred_diff_threshold > 0:
		fail_pred = np.abs(raw_pred_diffs) <= pred_diff_threshold
		n_excluded_pred_diff = int(np.sum(fail_pred))
		mask &= ~fail_pred

	if rel_delta_threshold is not None:
		fail_delta = relative_deltas > rel_delta_threshold
		# Count only samples not already dropped by the pred_diff filter,
		# so the two reported counts partition the exclusions.
		n_excluded_rel_delta = int(np.sum(mask & fail_delta))
		mask &= ~fail_delta

	return SampleFilter(
		mask=mask,
		n_total=n_samples,
		n_kept=int(np.sum(mask)),
		n_excluded_pred_diff=n_excluded_pred_diff,
		n_excluded_rel_delta=n_excluded_rel_delta,
	)


def normalize_signed(attributions, pred_diffs, flanks,
                      scheme=DEFAULT_NORM_SCHEME):
	"""Step 1: per-sample scaling, keeping the sign.

	Parameters
	----------
	attributions : (n, seq_len)
		Raw signed attributions a[s, :].
	pred_diffs : (n,)
		Signed d[s] = raw_predictions - raw_baseline_predictions.
	flanks : FlankLayout
	scheme : {"l1_mass", "pred_diff"}

	Returns
	-------
	(n, seq_len) float array of signed normalized attributions.
	"""
	check_scheme(scheme)

	attributions = np.asarray(attributions, dtype=np.float64)
	pred_diffs = np.asarray(pred_diffs, dtype=np.float64)

	if scheme == "l1_mass":
		denom = flanks.flank_mass(attributions)
	else:
		denom = np.abs(pred_diffs)

	if np.any(denom == 0):
		raise ValueError(
			f"{int(np.sum(denom == 0))} sample(s) have a zero {scheme} "
			f"normalization denominator; they should have been removed by "
			f"the pred_diff filter."
		)

	normed = attributions / denom[:, None]

	if not np.all(np.isfinite(normed)):
		raise ValueError(
			f"Non-finite values after {scheme} normalization "
			f"({int(np.sum(~np.isfinite(normed)))} entries)."
		)
	return normed


def combine_fwd_rc(fwd, rc, flanks):
	"""Step 2: flip the RC flanks into forward genomic order and average.

	Both inputs are (n_loci, seq_len) in model-input index space and must
	already be normalized (step 1). Non-flank positions of the result are
	left at zero -- they are ~zero by construction and are never plotted.
	"""
	lf_start, lf_end = flanks.lf_start, flanks.lf_end
	rf_start, rf_end = flanks.rf_start, flanks.rf_end

	avg = np.zeros_like(fwd)

	# Forward left flank pairs with the RC's right flank, reversed.
	avg[:, lf_start:lf_end] = (
		fwd[:, lf_start:lf_end] + rc[:, rf_start:rf_end][:, ::-1]
	) / 2
	# Forward right flank pairs with the RC's left flank, reversed.
	avg[:, rf_start:rf_end] = (
		fwd[:, rf_start:rf_end] + rc[:, lf_start:lf_end][:, ::-1]
	) / 2

	return avg


def pair_fwd_rc(rev_comp, ids=None):
	"""Return (fwd_indices, rc_indices) as 1:1 locus pairs.

	Asserts equal counts and, when `ids` is given, that the k-th forward
	and k-th RC sample are the same locus.
	"""
	rev_comp = np.asarray(rev_comp)
	fwd_indices = np.where(~rev_comp)[0]
	rc_indices = np.where(rev_comp)[0]

	if len(fwd_indices) != len(rc_indices):
		raise ValueError(
			f"Unequal forward ({len(fwd_indices)}) and RC "
			f"({len(rc_indices)}) sample counts. Cannot form 1:1 locus pairs."
		)
	if ids is not None and not np.array_equal(ids[fwd_indices], ids[rc_indices]):
		raise ValueError(
			"Forward/RC index pairing does not line up by id -- "
			"fwd_indices[k] and rc_indices[k] must be the same locus."
		)
	return fwd_indices, rc_indices


def per_locus_attributions(attributions, raw_predictions,
                            raw_baseline_predictions, relative_deltas,
                            rev_comp, flanks, *, ids=None,
                            scheme=DEFAULT_NORM_SCHEME,
                            pred_diff_threshold=0.05,
                            rel_delta_threshold=0.05,
                            sample_filter=None):
	"""Filter, normalize, RC-flip and average -- the whole pipeline.

	A locus is kept only if BOTH its orientations pass the sample filter.

	Parameters
	----------
	attributions : (n_samples, seq_len)
	raw_predictions, raw_baseline_predictions, relative_deltas : (n_samples,)
	rev_comp : (n_samples,) bool
	flanks : FlankLayout
	ids : (n_samples,) or None
		Locus ids; used to verify the forward/RC pairing and returned for
		the kept loci.
	scheme : {"l1_mass", "pred_diff"}
	pred_diff_threshold, rel_delta_threshold : float
		See filter_samples. Ignored if `sample_filter` is supplied.
	sample_filter : SampleFilter or None
		A precomputed filter (e.g. one a caller already built to report
		counts for an all-samples panel), reused instead of recomputing.

	Returns
	-------
	LocusAttributions
	"""
	check_scheme(scheme)

	raw_pred_diffs = (np.asarray(raw_predictions, dtype=np.float64) -
	                  np.asarray(raw_baseline_predictions, dtype=np.float64))

	if sample_filter is None:
		sample_filter = filter_samples(
			raw_pred_diffs, relative_deltas,
			pred_diff_threshold, rel_delta_threshold,
		)

	fwd_indices, rc_indices = pair_fwd_rc(rev_comp, ids=ids)
	n_loci_total = len(fwd_indices)

	pair_mask = sample_filter.mask[fwd_indices] & sample_filter.mask[rc_indices]
	if not np.any(pair_mask):
		raise ValueError("No locus pairs passed filters. Adjust thresholds.")

	kept_fwd_idx = fwd_indices[pair_mask]
	kept_rc_idx = rc_indices[pair_mask]

	fwd_raw = np.asarray(attributions[kept_fwd_idx], dtype=np.float64)
	rc_raw = np.asarray(attributions[kept_rc_idx], dtype=np.float64)

	# Raw mass, before normalization -- reported, not plotted.
	mass_fwd = flanks.flank_mass(fwd_raw)
	mass_rc = flanks.flank_mass(rc_raw)

	fwd_norm = normalize_signed(
		fwd_raw, raw_pred_diffs[kept_fwd_idx], flanks, scheme
	)
	rc_norm = normalize_signed(
		rc_raw, raw_pred_diffs[kept_rc_idx], flanks, scheme
	)

	avg_signed = combine_fwd_rc(fwd_norm, rc_norm, flanks)

	if not np.all(np.isfinite(avg_signed)):
		raise ValueError(
			"Non-finite values in the forward/RC averaged attributions."
		)

	return LocusAttributions(
		avg_signed=avg_signed,
		avg_abs=np.abs(avg_signed),
		locus_ids=(None if ids is None else np.asarray(ids)[kept_fwd_idx]),
		mass_fwd=mass_fwd,
		mass_rc=mass_rc,
		pred_diff_fwd=raw_pred_diffs[kept_fwd_idx],
		pred_diff_rc=raw_pred_diffs[kept_rc_idx],
		fwd_idx=kept_fwd_idx,
		rc_idx=kept_rc_idx,
		scheme=scheme,
		sample_filter=sample_filter,
		n_loci_total=n_loci_total,
	)
