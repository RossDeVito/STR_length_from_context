"""Correlate the presence/amount of nearby RepeatMasker elements -- by
class, and for any class at all -- in each STR's flanks with its
length/heterozygosity labels, over a range of flanking window sizes.

Like GC_content_and_gradient.py (and unlike the rest of this directory) this
script does not go through annotation_config.yaml / annotation_utils.py: the
repeat elements live in data/invalid_regions/mobile_elements.bed.gz, which is
an *exclusion* resource rather than a registered annotation, and the features
here are derived per-window rather than read as one value per locus.

IMPORTANT -- what the STR dataset already did with this same bed file
---------------------------------------------------------------------
data/scripts/create_STR_data_files/create_STR_data_files.py drops every locus
whose own [str_start, str_end) span overlaps ANY mobile element in this bed.
So by construction every locus analyzed here has exactly zero repeat overlap
on its own span (verified: 0 / 35,369 loci at str_len=4). Two consequences:

  1. The windows below are FLANK-ONLY -- the STR span is excluded, which is
     also the convention GC_content_and_gradient.py and seq_models/caduceus/
     data.py use. Including the span would only add a guaranteed-empty
     constant.
  2. Presence rates are censored relative to the genome at large: loci sitting
     INSIDE an element were removed, so what survives at small windows is
     specifically the set of STRs that sit at element BOUNDARIES. That turns
     out to be a large and very non-random group -- see "Reading the results"
     below. The presence-rate diagnostic figure exists so the effect sizes can
     be read against it.

Features
--------
For each window size n in WINDOW_SIZES, the window is the n-bp upstream flank
([str_start - n, str_start)) plus the n-bp downstream flank
([str_end, str_end + n)), each clipped to the chromosome, treated as one
combined window of up to 2n bp. For each repeat class two per-locus values are
derived from the class's merged (union'd) intervals:

  coverage: fraction of the window's available bp covered by that class,
            i.e. (covered bp upstream + covered bp downstream) / (window bp
            upstream + window bp downstream). Continuous.
  presence: whether that coverage is > 0, i.e. any element of the class
            overlaps either flank. Binary -- the direct reading of "is one
            nearby".

Repeat classes
--------------
The bed's 17 raw repClass values are folded (CLASS_GROUPING="canonical") to
the 8 classes the download.sh filter names: the "?" uncertainty suffix is
stripped (LTR? -> LTR) and the RNA subfamilies (snRNA/tRNA/rRNA/srpRNA/scRNA)
collapse into RNA. CLASS_GROUPING="raw" keeps all 17 instead, at the cost of
badly underpowered tests for the rare ones (scRNA/RNA/SINE? have only
hundreds to a couple thousand elements genome-wide).

An additional pseudo-class ANY is the union of ALL elements regardless of
class. It is a union, not the sum of the per-class coverages: elements of
different classes can overlap each other, and summing would double-count.

Statistics
----------
  coverage -> Spearman rho + two-sided p (scipy.stats.spearmanr), as in
              GC_content_and_gradient.py. Ties are pervasive here -- most loci
              have coverage exactly 0 at small windows -- and spearmanr's
              tie-averaged ranking handles that.
  presence -> Cliff's delta (present vs. absent loci) + two-sided Mann-Whitney
              U p-value, via the identity delta = 2*U/(n_x*n_y) - 1, the same
              helper used in str_anno_overlap_analysis.py. U gives tied pairs
              half credit, so this stays exact under ties -- which matters,
              since the length target (mode_copy_number) repeats across many
              loci. A (window, class) whose smaller group has < MIN_GROUP_N
              loci is recorded with a NaN effect and skipped.
              Sign convention: delta > 0 means loci WITH that class nearby
              have larger label values.

Every (window size, class, metric, target) combination run in one execution is
one test in the same family, so raw p-values are Bonferroni-corrected across
all of them (mirroring CLIFFS_DELTA_CORRECTION in str_anno_overlap_analysis.py
and the correction in GC_content_and_gradient.py); the plots mark each point
significant/not at SIGNIFICANCE_ALPHA on the corrected value.

The optional percentile bootstrap resamples loci with replacement and
recomputes the statistic from scratch each draw (not just re-indexing
precomputed ranks: which values tie changes with the resample). It defaults
OFF here -- unlike the GC script -- because this script runs ~300+ tests per
execution rather than ~36, and bootstrapping all of them costs ~20+ minutes.
Turn it on for a final figure.

Reading the results -- the boundary effect
-----------------------------------------
The effect sizes are largest at the SMALLEST windows (for str_len=4,
delta = +0.78 for ANY at n=1, ~ +0.72 at n=5, ~ +0.30 at n=5000), which is the
opposite of what a diffuse regional effect would look like. The abutting
analysis below (RUN_ABUTTING_ANALYSIS) exists to characterize that, and it
reports three things:

  1. 23.7% of str_len=4 loci sit exactly at an element boundary -- an element
     ending precisely at str_start or starting precisely at str_end. Their
     median mode_copy_number is 7.75 vs 4.25 for the rest (delta = +0.78).
  2. Those loci are motif-skewed: AAAT and ATTT are 61% of them vs 34% of all
     loci (1.98x and 1.54x enriched) -- the A-rich tracts at Alu poly-A tails,
     which RepeatMasker annotates right up to but not into.
  3. BUT that composition skew does NOT explain the label shift. Repeating the
     test within each of the top 10 motifs, where composition is fixed by
     construction, leaves it essentially intact: within-motif median delta
     +0.756 vs the pooled +0.781, every motif individually significant, and
     the non-A-rich motifs among the strongest (GTTT +0.83, CTTT +0.88).

So this is not a motif artifact. The most likely mechanism is ASCERTAINMENT,
from the interaction between where RepeatMasker places a boundary and the
span-overlap exclusion above. RepeatMasker ends an element where similarity to
its consensus ends, so an STR that is part of a repeat (an Alu A-tail, say) is
annotated as inside it and gets dropped -- UNLESS the tract has expanded past
the alignable region, which requires it to be long. The surviving boundary loci
are therefore enriched for long tracts by the selection rule itself.

That makes the small-window signal a property of how this dataset was built,
not evidence that repeat proximity predicts length in nature. The wider-window
results (n >= 500, where boundary loci are a small fraction of the present
group) are the ones to read as regional signal.

Figures
-------
Four, written to SAVE_DIR (or shown interactively when it is None):

  repeat_presence_vs_labels : Cliff's delta vs window size, one line per class
                              (ANY heavier, in black), one panel per target.
  repeat_coverage_vs_labels : the same for Spearman rho on covered fraction.
  repeat_presence_rate      : diagnostic -- the fraction of loci with each
                              class present, vs window size. Read the two
                              above against it: a class whose rate is still
                              near 0 has too few present loci to say much.
  repeat_abutting           : the three-panel boundary analysis (effect ->
                              candidate confound -> control). plot_abutting()
                              documents how to read each panel.

Configuration is via the editable constants below (no argparse), mirroring the
rest of this directory.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# ===========================================================================
# Configuration
# ===========================================================================

STR_LEN = 3
N_FLANKING = 10000  # must match the STR tsv's filename suffix

STR_DATA_PATH = f"../../data/STR_data/HipSTR_labeled_STRs/str_len_{STR_LEN}_n_flanking_{N_FLANKING}.tsv"
# Chromosome lengths only -- the .fa itself is never read here.
FAI_PATH = "../../data/reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai"
# 4-column bed (chrom, start, end, repClass) from data/invalid_regions/download.sh.
REPEAT_BED_PATH = "../../data/invalid_regions/mobile_elements.bed.gz"

# n-bp flank window(s), on EACH side of the STR (upstream and downstream) --
# i.e. a given n looks at up to 2*n bp per locus.
WINDOW_SIZES = [5, 10, 25, 50, 100, 250, 500, 1000, 5000]

# "canonical" -> fold the bed's 17 raw repClass values to the 8 named in
# download.sh (strip "?", collapse the RNA subfamilies). "raw" -> keep all 17.
CLASS_GROUPING = "canonical"

# Targets, mirroring str_anno_overlap_analysis.py / get_mean_baseline_performance.py.
TARGETS = {
	"length": "mode_copy_number",
	"heterozygosity": "heterozygosity",
}

# A presence test is skipped (NaN effect) unless BOTH the present and absent
# groups have at least this many loci. Small windows x rare classes hit this.
MIN_GROUP_N = 25

# Percentile bootstrap CI (over loci) for each effect size. OFF by default:
# this script runs ~300+ tests per execution (see module docstring), so a full
# bootstrap is a ~20+ minute run. Turn on for a final figure.
RUN_BOOTSTRAP = False
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

SEED = 147

# Significance threshold applied to the Bonferroni-corrected p-value (across
# every window_size x class x metric x target test run this execution) for the
# plots' significant/not-significant marker.
SIGNIFICANCE_ALPHA = 0.01

# Directly-abutting analysis: quantify the element-BOUNDARY loci (an element
# ending exactly at str_start, or starting exactly at str_end) that dominate
# the small-window signal, and test whether their label shift survives
# controlling for motif. See "Reading the results" in the module docstring.
RUN_ABUTTING_ANALYSIS = True
# How many motifs to show in the composition report/plot, and to stratify by.
ABUTTING_TOP_K_MOTIFS = 10

# Output. SAVE_DIR=None -> plt.show(); SAVE_RESULTS_PATH=None -> no json.
SAVE_DIR = None
SAVE_RESULTS_PATH = None


ANY_CLASS = "ANY"  # pseudo-class: union of all elements, any class

# Display/plot order for CLASS_GROUPING="canonical"; roughly by element count.
CANONICAL_CLASSES = [
	"SINE", "LINE", "LTR", "DNA", "Satellite", "Retroposon", "RC", "RNA",
]


# ===========================================================================
# Loading
# ===========================================================================

def load_str_data(path):
	"""Load HipSTR-labeled STRs, keeping one row per locus (drop the
	reverse-complement duplicate; it shares the same position/labels)."""
	df = pd.read_csv(path, sep="\t")
	df = df[~df["rev_comp"]].reset_index(drop=True)
	return df


def load_chrom_lengths(fai_path):
	"""{chrom: length} from a samtools .fai (col 0 = name, col 1 = length).

	Flank windows must be clipped to the chromosome so a locus near an end
	doesn't get credited with window bp that don't exist -- the coverage
	denominator would be wrong.
	"""
	fai = pd.read_csv(
		fai_path, sep="\t", header=None, usecols=[0, 1], names=["chrom", "length"],
	)
	return dict(zip(fai["chrom"], fai["length"]))


def fold_repeat_class(raw_class):
	"""Fold a raw repClass to its canonical class: drop the "?" uncertainty
	suffix (LTR? -> LTR), and collapse every RNA subfamily (snRNA, tRNA,
	rRNA, srpRNA, scRNA) into RNA."""
	base = raw_class.rstrip("?")
	return "RNA" if base.endswith("RNA") else base


def load_repeat_bed(path, keep_chroms, class_grouping):
	"""Load the 4-column repeat bed, restricted to `keep_chroms`, with a
	`repeat_class` column set by `class_grouping`."""
	bed_df = pd.read_csv(
		path, sep="\t", header=None,
		names=["chrom", "start", "end", "raw_class"], compression="infer",
	)
	bed_df = bed_df[bed_df["chrom"].isin(keep_chroms)].reset_index(drop=True)

	if class_grouping == "canonical":
		mapping = {c: fold_repeat_class(c) for c in bed_df["raw_class"].unique()}
		bed_df["repeat_class"] = bed_df["raw_class"].map(mapping)
	elif class_grouping == "raw":
		mapping = None
		bed_df["repeat_class"] = bed_df["raw_class"]
	else:
		raise ValueError(
			f"Unknown CLASS_GROUPING {class_grouping!r} (expected 'canonical' or 'raw')"
		)
	return bed_df, mapping


def class_order(bed_df, class_grouping):
	"""Plot/report order for the classes present, ANY last."""
	present = set(bed_df["repeat_class"].unique())
	if class_grouping == "canonical":
		ordered = [c for c in CANONICAL_CLASSES if c in present]
		ordered += sorted(present - set(CANONICAL_CLASSES))  # nothing expected
	else:
		counts = bed_df["repeat_class"].value_counts()
		ordered = list(counts.index)
	return ordered + [ANY_CLASS]


# ===========================================================================
# Merged-interval coverage index
# ===========================================================================

def merge_intervals(starts, ends):
	"""Sort and merge [start, end) intervals into a minimal disjoint set.

	Same result as the merge loop in GenomicIntersector.__init__
	(data/scripts/create_STR_data_files/create_STR_data_files.py), but
	vectorized -- this runs over ~4.8M bed rows, where a Python loop is the
	bottleneck. Abutting intervals (start == previous end) are merged too;
	that changes nothing about the union's bp, it just keeps the invariant
	"merged intervals are strictly separated", which covered_bp_before()
	relies on.

	RepeatMasker rows DO overlap and nest, so merging is required: summing
	raw interval lengths would double-count those bp.
	"""
	order = np.argsort(starts, kind="stable")
	s, e = starts[order], ends[order]

	running_max_end = np.maximum.accumulate(e)
	is_new = np.empty(len(s), dtype=bool)
	is_new[0] = True
	is_new[1:] = s[1:] > running_max_end[:-1]

	group_starts = np.flatnonzero(is_new)
	merged_starts = s[group_starts]
	# reduceat over the group boundaries: max end within each merged run.
	merged_ends = np.maximum.reduceat(running_max_end, group_starts)
	return merged_starts, merged_ends


def build_coverage_index(starts, ends):
	"""(merged_starts, merged_ends, prefix) for covered_bp_before().

	prefix[k] = total covered bp in merged intervals 0..k-1, so prefix has
	one more element than there are intervals.
	"""
	m_starts, m_ends = merge_intervals(starts, ends)
	prefix = np.concatenate([[0], np.cumsum(m_ends - m_starts)])
	return m_starts, m_ends, prefix


def covered_bp_before(index, pos):
	"""Total covered bp in [0, pos) for each position in `pos`, vectorized.

	Since the indexed intervals are disjoint and sorted, every interval
	before the one containing `pos` contributes its full length (the prefix
	sum) and at most one interval is partially counted.
	"""
	m_starts, m_ends, prefix = index
	# Last interval with start <= pos; -1 when pos precedes every interval.
	j = np.searchsorted(m_starts, pos, side="right") - 1
	valid = j >= 0
	jc = np.where(valid, j, 0)

	partial = np.minimum(m_ends[jc], pos) - m_starts[jc]
	partial = np.maximum(partial, 0)
	return np.where(valid, prefix[jc] + partial, 0).astype(np.int64)


def covered_bp_in(index, lo, hi):
	"""Covered bp within each half-open window [lo, hi)."""
	return covered_bp_before(index, hi) - covered_bp_before(index, lo)


def build_indices_by_class(bed_df, classes):
	"""{repeat_class: {chrom: coverage index}}, including the ANY union."""
	indices = {}
	for repeat_class in tqdm(classes, desc="  merging intervals"):
		if repeat_class == ANY_CLASS:
			sub = bed_df
		else:
			sub = bed_df[bed_df["repeat_class"] == repeat_class]

		per_chrom = {}
		for chrom, group in sub.groupby("chrom", observed=True):
			per_chrom[chrom] = build_coverage_index(
				group["start"].to_numpy(), group["end"].to_numpy()
			)
		indices[repeat_class] = per_chrom
	return indices


# ===========================================================================
# Per-locus features
# ===========================================================================

def compute_repeat_features(str_df, indices, classes, window_sizes, chrom_lens):
	"""Returns {window_size: {repeat_class: DataFrame(presence, coverage)}},
	one row per locus (aligned with str_df's index).

	Everything is vectorized per (class, chromosome); the only Python loops
	are over classes, chromosomes and window sizes, not loci.

	A locus/window with zero available bp (only possible for a locus flush
	against a chromosome end, which the >= N_FLANKING requirement makes
	unreachable in practice) gets coverage NaN and presence False.
	"""
	n_loci = len(str_df)
	chrom_arr = str_df["chrom"].to_numpy()
	start_arr = str_df["str_start"].to_numpy().astype(np.int64)
	end_arr = str_df["str_end"].to_numpy().astype(np.int64)

	coverage = {n: {c: np.full(n_loci, np.nan) for c in classes} for n in window_sizes}
	presence = {n: {c: np.zeros(n_loci, dtype=bool) for c in classes} for n in window_sizes}

	for chrom in tqdm(np.unique(chrom_arr), desc="  per-locus features"):
		rows = np.nonzero(chrom_arr == chrom)[0]
		chrom_len = chrom_lens[chrom]
		starts, ends = start_arr[rows], end_arr[rows]

		for n in window_sizes:
			# Upstream [start - n, start), downstream [end, end + n), each
			# clipped to the chromosome (see load_chrom_lengths).
			up_lo = np.maximum(starts - n, 0)
			up_hi = np.minimum(starts, chrom_len)
			down_lo = np.maximum(ends, 0)
			down_hi = np.minimum(ends + n, chrom_len)

			window_bp = (
				np.maximum(up_hi - up_lo, 0) + np.maximum(down_hi - down_lo, 0)
			)
			has_window = window_bp > 0

			for repeat_class in classes:
				index = indices[repeat_class].get(chrom)
				if index is None:
					# No elements of this class on this chromosome: coverage 0
					# where a window exists, NaN where none does.
					covered = np.zeros(len(rows), dtype=np.int64)
				else:
					covered = (
						covered_bp_in(index, up_lo, up_hi)
						+ covered_bp_in(index, down_lo, down_hi)
					)

				frac = np.where(has_window, covered / np.maximum(window_bp, 1), np.nan)
				coverage[n][repeat_class][rows] = frac
				presence[n][repeat_class][rows] = has_window & (covered > 0)

	return {
		n: {
			c: pd.DataFrame({"presence": presence[n][c], "coverage": coverage[n][c]})
			for c in classes
		}
		for n in window_sizes
	}


# ===========================================================================
# Directly abutting elements
# ===========================================================================
#
# The window sweep above shows its largest effects at its SMALLEST windows,
# which is backwards for a diffuse regional effect. The reason is the set of
# loci an element touches but does not enter: because every locus overlapping
# an element was dropped when the dataset was built (see the module docstring),
# an element covering the base immediately before str_start must END exactly at
# str_start. Those boundary loci are what a tiny window actually selects.
#
# The functions here identify them explicitly (rather than inferring them from
# the n=1 window), test whether their labels differ, and control for the
# obvious confound -- that boundary loci are motif-skewed. At str_len=4 the
# control comes back negative: the shift survives within every motif, so it is
# not composition. See "Reading the results" in the module docstring.

def _in_sorted(sorted_arr, values):
	"""Vectorized membership test: is each of `values` present in the sorted
	array `sorted_arr`?"""
	if len(sorted_arr) == 0:
		return np.zeros(len(values), dtype=bool)
	idx = np.searchsorted(sorted_arr, values, side="left")
	idx = np.minimum(idx, len(sorted_arr) - 1)
	return sorted_arr[idx] == values


def compute_abutting_flags(str_df, indices, classes):
	"""Returns {repeat_class: DataFrame(abuts_upstream, abuts_downstream,
	abuts)}, one row per locus (aligned with str_df's index).

	`abuts_upstream` means some merged interval of that class ends exactly at
	str_start; `abuts_downstream`, that one starts exactly at str_end.

	Merging within a class is safe for this test: a merged interval's end is
	always some member element's end (the max over the run), so an exact
	boundary match survives merging and merging cannot invent one. And no
	merged interval can straddle the STR -- an interval ending at str_start and
	one starting at str_end are separated by the STR's own length, so they
	never merge into one.
	"""
	n_loci = len(str_df)
	chrom_arr = str_df["chrom"].to_numpy()
	start_arr = str_df["str_start"].to_numpy().astype(np.int64)
	end_arr = str_df["str_end"].to_numpy().astype(np.int64)

	up = {c: np.zeros(n_loci, dtype=bool) for c in classes}
	down = {c: np.zeros(n_loci, dtype=bool) for c in classes}

	for chrom in np.unique(chrom_arr):
		rows = np.nonzero(chrom_arr == chrom)[0]
		for repeat_class in classes:
			index = indices[repeat_class].get(chrom)
			if index is None:
				continue
			m_starts, m_ends, _ = index
			# m_ends is sorted because the merged intervals are disjoint.
			up[repeat_class][rows] = _in_sorted(m_ends, start_arr[rows])
			down[repeat_class][rows] = _in_sorted(m_starts, end_arr[rows])

	return {
		c: pd.DataFrame({
			"abuts_upstream": up[c],
			"abuts_downstream": down[c],
			"abuts": up[c] | down[c],
		})
		for c in classes
	}


def abutting_label_tests(str_df, abutting, classes, min_group_n):
	"""Cliff's delta of each label for abutting vs. non-abutting loci, per
	repeat class. Bonferroni-corrected across this family only -- these are a
	separate family from the window sweep, and pooling them would over-correct
	both."""
	records = []
	for repeat_class in classes:
		flag = abutting[repeat_class]["abuts"].to_numpy()
		for target_name, label_col in TARGETS.items():
			labels = str_df[label_col].to_numpy()
			stat = cliffs_delta_with_ci(
				flag, labels, min_group_n, False, 0, CI_LEVEL, None
			)
			stat.update({
				"repeat_class": repeat_class, "target": target_name,
				"median_abutting": float(np.nanmedian(labels[flag])) if flag.any() else np.nan,
				"median_other": float(np.nanmedian(labels[~flag])) if (~flag).any() else np.nan,
			})
			records.append(stat)

	df = pd.DataFrame(records)
	return _bonferroni(df)


def abutting_motif_composition(str_df, abutting, top_k):
	"""Motif share among ANY-abutting loci vs. all loci, for the motifs most
	common among the abutting ones. The enrichment column is what shows the
	confound: a motif over-represented among abutting loci carries its own
	label distribution into the comparison."""
	flag = abutting[ANY_CLASS]["abuts"].to_numpy()
	abut_share = str_df.loc[flag, "motif"].value_counts(normalize=True)
	all_share = str_df["motif"].value_counts(normalize=True)

	motifs = list(abut_share.head(top_k).index)
	return pd.DataFrame({
		"motif": motifs,
		"share_abutting": [float(abut_share.get(m, 0.0)) for m in motifs],
		"share_all": [float(all_share.get(m, 0.0)) for m in motifs],
		"enrichment": [
			float(abut_share.get(m, 0.0) / all_share[m]) if all_share.get(m, 0) else np.nan
			for m in motifs
		],
		"n_abutting": [int((str_df.loc[flag, "motif"] == m).sum()) for m in motifs],
	})


def abutting_within_motif_tests(str_df, abutting, motifs, min_group_n):
	"""The control for the composition confound: repeat the ANY-abutting vs.
	non-abutting Cliff's delta WITHIN each motif, where composition is held
	fixed by construction.

	If the pooled effect were purely compositional, these within-motif deltas
	would collapse toward zero; whatever survives is the part of the shift that
	is not explained by which motifs sit at element boundaries. Corrected
	across this family only, as in abutting_label_tests.
	"""
	flag = abutting[ANY_CLASS]["abuts"].to_numpy()
	motif_arr = str_df["motif"].to_numpy()

	records = []
	for motif in motifs:
		sel = motif_arr == motif
		for target_name, label_col in TARGETS.items():
			labels = str_df[label_col].to_numpy()[sel]
			stat = cliffs_delta_with_ci(
				flag[sel], labels, min_group_n, False, 0, CI_LEVEL, None
			)
			stat.update({"motif": motif, "target": target_name})
			records.append(stat)

	df = pd.DataFrame(records)
	return _bonferroni(df)


def _bonferroni(df):
	"""Bonferroni-correct `p_value` across the rows that were actually tested
	(a skipped test was never evaluated, so counting it would over-correct the
	rest), adding p_value_bonferroni / significant columns."""
	testable = df["p_value"].notna()
	df["p_value_bonferroni"] = np.nan
	df["significant"] = False
	if testable.any():
		_, p_bonf, _, _ = multipletests(
			df.loc[testable, "p_value"].to_numpy(),
			alpha=SIGNIFICANCE_ALPHA, method="bonferroni",
		)
		df.loc[testable, "p_value_bonferroni"] = p_bonf
		df.loc[testable, "significant"] = p_bonf < SIGNIFICANCE_ALPHA
	return df


def run_abutting_analysis(str_df, indices, classes):
	"""Full abutting analysis + report. Returns the three result frames."""
	print("\n" + "=" * 78)
	print("DIRECTLY ABUTTING ELEMENTS  (element boundary exactly at the STR edge)")
	print("=" * 78)

	abutting = compute_abutting_flags(str_df, indices, classes)

	print("\n--- how many loci sit at an element boundary ---")
	for repeat_class in classes:
		a = abutting[repeat_class]
		n_abut = int(a["abuts"].sum())
		print(
			f"  {repeat_class:<12} {n_abut:>7,} / {len(str_df):,} ({n_abut / len(str_df):7.3%})"
			f"   upstream={int(a['abuts_upstream'].sum()):>6,}"
			f"  downstream={int(a['abuts_downstream'].sum()):>6,}"
			f"  both={int((a['abuts_upstream'] & a['abuts_downstream']).sum()):>5,}"
		)

	label_df = abutting_label_tests(str_df, abutting, classes, MIN_GROUP_N)
	print(f"\n--- labels: abutting vs non-abutting (Bonferroni over this family, "
	      f"alpha={SIGNIFICANCE_ALPHA}) ---")
	for _, row in label_df.iterrows():
		if not np.isfinite(row["effect"]):
			print(f"  {row['repeat_class']:<12} vs {row['target']:<15} SKIPPED "
			      f"(n_abutting={int(row['n_present'])} < {MIN_GROUP_N})")
			continue
		print(
			f"  {row['repeat_class']:<12} vs {row['target']:<15} "
			f"n_abut={int(row['n_present']):>6,}  delta={row['effect']:+.4f}  "
			f"p_bonf={row['p_value_bonferroni']:.3g}"
			f"{'*' if row['significant'] else ' '}  "
			f"median {row['median_abutting']:.2f} vs {row['median_other']:.2f}"
		)

	motif_df = abutting_motif_composition(str_df, abutting, ABUTTING_TOP_K_MOTIFS)
	print(f"\n--- motif composition of {ANY_CLASS}-abutting loci (top "
	      f"{ABUTTING_TOP_K_MOTIFS}) ---")
	print(f"  {'motif':<10} {'abutting':>9} {'all loci':>9} {'enrich':>7}  n_abutting")
	for _, row in motif_df.iterrows():
		print(
			f"  {row['motif']:<10} {row['share_abutting']:>8.2%} "
			f"{row['share_all']:>8.2%} {row['enrichment']:>7.2f}x  "
			f"{int(row['n_abutting']):>7,}"
		)

	within_df = abutting_within_motif_tests(
		str_df, abutting, list(motif_df["motif"]), MIN_GROUP_N
	)
	print("\n--- control: same test WITHIN each motif (composition held fixed) ---")
	for _, row in within_df.iterrows():
		if not np.isfinite(row["effect"]):
			print(f"  {row['motif']:<10} vs {row['target']:<15} SKIPPED "
			      f"(smaller group < {MIN_GROUP_N})")
			continue
		print(
			f"  {row['motif']:<10} vs {row['target']:<15} n={int(row['n']):>6,}  "
			f"delta={row['effect']:+.4f}  p_bonf={row['p_value_bonferroni']:.3g}"
			f"{'*' if row['significant'] else ' '}"
		)

	for target_name in TARGETS:
		pooled = label_df[
			(label_df["repeat_class"] == ANY_CLASS) & (label_df["target"] == target_name)
		]["effect"].iloc[0]
		w = within_df[within_df["target"] == target_name]
		w = w[np.isfinite(w["effect"])]
		if not w.empty:
			print(
				f"\n  {target_name}: pooled {ANY_CLASS}-abutting delta={pooled:+.4f}; "
				f"within-motif median={w['effect'].median():+.4f} "
				f"(range {w['effect'].min():+.4f} to {w['effect'].max():+.4f}, "
				f"{len(w)} motifs)"
			)

	plot_abutting(label_df, motif_df, within_df, classes)
	return label_df, motif_df, within_df


# ===========================================================================
# Statistics
# ===========================================================================

def cliffs_delta(x, y):
	"""Cliff's delta effect size (P(X>Y) - P(X<Y)) via the Mann-Whitney U
	identity delta = 2*U/(n_x*n_y) - 1, copied from
	str_anno_overlap_analysis.py. U already gives tied pairs half credit, so
	this is exact under ties -- important since `length` (mode_copy_number)
	has many repeated values across loci. Returns (delta, p_value) from the
	same two-sided Mann-Whitney U test.

	Interpretation thresholds (Romano et al. 2006): |delta| < 0.147
	negligible, < 0.33 small, < 0.474 medium, >= 0.474 large.
	"""
	u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
	delta = 2 * u_stat / (len(x) * len(y)) - 1
	return delta, p_value


def spearman_with_ci(x, y, run_bootstrap, n_boot, ci_level, rng):
	"""Spearman rho + two-sided p-value on the paired, finite entries of x
	and y, plus an optional percentile bootstrap CI over loci.

	Each bootstrap draw resamples loci with replacement and recomputes rho
	from scratch (not just re-indexing precomputed ranks): which values tie
	changes with the resample, and scipy's tie-averaged ranking needs to see
	that to stay correct. Mirrors GC_content_and_gradient.py.
	"""
	mask = np.isfinite(x) & np.isfinite(y)
	x, y = x[mask], y[mask]
	n = len(x)

	if n < 3 or np.all(x == x[0]):
		# Constant coverage (every locus 0) -- rho is undefined, not zero.
		return {"n": n, "effect": np.nan, "p_value": np.nan,
		        "ci_lo": np.nan, "ci_hi": np.nan}

	rho, p_value = spearmanr(x, y)

	ci_lo, ci_hi = np.nan, np.nan
	if run_bootstrap:
		boot_idx = rng.integers(0, n, size=(n_boot, n))
		boot = np.array([spearmanr(x[idx], y[idx])[0] for idx in boot_idx])
		alpha = 1 - ci_level
		ci_lo = float(np.nanpercentile(boot, 100 * (alpha / 2)))
		ci_hi = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))

	return {"n": n, "effect": float(rho), "p_value": float(p_value),
	        "ci_lo": ci_lo, "ci_hi": ci_hi}


def cliffs_delta_with_ci(flag, labels, min_group_n, run_bootstrap, n_boot,
                         ci_level, rng):
	"""Cliff's delta of `labels` for flag=True vs. flag=False loci, plus an
	optional percentile bootstrap CI over loci.

	Returns NaN effect/p when either group has < min_group_n loci -- at small
	windows the present group is genuinely tiny for rare classes (see the
	module docstring on dataset construction), and a delta from a handful of
	loci is noise.
	"""
	mask = np.isfinite(labels)
	flag, labels = flag[mask], labels[mask]
	n_present = int(flag.sum())
	n_absent = int(len(flag) - n_present)

	base = {"n": len(flag), "n_present": n_present,
	        "presence_rate": float(n_present / len(flag)) if len(flag) else np.nan}

	if min(n_present, n_absent) < min_group_n:
		return {**base, "effect": np.nan, "p_value": np.nan,
		        "ci_lo": np.nan, "ci_hi": np.nan}

	delta, p_value = cliffs_delta(labels[flag], labels[~flag])

	ci_lo, ci_hi = np.nan, np.nan
	if run_bootstrap:
		n = len(flag)
		boot_idx = rng.integers(0, n, size=(n_boot, n))
		boot = []
		for idx in boot_idx:
			f, lab = flag[idx], labels[idx]
			# A resample can wipe out one group entirely for a rare class.
			if f.all() or not f.any():
				boot.append(np.nan)
			else:
				boot.append(cliffs_delta(lab[f], lab[~f])[0])
		boot = np.array(boot, dtype=float)
		alpha = 1 - ci_level
		ci_lo = float(np.nanpercentile(boot, 100 * (alpha / 2)))
		ci_hi = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))

	return {**base, "effect": float(delta), "p_value": float(p_value),
	        "ci_lo": ci_lo, "ci_hi": ci_hi}


# ===========================================================================
# Main
# ===========================================================================

def main():
	rng = np.random.default_rng(SEED)

	print("=" * 78)
	print(f"Nearby repeat elements vs STR labels  [str_len={STR_LEN}]")
	print("=" * 78)

	str_df = load_str_data(STR_DATA_PATH)
	print(f"\nLoaded {len(str_df):,} loci from {STR_DATA_PATH}")

	chrom_lens = load_chrom_lengths(FAI_PATH)
	str_chroms = set(str_df["chrom"].unique())
	missing = str_chroms - set(chrom_lens)
	if missing:
		raise KeyError(f"Chromosomes in the STR data missing from {FAI_PATH}: {sorted(missing)}")

	print(f"\nLoading repeat elements from {REPEAT_BED_PATH}...")
	bed_df, fold_map = load_repeat_bed(REPEAT_BED_PATH, str_chroms, CLASS_GROUPING)
	print(f"  {len(bed_df):,} elements on the {len(str_chroms)} chromosome(s) with STRs")

	if fold_map is not None:
		folded = sorted({f"{raw} -> {new}" for raw, new in fold_map.items() if raw != new})
		if folded:
			print(f"  folded raw repClass values: {', '.join(folded)}")

	classes = class_order(bed_df, CLASS_GROUPING)
	counts = bed_df["repeat_class"].value_counts()
	print(f"  {len(classes) - 1} classes (+ {ANY_CLASS}):")
	for c in classes:
		n_elems = len(bed_df) if c == ANY_CLASS else int(counts.get(c, 0))
		print(f"    {c:<12} {n_elems:>10,} elements")

	print(f"\nBuilding merged-interval indices...")
	indices = build_indices_by_class(bed_df, classes)

	print(f"\nComputing per-locus features over window sizes {WINDOW_SIZES} bp "
	      f"(each side)...")
	features = compute_repeat_features(
		str_df, indices, classes, WINDOW_SIZES, chrom_lens
	)

	# --- tests --------------------------------------------------------------

	tasks = [
		(n, repeat_class, metric, target_name, label_col)
		for n in WINDOW_SIZES
		for repeat_class in classes
		for metric in ("presence", "coverage")
		for target_name, label_col in TARGETS.items()
	]

	if RUN_BOOTSTRAP:
		print(f"\nBootstrapping each effect size ({N_BOOTSTRAP} resamples, "
		      f"{int(CI_LEVEL * 100)}% CI) -- this is the slow path.")

	results = []
	for n, repeat_class, metric, target_name, label_col in tqdm(tasks, desc="  effect sizes"):
		labels = str_df[label_col].to_numpy()
		feat = features[n][repeat_class]

		if metric == "presence":
			stat = cliffs_delta_with_ci(
				feat["presence"].to_numpy(), labels, MIN_GROUP_N,
				RUN_BOOTSTRAP, N_BOOTSTRAP, CI_LEVEL, rng,
			)
		else:
			stat = spearman_with_ci(
				feat["coverage"].to_numpy(), labels,
				RUN_BOOTSTRAP, N_BOOTSTRAP, CI_LEVEL, rng,
			)

		stat.update({
			"window_size": n, "repeat_class": repeat_class,
			"metric": metric, "target": target_name,
		})
		results.append(stat)

	results_df = pd.DataFrame(results)

	# --- Bonferroni correction across every test actually run ---------------
	# Tests skipped for lack of group size (NaN p) are not part of the family:
	# they were never evaluated, so counting them would over-correct the rest.

	n_tests = int(results_df["p_value"].notna().sum())
	n_skipped = int(len(results_df) - n_tests)
	results_df = _bonferroni(results_df)

	# --- report -------------------------------------------------------------

	print("\n" + "=" * 78)
	print(f"RESULTS  (Bonferroni over {n_tests} tests, alpha={SIGNIFICANCE_ALPHA}"
	      + (f"; {n_skipped} skipped for group size < {MIN_GROUP_N} or constant coverage"
	         if n_skipped else "") + ")")
	print("=" * 78)

	effect_name = {"presence": "delta", "coverage": "rho"}
	for metric in ("presence", "coverage"):
		print(f"\n--- {metric} "
		      + ("(Cliff's delta, present vs absent)" if metric == "presence"
		         else "(Spearman rho on covered fraction)") + " ---")
		sub = results_df[results_df["metric"] == metric]
		for _, row in sub.iterrows():
			if not np.isfinite(row["effect"]):
				if metric == "presence":
					n_present = int(row["n_present"])
					why = (f"n_present={n_present}, n_absent={int(row['n']) - n_present}"
					       f" -- smaller group < {MIN_GROUP_N}")
				else:
					why = "coverage constant across loci"
				print(
					f"window={int(row['window_size']):>5}bp  {row['repeat_class']:<12} "
					f"vs {row['target']:<15} SKIPPED ({why})"
				)
				continue
			ci_str = (
				f"  {int(CI_LEVEL * 100)}% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]"
				if RUN_BOOTSTRAP else ""
			)
			sig_mark = "*" if row["significant"] else " "
			rate_str = (
				f"  present={row['presence_rate']:.3%}"
				if metric == "presence" else ""
			)
			print(
				f"window={int(row['window_size']):>5}bp  {row['repeat_class']:<12} "
				f"vs {row['target']:<15} n={int(row['n']):<7} "
				f"{effect_name[metric]}={row['effect']:+.4f}  p={row['p_value']:.3g}  "
				f"p_bonf={row['p_value_bonferroni']:.3g}{sig_mark}{ci_str}{rate_str}"
			)

	if SAVE_RESULTS_PATH:
		os.makedirs(os.path.dirname(os.path.abspath(SAVE_RESULTS_PATH)), exist_ok=True)
		with open(SAVE_RESULTS_PATH, "w") as f:
			json.dump(
				{
					"str_len": STR_LEN,
					"n_flanking": N_FLANKING,
					"window_sizes": WINDOW_SIZES,
					"class_grouping": CLASS_GROUPING,
					"classes": classes,
					"targets": TARGETS,
					"min_group_n": MIN_GROUP_N,
					"run_bootstrap": RUN_BOOTSTRAP,
					"n_bootstrap": N_BOOTSTRAP,
					"ci_level": CI_LEVEL,
					"seed": SEED,
					"n_tests": n_tests,
					"n_skipped": n_skipped,
					"significance_alpha": SIGNIFICANCE_ALPHA,
					"results": json.loads(results_df.to_json(orient="records")),
				},
				f, indent=2,
			)
		print(f"\nWrote {SAVE_RESULTS_PATH}")

	plot_effect_sizes(results_df, classes, "presence",
	                  "Cliff's delta (present - absent)")
	plot_effect_sizes(results_df, classes, "coverage",
	                  "Spearman rho (covered fraction)")
	plot_presence_rates(results_df, classes)

	if RUN_ABUTTING_ANALYSIS:
		abut_label_df, abut_motif_df, abut_within_df = run_abutting_analysis(
			str_df, indices, classes
		)
		if SAVE_RESULTS_PATH:
			_append_json(SAVE_RESULTS_PATH, "abutting", {
				"top_k_motifs": ABUTTING_TOP_K_MOTIFS,
				"by_class": json.loads(abut_label_df.to_json(orient="records")),
				"motif_composition": json.loads(abut_motif_df.to_json(orient="records")),
				"within_motif": json.loads(abut_within_df.to_json(orient="records")),
			})
			print(f"\nAdded the abutting results to {SAVE_RESULTS_PATH}")


# ===========================================================================
# Plot
# ===========================================================================

def _append_json(path, key, payload):
	"""Add a top-level key to the already-written results json, so the
	abutting analysis (which runs after the main results are saved) lands in
	the same file rather than a second one."""
	with open(path) as f:
		data = json.load(f)
	data[key] = payload
	with open(path, "w") as f:
		json.dump(data, f, indent=2)


def _show_or_save(fig, name, save_dir):
	"""Mirrors _show_or_save in str_anno_overlap_analysis.py."""
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, f"{name}.png")
		fig.savefig(path, dpi=150, bbox_inches="tight")
		plt.close(fig)
		print(f"  saved {path}")
	else:
		plt.show()


def build_class_colors(classes):
	"""One color per repeat class; ANY gets black (it is drawn heavier, as
	the headline "is anything nearby" series)."""
	real = [c for c in classes if c != ANY_CLASS]
	cmap = plt.get_cmap("tab10" if len(real) <= 10 else "tab20")
	colors = {c: cmap(i % cmap.N) for i, c in enumerate(real)}
	colors[ANY_CLASS] = "black"
	return colors


def plot_effect_sizes(results_df, classes, metric, ylabel):
	"""One panel per target: effect size vs. window size, one line per repeat
	class (ANY heavier, in black). Points are a filled circle if significant
	(Bonferroni-corrected p < SIGNIFICANCE_ALPHA) or an open circle if not;
	skipped tests simply have no point."""
	targets = list(TARGETS.keys())
	colors = build_class_colors(classes)

	# sharey: the two targets' panels are the same effect size on the same
	# scale, so a shared axis makes their magnitudes directly comparable at a
	# glance instead of each panel silently auto-scaling to its own range.
	fig, axes = plt.subplots(1, len(targets), figsize=(7.0 * len(targets), 5.5),
	                         sharey=True, squeeze=False)
	axes = axes[0]

	for ax, target_name in zip(axes, targets):
		sub = results_df[
			(results_df["target"] == target_name) & (results_df["metric"] == metric)
		]
		for repeat_class in classes:
			m = sub[sub["repeat_class"] == repeat_class].sort_values("window_size")
			m = m[np.isfinite(m["effect"])]
			if m.empty:
				continue

			is_any = repeat_class == ANY_CLASS
			yerr = None
			if RUN_BOOTSTRAP:
				yerr = np.abs(np.vstack(
					[m["effect"] - m["ci_lo"], m["ci_hi"] - m["effect"]]
				))
			# marker="None" here: markers are drawn separately below, per
			# point, so significant/not can use different marker styles.
			ax.errorbar(
				m["window_size"], m["effect"], yerr=yerr, marker="None", capsize=3,
				color=colors[repeat_class], label=repeat_class,
				lw=2.5 if is_any else 1.5, zorder=2 if is_any else 1,
			)

			sig = m[m["significant"]]
			not_sig = m[~m["significant"]]
			s = 60 if is_any else 40
			ax.scatter(sig["window_size"], sig["effect"], marker="o",
			           color=colors[repeat_class], s=s, zorder=4 if is_any else 3)
			ax.scatter(not_sig["window_size"], not_sig["effect"], marker="o",
			           facecolors="none", edgecolors=colors[repeat_class], s=s,
			           linewidths=1.5, zorder=4 if is_any else 3)

		ax.axhline(0, color="gray", lw=1, ls=":")
		ax.set_xscale("log")
		ax.set_xlabel("Flank window size (bp, each side)")
		ax.set_ylabel(ylabel)
		ax.set_title(f"vs {TARGETS[target_name].replace('_', ' ')}")

		sig_handles = [
			Line2D([0], [0], marker="o", linestyle="None", color="gray",
			       markersize=7,
			       label=f"significant (Bonferroni, α={SIGNIFICANCE_ALPHA})"),
			Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none",
			       markeredgecolor="gray", markersize=7, label="not significant"),
		]
		handles, _ = ax.get_legend_handles_labels()
		ax.legend(handles=handles + sig_handles, fontsize=8, ncol=2)

	fig.suptitle(
		f"Nearby repeat elements ({metric}) vs STR labels  [str_len={STR_LEN}, "
		f"grouping={CLASS_GROUPING}"
		+ (f", {int(CI_LEVEL * 100)}% bootstrap CI, {N_BOOTSTRAP} resamples"
		   if RUN_BOOTSTRAP else "") + "]"
	)
	plt.tight_layout()
	_show_or_save(fig, f"repeat_{metric}_vs_labels_str_len_{STR_LEN}", SAVE_DIR)


def plot_presence_rates(results_df, classes):
	"""Diagnostic (not a hypothesis test): fraction of loci with at least one
	element of each class in their flanks, vs. window size.

	This is what makes the effect-size figures readable. Loci overlapping ANY
	mobile element were excluded when the STR dataset was built, so a nonzero
	rate at a tiny window means the STR sits exactly at an element BOUNDARY,
	not inside one -- a large, non-random group (see the module docstring's
	"Reading the results"). A class whose curve is still near 0 at a given
	window has too few present loci for its delta to mean much (those below
	MIN_GROUP_N are skipped outright).
	"""
	colors = build_class_colors(classes)
	# presence_rate is a property of (window, class) alone; take one target.
	first_target = next(iter(TARGETS))
	sub = results_df[
		(results_df["metric"] == "presence") & (results_df["target"] == first_target)
	]

	fig, ax = plt.subplots(figsize=(7.5, 5.5))
	for repeat_class in classes:
		m = sub[sub["repeat_class"] == repeat_class].sort_values("window_size")
		if m.empty:
			continue
		is_any = repeat_class == ANY_CLASS
		ax.plot(m["window_size"], m["presence_rate"], marker="o",
		        color=colors[repeat_class], label=repeat_class,
		        lw=2.5 if is_any else 1.5, markersize=6 if is_any else 4)

	ax.set_xscale("log")
	# Linear y over the full [0, 1] it can take: the rate is a proportion, and
	# a shared fixed scale keeps curves comparable across runs/motif lengths.
	ax.set_ylim(0, 1)
	ax.set_xlabel("Flank window size (bp, each side)")
	ax.set_ylabel("Fraction of loci with >=1 element in flanks")
	ax.set_title(f"Repeat element presence rate by flank window  [str_len={STR_LEN}]")
	ax.legend(fontsize=8, ncol=2)
	plt.tight_layout()
	_show_or_save(fig, f"repeat_presence_rate_str_len_{STR_LEN}", SAVE_DIR)


def plot_abutting(label_df, motif_df, within_df, classes):
	"""Three panels, read left to right as one argument: here is an effect,
	here is the obvious thing that could explain it away, here is the test of
	whether it does.

	(a) THE EFFECT -- "do boundary loci have different labels?"
	    One row per repeat class, one bar per target. The bar is Cliff's delta
	    between loci that abut an element of that class and loci that do not.
	    Positive = abutting loci have LARGER labels (longer tracts / higher
	    heterozygosity); 0 = no difference. Faded bars are not significant at
	    SIGNIFICANCE_ALPHA after Bonferroni; classes with too few abutting loci
	    to test (< MIN_GROUP_N) are omitted entirely. As a magnitude guide
	    (Romano et al. 2006): |delta| < 0.147 negligible, < 0.33 small,
	    < 0.474 medium, >= 0.474 large.

	(b) THE CANDIDATE CONFOUND -- "are boundary loci just a different mix of
	    motifs?"
	    For the ABUTTING_TOP_K_MOTIFS most common motifs among ANY-abutting
	    loci, two bars: that motif's share of the abutting loci (colored) vs.
	    its share of all loci (gray). The "N.NNx" label at the right is the
	    ratio of the two. Above 1x the motif is over-represented at element
	    boundaries, below 1x under-represented. This matters because motifs
	    differ in their own typical length, so a skewed mix could by itself
	    produce panel (a) without repeat proximity doing anything.

	(c) THE CONTROL -- "does the effect survive holding motif fixed?"
	    Panel (a)'s ANY test, recomputed separately within each motif from
	    panel (b), where the composition skew cannot operate by construction.
	    The dashed vertical line is the pooled (all-motif) delta from (a).
	    Bars sitting AT the dashed line mean composition explains none of the
	    pooled effect. Bars collapsing toward 0 would mean it explained most of
	    it. At str_len=4 they sit at the line -- so the boundary effect is real
	    and needs a different explanation; the module docstring's "Reading the
	    results" gives the ascertainment one.
	"""
	targets = list(TARGETS.keys())
	target_colors = dict(zip(targets, ["#3170ad", "#e8443a"]))

	fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

	# --- (a) per-class delta --------------------------------------------
	ax = axes[0]
	shown = [c for c in classes
	         if np.isfinite(label_df[label_df["repeat_class"] == c]["effect"]).any()]
	y = np.arange(len(shown))
	h = 0.8 / max(len(targets), 1)
	for k, target_name in enumerate(targets):
		sub = label_df[label_df["target"] == target_name].set_index("repeat_class")
		vals = [sub["effect"].get(c, np.nan) for c in shown]
		sig = [bool(sub["significant"].get(c, False)) for c in shown]
		bars = ax.barh(y + k * h - 0.4 + h / 2, vals, height=h,
		               color=target_colors[target_name], label=target_name,
		               edgecolor="black", linewidth=0.5)
		for bar, s in zip(bars, sig):
			if not s:
				bar.set_alpha(0.35)
	ax.set_yticks(y)
	ax.set_yticklabels(shown)
	ax.invert_yaxis()
	ax.axvline(0, color="gray", lw=1, ls=":")
	ax.set_xlabel("Cliff's delta (abutting - non-abutting)")
	ax.set_title("(a) Label shift for boundary loci, by class")
	ax.legend(fontsize=8, title=f"faded = n.s. (α={SIGNIFICANCE_ALPHA})",
	          title_fontsize=7)

	# --- (b) motif composition ------------------------------------------
	ax = axes[1]
	y = np.arange(len(motif_df))
	ax.barh(y - 0.2, motif_df["share_abutting"], height=0.4,
	        color="#b5651d", edgecolor="black", linewidth=0.5,
	        label=f"{ANY_CLASS}-abutting loci")
	ax.barh(y + 0.2, motif_df["share_all"], height=0.4,
	        color="#bbbbbb", edgecolor="black", linewidth=0.5, label="all loci")
	for yi, enr in zip(y, motif_df["enrichment"]):
		ax.text(max(motif_df["share_abutting"].max(), motif_df["share_all"].max()) * 1.02,
		        yi, f"{enr:.2f}x", va="center", fontsize=7.5)
	ax.set_yticks(y)
	ax.set_yticklabels(motif_df["motif"])
	ax.invert_yaxis()
	ax.set_xlabel("Share of loci")
	ax.set_title("(b) Motif composition of boundary loci")
	ax.legend(fontsize=8)

	# --- (c) within-motif control ---------------------------------------
	ax = axes[2]
	motifs = list(motif_df["motif"])
	y = np.arange(len(motifs))
	for k, target_name in enumerate(targets):
		sub = within_df[within_df["target"] == target_name].set_index("motif")
		vals = [sub["effect"].get(m, np.nan) for m in motifs]
		sig = [bool(sub["significant"].get(m, False)) for m in motifs]
		bars = ax.barh(y + k * h - 0.4 + h / 2, vals, height=h,
		               color=target_colors[target_name], label=target_name,
		               edgecolor="black", linewidth=0.5)
		for bar, s in zip(bars, sig):
			if not s:
				bar.set_alpha(0.35)

		pooled = label_df[
			(label_df["repeat_class"] == ANY_CLASS) & (label_df["target"] == target_name)
		]["effect"].iloc[0]
		ax.axvline(pooled, color=target_colors[target_name], lw=1.5, ls="--",
		           label=f"{target_name}: pooled = {pooled:+.3f}")
	ax.set_yticks(y)
	ax.set_yticklabels(motifs)
	ax.invert_yaxis()
	ax.axvline(0, color="gray", lw=1, ls=":")
	ax.set_xlabel("Cliff's delta within motif")
	ax.set_title("(c) Control: same test within each motif\n(bars at the dashed line = not a composition effect)", fontsize=10)
	ax.legend(fontsize=7)

	fig.suptitle(
		f"Loci directly abutting a repeat element  [str_len={STR_LEN}, "
		f"grouping={CLASS_GROUPING}]"
	)
	plt.tight_layout()
	_show_or_save(fig, f"repeat_abutting_str_len_{STR_LEN}", SAVE_DIR)


if __name__ == "__main__":
	main()
