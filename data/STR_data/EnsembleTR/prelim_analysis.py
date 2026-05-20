""" Preliminary EnsembleTR analysis.

Looks at:
	- Missing rate by superpopulation
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_afreq(afreq_str):
	"""Parse the JSON allele frequency dict. Returns (alleles, freqs) as
	np.arrays of floats, or (None, None) if empty/missing."""
	if pd.isna(afreq_str) or afreq_str == '' or afreq_str == '{}':
		return None, None
	d = json.loads(afreq_str)
	if len(d) == 0:
		return None, None
	alleles = np.array([float(k) for k in d.keys()])
	freqs = np.array(list(d.values()), dtype=float)
	return alleles, freqs


def mean_length(alleles, freqs):
	if alleles is None:
		return np.nan
	return np.sum(alleles * freqs)


def median_length(alleles, freqs):
	"""Frequency-weighted median. Returns the smallest allele such that
	cumulative frequency >= 0.5."""
	if alleles is None:
		return np.nan
	order = np.argsort(alleles)
	sorted_alleles = alleles[order]
	sorted_freqs = freqs[order]
	cum = np.cumsum(sorted_freqs)
	idx = np.searchsorted(cum, 0.5)
	# Clip for numerical safety (cumsum might end at 0.9999...)
	idx = min(idx, len(sorted_alleles) - 1)
	return sorted_alleles[idx]


def mode_length(alleles, freqs):
	"""Most frequent allele. Ties broken by taking the smallest allele."""
	if alleles is None:
		return np.nan
	max_freq = freqs.max()
	tied = alleles[freqs == max_freq]
	return tied.min()


def shannon_entropy(freqs):
	"""Shannon entropy in bits. Returns NaN if freqs is None."""
	if freqs is None:
		return np.nan
	# Guard against zero-probability alleles (shouldn't happen in practice)
	freqs = freqs[freqs > 0]
	if len(freqs) == 0:
		return np.nan
	return -np.sum(freqs * np.log2(freqs))


def heterozygosity(freqs):
	"""Gini-Simpson heterozygosity = 1 - sum(p^2)."""
	if freqs is None:
		return np.nan
	return 1.0 - np.sum(freqs ** 2)


def get_clip(col):
	if col == 'het':
		return (0, 1)
	return (0, None)


def plot_by_population(
	long_df,
	motif_length,
	length_stat='median_len',
	populations=None,
	output_path=None,
):
	"""One figure for a single motif length.

	4 panels of KDEs by population: num_called, length_stat, het, entropy.
	"""
	if populations is None:
		populations = ['AFR', 'AMR', 'EAS', 'SAS', 'EUR', 'H3Africa']

	assert length_stat in ('mean_len', 'median_len', 'mode_len')

	subset = long_df[long_df['motif_len'] == motif_length]

	# Drop populations with no data for this motif length
	pops_with_data = [
		p for p in populations
		if subset[subset['population'] == p]['num_called'].sum() > 0
	]

	quantities = [
		('num_called', 'Samples called'),
		(length_stat, length_stat.replace('_', ' ').title()),
		('het', 'Heterozygosity'),
		('entropy', 'Entropy (bits)'),
	]

	fig, axes = plt.subplots(1, 4, figsize=(20, 4))

	for ax, (col, label) in zip(axes, quantities):
		plot_data = subset[subset['population'].isin(pops_with_data)]
		plot_data = plot_data[['population', col]].dropna(subset=[col])

		sns.kdeplot(
			data=plot_data,
			x=col,
			hue='population',
			hue_order=pops_with_data,
			common_norm=False,
			ax=ax,
			clip=get_clip(col),
		)

		ax.set_title(label)
		ax.set_xlabel(label)

	fig.suptitle(f'Motif length = {motif_length}', y=0.95, fontsize=14)
	fig.tight_layout()

	if output_path:
		fig.savefig(output_path, dpi=150, bbox_inches='tight')
	return fig


if __name__ == '__main__':

	str_info_path = 'ensembletr_raw_data/repeat_info.tsv'
	str_calls_path = 'ensembletr_raw_data/afreq_het.tsv'
	 
	motif_lens = [1, 2, 4]

	# Load files
	str_df = pd.read_csv(
		str_info_path,
		sep='\t'
	)
	print(f"Loaded {len(str_df)} STRs from {str_info_path}")
	calls_df = pd.read_csv(
		str_calls_path,
		sep='\t'
	)
	print(f"Loaded {len(calls_df)} STR calls from {str_calls_path}")

	# Merge on repeat_id without creating duplicate columns
	str_df = str_df.merge(
		calls_df[[
			'ID', 'afreq_AFR', 'het_AFR', 'numcalled_AFR',
			'afreq_AMR', 'het_AMR', 'numcalled_AMR',
			'afreq_EAS', 'het_EAS', 'numcalled_EAS',
			'afreq_SAS', 'het_SAS', 'numcalled_SAS',
			'afreq_EUR', 'het_EUR', 'numcalled_EUR',
			'afreq_H3Africa', 'het_H3Africa', 'numcalled_H3Africa',
		]],
		on='ID',
		how='inner'
	)
	print(f"After merging, have {len(str_df)} STRs with calls")

	del calls_df  # Free memory

	# Add motif length column
	str_df['motif_len'] = str_df['Motif'].str.len()

	print("\nMotif length counts:")
	print(str_df['motif_len'].value_counts().sort_index().head(10))

	# Filter to motif lengths 1 and 2 only for now (to match our training data)
	str_df = str_df[str_df['motif_len'].isin(motif_lens)]

	# Create long format dataframe of counts and allele frequency by superpopulation
	long_dfs = []
	populations = ['AFR', 'AMR', 'EAS', 'SAS', 'EUR', 'H3Africa']

	for pop in populations:
		long_dfs.append(
			str_df[[
				'ID',
				f'numcalled_{pop}',
				f'afreq_{pop}',
			]].rename(columns={
				f'numcalled_{pop}': 'num_called',
				f'afreq_{pop}': 'afreq',
			}).assign(
				population=pop
			)
		)
	long_df = pd.concat(long_dfs, ignore_index=True)

	print("Zero-counts by population (counts and %):")
	for pop in populations:
		pop_counts = long_df[long_df['population'] == pop]
		n_zero = (pop_counts['num_called'] == 0).sum()
		n_total = len(pop_counts)
		print(f"{pop}: {n_zero} / {n_total} ({n_zero / n_total:.2%})")
		  
	# Compute mean and dispersion stats
	parsed = long_df['afreq'].apply(parse_afreq)
	long_df['_alleles'] = parsed.apply(lambda x: x[0])
	long_df['_freqs'] = parsed.apply(lambda x: x[1])

	long_df['entropy'] = long_df['_freqs'].apply(shannon_entropy)
	long_df['het'] = long_df['_freqs'].apply(heterozygosity)
	long_df['mean_len'] = long_df.apply(lambda r: mean_length(r['_alleles'], r['_freqs']), axis=1)
	long_df['median_len'] = long_df.apply(lambda r: median_length(r['_alleles'], r['_freqs']), axis=1)
	long_df['mode_len'] = long_df.apply(lambda r: mode_length(r['_alleles'], r['_freqs']), axis=1)

	long_df = long_df.drop(columns=['_alleles', '_freqs'])
	 
	print("\nComputed stats by population")

	# Find a reasonable min_num_called_total threshold

	long_df = long_df.merge(str_df[['ID', 'motif_len']], on='ID')

	# Sum numcalled across populations for each locus, per motif length
	total_called_per_locus = (
		long_df.groupby(['ID', 'motif_len'])['num_called']
		.sum()
		.reset_index()
		.rename(columns={'num_called': 'num_called_total'})
	)

	candidate_thresholds = [100, 250, 500, 1000, 1500, 2000]
	motif_lens_to_plot = [1, 2]

	# Summary table: fraction kept at each threshold
	print("\nFraction of loci kept at each threshold:")
	print(f"{'threshold':>10}", end="")
	for motif_len in motif_lens_to_plot:
		print(f"{f'motif_len={motif_len}':>16}", end="")
	print()
	for thresh in candidate_thresholds:
		print(f"{thresh:>10}", end="")
		for motif_len in motif_lens_to_plot:
			subset = total_called_per_locus[
				total_called_per_locus['motif_len'] == motif_len
			]
			frac = (subset['num_called_total'] >= thresh).mean()
			print(f"{frac:>16.1%}", end="")
		print()

	# How many loci have calls in each pop, and in all pops?
	for pop in populations:
		has_calls = (long_df[long_df['population'] == pop]['num_called'] > 0)
		print(f"{pop}: {has_calls.mean():.1%} of loci have calls")

	# Loci with all 6 pops called
	locus_pop_counts = (
		long_df[long_df['num_called'] > 0]
		.groupby(['ID', 'motif_len'])
		.size()
		.reset_index(name='n_pops')
	)
	for motif_len in [1, 2]:
		subset = locus_pop_counts[locus_pop_counts['motif_len'] == motif_len]
		frac_all_6 = (subset['n_pops'] == 6).mean()
		frac_at_least_5 = (subset['n_pops'] >= 5).mean()
		print(
			f"motif_len={motif_len}: all 6 pops = {frac_all_6:.1%}, "
			f"≥5 pops = {frac_at_least_5:.1%}"
		)

	# # Plot by population for each motif length
	# for motif_len in motif_lens:
	# 	fig = plot_by_population(
	# 		long_df,
	# 		motif_length=motif_len,
	# 		length_stat='median_len',
	# 		# output_path=f'prelim_motif_len_{motif_len}.png',
	# 	)
	# 	plt.show()