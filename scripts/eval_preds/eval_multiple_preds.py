""" Evaluate predictions from multiple models. 

Also plots against baseline overall and by motif means, created by
get_baseline_performance.py and saved in 
predictions/baseline/mean_performance.json
"""

import os
import json
from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import scipy.stats as stats
from matplotlib.colors import LogNorm


def get_metrics(pred, true):
	mse = sk_metrics.mean_squared_error(true, pred)
	mape = sk_metrics.mean_absolute_percentage_error(true, pred)
	mae = sk_metrics.mean_absolute_error(true, pred)
	r2 = sk_metrics.r2_score(true, pred)
	pearson_r, pearson_p = stats.pearsonr(true, pred)
	spearman_r, spearman_p = stats.spearmanr(true, pred)

	return {
		"MSE": mse,
		"MAE": mae,
		"MAPE": mape,
		"R2": r2,
		"Pearson_r": pearson_r,
		"Pearson_p": pearson_p,
		"Spearman_r": spearman_r,
		"Spearman_p": spearman_p,
	}


if __name__ == '__main__':

	# by_len_metric = "Spearman_r"
	# by_len_metric = "MAPE"
	by_len_metric = "R2"

	pred_dir = {
		1: "predictions/soft_prompt/str1/v2",
		2: "predictions/soft_prompt/str2/v2"
	}

	model_names = [
		"str1_l1m_f100_p128_log_2026-03-29_16-23-14",
		"str1_l1m_f2000_p128_log_2026-03-28_01-17-25",

		"str2_l1m_f100_p128_log_2026-03-27_13-08-24",
		"str2_l1m_f2000_p128_log_2026-03-27_13-08-24",
		"str2_l1m_f4000_p128_log_2026-03-27_13-08-49_resumed_epoch62",
		"str2_l1m_f6000_p128_log_2026-03-27_13-09-31_resumed_epoch40"
	]

	lin_pred_dir = {
		1: "predictions/baseline/lin/str1",
		2: "predictions/baseline/lin/str2"
	}

	lin_model_names = [
		"str1_f100_log",
		"str1_f2000_log",
		"str1_f4000_log",
		"str1_f6000_log",
		"str1_f8000_log",
		"str2_f100_log",
		"str2_f2000_log",
		"str2_f4000_log",
		"str2_f6000_log",
		"str2_f8000_log",
	]

	results = []

	# ------------------------------------------------------------------
	# HyenaDNA models
	# ------------------------------------------------------------------

	for model_name in model_names:

		if model_name.startswith("str1_"):
			pred_path = os.path.join(pred_dir[1], model_name, "predictions_test.tsv")
		elif model_name.startswith("str2_"):
			pred_path = os.path.join(pred_dir[2], model_name, "predictions_test.tsv")
		else:
			raise ValueError(f"Unknown model prefix for model: {model_name}")

		# Load predictions
		pred_df = pd.read_csv(pred_path, sep="\t")

		# Compute metrics
		metrics = get_metrics(
			pred=pred_df["pred_length"],
			true=pred_df["true_length"]
		)

		# Get attributes from model name
		# str{str_len}_l1m_f{n_flank}_p{prompt_len}...
		str_len = model_name.split("_")[0][3:]
		n_flank = model_name.split("_")[2][1:]
		prompt_len = model_name.split("_")[3][1:]

		results.append({
			"model": model_name,
			"model_type": "hyenaDNA",
			"str_len": str_len,
			"n_flank": n_flank,
			"prompt_len": prompt_len,
			**metrics
		})

	# ------------------------------------------------------------------
	# Linear baseline models
	# ------------------------------------------------------------------

	for model_name in lin_model_names:

		if model_name.startswith("str1_"):
			pred_path = os.path.join(lin_pred_dir[1], model_name, "predictions_test.tsv")
		elif model_name.startswith("str2_"):
			pred_path = os.path.join(lin_pred_dir[2], model_name, "predictions_test.tsv")
		else:
			raise ValueError(f"Unknown model prefix for linear model: {model_name}")

		# Load predictions
		pred_df = pd.read_csv(pred_path, sep="\t")

		# Compute metrics
		metrics = get_metrics(
			pred=pred_df["pred_length"],
			true=pred_df["true_length"]
		)

		# Get attributes from model name
		# str{str_len}_f{n_flank}_log
		parts = model_name.split("_")
		str_len = parts[0][3:]
		n_flank = parts[1][1:]

		results.append({
			"model": model_name,
			"model_type": "linear",
			"str_len": str_len,
			"n_flank": n_flank,
			"prompt_len": None,
			**metrics
		})

	results_df = pd.DataFrame(results)
	results_df["n_flank"] = results_df["n_flank"].astype(int)

	hyena_df = results_df[results_df["model_type"] == "hyenaDNA"].copy()
	hyena_df["prompt_len"] = hyena_df["prompt_len"].astype(int)
	lin_df = results_df[results_df["model_type"] == "linear"]

	str1_hyena_df = hyena_df[hyena_df["str_len"] == "1"]
	str2_hyena_df = hyena_df[hyena_df["str_len"] == "2"]
	str1_lin_df = lin_df[lin_df["str_len"] == "1"]
	str2_lin_df = lin_df[lin_df["str_len"] == "2"]

	# Load baseline performance JSON
	with open('predictions/baseline/mean_performance.json', 'r') as file:
		baseline_perf = json.load(file)

	print("Mean baseline performance")
	pprint(baseline_perf)

	# Print metrics
	display_cols = [
		"model", "model_type",
		"MSE", "MAE", "MAPE", "R2",
		"Pearson_r", "Pearson_p",
		"Spearman_r", "Spearman_p"
	]
	print("\nLength 1 STR Results:")
	str1_df = results_df[results_df["str_len"] == "1"]
	if len(str1_df) > 0:
		print(str1_df[display_cols].to_string(
			index=False,
			max_colwidth=30,
		))

	print("\nLength 2 STR Results:")
	str2_df = results_df[results_df["str_len"] == "2"]
	if len(str2_df) > 0:
		print(str2_df[display_cols].to_string(
			index=False,
			max_colwidth=30,
		))

	# ------------------------------------------------------------------
	# Plot performance by flank length
	# ------------------------------------------------------------------

	for str_len_str, hyena_sub, lin_sub in [
		("1", str1_hyena_df, str1_lin_df),
		("2", str2_hyena_df, str2_lin_df),
	]:
		if len(hyena_sub) <= 1 and len(lin_sub) == 0:
			continue

		plt.figure(figsize=(8, 6))

		# HyenaDNA line
		if len(hyena_sub) > 1:
			sns.lineplot(
				data=hyena_sub,
				x="n_flank",
				y=by_len_metric,
				marker="o",
				label="HyenaDNA",
			)
		elif len(hyena_sub) == 1:
			plt.scatter(
				hyena_sub["n_flank"],
				hyena_sub[by_len_metric],
				marker="o",
				label="HyenaDNA",
				zorder=5,
			)

		# Linear baseline points
		if len(lin_sub) > 1:
			sns.lineplot(
				data=lin_sub,
				x="n_flank",
				y=by_len_metric,
				marker="s",
				label="Linear (Ridge)",
			)
		elif len(lin_sub) == 1:
			plt.scatter(
				lin_sub["n_flank"],
				lin_sub[by_len_metric],
				marker="s",
				label="Linear (Ridge)",
				zorder=5,
			)

		# Mean baselines
		overall_mean_val = baseline_perf[str_len_str]["overall_mean"].get(by_len_metric)
		if overall_mean_val is not None and not np.isnan(overall_mean_val):
			plt.axhline(
				y=overall_mean_val,
				color='red',
				linestyle='--',
				label='Overall Mean',
			)

		motif_mean_val = baseline_perf[str_len_str]["motif_mean"].get(by_len_metric)
		if motif_mean_val is not None and not np.isnan(motif_mean_val):
			plt.axhline(
				y=motif_mean_val,
				color='orange',
				linestyle='--',
				label='STR Motif Mean',
			)

		# Collect all plotted values for y-axis scaling
		all_vals = []
		all_vals.extend(hyena_sub[by_len_metric].tolist())
		all_vals.extend(lin_sub[by_len_metric].tolist())
		if overall_mean_val is not None and not np.isnan(overall_mean_val):
			all_vals.append(overall_mean_val)
		if motif_mean_val is not None and not np.isnan(motif_mean_val):
			all_vals.append(motif_mean_val)

		plt.title(f"Length {str_len_str} STR Prediction Performance by Flank Length")
		plt.xlabel("Flank Length (bp)")
		plt.ylabel(by_len_metric)
		plt.ylim(bottom=0, top=1.05 * max(all_vals))
		plt.grid(True)
		plt.legend(loc='lower right')
		plt.show()