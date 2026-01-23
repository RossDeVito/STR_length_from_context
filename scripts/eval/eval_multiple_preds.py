""" Evaluate predictions from multiple models. """

import os
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

	# pred_dir = "predictions/soft_prompt/str2/tscc_dev"
	# model_names = [
	# 	"dev4Lb_2025-12-05_11-14-13",
	# 	"dev4Lb_m450_2025-12-06_16-44-44",
	# 	"dev5_m450_2025-12-08_02-35-32",
	# 	"dev5L_l1m_2025-12-08_00-22-52",
	# 	"dev5L_m450_2025-12-07_21-46-02",
	# ]

	pred_dir = "predictions/soft_prompt/str2/tscc_v1"
	model_names = [
		"str2_l1m_f100_p128_log_2026-01-12_13-28-45",
		"str2_l1m_f1000_p128_log_2026-01-14_15-46-56",
		"str2_l1m_f2000_p128_log_2026-01-12_13-28-53",
		"str2_l1m_f4000_p128_log_2026-01-12_13-21-40_resumed_epoch62",
	]

	results = []

	for model_name in model_names:
		pred_path = os.path.join(pred_dir, model_name, "predictions_test.tsv")

		# Load predictions
		pred_df = pd.read_csv(pred_path, sep="\t")

		# Compute metrics
		metrics = get_metrics(
			pred=pred_df["pred_length"],
			true=pred_df["true_length"]
		)

		# Get attributes from model name
		# str{str_len}_{backbone_model}_f{n_flank}_p{prompt_len}...
		str_len = model_name.split("_")[0][3:]
		n_flank = model_name.split("_")[2][1:]
		prompt_len = model_name.split("_")[3][1:]

		results.append({
			"model": model_name,
			"str_len": str_len,
			"n_flank": n_flank,
			"prompt_len": prompt_len,
			**metrics
		})

	results_df = pd.DataFrame(results)
	results_df["n_flank"] = results_df["n_flank"].astype(int)
	results_df["prompt_len"] = results_df["prompt_len"].astype(int)

	str1_res_df = results_df[results_df["str_len"] == "1"]
	str2_res_df = results_df[results_df["str_len"] == "2"]

	# Print metrics
	display_cols = [
		"model",
		"MSE", "MAE", "MAPE", "R2",
		"Pearson_r", "Pearson_p",
		"Spearman_r", "Spearman_p"
	]
	print("Length 1 STR Results:")
	print(str1_res_df[display_cols].to_string(index=False))
	print("\nLength 2 STR Results:")
	print(str2_res_df[display_cols].to_string(index=False))


	# Plot performance by flank length

	by_len_metric = "Spearman_r"
	# by_len_metric = "Pearson_r"
	# by_len_metric = "MAE"

	if len(str1_res_df) > 1:
		plt.figure(figsize=(8,6))
		sns.lineplot(
			data=str1_res_df,
			x="n_flank",
			y=by_len_metric,
			marker="o"
		)
		plt.title("Length 1 STR Prediction Performance by Flank Length")
		plt.xlabel("Flank Length (bp)")
		plt.ylim(
			bottom=0,
			top=1.05 * str1_res_df[by_len_metric].max()
		)
		plt.grid(True)
		plt.show()

	if len(str2_res_df) > 1:
		plt.figure(figsize=(8,6))
		sns.lineplot(
			data=str2_res_df,
			x="n_flank",
			y=by_len_metric,
			marker="o"
		)
		plt.title("Length 2 STR Prediction Performance by Flank Length")
		plt.xlabel("Flank Length (bp)")
		plt.ylim(
			bottom=0,
			top=1.05 * str2_res_df[by_len_metric].max()
		)
		plt.grid(True)
		plt.show()
