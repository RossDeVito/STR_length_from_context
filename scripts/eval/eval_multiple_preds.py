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
	rmse = np.sqrt(mse)
	mae = sk_metrics.mean_absolute_error(true, pred)
	r2 = sk_metrics.r2_score(true, pred)
	pearson_r, pearson_p = stats.pearsonr(true, pred)
	spearman_r, spearman_p = stats.spearmanr(true, pred)

	return {
		"MSE": mse,
		"RMSE": rmse,
		"MAE": mae,
		"R2": r2,
		"Pearson_r": pearson_r,
		"Pearson_p": pearson_p,
		"Spearman_r": spearman_r,
		"Spearman_p": spearman_p,
	}


if __name__ == '__main__':

	pred_dir = "predictions/soft_prompt/str2/tscc_v1"
	model_names = [
		"str2_l1m_f100_p128_log_2026-01-12_13-28-45",
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

		results.append({
			"model": model_name,
			**metrics
		})

	results_df = pd.DataFrame(results)
	print(results_df)