""" Evaluate predictions.

Includes computing metrics and plotting.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import scipy.stats as stats


def plot_density_scatter(ax, df):
    """Plot 1: 2D Histogram/Density Plot"""
    # Create 2D histogram
    # cmin=1 prevents plotting empty white space
    h = ax.hist2d(df['true_length'], df['pred_length'], bins=50, cmap='Blues', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')
    
    # Add identity line (Perfect prediction)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=10, label='Perfect Prediction')
    
    ax.set_title('Density of Predictions')
    ax.set_xlabel('True Length')
    ax.set_ylabel('Predicted Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_box_by_length(ax, df):
    """Plot 2: Boxplot grouped by True Length"""
    # Group predictions by true length
    # If you have too many x-categories, you might bin them:
    # df['len_bin'] = pd.cut(df['true_length'], bins=10)
    
    sns.boxplot(x='true_length', y='pred_length', data=df, ax=ax, 
                color='lightblue', showfliers=False) # Hide outliers to see the median better
    
    # Add identity line approximation (diagonal across boxes)
    # Since x-axis is categorical 0..N, we map true values to 0..N indices
    unique_lens = sorted(df['true_length'].unique())
    ax.plot(range(len(unique_lens)), unique_lens, 'r--', lw=2, label='Target')
    
    # Clean up x-labels if too crowded
    if len(unique_lens) > 20:
        for ind, label in enumerate(ax.get_xticklabels()):
            if ind % 5 == 0:  # every 5th label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)

    ax.set_title('Distribution of Predictions per True Length')
    ax.set_ylabel('Predicted Length')

def plot_residuals(ax, df):
    """Plot 3: Residuals (Bias Analysis)"""
    residuals = df['pred_length'] - df['true_length']
    
    sns.scatterplot(x=df['true_length'], y=residuals, ax=ax, alpha=0.3)
    ax.axhline(0, color='r', linestyle='--')
    
    ax.set_title('Residuals (Pred - True)')
    ax.set_xlabel('True Length')
    ax.set_ylabel('Error')
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
	
	pred_dir = 'predictions/soft_prompt/str2/tscc_dev/dev4Lb_m450_2025-12-06_16-44-44'
	pred_path = os.path.join(pred_dir, 'predictions_test.tsv')

	# Load predictions
	pred_df = pd.read_csv(pred_path, sep='\t')

	# Compute metrics
	# MSE
	mse = sk_metrics.mean_squared_error(
		pred_df['true_length'],
		pred_df['pred_length']
	)
	print(f"MSE: {mse:.4f}")

	# MAE
	mae = sk_metrics.mean_absolute_error(
		pred_df['true_length'],
		pred_df['pred_length']
	)
	print(f"MAE: {mae:.4f}")

	# Median AE
	med_ae = sk_metrics.median_absolute_error(
		pred_df['true_length'],
		pred_df['pred_length']
	)
	print(f"Median AE: {med_ae:.4f}")

	# R^2
	r2 = sk_metrics.r2_score(
		pred_df['true_length'],
		pred_df['pred_length']
	)
	print(f"R^2: {r2:.4f}")

	# Pearson correlation with p-value
	pearson_r, pearson_p = stats.pearsonr(
		pred_df['true_length'],
		pred_df['pred_length']
	)
	print(f"Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

	# Spearman correlation with p-value
	spearman_r, spearman_p = stats.spearmanr(
		pred_df['true_length'],
		pred_df['pred_length']
	)
	print(f"Spearman r: {spearman_r:.4f}, p-value: {spearman_p:.4e}")


	# Plot true vs predicted
	plt.figure(figsize=(6,6))
	sns.scatterplot(
		x='true_length',
		y='pred_length',
		data=pred_df,
		alpha=0.25
	)
	plt.show()
     
	
	sns.set_theme(style="whitegrid")
	fig, axes = plt.subplots(1, 3, figsize=(20, 6))

	plot_density_scatter(axes[0], pred_df)
	plot_box_by_length(axes[1], pred_df)
	plot_residuals(axes[2], pred_df)

	# Add metrics to the figure title
	plt.suptitle(f"Model Eval | Pearson: {pearson_r:.3f} | Spearman: {spearman_r:.3f} | MAE: {mae:.3f}", fontsize=14)
	
	plt.tight_layout()
	plt.show()