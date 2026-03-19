""" Loads trained model then plots token embeddings in 2D space with PCA. """

import os
# import yaml

import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


PROMPT_START_ID = 16


if __name__ == "__main__":

	# Options
	model_out_dir = "../training/output/soft_prompt"
	str_version = "str2"
	model_version = "v1"
	model_desc = "str2_l1m_f2000_p128_log_2026-01-12_13-28-53"


	# Load model and config
	model_dir = os.path.join(
		model_out_dir, str_version, model_version, model_desc
	)

	# Find best model checkpoint
	checkpoints_dir = os.path.join(model_dir, "checkpoints")
	best_model_path_options = [
		os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir)
		if f.startswith("best-model-epoch=")
	]
	assert len(best_model_path_options) == 1, f"Expected exactly one best-model checkpoint, found: {best_model_path_options}"
	best_model_path = best_model_path_options[0]
	print(f"Loading model from checkpoint: {best_model_path}")

	# Determine device (cuda > mps > cpu)
	model = torch.load(best_model_path, map_location="cpu")

	# config_path = os.path.join(model_dir, "config.yaml")
	# with open(config_path, "r") as f:
	# 	config = yaml.safe_load(f)


	# Extract token embeddings
	tokenizer = AutoTokenizer.from_pretrained(
		model['datamodule_hyper_parameters']['tokenizer_checkpoint'],
		trust_remote_code=True
	)

	embeddings = model["state_dict"]["hyena_model.backbone.embeddings.word_embeddings.weight"].cpu().numpy()


	# Evaluate embedding space with PCA
	base_emb = embeddings[[
		tokenizer.get_vocab()['A'],
		tokenizer.get_vocab()['C'],
		tokenizer.get_vocab()['G'],
		tokenizer.get_vocab()['T'],
		tokenizer.get_vocab()['N'],
	]]

	all_prompt_ids = torch.arange(
		PROMPT_START_ID,
		PROMPT_START_ID 
		+ model['datamodule_hyper_parameters']['n_prefix_prompt_tokens']
		+ model['datamodule_hyper_parameters']['n_str_prompt_tokens'],
	)
	prefix_prompt_ids = all_prompt_ids[
		:model['datamodule_hyper_parameters']['n_prefix_prompt_tokens']
	]
	str_prompt_ids = all_prompt_ids[
		model['datamodule_hyper_parameters']['n_prefix_prompt_tokens']:
	]

	prefix_emb = embeddings[prefix_prompt_ids]
	in_str_emb = embeddings[str_prompt_ids]


	# Run PCA on just base embeddings and on all three categories
	base_pca = PCA(n_components=2)
	base_only_emb_2d = base_pca.fit_transform(base_emb)

	all_pca = PCA(n_components=2)
	all_pca.fit(np.concatenate([base_emb, prefix_emb, in_str_emb], axis=0))

	base_emb_2d = all_pca.transform(base_emb)
	prefix_emb_2d = all_pca.transform(prefix_emb)
	in_str_emb_2d = all_pca.transform(in_str_emb)


	# Plot just the base embeddings in base space
	plt.figure(figsize=(6, 6))
	sns.scatterplot(x=base_only_emb_2d[:, 0], y=base_only_emb_2d[:, 1], s=100)
	for i, base in enumerate(['A', 'C', 'G', 'T', 'N']):
		plt.text(base_only_emb_2d[i, 0]+0.01, base_only_emb_2d[i, 1]+0.01, base, fontsize=12)
	plt.title("PCA of Base Token Embeddings in Base Embedding Space")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.gca().set_aspect("equal", adjustable="box")
	plt.axis('equal')
	plt.show()


	# Plot just the base embeddings
	plt.figure(figsize=(6, 6))
	sns.scatterplot(x=base_emb_2d[:, 0], y=base_emb_2d[:, 1], s=100)
	for i, base in enumerate(['A', 'C', 'G', 'T', 'N']):
		plt.text(base_emb_2d[i, 0]+0.01, base_emb_2d[i, 1]+0.01, base, fontsize=12)
	plt.title("PCA of Base Token Embeddings in PCA Space of All Embeddings")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.gca().set_aspect("equal", adjustable="box")
	plt.axis('equal')
	plt.show()


	# Plot all embeddings with categories highlighted
	plt.figure(figsize=(6, 6))	
	sns.scatterplot(x=prefix_emb_2d[:, 0], y=prefix_emb_2d[:, 1], s=100, color='blue', label='Prefix Prompt Tokens')
	sns.scatterplot(x=in_str_emb_2d[:, 0], y=in_str_emb_2d[:, 1], s=100, color='green', label='In-STR Prompt Tokens')
	sns.scatterplot(x=base_emb_2d[:, 0], y=base_emb_2d[:, 1], s=100, color='red', label='Base Tokens')
	for i, base in enumerate(['A', 'C', 'G', 'T', 'N']):
		plt.text(base_emb_2d[i, 0]+0.01, base_emb_2d[i, 1]+0.01, base, fontsize=12)
	plt.title("PCA of Token Embeddings")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.gca().set_aspect("equal", adjustable="box")
	plt.axis('equal')
	plt.legend()
	plt.show()








	# # Evaluate base embeddings
	# base_emb = embeddings[[
	# 	tokenizer.get_vocab()['A'],
	# 	tokenizer.get_vocab()['C'],
	# 	tokenizer.get_vocab()['G'],
	# 	tokenizer.get_vocab()['T'],
	# 	tokenizer.get_vocab()['N'],
	# ]]

	# # Run PCA on base embeddings
	# base_pca = PCA(n_components=2)
	# base_emb_2d = base_pca.fit_transform(base_emb)

	# # Plot base embeddings
	# plt.figure(figsize=(6, 6))
	# sns.scatterplot(x=base_emb_2d[:, 0], y=base_emb_2d[:, 1], s=100)
	# for i, base in enumerate(['A', 'C', 'G', 'T', 'N']):
	# 	plt.text(base_emb_2d[i, 0]+0.01, base_emb_2d[i, 1]+0.01, base, fontsize=12)
	# plt.title("PCA of Base Token Embeddings")
	# plt.xlabel("PC1")
	# plt.ylabel("PC2")
	# plt.gca().set_aspect("equal", adjustable="box")
	# plt.axis('equal')	
	# plt.show()


	# # Run PCA on all embeddings then plot base tokens in that space
	# pca = PCA(n_components=2)
	# emb_2d = pca.fit_transform(embeddings)
	
	# base_embs_2d = pca.transform(base_emb)

	# plt.figure(figsize=(6, 6))
	# sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], s=10, alpha=0.5)
	# sns.scatterplot(x=base_embs_2d[:, 0], y=base_embs_2d[:, 1], s=100, color='red')
	# for i, base in enumerate(['A', 'C', 'G', 'T', 'N']):
	# 	plt.text(base_embs_2d[i, 0]+0.01, base_embs_2d[i, 1]+0.01, base, fontsize=12)
	# plt.title("PCA of All Token Embeddings (Base Tokens Highlighted)")
	# plt.xlabel("PC1")
	# plt.ylabel("PC2")
	# plt.gca().set_aspect("equal", adjustable="box")
	# plt.axis('equal')
	# plt.show()


	
