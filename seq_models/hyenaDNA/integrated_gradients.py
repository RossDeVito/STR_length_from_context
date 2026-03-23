""" Compute Integrated Gradients attribution scores for STR length model.

Uses Captum to compute per-position attribution scores with respect to
the embedding layer. The baseline for flanking DNA positions is the mean
embedding of A, C, G, T. STR bases, prompt positions, and STR gap prompts
retain their actual embeddings (producing zero attribution by construction).

This means attributions measure: "how much does knowing the specific
flanking base at this position (vs. a generic base) contribute to the
predicted STR length?"

Args:
	--config: Path to YAML config file.
	--output_dir: Directory to save outputs.
	--cpu: Force use of CPU, even if CUDA/MPS is available.

Config keys:
	data_path (str): Path to STR data file.
	ref_path (str): Path to human reference genome file.
	model_path (str): Path to model checkpoint.
	desc (str): Output name/path relative to output_dir.
	split (str): Split field value in data file to run on.
	batch_size (int): Captum internal_batch_size.
	num_workers (int): DataLoader num_workers.
	n_steps (int): Captum IG integration steps.
"""

import argparse
import datetime
import os
import json
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients
import pandas as pd
from tqdm import tqdm

from seq_models.hyenaDNA.model import STRLengthModel
from seq_models.hyenaDNA.data import STRLengthDataset
from seq_models.hyenaDNA.hyenaDNA_info import PROMPT_START_ID


# ---------------------------------------------------------------------------
# Embedding-space model wrapper
# ---------------------------------------------------------------------------

class EmbeddingForwardWrapper(nn.Module):
	"""Wraps STRLengthModel to accept embeddings instead of token IDs.

	Registers a forward hook on the embedding layer that replaces the
	lookup output with the provided embedding tensor. This lets Captum
	interpolate in continuous embedding space while the rest of the model
	(Hyena backbone, pooling, head) runs normally.
	"""

	def __init__(self, str_model):
		super().__init__()
		self.str_model = str_model
		self.embed_layer = (
			str_model.hyena_model.backbone.embeddings.word_embeddings
		)

	def forward(self, input_embeds):
		# Hook replaces the embedding lookup output with our tensor
		handle = self.embed_layer.register_forward_hook(
			lambda mod, inp, out: input_embeds
		)

		# Dummy input_ids -- shape must match but values are ignored
		dummy_ids = torch.zeros(
			input_embeds.shape[0], input_embeds.shape[1],
			dtype=torch.long, device=input_embeds.device,
		)

		output = self.str_model(dummy_ids)
		handle.remove()
		return output


# ---------------------------------------------------------------------------
# Baseline construction
# ---------------------------------------------------------------------------

def construct_baseline(
	input_ids,
	embed_layer,
	mean_base_emb,
	n_prefix_prompt,
	n_flanking_bp,
	n_str_bp,
	n_str_prompt,
):
	"""Build baseline embeddings for a batch of input_ids.

	Only flanking DNA positions receive the mean A/C/G/T embedding.
	Prefix prompts, STR bases (left and right), STR gap prompts all
	retain their actual embeddings, producing zero IG attribution.

	Sequence layout:
		[prefix_prompt | left_flank | left_str | str_prompt | right_str | right_flank]

	Args:
		input_ids: (batch, seq_len) token IDs.
		embed_layer: The model's nn.Embedding layer.
		mean_base_emb: (hidden_dim,) mean of A, C, G, T embeddings.
		n_prefix_prompt: Number of prefix prompt tokens.
		n_flanking_bp: Number of flanking base pairs on each side.
		n_str_bp: Number of STR base pairs shown on each side.
		n_str_prompt: Number of STR gap prompt tokens.

	Returns:
		Tensor of shape (batch, seq_len, hidden_dim).
	"""
	with torch.no_grad():
		actual_embeds = embed_layer(input_ids)

	baseline = actual_embeds.clone()

	# Left flank: positions [n_prefix_prompt, n_prefix_prompt + n_flanking_bp)
	lf_start = n_prefix_prompt
	lf_end = lf_start + n_flanking_bp
	baseline[:, lf_start:lf_end] = mean_base_emb

	# Right flank: last n_flanking_bp positions
	rf_start = (
		n_prefix_prompt + n_flanking_bp + n_str_bp + n_str_prompt + n_str_bp
	)
	rf_end = rf_start + n_flanking_bp
	baseline[:, rf_start:rf_end] = mean_base_emb

	return baseline


# ---------------------------------------------------------------------------
# Position labelling
# ---------------------------------------------------------------------------

def make_position_labels(
	seq_len, n_prefix_prompt, n_flanking_bp, n_str_bp, n_str_prompt,
):
	"""Create a string label for each position in the input sequence.

	Returns:
		List[str] of length seq_len.  Values are one of:
			'prefix_prompt', 'left_flank', 'left_str', 'str_prompt',
			'right_str', 'right_flank'.
	"""
	labels = []
	labels += ["prefix_prompt"] * n_prefix_prompt
	labels += ["left_flank"] * n_flanking_bp
	labels += ["left_str"] * n_str_bp
	labels += ["str_prompt"] * n_str_prompt
	labels += ["right_str"] * n_str_bp
	labels += ["right_flank"] * n_flanking_bp
	assert len(labels) == seq_len, (
		f"Position label length {len(labels)} != seq_len {seq_len}"
	)
	return labels


# ---------------------------------------------------------------------------
# Sequence decoding
# ---------------------------------------------------------------------------

def decode_sequence(input_ids_1d, tokenizer, position_labels):
	"""Decode token IDs to a string, using X for prompt positions.

	Args:
		input_ids_1d: (seq_len,) numpy array of token IDs.
		tokenizer: HuggingFace tokenizer.
		position_labels: List[str] of position types.

	Returns:
		str of length seq_len.
	"""
	chars = []
	id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

	for tok_id, label in zip(input_ids_1d, position_labels):
		if "prompt" in label:
			chars.append("X")
		else:
			token = id_to_token.get(int(tok_id), "?")
			chars.append(token)

	return "".join(chars)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(
		description="Compute IG attribution scores for STR length model."
	)
	parser.add_argument(
		"--config", type=str, required=True,
		help="Path to the configuration YAML file.",
	)
	parser.add_argument(
		"--output_dir", type=str, default=".",
		help="Directory to save outputs.",
	)
	parser.add_argument(
		"--cpu", action="store_true",
		help="Force use of CPU, even if CUDA/MPS is available.",
	)
	args = parser.parse_args()

	# ------------------------------------------------------------------
	# Load config
	# ------------------------------------------------------------------
	print(f"Loading config from {args.config}")
	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	# ------------------------------------------------------------------
	# Device selection
	# ------------------------------------------------------------------
	if args.cpu:
		device = torch.device("cpu")
		print("CPU override flag set. Using CPU.")
	elif torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"Using CUDA ({torch.cuda.get_device_name(0)})")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
		print("Using MPS")
	else:
		device = torch.device("cpu")
		print("No GPU detected. Using CPU.")

	# ------------------------------------------------------------------
	# Set up output directory
	# ------------------------------------------------------------------
	run_dir = os.path.join(args.output_dir, config["desc"])
	os.makedirs(run_dir, exist_ok=True)

	config_save_path = os.path.join(run_dir, "config.yaml")
	with open(config_save_path, "w") as f:
		yaml.dump(config, f)
	print(f"Outputs will be saved to: {run_dir}")

	# ------------------------------------------------------------------
	# Load model checkpoint & extract saved hyper-parameters
	# ------------------------------------------------------------------
	model_path = config["model_path"]
	print(f"Loading model from {model_path}")

	model = STRLengthModel.load_from_checkpoint(
		model_path, map_location="cpu"
	)
	model.eval()
	model.to(device)

	model_hp = model.hparams

	# Also load raw checkpoint for datamodule hyper-parameters
	raw_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
	dm_hp = raw_ckpt.get("datamodule_hyper_parameters", {})

	# ------------------------------------------------------------------
	# Resolve sequence layout dimensions from saved hyper-parameters
	# ------------------------------------------------------------------
	n_prefix_prompt = model_hp.get(
		"n_prefix_prompt_tokens",
		dm_hp.get("n_prefix_prompt_tokens", 0),
	)
	n_str_prompt = model_hp.get(
		"n_str_prompt_tokens",
		dm_hp.get("n_str_prompt_tokens", 0),
	)
	n_flanking_bp = dm_hp.get("n_flanking_bp", model_hp.get("n_flanking_bp"))
	n_str_bp = dm_hp.get("n_str_bp", model_hp.get("n_str_bp"))
	tokenizer_checkpoint = dm_hp.get(
		"tokenizer_checkpoint",
		model_hp.get("hyenaDNA_checkpoint"),
	)

	seq_len = (
		n_prefix_prompt
		+ n_flanking_bp + n_str_bp
		+ n_str_prompt
		+ n_str_bp + n_flanking_bp
	)

	print(f"Sequence layout: "
		  f"prefix_prompt={n_prefix_prompt}, "
		  f"left_flank={n_flanking_bp}, "
		  f"left_str={n_str_bp}, "
		  f"str_prompt={n_str_prompt}, "
		  f"right_str={n_str_bp}, "
		  f"right_flank={n_flanking_bp}, "
		  f"total={seq_len}")

	# ------------------------------------------------------------------
	# Create dataset for the requested split
	# ------------------------------------------------------------------
	print(f"Loading tokenizer from {tokenizer_checkpoint}")
	tokenizer = AutoTokenizer.from_pretrained(
		tokenizer_checkpoint, trust_remote_code=True
	)

	print(f"Loading data from {config['data_path']} (split={config['split']})")
	full_df = pd.read_csv(config["data_path"], sep="\t")
	split_df = full_df[full_df["split"] == config["split"]].reset_index(
		drop=True
	)
	print(f"  {len(split_df)} samples in '{config['split']}' split")

	dataset = STRLengthDataset(
		str_df=split_df,
		ref_path=config["ref_path"],
		tokenizer=tokenizer,
		n_flanking_bp=n_flanking_bp,
		n_str_bp=n_str_bp,
		n_prefix_prompt_tokens=n_prefix_prompt,
		n_str_prompt_tokens=n_str_prompt,
	)

	loader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		num_workers=config.get("num_workers", 4),
		pin_memory=(device.type == "cuda"),
	)

	# ------------------------------------------------------------------
	# Compute mean-ACGT baseline embedding
	# ------------------------------------------------------------------
	embed_layer = model.hyena_model.backbone.embeddings.word_embeddings
	vocab = tokenizer.get_vocab()

	with torch.no_grad():
		base_ids = torch.tensor(
			[vocab["A"], vocab["C"], vocab["G"], vocab["T"]],
			dtype=torch.long,
			device=device,
		)
		base_embeds = embed_layer(base_ids)
		mean_base_emb = base_embeds.mean(dim=0)

	print(f"Mean-ACGT baseline embedding computed "
		  f"(hidden_dim={mean_base_emb.shape[0]})")

	# ------------------------------------------------------------------
	# Set up Captum IG
	# ------------------------------------------------------------------
	wrapper = EmbeddingForwardWrapper(model)
	wrapper.eval()

	ig = IntegratedGradients(wrapper)

	n_steps = config.get("n_steps", 50)
	internal_batch_size = config.get("batch_size", 32)
	ig_method = config.get("method", "gausslegendre")

	print(f"IG config: n_steps={n_steps}, "
		  f"internal_batch_size={internal_batch_size}, "
		  f"method={ig_method}")

	# ------------------------------------------------------------------
	# Build position labels (same for all samples)
	# ------------------------------------------------------------------
	position_labels = make_position_labels(
		seq_len, n_prefix_prompt, n_flanking_bp, n_str_bp, n_str_prompt,
	)

	# ------------------------------------------------------------------
	# Run IG
	# ------------------------------------------------------------------
	n_samples = len(dataset)
	print(f"--- Computing Integrated Gradients ({n_samples} samples) ---")

	attributions_list = []
	input_ids_list = []
	sequences_list = []
	predictions_list = []
	baseline_predictions_list = []
	raw_predictions_list = []
	raw_baseline_predictions_list = []
	labels_list = []
	deltas_list = []

	for i, batch in enumerate(tqdm(loader, desc="IG attribution")):
		input_ids = batch["input_ids"].to(device)
		label = batch["label"]

		# Actual embeddings
		with torch.no_grad():
			actual_embeds = embed_layer(input_ids)

		# Baseline embeddings (only flanking positions replaced)
		baseline_embeds = construct_baseline(
			input_ids, embed_layer, mean_base_emb,
			n_prefix_prompt, n_flanking_bp, n_str_bp, n_str_prompt,
		)

		# Model predictions for both input and baseline
		with torch.no_grad():
			raw_pred = wrapper(actual_embeds)
			raw_baseline_pred = wrapper(baseline_embeds)

			if model_hp.get("log_transform", False):
				pred = torch.expm1(raw_pred)
				baseline_pred = torch.expm1(raw_baseline_pred)
			else:
				pred = raw_pred
				baseline_pred = raw_baseline_pred

		# Compute IG attributions
		attrs, delta = ig.attribute(
			actual_embeds,
			baselines=baseline_embeds,
			n_steps=n_steps,
			method=ig_method,
			internal_batch_size=internal_batch_size,
			return_convergence_delta=True,
		)

		# Reduce to per-position: sum across hidden dim
		pos_attrs = attrs.sum(dim=-1).squeeze(0)

		# Decode sequence to string
		ids_np = input_ids.squeeze(0).cpu().numpy()
		seq_str = decode_sequence(ids_np, tokenizer, position_labels)

		# Accumulate
		attributions_list.append(pos_attrs.detach().cpu().numpy())
		input_ids_list.append(ids_np)
		sequences_list.append(seq_str)
		predictions_list.append(pred.item())
		baseline_predictions_list.append(baseline_pred.item())
		raw_predictions_list.append(raw_pred.item())
		raw_baseline_predictions_list.append(raw_baseline_pred.item())
		labels_list.append(label.item())
		deltas_list.append(delta.item())

	# Stack into arrays
	all_attributions = np.stack(attributions_list)
	all_input_ids = np.stack(input_ids_list)
	all_sequences = np.array(sequences_list)
	all_predictions = np.array(predictions_list, dtype=np.float32)
	all_baseline_predictions = np.array(
		baseline_predictions_list, dtype=np.float32
	)
	all_raw_predictions = np.array(raw_predictions_list, dtype=np.float32)
	all_raw_baseline_predictions = np.array(
		raw_baseline_predictions_list, dtype=np.float32
	)
	all_labels = np.array(labels_list, dtype=np.float32)
	all_deltas = np.array(deltas_list, dtype=np.float32)

	# Per-sample relative convergence delta
	# Both delta and denominator are in raw model output space
	# (log space if log_transform=True)
	raw_pred_diffs = np.abs(all_raw_predictions - all_raw_baseline_predictions)
	safe_diffs = np.where(raw_pred_diffs > 1e-6, raw_pred_diffs, 1.0)
	all_relative_deltas = np.abs(all_deltas) / safe_diffs

	# ------------------------------------------------------------------
	# Save results
	# ------------------------------------------------------------------
	npz_path = os.path.join(run_dir, "attributions.npz")

	np.savez_compressed(
		npz_path,
		attributions=all_attributions,
		input_ids=all_input_ids,
		sequences=all_sequences,
		predictions=all_predictions,
		baseline_predictions=all_baseline_predictions,
		raw_predictions=all_raw_predictions,
		raw_baseline_predictions=all_raw_baseline_predictions,
		labels=all_labels,
		convergence_deltas=all_deltas,
		relative_convergence_deltas=all_relative_deltas,
		position_labels=np.array(position_labels),
		hipstr_names=split_df["HipSTR_name"].values,
		rev_comp=split_df["rev_comp"].values,
	)
	print(f"Attributions saved to {npz_path}")

	# ------------------------------------------------------------------
	# Compute and save convergence report
	# ------------------------------------------------------------------
	# Delta is in the model's raw output space (log space if
	# log_transform=True), so the denominator must match.
	# Per-sample relative deltas already computed above.
	convergence = {
		"absolute_delta": {
			"mean": float(np.mean(np.abs(all_deltas))),
			"median": float(np.median(np.abs(all_deltas))),
			"max": float(np.max(np.abs(all_deltas))),
			"std": float(np.std(np.abs(all_deltas))),
		},
		"relative_delta": {
			"mean": float(np.mean(all_relative_deltas)),
			"median": float(np.median(all_relative_deltas)),
			"max": float(np.max(all_relative_deltas)),
			"std": float(np.std(all_relative_deltas)),
			"pct_above_5pct": float(
				np.mean(all_relative_deltas > 0.05) * 100
			),
			"pct_above_1pct": float(
				np.mean(all_relative_deltas > 0.01) * 100
			),
		},
		"raw_prediction_diff_F_input_minus_F_baseline": {
			"mean": float(np.mean(raw_pred_diffs)),
			"median": float(np.median(raw_pred_diffs)),
			"min": float(np.min(raw_pred_diffs)),
			"max": float(np.max(raw_pred_diffs)),
		},
		"log_transform": bool(model_hp.get("log_transform", False)),
		"n_samples": n_samples,
		"n_steps": n_steps,
		"method": ig_method,
	}

	convergence_path = os.path.join(run_dir, "convergence.json")
	with open(convergence_path, "w") as f:
		json.dump(convergence, f, indent=4)
	print(f"Convergence report saved to {convergence_path}")

	print("\n--- Convergence Summary ---")
	print(f"  Absolute delta  -- "
		  f"mean: {convergence['absolute_delta']['mean']:.4f}, "
		  f"median: {convergence['absolute_delta']['median']:.4f}, "
		  f"max: {convergence['absolute_delta']['max']:.4f}")
	print(f"  Relative delta  -- "
		  f"mean: {convergence['relative_delta']['mean']:.4f}, "
		  f"median: {convergence['relative_delta']['median']:.4f}, "
		  f"max: {convergence['relative_delta']['max']:.4f}")
	print(f"  Samples > 5% relative delta: "
		  f"{convergence['relative_delta']['pct_above_5pct']:.1f}%")
	print(f"  Samples > 1% relative delta: "
		  f"{convergence['relative_delta']['pct_above_1pct']:.1f}%")

	if convergence["relative_delta"]["mean"] > 0.05:
		print("  WARNING: Mean relative delta > 5%. "
			  "Consider increasing n_steps.")
	else:
		print("  Convergence looks good.")

	# ------------------------------------------------------------------
	# Save run metadata
	# ------------------------------------------------------------------
	meta = {
		"config": config,
		"model_hyperparameters": {
			k: v for k, v in dict(model_hp).items()
			if isinstance(v, (int, float, str, bool, type(None)))
		},
		"datamodule_hyperparameters": {
			k: v for k, v in dm_hp.items()
			if isinstance(v, (int, float, str, bool, type(None)))
		},
		"sequence_layout": {
			"n_prefix_prompt": n_prefix_prompt,
			"n_flanking_bp": n_flanking_bp,
			"n_str_bp": n_str_bp,
			"n_str_prompt": n_str_prompt,
			"seq_len": seq_len,
			"order": [
				"prefix_prompt", "left_flank", "left_str",
				"str_prompt", "right_str", "right_flank",
			],
		},
		"ig_config": {
			"n_steps": n_steps,
			"internal_batch_size": internal_batch_size,
			"method": ig_method,
			"baseline": "mean_ACGT_embedding (flanking only)",
		},
		"n_samples": n_samples,
		"device": str(device),
		"timestamp": datetime.datetime.now().isoformat(),
	}

	meta_path = os.path.join(run_dir, "meta.json")
	with open(meta_path, "w") as f:
		json.dump(meta, f, indent=4)
	print(f"Metadata saved to {meta_path}")

	print("\n--- Done ---")


if __name__ == "__main__":
	main()