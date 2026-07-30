""" Compute Integrated Gradients attribution scores for the Caduceus STR
length/variation model.

Uses Captum to compute per-position attribution scores with respect to the
Caduceus embedding layer. The baseline for flanking DNA positions is the mean
embedding of A, C, G, T. STR bases (synthesized by tiling the reference
repeat unit), prompt positions (prefix / STR-gap / suffix), retain their
actual embeddings (producing zero attribution by construction).

This means attributions measure: "how much does knowing the specific
flanking base at this position (vs. a generic base) contribute to the
predicted output?"

The checkpoint may be single- or multi-task (active targets are whatever the
model was trained with, e.g. 'length' and/or 'variation' -- see
seq_models/caduceus/model.py). Since Captum needs a single scalar target per
attribute() call, this script runs IG once per active task per sample (all in
one invocation), reusing the same actual/baseline embeddings across tasks.

Args:
	--config: Path to YAML config file.
	--output_dir: Directory to save outputs.
	--cpu: Force use of CPU, even if CUDA/MPS is available.

Config keys:
	model_dir (str): Trained model directory, relative to
		scripts/training/output/caduceus/ (contains checkpoints/ and
		config.yaml) -- the same convention as
		seq_models/caduceus/region_permutation_importance.py.
	desc (str): Output name/path relative to output_dir.
	split (str): Split to run on ('test', 'val', 'train').
	batch_size (int): Override the model's data-loading batch size (data is
		still loaded then run through IG one sample at a time).
	num_workers (int): DataLoader num_workers.
	n_steps (int): Captum IG integration steps.
	internal_batch_size (int): Captum internal batch size for interpolation
		steps. Defaults to `batch_size`.
	method (str): Captum IG integration method. Default "gausslegendre".
	subsample_loci (int, optional): If set, deterministically restrict the
		split to this many unique loci (both fwd/RC orientations kept per
		locus, via region_permutation_importance.subsample_dataset) -- for
		fast, reproducible method/n_steps comparison runs.
	seed (int): Random seed for `subsample_loci`. Default 42.

Outputs (written to {output_dir}/{desc}/):
	config.yaml
		Echo of the resolved run config (as loaded, before any mutation).
	attributions.npz
		Shared arrays (one row per sample, i.e. per forward/reverse-complement
		orientation of a locus):
			input_ids                (n_samples, seq_len) int -- token ids.
			sequences                (n_samples,) str -- decoded bases, with
				'X' at every prompt-token position.
			position_labels          (seq_len,) str -- one of 'prefix_prompt',
				'left_flank', 'left_str', 'str_prompt', 'right_str',
				'right_flank', 'suffix_prompt' per position.
			ids                      (n_samples,) str -- locus id (the data
				file's 'ID' column; shared by a locus's fwd/RC pair).
			rev_comp                 (n_samples,) bool.
		Then, for EACH active task name (e.g. 'length', 'variation'), arrays
		suffixed "_{task}":
			attributions_{task}          (n_samples, seq_len) float -- IG
				attribution per position (embedding-dim summed), attributing
				that task's training-space output.
			predictions_{task}           (n_samples,) float -- native-unit
				prediction for the actual input (inverse-transformed).
			baseline_predictions_{task}  (n_samples,) float -- native-unit
				prediction for the baseline input.
			raw_predictions_{task}       (n_samples,) float -- training-space
				(pre-inverse-transform) prediction for the actual input; this
				is the space Captum's convergence delta lives in.
			raw_baseline_predictions_{task} (n_samples,) float -- training-
				space prediction for the baseline input.
			labels_{task}                 (n_samples,) float -- native-unit
				ground-truth label for that task.
			convergence_deltas_{task}     (n_samples,) float -- Captum
				convergence delta (training-space).
			relative_convergence_deltas_{task} (n_samples,) float -- abs
				delta divided by abs(raw_prediction - raw_baseline_prediction)
				(training-space), i.e. delta as a fraction of the quantity
				being attributed.
	convergence.json
		{task_name: {"absolute_delta": {...}, "relative_delta": {...},
		"raw_prediction_diff_F_input_minus_F_baseline": {...},
		"transform": transform_name}, ..., "n_samples", "n_steps", "method"}.
	meta.json
		{"config": <run config>, "model_dir": ..., "task_names": [...],
		"transforms": {task: transform_name}, "targets": {task: source_col},
		"sequence_layout": {n_prefix_prompt, n_flanking_bp, n_str_bp,
		n_str_prompt, n_suffix_prompt, seq_len, "order": [7 segment names]},
		"ig_config": {n_steps, internal_batch_size, method, baseline,
		subsample_loci, seed},
		"n_samples", "device", "timestamp"}.
"""

import argparse
import datetime
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
import yaml
from tqdm import tqdm

from seq_models.caduceus.model import inverse_transform
from seq_models.caduceus.region_permutation_importance import (
	load_model_and_data,
	subsample_dataset,
)


# ---------------------------------------------------------------------------
# Embedding-space model wrapper
# ---------------------------------------------------------------------------

class EmbeddingForwardWrapper(nn.Module):
	"""Wraps STRLengthModel to accept embeddings instead of token IDs.

	Registers a forward hook on the (SplitEmbedding) embedding layer that
	replaces the lookup output with the provided embedding tensor. This lets
	Captum interpolate in continuous embedding space while the rest of the
	model (Caduceus backbone, pooling, heads) runs normally.

	Returns a single (batch, n_tasks) tensor (rather than STRLengthModel's
	native dict output) so Captum's `target=task_idx` can select which task
	to attribute.
	"""

	def __init__(self, str_model, task_names):
		super().__init__()
		self.str_model = str_model
		self.task_names = task_names
		self.embed_layer = (
			str_model.caduceus.backbone.embeddings.word_embeddings
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

		out = self.str_model(dummy_ids)
		handle.remove()
		return torch.stack([out[name] for name in self.task_names], dim=-1)


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

	Only flanking DNA positions receive the mean A/C/G/T embedding. Prefix
	prompt, STR bases (left and right, synthesized by tiling the reference
	repeat unit), STR-gap prompt, and suffix prompt all retain their actual
	embeddings, producing zero IG attribution.

	Sequence layout:
		[prefix_prompt | left_flank | left_str | str_prompt | right_str
		 | right_flank | suffix_prompt]

	Args:
		input_ids: (batch, seq_len) token IDs.
		embed_layer: The model's embedding layer (SplitEmbedding).
		mean_base_emb: (hidden_dim,) mean of A, C, G, T embeddings.
		n_prefix_prompt: Number of prefix prompt tokens.
		n_flanking_bp: Number of flanking base pairs on each side.
		n_str_bp: Number of synthesized STR base pairs shown on each side.
		n_str_prompt: Number of STR gap prompt tokens.

	Returns:
		Tensor of shape (batch, seq_len, hidden_dim).
	"""
	with torch.no_grad():
		actual_embeds = embed_layer(input_ids)

	baseline = actual_embeds.clone()

	# Left flank: positions [n_prefix_prompt, n_prefix_prompt + n_flanking_bp).
	lf_start = n_prefix_prompt
	lf_end = lf_start + n_flanking_bp
	baseline[:, lf_start:lf_end] = mean_base_emb

	# Right flank.
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
	n_suffix_prompt,
):
	"""Create a string label for each position in the input sequence.

	Returns:
		List[str] of length seq_len. Values are one of:
			'prefix_prompt', 'left_flank', 'left_str', 'str_prompt',
			'right_str', 'right_flank', 'suffix_prompt'.
	"""
	labels = []
	labels += ["prefix_prompt"] * n_prefix_prompt
	labels += ["left_flank"] * n_flanking_bp
	labels += ["left_str"] * n_str_bp
	labels += ["str_prompt"] * n_str_prompt
	labels += ["right_str"] * n_str_bp
	labels += ["right_flank"] * n_flanking_bp
	labels += ["suffix_prompt"] * n_suffix_prompt
	assert len(labels) == seq_len, (
		f"Position label length {len(labels)} != seq_len {seq_len}"
	)
	return labels


# ---------------------------------------------------------------------------
# Sequence decoding
# ---------------------------------------------------------------------------

def decode_sequence(input_ids_1d, id_to_token, position_labels):
	"""Decode token IDs to a string, using X for prompt positions.

	Args:
		input_ids_1d: (seq_len,) numpy array of token IDs.
		id_to_token: dict mapping token id -> token string.
		position_labels: List[str] of position types.

	Returns:
		str of length seq_len.
	"""
	chars = []
	for tok_id, label in zip(input_ids_1d, position_labels):
		if "prompt" in label:
			chars.append("X")
		else:
			chars.append(id_to_token.get(int(tok_id), "?"))
	return "".join(chars)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(
		description="Compute IG attribution scores for the Caduceus STR "
		"length/variation model."
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

	# pytorch_lightning sets __spec__ assumptions; mirror predict.py guard.
	global __spec__
	__spec__ = None

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
	# Load model + data (predict.py / region_permutation_importance.py
	# pattern: read the model's own config.yaml, build the data module from
	# it, load the best checkpoint).
	# ------------------------------------------------------------------
	model, model_config, dataset, _ = load_model_and_data(config)
	model.to(device)
	# 32-bit for numerical stability of the IG interpolation/backward passes,
	# regardless of the precision used during training (mirrors predict.py).
	model = model.float()

	task_names = list(model.task_names)
	transforms = model.transforms
	targets = model.targets
	print(f"Active targets: {task_names}")
	print(f"Transforms: {dict(transforms)}")

	subsample_loci = config.get("subsample_loci", None)
	seed = config.get("seed", 42)
	if subsample_loci is not None:
		dataset = subsample_dataset(dataset, int(subsample_loci), seed)

	tokenizer = dataset.tokenizer

	# ------------------------------------------------------------------
	# Resolve sequence layout dimensions from the model's data config
	# ------------------------------------------------------------------
	n_prefix_prompt = model_config.get("n_prefix_prompt_tokens", 0)
	n_str_prompt = model_config.get("n_str_prompt_tokens", 0)
	n_suffix_prompt = model_config.get("n_suffix_prompt_tokens", 0)
	n_flanking_bp = model_config["n_flanking_bp"]
	n_str_bp = model_config["n_str_bp"]

	seq_len = (
		n_prefix_prompt
		+ n_flanking_bp + n_str_bp
		+ n_str_prompt
		+ n_str_bp + n_flanking_bp
		+ n_suffix_prompt
	)

	print(f"Sequence layout: "
		  f"prefix_prompt={n_prefix_prompt}, "
		  f"left_flank={n_flanking_bp}, "
		  f"left_str={n_str_bp}, "
		  f"str_prompt={n_str_prompt}, "
		  f"right_str={n_str_bp}, "
		  f"right_flank={n_flanking_bp}, "
		  f"suffix_prompt={n_suffix_prompt}, "
		  f"total={seq_len}")

	position_labels = make_position_labels(
		seq_len, n_prefix_prompt, n_flanking_bp, n_str_bp, n_str_prompt,
		n_suffix_prompt,
	)

	# ------------------------------------------------------------------
	# DataLoader for the requested split (one sample at a time, as IG runs
	# per-sample regardless of how the data was batched for loading)
	# ------------------------------------------------------------------
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
	embed_layer = model.caduceus.backbone.embeddings.word_embeddings
	vocab = tokenizer.get_vocab()
	id_to_token = {v: k for k, v in vocab.items()}

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
	wrapper = EmbeddingForwardWrapper(model, task_names)
	wrapper.eval()

	ig = IntegratedGradients(wrapper)

	n_steps = config.get("n_steps", 50)
	internal_batch_size = config.get(
		"internal_batch_size", config.get("batch_size", 32)
	)
	ig_method = config.get("method", "gausslegendre")

	print(f"IG config: n_steps={n_steps}, "
		  f"internal_batch_size={internal_batch_size}, "
		  f"method={ig_method}")

	# ------------------------------------------------------------------
	# Run IG
	# ------------------------------------------------------------------
	n_samples = len(dataset)
	print(f"--- Computing Integrated Gradients "
		  f"({n_samples} samples, tasks={task_names}) ---")

	input_ids_list = []
	sequences_list = []
	locus_ids_list = []
	rev_comp_list = []

	attributions_acc = {t: [] for t in task_names}
	predictions_acc = {t: [] for t in task_names}
	baseline_predictions_acc = {t: [] for t in task_names}
	raw_predictions_acc = {t: [] for t in task_names}
	raw_baseline_predictions_acc = {t: [] for t in task_names}
	labels_acc = {t: [] for t in task_names}
	deltas_acc = {t: [] for t in task_names}

	for batch in tqdm(loader, desc="IG attribution"):
		input_ids = batch["input_ids"].to(device)

		# Actual embeddings
		with torch.no_grad():
			actual_embeds = embed_layer(input_ids)

		# Baseline embeddings (only flanking positions replaced)
		baseline_embeds = construct_baseline(
			input_ids, embed_layer, mean_base_emb,
			n_prefix_prompt, n_flanking_bp, n_str_bp, n_str_prompt,
		)

		# Model predictions (all tasks at once) for both input and baseline
		with torch.no_grad():
			raw_pred_all = wrapper(actual_embeds)          # (1, n_tasks)
			raw_baseline_pred_all = wrapper(baseline_embeds)

		ids_np = input_ids.squeeze(0).cpu().numpy()
		seq_str = decode_sequence(ids_np, id_to_token, position_labels)

		input_ids_list.append(ids_np)
		sequences_list.append(seq_str)
		locus_ids_list.append(str(batch["id"][0]))
		rev_comp_list.append(bool(batch["rev_comp"].item()))

		for t_idx, task in enumerate(task_names):
			raw_pred = raw_pred_all[0, t_idx]
			raw_baseline_pred = raw_baseline_pred_all[0, t_idx]
			pred = inverse_transform(transforms[task], raw_pred)
			baseline_pred = inverse_transform(transforms[task], raw_baseline_pred)

			# Compute IG attributions for this task
			attrs, delta = ig.attribute(
				actual_embeds,
				baselines=baseline_embeds,
				target=t_idx,
				n_steps=n_steps,
				method=ig_method,
				internal_batch_size=internal_batch_size,
				return_convergence_delta=True,
			)

			# Reduce to per-position: sum across hidden dim
			pos_attrs = attrs.sum(dim=-1).squeeze(0)

			attributions_acc[task].append(pos_attrs.detach().cpu().numpy())
			predictions_acc[task].append(pred.item())
			baseline_predictions_acc[task].append(baseline_pred.item())
			raw_predictions_acc[task].append(raw_pred.item())
			raw_baseline_predictions_acc[task].append(raw_baseline_pred.item())
			labels_acc[task].append(batch[task].item())
			deltas_acc[task].append(delta.item())

	# ------------------------------------------------------------------
	# Stack shared arrays
	# ------------------------------------------------------------------
	save_arrays = {
		"input_ids": np.stack(input_ids_list),
		"sequences": np.array(sequences_list),
		"position_labels": np.array(position_labels),
		"ids": np.array(locus_ids_list),
		"rev_comp": np.array(rev_comp_list, dtype=bool),
	}

	# ------------------------------------------------------------------
	# Stack per-task arrays, compute convergence stats
	# ------------------------------------------------------------------
	convergence = {}

	for task in task_names:
		attributions_task = np.stack(attributions_acc[task])
		predictions_task = np.array(predictions_acc[task], dtype=np.float32)
		baseline_predictions_task = np.array(
			baseline_predictions_acc[task], dtype=np.float32
		)
		raw_predictions_task = np.array(
			raw_predictions_acc[task], dtype=np.float32
		)
		raw_baseline_predictions_task = np.array(
			raw_baseline_predictions_acc[task], dtype=np.float32
		)
		labels_task = np.array(labels_acc[task], dtype=np.float32)
		deltas_task = np.array(deltas_acc[task], dtype=np.float32)

		# Per-sample relative convergence delta. Both delta and denominator
		# are in raw model output space (the task's training-space, e.g.
		# log1p space for a log1p-transformed target).
		raw_pred_diffs = np.abs(raw_predictions_task - raw_baseline_predictions_task)
		safe_diffs = np.where(raw_pred_diffs > 1e-6, raw_pred_diffs, 1.0)
		relative_deltas_task = np.abs(deltas_task) / safe_diffs

		save_arrays[f"attributions_{task}"] = attributions_task
		save_arrays[f"predictions_{task}"] = predictions_task
		save_arrays[f"baseline_predictions_{task}"] = baseline_predictions_task
		save_arrays[f"raw_predictions_{task}"] = raw_predictions_task
		save_arrays[f"raw_baseline_predictions_{task}"] = raw_baseline_predictions_task
		save_arrays[f"labels_{task}"] = labels_task
		save_arrays[f"convergence_deltas_{task}"] = deltas_task
		save_arrays[f"relative_convergence_deltas_{task}"] = relative_deltas_task

		convergence[task] = {
			"transform": transforms[task],
			"absolute_delta": {
				"mean": float(np.mean(np.abs(deltas_task))),
				"median": float(np.median(np.abs(deltas_task))),
				"max": float(np.max(np.abs(deltas_task))),
				"std": float(np.std(np.abs(deltas_task))),
			},
			"relative_delta": {
				"mean": float(np.mean(relative_deltas_task)),
				"median": float(np.median(relative_deltas_task)),
				"max": float(np.max(relative_deltas_task)),
				"std": float(np.std(relative_deltas_task)),
				"pct_above_5pct": float(
					np.mean(relative_deltas_task > 0.05) * 100
				),
				"pct_above_1pct": float(
					np.mean(relative_deltas_task > 0.01) * 100
				),
			},
			"raw_prediction_diff_F_input_minus_F_baseline": {
				"mean": float(np.mean(raw_pred_diffs)),
				"median": float(np.median(raw_pred_diffs)),
				"min": float(np.min(raw_pred_diffs)),
				"max": float(np.max(raw_pred_diffs)),
			},
		}

		print(f"\n--- Convergence Summary [{task}] ---")
		print(f"  Absolute delta  -- "
			  f"mean: {convergence[task]['absolute_delta']['mean']:.4f}, "
			  f"median: {convergence[task]['absolute_delta']['median']:.4f}, "
			  f"max: {convergence[task]['absolute_delta']['max']:.4f}")
		print(f"  Relative delta  -- "
			  f"mean: {convergence[task]['relative_delta']['mean']:.4f}, "
			  f"median: {convergence[task]['relative_delta']['median']:.4f}, "
			  f"max: {convergence[task]['relative_delta']['max']:.4f}")
		print(f"  Samples > 5% relative delta: "
			  f"{convergence[task]['relative_delta']['pct_above_5pct']:.1f}%")
		print(f"  Samples > 1% relative delta: "
			  f"{convergence[task]['relative_delta']['pct_above_1pct']:.1f}%")
		if convergence[task]["relative_delta"]["mean"] > 0.05:
			print(f"  WARNING: Mean relative delta > 5% for '{task}'. "
				  "Consider increasing n_steps.")
		else:
			print("  Convergence looks good.")

	convergence["n_samples"] = n_samples
	convergence["n_steps"] = n_steps
	convergence["method"] = ig_method

	# ------------------------------------------------------------------
	# Save results
	# ------------------------------------------------------------------
	npz_path = os.path.join(run_dir, "attributions.npz")
	np.savez_compressed(npz_path, **save_arrays)
	print(f"\nAttributions saved to {npz_path}")

	convergence_path = os.path.join(run_dir, "convergence.json")
	with open(convergence_path, "w") as f:
		json.dump(convergence, f, indent=4)
	print(f"Convergence report saved to {convergence_path}")

	# ------------------------------------------------------------------
	# Save run metadata
	# ------------------------------------------------------------------
	meta = {
		"config": config,
		"model_dir": config["model_dir"],
		"task_names": task_names,
		"transforms": {t: transforms[t] for t in task_names},
		"targets": {t: targets[t] for t in task_names},
		"sequence_layout": {
			"n_prefix_prompt": n_prefix_prompt,
			"n_flanking_bp": n_flanking_bp,
			"n_str_bp": n_str_bp,
			"n_str_prompt": n_str_prompt,
			"n_suffix_prompt": n_suffix_prompt,
			"seq_len": seq_len,
			"order": [
				"prefix_prompt", "left_flank", "left_str", "str_prompt",
				"right_str", "right_flank", "suffix_prompt",
			],
		},
		"ig_config": {
			"n_steps": n_steps,
			"internal_batch_size": internal_batch_size,
			"method": ig_method,
			"baseline": "mean_ACGT_embedding (flanking only)",
			"subsample_loci": subsample_loci,
			"seed": seed,
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
