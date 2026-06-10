""" Full fine-tuning of Caduceus-Ph for STR length / variation prediction.

Requires a YAML config file specifying data/model/training params.

Args:
	--config: Path to YAML config file.
	--output_dir: Directory to save outputs (models, logs).
	--cpu: Force use of CPU (note: the Caduceus backbone needs CUDA mamba kernels).
	--dev_subset_n: Override config dev_subset_n (cap every split to N rows).

Checkpoint selection and early stopping monitor `val_fixed_loss` (the fixed,
variance-normalized validation loss), NOT the uncertainty-weighted `val_loss`,
which includes the learned log-variance terms and isn't comparable across steps.
The per-task variance normalizers (`monitor_norm`) are computed once from the val
split and written into the saved config so selection is reproducible/resumable.
"""

import argparse
import datetime
import json
import os

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
	EarlyStopping,
	LearningRateMonitor,
	ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from seq_models.caduceus.data import create_data_module
from seq_models.caduceus.model import (
	DEFAULT_TARGETS,
	compute_monitor_norms,
	create_model,
	resolve_transforms,
)


if __name__ == "__main__":

	__spec__ = None

	parser = argparse.ArgumentParser(description="Fine-tune Caduceus STRLengthModel")
	parser.add_argument(
		"--config", type=str, required=True,
		help="Path to the configuration YAML file."
	)
	parser.add_argument(
		"--output_dir", type=str, default=".",
		help="Directory to save outputs (models, logs)."
	)
	parser.add_argument(
		"--cpu", action="store_true",
		help="Force use of CPU (the Caduceus backbone needs CUDA mamba kernels)."
	)
	parser.add_argument(
		"--dev_subset_n", type=int, default=None,
		help="Override config dev_subset_n (cap every split to N rows)."
	)
	args = parser.parse_args()

	# Load config
	print(f"Loading config from {args.config}")
	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	if args.dev_subset_n is not None:
		config["dev_subset_n"] = args.dev_subset_n

	# Set up output directory
	experiment_name = config.get("experiment_name", "fine_tune")
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	experiment_path = os.path.join(
		args.output_dir, f"{experiment_name}_{timestamp}"
	)
	os.makedirs(experiment_path, exist_ok=True)

	# Create DataModule
	print("Initializing DataModule...")
	datamodule = create_data_module(config)
	datamodule.setup()

	# Precompute and freeze the fixed-loss monitor normalizers (per-task variance
	# of the transformed target on the val split), unless already in config. This
	# keeps checkpoint selection comparable across steps and reproducible on resume.
	if "monitor_norm" not in config or config["monitor_norm"] is None:
		targets = config.get("targets", DEFAULT_TARGETS)
		transforms = resolve_transforms(
			list(targets.keys()), config.get("target_transforms")
		)
		config["monitor_norm"] = compute_monitor_norms(
			datamodule.val_dataset.str_df, targets, transforms
		)
		print(f"Computed monitor_norm from val split: {config['monitor_norm']}")

	# Save the (now monitor_norm-augmented) config used.
	config_save_path = os.path.join(experiment_path, "config.yaml")
	with open(config_save_path, "w") as f:
		yaml.dump(config, f)

	logger = TensorBoardLogger(save_dir=experiment_path, name="", version=".")
	print(f"Outputs will be saved to: {experiment_path}")

	# Create model
	print("Initializing Model...")
	model = create_model(config)

	# Callbacks. Selection/early-stop monitor the fixed-weight val loss.
	print("Setting up callbacks...")
	checkpoint_dir = os.path.join(experiment_path, "checkpoints")
	checkpoint_callback = ModelCheckpoint(
		dirpath=checkpoint_dir,
		filename="best-model-{epoch:02d}",
		monitor="val_fixed_loss",
		mode="min",
		save_top_k=1,
		save_last=True,
		verbose=True,
	)
	early_stop_callback = EarlyStopping(
		monitor="val_fixed_loss",
		patience=config.get("patience_early_stopping", 10),
		verbose=True,
		mode="min",
	)
	lr_monitor = LearningRateMonitor(logging_interval="epoch")
	callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

	# Hardware / precision
	if args.cpu:
		print("--- CPU override flag set. Using CPU. ---")
		accelerator, devices = "cpu", 1
	else:
		print("--- Using 'auto' accelerator (GPU/MPS if available) ---")
		accelerator, devices = "auto", 1

	requested_precision = config.get("precision", "32-true")
	if "bf16" in requested_precision and not args.cpu:
		if torch.cuda.is_available():
			# `including_emulation` only exists on newer torch; keep it simple/portable.
			if torch.cuda.is_bf16_supported():
				print(f"Hardware supports BF16. Using: {requested_precision}")
			else:
				print("Hardware does NOT support BF16; downgrading to '16-mixed'.")
				config["precision"] = "16-mixed"
		elif torch.backends.mps.is_available():
			print("MPS detected; downgrading precision to '16-mixed'.")
			config["precision"] = "16-mixed"

	trainer = pl.Trainer(
		max_epochs=config.get("max_epochs", 100),
		accelerator=accelerator,
		devices=devices,
		logger=logger,
		callbacks=callbacks,
		gradient_clip_val=config.get("gradient_clip_val", 0.0),
		precision=config.get("precision", "32-true"),
		accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
	)

	# Train
	print("--- Starting training ---")
	trainer.fit(model, datamodule=datamodule)
	print("--- Training complete ---")

	# Evaluate on test set with the best checkpoint
	print("--- Running evaluation on test set ---")
	test_results = trainer.test(datamodule=datamodule, ckpt_path="best")
	print("--- Evaluation complete ---")

	if test_results:
		test_metrics = test_results[0]
		test_metrics["best_model_path"] = checkpoint_callback.best_model_path
		json_path = os.path.join(experiment_path, "test_results.json")
		print(f"Saving test results to {json_path}")
		with open(json_path, "w") as f:
			json.dump(test_metrics, f, indent=4)

	print("\nFind your logs in TensorBoard:")
	print(f"tensorboard --logdir={os.path.abspath(experiment_path)}")
	print(f"\nBest model saved to: {checkpoint_callback.best_model_path}")
