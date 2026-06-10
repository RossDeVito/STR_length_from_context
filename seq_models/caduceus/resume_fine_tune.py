""" Resume Caduceus fine-tuning from a previous experiment directory.

1. Locates the latest checkpoint and config in --resume_dir.
2. Creates a NEW experiment directory (logs start fresh there).
3. Resumes training from the checkpoint.

Selection/early-stop monitor `val_fixed_loss` (matching fine_tune.py). The
`monitor_norm` normalizers were written into config.yaml by fine_tune.py, so they
are reused unchanged on resume.

Usage:
	python -m seq_models.caduceus.resume_fine_tune \
		--resume_dir scripts/training/output/caduceus/<run> \
		--output_dir scripts/training/output/caduceus
"""

import argparse
import glob
import os
import shutil

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
from seq_models.caduceus.model import create_model


def find_resume_checkpoint(resume_dir):
	"""Find the checkpoint to resume from (prefer last.ckpt)."""
	ckpt_dir = os.path.join(resume_dir, "checkpoints")
	if not os.path.exists(ckpt_dir):
		raise FileNotFoundError(f"No 'checkpoints' folder found in {resume_dir}")

	last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
	if os.path.exists(last_ckpt):
		return last_ckpt

	ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
	if not ckpts:
		raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
	latest_ckpt = max(ckpts, key=os.path.getmtime)
	print(f"   > 'last.ckpt' not found. Falling back to latest: {os.path.basename(latest_ckpt)}")
	return latest_ckpt


def parse_args():
	parser = argparse.ArgumentParser(description="Resume Caduceus fine-tuning.")
	parser.add_argument("--resume_dir", type=str, required=True, help="Path to interrupted run.")
	parser.add_argument("--output_dir", type=str, required=True, help="Parent dir for new output.")
	parser.add_argument("--cpu", action="store_true", help="Force CPU.")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	if not os.path.exists(args.resume_dir):
		raise FileNotFoundError(f"Resume directory not found: {args.resume_dir}")

	config_path = os.path.join(args.resume_dir, "config.yaml")
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"config.yaml not found in {args.resume_dir}")

	ckpt_path = find_resume_checkpoint(args.resume_dir)
	print(f"Resuming from: {ckpt_path}")

	try:
		checkpoint_header = torch.load(ckpt_path, map_location="cpu")
		resume_epoch = checkpoint_header.get("epoch", 0)
		print(f"Checkpoint detected: Epoch {resume_epoch}")
	except Exception as e:
		raise RuntimeError(f"Failed to read checkpoint: {e}")

	# New experiment directory
	old_exp_name = os.path.basename(os.path.normpath(args.resume_dir))
	new_exp_name = f"{old_exp_name}_resumed_epoch{resume_epoch}"
	new_experiment_path = os.path.join(args.output_dir, new_exp_name)
	os.makedirs(new_experiment_path, exist_ok=True)
	print(f"New experiment directory: {new_experiment_path}")

	new_config_path = os.path.join(new_experiment_path, "config.yaml")
	shutil.copy(config_path, new_config_path)
	with open(new_config_path, "r") as f:
		config = yaml.safe_load(f)

	logger = TensorBoardLogger(save_dir=args.output_dir, name=new_exp_name, version=".")

	print("Initializing Data/Model...")
	datamodule = create_data_module(config)
	model = create_model(config)

	print("Setting up callbacks...")
	new_checkpoint_dir = os.path.join(new_experiment_path, "checkpoints")
	checkpoint_callback = ModelCheckpoint(
		dirpath=new_checkpoint_dir,
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
		accelerator, devices = "cpu", 1
	else:
		accelerator, devices = "auto", 1

	requested_precision = config.get("precision", "32-true")
	if "bf16" in requested_precision and not args.cpu:
		if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
			print("Downgrading precision to 16-mixed (Ampere+ required for BF16)")
			config["precision"] = "16-mixed"
		elif torch.backends.mps.is_available():
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

	print("--- Resuming training... ---")
	trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
	print("--- Training complete ---")

	with open(os.path.join(new_experiment_path, "TRAINING_COMPLETED.txt"), "w") as f:
		f.write("Training completed successfully.\n")
