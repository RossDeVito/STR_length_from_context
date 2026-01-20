""" Resume training from a previous experiment directory.

1. Locates the latest checkpoint and config in --resume_dir.
2. Extracts the global_step from the checkpoint.
3. Creates a new experiment directory: {old_name}_resumed_step{step}.
4. Cleans old logs to ensure correct wall time.
5. Resumes training.

Usage:
	python scripts/training/resume_training.py \
		--resume_dir scripts/training/output/soft_prompt_run_1 \
		--output_dir scripts/training/output
"""

import argparse
import os
import shutil
import glob
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
	ModelCheckpoint,
	EarlyStopping,
	LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorboard.compat.proto import event_pb2

from seq_models.hyenaDNA.model import create_model
from seq_models.hyenaDNA.data import create_data_module


def find_resume_checkpoint(resume_dir):
	"""Finds the best checkpoint to resume from in the resume_dir."""
	ckpt_dir = os.path.join(resume_dir, "checkpoints")
	if not os.path.exists(ckpt_dir):
		raise FileNotFoundError(f"No 'checkpoints' folder found in {resume_dir}")

	# Priority 1: 'last.ckpt'
	last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
	if os.path.exists(last_ckpt):
		return last_ckpt

	# Priority 2: Latest 'best-model'
	ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
	if not ckpts:
		raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
	
	latest_ckpt = max(ckpts, key=os.path.getmtime)
	return latest_ckpt


def clean_and_copy_logs(old_dir, new_dir, resume_step):
	"""
	Reads event logs from old_dir, filters out events >= resume_step,
	and writes the clean history to new_dir.
	"""
	print(f"   > Attempting to unify logs (Cutoff step: {resume_step})...")
	
	files = glob.glob(os.path.join(old_dir, "events.out.tfevents*"))
	if not files:
		print("   > Warning: No old event file found. Skipping log unification.")
		return
	
	# Take the latest file
	files.sort(key=os.path.getmtime)
	old_file_path = files[-1]

	# Setup Writer in new dir
	# filename_suffix ensures the file is distinct from the new training logs
	# TensorBoard will read this file first (due to timestamp) then the new one.
	writer = tf.summary.create_file_writer(new_dir, filename_suffix=".history_cleaned")

	total_events = 0
	kept_events = 0

	with writer.as_default():
		try:
			# Iterate raw events using TF's iterator
			for e in tf.compat.v1.train.summary_iterator(old_file_path):
				total_events += 1
				
				# Keep metadata (step 0) and any steps strictly before resume
				if e.step < resume_step:
					if e.HasField('summary'):
						for value in e.summary.value:
							if value.HasField('simple_value'):
								tf.summary.scalar(value.tag, value.simple_value, step=e.step)
					kept_events += 1
		except Exception as e:
			raise RuntimeError(f"Failed to parse old logs: {e}")
	
	writer.close()
	print(f"   > Log unification complete.")
	print(f"   > Processed {total_events} events. Kept {kept_events} (Dropped {total_events - kept_events} orphaned steps).")


def parse_args():
	parser = argparse.ArgumentParser(description="Resume training from previous run.")
	parser.add_argument(
		"--resume_dir", 
		type=str, 
		required=True, 
		help="Directory of the interrupted run (containing config.yaml and checkpoints/)."
	)
	parser.add_argument(
		"--output_dir", 
		type=str, 
		required=True, 
		help="Parent directory where the new '_resumed' folder will be created."
	)
	parser.add_argument(
		"--cpu", 
		action="store_true", 
		help="Force use of CPU."
	)
	return parser.parse_args()


if __name__ == "__main__":

	args = parse_args()

	# Validate input and find paths
	if not os.path.exists(args.resume_dir):
		raise FileNotFoundError(f"Resume directory does not exist: {args.resume_dir}")

	config_path = os.path.join(args.resume_dir, "config.yaml")
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"config.yaml not found in {args.resume_dir}")

	ckpt_path = find_resume_checkpoint(args.resume_dir)
	print(f"Resuming from checkpoint: {ckpt_path}")

	# Get resume step from checkpoint
	try:
		# Map location CPU to avoid loading model to GPU just to read header
		checkpoint_header = torch.load(ckpt_path, map_location="cpu")
		resume_step = checkpoint_header.get('global_step', 0)
		resume_epoch = checkpoint_header.get('epoch', 0)
		print(f"Checkpoint detected: Epoch {resume_epoch} | Global Step {resume_step}")
	except Exception as e:
		raise RuntimeError(f"Failed to read checkpoint header: {e}")

	# Create new experiment directory
	old_exp_name = os.path.basename(os.path.normpath(args.resume_dir))
	new_exp_name = f"{old_exp_name}_resumed_epoch{resume_epoch}"
	
	new_experiment_path = os.path.join(args.output_dir, new_exp_name)
	os.makedirs(new_experiment_path, exist_ok=True)
	
	print(f"New experiment directory: {new_experiment_path}")

	# Copy config
	new_config_path = os.path.join(new_experiment_path, "config.yaml")
	shutil.copy(config_path, new_config_path)
	
	with open(new_config_path, 'r') as f:
		config = yaml.safe_load(f)

	# Log unification
	try:
		clean_and_copy_logs(args.resume_dir, new_experiment_path, resume_step)
	except Exception as e:
		print(f"\nError: Failed to unify logs")
		print(f"Error details: {e}")
		# Clean up the empty directory we just created so we don't leave junk
		shutil.rmtree(new_experiment_path)
		raise e

	# Initialize components
	logger = TensorBoardLogger(
		save_dir=args.output_dir, 
		name=new_exp_name,
		version="." 
	)

	print("Initializing DataModule...")
	datamodule = create_data_module(config)

	print("Initializing Model...")
	model = create_model(config)

	# Setup Callbacks ---
	print("Setting up callbacks...")
	new_checkpoint_dir = os.path.join(new_experiment_path, "checkpoints")
	
	checkpoint_callback = ModelCheckpoint(
		dirpath=new_checkpoint_dir,
		filename="best-model-{epoch:02d}",
		monitor="val_loss",
		mode="min",
		save_top_k=1,
		save_last=True,
		verbose=True
	)

	early_stop_callback = EarlyStopping(
		monitor="val_loss",
		patience=config.get("patience_early_stopping", 10),
		verbose=True,
		mode="min"
	)
	
	lr_monitor = LearningRateMonitor(logging_interval="epoch")
	callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

	# --- 7. Hardware Setup ---
	if args.cpu:
		accelerator = "cpu"
		devices = 1
	else:
		accelerator = "auto"
		devices = 1

	requested_precision = config.get("precision", "32-true")
	if "bf16" in requested_precision and not args.cpu:
		if torch.cuda.is_available():
			major, minor = torch.cuda.get_device_capability()
			if major < 8:
				print(f"Hardware Cap {major}.{minor}: Downgrading to 16-mixed")
				config["precision"] = "16-mixed"
		elif torch.backends.mps.is_available():
			config["precision"] = "16-mixed"

	# --- 8. Resume Training ---
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

	print(f"--- Resuming training... ---")
	trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
	print("--- Training complete ---")