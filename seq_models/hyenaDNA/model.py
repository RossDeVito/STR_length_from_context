""" HyenaDNA based models for STR length prediction.

- Loads HyenaDNA model from Hugging Face.
- Adds a regression head for STR length prediction.
- Modifies embedding matrix to add soft prompt tokens.
- Implements soft prompting and full fine-tuning strategies.

create_model() can be used to create the model from a config dict.
"""

import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
	AutoModelForSequenceClassification,
	get_linear_schedule_with_warmup,
	get_cosine_schedule_with_warmup
)
from torchmetrics import MeanSquaredError, MetricCollection
from torchmetrics.regression import PearsonCorrCoef, R2Score, SpearmanCorrCoef

from seq_models.hyenaDNA.hyenaDNA_info import PROMPT_START_ID, MAX_SEQ_LENGTH


def create_model(config):
	"""
	Factory function to create the STRLengthModel from a config dict.

	Args:
		config (dict): Configuration dictionary.

	Required config keys:
		hyenaDNA_checkpoint (str): HF checkpoint for the HyenaDNA model.
		n_prompt_tokens (int): The number of soft prompt tokens to add.
		n_flanking_bp (int): Number of flanking base pairs to include on
			each side of the STR. Used to verify model can handle input length.
		n_str_bp (int): Number of base pairs of the STR to include on
			each side of the STR. Used to verify model can handle input length.
	
	Optional config keys:
		tuning_strategy (str): 'soft_prompt' or 'full_finetune'.
			Default: 'soft_prompt'.
		
		optimizer_name (str): 'Adam' or 'AdamW'. Default: 'AdamW'.
		lr (float): Learning rate for new components (head, prompt). Default: 1e-4.
		backbone_lr (float): Learning rate for the pretrained backbone. Default: 1e-5.
		weight_decay (float): Weight decay for the optimizer. Default: 0.01.

		scheduler_name (str): 'ReduceLROnPlateau', 'linear_with_warmup', 
							  'cosine_with_warmup', or 'None'. 
							  Default: 'ReduceLROnPlateau'.
		warmup_steps (int): Steps for linear/cosine warmup. Default: 100.
		scheduler_patience (int): Patience for ReduceLROnPlateau. Default: 3.
		scheduler_factor (float): Factor for ReduceLROnPlateau. Default: 0.1.
	"""
	# Verify that the model can handle the input length
	total_input_length = (
		config["n_flanking_bp"] * 2
		+ config["n_str_bp"] * 2
		+ config["n_prompt_tokens"]
	)
	
	if config["hyenaDNA_checkpoint"] not in MAX_SEQ_LENGTH.keys():
		raise ValueError(
			f"Unknown hyenaDNA_checkpoint: {config['hyenaDNA_checkpoint']}"
		)
	
	if total_input_length > MAX_SEQ_LENGTH[config["hyenaDNA_checkpoint"]]:
		raise ValueError(
			f"Input length {total_input_length} exceeds max for {config['hyenaDNA_checkpoint']}: {MAX_SEQ_LENGTH[config['hyenaDNA_checkpoint']]}"
		)

	return STRLengthModel(
		hyenaDNA_checkpoint=config["hyenaDNA_checkpoint"],
		n_prompt_tokens=config["n_prompt_tokens"],
		
		tuning_strategy=config.get("tuning_strategy", "soft_prompt"),
		
		optimizer_name=config.get("optimizer_name", "AdamW"),
		lr=float(config.get("lr", 1e-4)),
		backbone_lr=float(config.get("backbone_lr", 1e-5)),
		weight_decay=float(config.get("weight_decay", 0.01)),

		scheduler_name=config.get("scheduler_name", "ReduceLROnPlateau"),
		warmup_steps=config.get("warmup_steps", 100),
		scheduler_patience=config.get("scheduler_patience", 3),
		scheduler_factor=config.get("scheduler_factor", 0.1)
	)


class STRLengthModel(pl.LightningModule):
	""" 
	LightningModule for STR length regression with soft prompting,
	built on heyenaDNA AutoModelForSequenceClassification.
	"""
	
	def __init__(
		self,
		# Model
		hyenaDNA_checkpoint: str,
		n_prompt_tokens: int,
		tuning_strategy: str,
		
		# Optimizer
		optimizer_name: str,
		lr: float,
		backbone_lr: float,
		weight_decay: float,
		
		# Scheduler
		scheduler_name: str,
		warmup_steps: int,
		scheduler_patience: int,
		scheduler_factor: float
	):
		""" Initialize STRLengthModel.

		Sets gradient checkpointing to True for memory efficiency.

		Args:
			model_checkpoint (str): HF checkpoint name.
			n_prompt_tokens (int): Number of soft prompt tokens.
			tuning_strategy (str): 'soft_prompt' or 'full_finetune'.
			
			optimizer_name (str): 'Adam', 'AdamW', etc.
			lr (float): Main learning rate (for head, prompt).
			backbone_lr (float): Learning rate for backbone (in full_finetune).
			weight_decay (float): Weight decay for the optimizer.
			
			scheduler_name (str): 'ReduceLROnPlateau', 'linear_with_warmup',
								'cosine_with_warmup', or 'None'.
			warmup_steps (int): Patience for linear/cosine warmup.
			scheduler_patience (int): Patience for ReduceLROnPlateau.
			scheduler_factor (float): Factor for ReduceLROnPlateau.
		"""

		super().__init__()

		# Save hyperparameters
		self.save_hyperparameters()

		# Load heyenaDNA model for regression
		self.model = AutoModelForSequenceClassification.from_pretrained(
			hyenaDNA_checkpoint,
			trust_remote_code=True,
			num_labels=1  # Create new regression head
		)

		# Enable gradient checkpointing for memory efficiency
		self.model.gradient_checkpointing_enable()

		# Add soft prompt tokens to embedding matrix
		self.model.resize_token_embeddings(
			PROMPT_START_ID + n_prompt_tokens
		)

		# Set gradient tracking based on tuning strategy
		if self.hparams.tuning_strategy == "soft_prompt":

			embed_param = self.model.get_input_embeddings().weight
			
			# Freeze all parameters in the backbone
			for param in self.model.hyena.parameters():
				param.requires_grad = False
			
			# Unfreeze the entire embedding parameter
			embed_param.requires_grad = True
			
			# And register a hook that will zero out the gradients
			# for the part of the tensor we want to keep frozen.
			def zero_grad_hook(grad):
				# This line zeros the gradients for tokens 0 thru 15
				grad[:PROMPT_START_ID].fill_(0)
				return grad
			
			embed_param.register_hook(zero_grad_hook)
		
		elif self.hparams.tuning_strategy == "full_finetune":
			pass # All params are trainable
		
		else:
			raise ValueError(
				f"Unknown tuning_strategy: {self.hparams.tuning_strategy}"
			)
		
		# Define loss and metrics
		self.loss_fn = nn.MSELoss()
		
		metrics = MetricCollection({
			'rmse': MeanSquaredError(squared=False),
			'pearson': PearsonCorrCoef(),
			'spearman': SpearmanCorrCoef(),
			'r2': R2Score()
		})

		self.train_metrics = metrics.clone(prefix='train_')
		self.val_metrics = metrics.clone(prefix='val_')
		self.test_metrics = metrics.clone(prefix='test_')

	def forward(self, input_ids):
		return self.model(input_ids=input_ids)
	
	def train(self, mode: bool = True):
		"""
		Override the LightningModule.train() method to
		keep the backbone in eval mode during soft-prompting.
		"""
		# Set the train mode for the whole module
		# (this will put self.model.score in train mode)
		super().train(mode)
		
		# If we are soft-prompting, force the backbone
		# back into eval() mode to disable dropouts.
		if self.hparams.tuning_strategy == "soft_prompt":
			self.model.hyena.eval()

	def _common_step(self, batch):
		input_ids = batch["input_ids"]
		labels = batch["label"] 
		outputs = self(input_ids) 
		logits = outputs.logits.squeeze(1)
		loss = self.loss_fn(logits, labels)
		return loss, logits, labels
	
	def training_step(self, batch, batch_idx):
		loss, logits, labels = self._common_step(batch)
		
		# Log training loss (for overfitting check)
		self.log(
			"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
			batch_size=batch["input_ids"].shape[0]
		)
		
		# Update and log training metrics (at epoch end)
		self.train_metrics.update(logits, labels)
		self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
		
		return loss

	def validation_step(self, batch, batch_idx):
		loss, logits, labels = self._common_step(batch)
		
		# Log validation loss
		self.log(
			"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
			batch_size=batch["input_ids"].shape[0]
		)
		
		# Update and log validation metrics
		self.val_metrics.update(logits, labels)
		self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

	def test_step(self, batch, batch_idx):
		"""
		Runs at the end of training, on the test set.
		(Called with trainer.test())
		"""
		loss, logits, labels = self._common_step(batch)
		
		# Log test loss
		self.log(
			"test_loss", loss, on_step=False, on_epoch=True,
			batch_size=batch["input_ids"].shape[0]
		)
		
		# Update and log test metrics
		self.test_metrics.update(logits, labels)
		self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

	def configure_optimizers(self):
		
		try:
			opt_class = getattr(torch.optim, self.hparams.optimizer_name)
		except AttributeError:
			raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_name}")

		# Create parameter groups
		if self.hparams.tuning_strategy == "soft_prompt":
			optimizer = opt_class(
				self.parameters(), 
				lr=self.hparams.lr, 
				weight_decay=self.hparams.weight_decay
			)

		elif self.hparams.tuning_strategy == "full_finetune":
			# Group 1 (High LR): The regression head
			group1 = {
				"params": self.model.score.parameters(),
				"lr": self.hparams.lr
			}
			# Group 2 (Low LR): The entire backbone, including the
			# pretrained prompt tokens (using soft_prompt strategy first)
			group2 = {
				"params": self.model.hyena.parameters(), 
				"lr": self.hparams.backbone_lr
			}
			optimizer = opt_class(
				[group1, group2], 
				weight_decay=self.hparams.weight_decay
			)

		# Scheduler setup
		if self.hparams.scheduler_name == "ReduceLROnPlateau":
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer,
				mode="min",
				factor=self.hparams.scheduler_factor,
				patience=self.hparams.scheduler_patience,
			)
			return {
				"optimizer": optimizer,
				"lr_scheduler": {
					"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"
				}
			}
		
		total_steps = 0
		if self.hparams.scheduler_name in ["linear_with_warmup", "cosine_with_warmup"]:
			try:
				total_steps = self.trainer.estimated_stepping_batches
			except AttributeError:
				print("Warning: Could not get estimated_stepping_batches. Using fallback.")
				try:
					total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
				except Exception:
					print("Warning: Could not compute total_steps from dataloader. Using default of 10000.")
					total_steps = 10000
		
		if self.hparams.scheduler_name == "cosine_with_warmup":
			scheduler = get_cosine_schedule_with_warmup(
				optimizer,
				num_warmup_steps=self.hparams.warmup_steps,
				num_training_steps=total_steps
			)
			return {
				"optimizer": optimizer,
				"lr_scheduler": {
					"scheduler": scheduler, "interval": "step", "frequency": 1
				}
			}
			
		elif self.hparams.scheduler_name == "linear_with_warmup":
			scheduler = get_linear_schedule_with_warmup(
				optimizer,
				num_warmup_steps=self.hparams.warmup_steps,
				num_training_steps=total_steps
			)
			return {
				"optimizer": optimizer,
				"lr_scheduler": {
					"scheduler": scheduler, "interval": "step", "frequency": 1
				}
			}

		elif self.hparams.scheduler_name == "None":
			return optimizer
		
		else:
			raise ValueError(f"Unknown scheduler: {self.hparams.scheduler_name}")
