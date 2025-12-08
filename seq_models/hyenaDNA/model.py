""" HyenaDNA based models for STR length prediction.

- Loads HyenaDNA model from Hugging Face.
- Adds a regression head for STR length prediction.
- Modifies embedding matrix to add soft prompt tokens.
- Implements soft prompting and full fine-tuning strategies.

create_model() can be used to create the model from a config dict.
"""

import datetime

from networkx import config
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
	AutoModel,
	AutoConfig,
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
		n_prefix_prompt_tokens (int): Number of prompt tokens before the
			sequence.
		n_str_prompt_tokens (int): Number of prompt tokens in the STR gap.
		n_flanking_bp (int): Number of flanking base pairs to include on
			each side of the STR. Used to verify model can handle input length.
		n_str_bp (int): Number of base pairs of the STR to include on
			each side of the STR. Used to verify model can handle input length.
	
	Optional config keys:
		tuning_strategy (str): 'soft_prompt' or 'full_finetune'.
			Default: 'soft_prompt'.
		log_transform (bool): Whether to log-transform the target lengths.
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
		use_gradient_checkpointing (bool): Whether to use gradient checkpointing.
			Default: False.
	"""

	has_legacy = "n_prompt_tokens" in config
	has_new = "n_prefix_prompt_tokens" in config or "n_str_prompt_tokens" in config

	n_prefix = 0
	n_str = 0

	if has_new:
		n_prefix = config.get("n_prefix_prompt_tokens", 0)
		n_str = config.get("n_str_prompt_tokens", 0)
	elif has_legacy:
		# Legacy behavior: all tokens go into the STR gap
		n_str = config["n_prompt_tokens"]
		n_prefix = 0
		
	# Verify that the model can handle the input length
	total_input_length = (
		config["n_flanking_bp"] * 2
        + config["n_str_bp"] * 2
        + n_prefix
        + n_str
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
		n_prefix_prompt_tokens=n_prefix,
        n_str_prompt_tokens=n_str,
		log_transform=config.get("log_transform", False),

		head_hidden_layers=config.get("head_hidden_layers", None), # e.g. [128, 64]
		head_dropout=float(config.get("head_dropout", 0.1)),
		use_attention_pooling=config.get("use_attention_pooling", False),
		attention_pooling_num_heads=config.get("attention_pooling_num_heads", 4),
		
		tuning_strategy=config.get("tuning_strategy", "soft_prompt"),
		
		optimizer_name=config.get("optimizer_name", "AdamW"),
		lr=float(config.get("lr", 1e-4)),
		backbone_lr=float(config.get("backbone_lr", 1e-5)),
		weight_decay=float(config.get("weight_decay", 0.01)),

		scheduler_name=config.get("scheduler_name", "ReduceLROnPlateau"),
		warmup_steps=config.get("warmup_steps", 100),
		scheduler_patience=config.get("scheduler_patience", 3),
		scheduler_factor=config.get("scheduler_factor", 0.1),
		use_gradient_checkpointing=config.get("gradient_checkpointing", False),
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
		n_prefix_prompt_tokens: int,
        n_str_prompt_tokens: int,
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
		scheduler_factor: float,

		use_gradient_checkpointing: bool = False,
		log_transform: bool = False,

		head_hidden_layers = None,
		head_dropout: float = 0.1,
		use_attention_pooling: bool = False,
		attention_pooling_num_heads: int = 4,
	):
		""" Initialize STRLengthModel.

		Sets gradient checkpointing to True for memory efficiency.

		Args:
			model_checkpoint (str): HF checkpoint name.
			n_prefix_prompt_tokens (int): Number of soft prompt tokens at
				the start.
			n_str_prompt_tokens (int): Number of soft prompt tokens in the
				STR gap.
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

		# Load base heyenaDNA model
		config = AutoConfig.from_pretrained(
			hyenaDNA_checkpoint,
			trust_remote_code=True
		)
		self.hidden_size = config.d_model
		
		self.hyena_model = AutoModel.from_pretrained(
			hyenaDNA_checkpoint,
			config=config,
			trust_remote_code=True
		)

		# Enable gradient checkpointing for memory efficiency
		if use_gradient_checkpointing:
			self.hyena_model.gradient_checkpointing_enable()

		# Resize Embeddings manually to add soft prompt tokens
		old_embed = self.hyena_model.backbone.embeddings.word_embeddings
		
		# Sanity Check
		assert old_embed.embedding_dim == self.hidden_size, \
			f"Dim mismatch: {old_embed.embedding_dim} vs {self.hidden_size}"

		n_total_prompt_tokens = n_prefix_prompt_tokens + n_str_prompt_tokens
		new_num_tokens = PROMPT_START_ID + n_total_prompt_tokens
	
		new_embed = nn.Embedding(
			new_num_tokens, self.hidden_size, padding_idx=old_embed.padding_idx
		)
		
		# Copy weights
		with torch.no_grad():
			new_embed.weight[:old_embed.num_embeddings] = old_embed.weight
		
		# Overwrite in hierarchy
		self.hyena_model.backbone.embeddings.word_embeddings = new_embed
		self.hyena_model.config.vocab_size = new_num_tokens

		# Verify resize
		assert self.hyena_model.backbone.embeddings.word_embeddings.weight.shape == (new_num_tokens, self.hidden_size)

		# Set up attention pooling if specified
		self.attn_pooling_layer = None
		self.pooling_query = None
		
		if use_attention_pooling:
			# Learnable Query Vector (1, 1, hidden_dim)
			self.pooling_query = nn.Parameter(
				torch.randn(1, 1, self.hidden_size)
			)
			self.attn_pooling_layer = nn.MultiheadAttention(
				embed_dim=self.hidden_size,
				num_heads=attention_pooling_num_heads,
				batch_first=True
			)

		# Set up regression head
		layers = []
		input_dim = self.hidden_size
		
		if head_hidden_layers:
			for hidden_dim in head_hidden_layers:
				layers.append(nn.Linear(input_dim, hidden_dim))
				layers.append(nn.ReLU())
				layers.append(nn.Dropout(head_dropout))
				input_dim = hidden_dim
		
		# Final projection to scalar
		layers.append(nn.Linear(input_dim, 1))
		self.head = nn.Sequential(*layers)

		# Set gradient tracking based on tuning strategy
		if self.hparams.tuning_strategy == "soft_prompt":
			
			# Freeze all parameters in the backbone
			for param in self.hyena_model.parameters():
				param.requires_grad = False
			
			# Unfreeze the entire embedding parameter
			embed_param = self.hyena_model.backbone.embeddings.word_embeddings.weight
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
			'mse': MeanSquaredError(),
			'rmse': MeanSquaredError(squared=False),
			'pearson': PearsonCorrCoef(),
			'spearman': SpearmanCorrCoef(),
			'r2': R2Score()
		})

		self.train_metrics = metrics.clone(prefix='train_')
		self.val_metrics = metrics.clone(prefix='val_')
		self.test_metrics = metrics.clone(prefix='test_')

	def forward(self, input_ids):
		# hyena_model Forward: last_hidden_state (Batch, SeqLen, Hidden)
		outputs = self.hyena_model(input_ids=input_ids)
		sequence_output = outputs.last_hidden_state

		# Pooling
		if self.hparams.use_attention_pooling:
			batch_size = input_ids.shape[0]
			# Expand query to batch size: (Batch, 1, Hidden)
			query = self.pooling_query.expand(batch_size, -1, -1)
			# Attend to sequence. Output: (Batch, 1, Hidden)
			attn_output, _ = self.attn_pooling_layer(
				query, sequence_output, sequence_output
			)
			pooled_output = attn_output.squeeze(1)
		else:
			# Mean Pooling: (Batch, Hidden)
			pooled_output = torch.mean(sequence_output, dim=1)

		# Output Head
		logits = self.head(pooled_output)
		return logits.squeeze(1)
	
	def train(self, mode: bool = True):
		"""
		Override the LightningModule.train() method to
		keep the backbone in eval mode during soft-prompting.
		"""
		super().train(mode)
		
		# If we are soft-prompting, force the backbone
		# back into eval() mode to disable dropouts.
		if self.hparams.tuning_strategy == "soft_prompt":
			self.hyena_model.eval()

	def _common_step(self, batch):
		input_ids = batch["input_ids"]
		labels = batch["label"]

		logits = self(input_ids)

		if self.hparams.log_transform:
			optimization_targets = torch.log1p(labels)
			loss = self.loss_fn(logits, optimization_targets)
			preds = torch.expm1(logits).detach()
		else:
			loss = self.loss_fn(logits, labels)
			preds = logits.detach()
			
		return loss, preds, labels
	
	def training_step(self, batch, batch_idx):
		loss, preds, labels = self._common_step(batch)
		
		# Log training loss (for overfitting check)
		self.log(
			"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
			batch_size=batch["input_ids"].shape[0]
		)
		
		# Update and log training metrics (at epoch end)
		self.train_metrics.update(preds, labels)
		self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
		
		return loss

	def validation_step(self, batch, batch_idx):
		loss, preds, labels = self._common_step(batch)
		
		# Log validation loss
		self.log(
			"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
			batch_size=batch["input_ids"].shape[0]
		)
		
		# Update and log validation metrics
		self.val_metrics.update(preds, labels)
		self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

	def test_step(self, batch, batch_idx):
		loss, preds, labels = self._common_step(batch)
		
		# Log test loss
		self.log(
			"test_loss", loss, on_step=False, on_epoch=True,
			batch_size=batch["input_ids"].shape[0]
		)
		
		# Update and log test metrics
		self.test_metrics.update(preds, labels)
		self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		"""Returns predictions in real units (base pairs)."""
		input_ids = batch["input_ids"]
		outputs = self(input_ids)
		logits = outputs.logits.squeeze(1)
		
		if self.hparams.log_transform:
			return torch.expm1(logits)
		else:
			return logits

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
			high_lr_params = list(self.head.parameters())
			high_lr_params += list(self.hyena_model.backbone.embeddings.word_embeddings.parameters())
			
			if self.hparams.use_attention_pooling:
				high_lr_params.append(self.pooling_query)
				high_lr_params += list(self.attn_pooling_layer.parameters())

			high_lr_ids = {id(p) for p in high_lr_params}
			backbone_params = []
			for param in self.hyena_model.parameters():
				if param.requires_grad and id(param) not in high_lr_ids:
					backbone_params.append(param)
			
			optimizer = opt_class(
				[
					{"params": high_lr_params, "lr": self.hparams.lr},
					{"params": backbone_params, "lr": self.hparams.backbone_lr}
				], 
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
