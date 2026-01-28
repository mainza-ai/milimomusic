"""
HeartMuLa LoRA Fine-Tuning Trainer

Implements LoRA-based training for HeartMuLa music generation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import json
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    rank: int = 32  # Increased for more expressive power
    alpha: float = 64.0  # Typically 2x rank
    dropout: float = 0.05
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Target attention projections in backbone transformer
            self.target_modules = [
                'q_proj', 'k_proj', 'v_proj', 'output_proj'
            ]


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    # Increased epochs and learning rate for better convergence on small datasets
    epochs: int = 10
    learning_rate: float = 3e-4  # Higher LR for faster learning
    batch_size: int = 1
    gradient_accumulation_steps: int = 1  # Reduced for small datasets
    warmup_steps: int = 10  # Reduced warmup for small datasets
    max_grad_norm: float = 1.0
    save_steps: int = 500
    log_steps: int = 1  # Log more frequently
    use_gradient_checkpointing: bool = True
    lora: LoRAConfig = None
    
    def __post_init__(self):
        if self.lora is None:
            self.lora = LoRAConfig()


class LoRALinear(nn.Module):
    """LoRA adapter for linear layers."""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Get dtype from original layer
        dtype = original_layer.weight.dtype
        device = original_layer.weight.device
        
        # LoRA weights - match dtype and device of original layer
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
        self.lora_dropout = nn.Dropout(p=dropout)
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original forward
        result = self.original_layer(x)
        
        # LoRA forward - ensure dtype matches input for mixed precision
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        
        lora_out = self.lora_dropout(x) @ lora_A.T @ lora_B.T
        result = result + lora_out * self.scaling
        
        return result
    
    def merge_weights(self):
        """Merge LoRA weights into original layer."""
        with torch.no_grad():
            self.original_layer.weight.add_(
                (self.lora_B @ self.lora_A) * self.scaling
            )
    
    def get_lora_state_dict(self):
        """Get only the LoRA weights."""
        return {
            'lora_A': self.lora_A.data,
            'lora_B': self.lora_B.data,
            'rank': self.rank,
            'alpha': self.alpha
        }


class HeartMuLaLoRATrainer:
    """Trainer for HeartMuLa with LoRA adapters."""
    
    def __init__(
        self,
        model_path: str,
        config: TrainingConfig,
        device: str = "cpu",
        log_callback: Optional[Callable[[str], None]] = None
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.log_callback = log_callback or (lambda x: None)
        
        self.model = None
        self.tokenizer = None
        self.lora_layers: Dict[str, LoRALinear] = {}
        self.optimizer = None
        self.scheduler = None
        
        self.global_step = 0
        self.current_epoch = 0
        
    def log(self, message: str):
        """Log a message."""
        logger.info(message)
        self.log_callback(message)
    
    def _load_model(self):
        """Load HeartMuLa model."""
        self.log(f"Loading HeartMuLa model from {self.model_path}")
        
        from heartlib.heartmula.modeling_heartmula import HeartMuLa
        from heartlib.heartmula.configuration_heartmula import HeartMuLaConfig
        
        # Load config
        config_path = Path(self.model_path) / "HeartMuLa-oss-3B" / "config.json"
        with open(config_path, 'r') as f:
            model_config = HeartMuLaConfig(**json.load(f))
        
        # Load model
        self.model = HeartMuLa.from_pretrained(
            str(Path(self.model_path) / "HeartMuLa-oss-3B"),
            config=model_config,
            torch_dtype=torch.float32  # Need float32 for training
        ).to(self.device)
        
        # Load tokenizer
        tokenizer_path = Path(self.model_path) / "tokenizer.json"
        if tokenizer_path.exists():
            with open(tokenizer_path, 'r') as f:
                self.tokenizer = json.load(f)
        
        self.log("Model loaded successfully")
    
    def _inject_lora(self):
        """Inject LoRA adapters into target modules."""
        self.log(f"Injecting LoRA adapters (rank={self.config.lora.rank})")
        
        lora_config = self.config.lora
        count = 0
        
        # Inject into backbone
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this is a target module
                for target in lora_config.target_modules:
                    if target in name:
                        # Get parent module and attribute name
                        parts = name.rsplit('.', 1)
                        if len(parts) == 2:
                            parent_name, attr_name = parts
                            parent = dict(self.model.backbone.named_modules())[parent_name]
                        else:
                            parent = self.model.backbone
                            attr_name = name
                        
                        # Replace with LoRA layer
                        lora_layer = LoRALinear(
                            module,
                            rank=lora_config.rank,
                            alpha=lora_config.alpha,
                            dropout=lora_config.dropout
                        ).to(self.device)
                        setattr(parent, attr_name, lora_layer)
                        self.lora_layers[f"backbone.{name}"] = lora_layer
                        count += 1
                        break
        
        self.log(f"Injected {count} LoRA adapters")
        
        # Enable gradient checkpointing if configured
        if self.config.use_gradient_checkpointing:
            self.log("Enabling gradient checkpointing")
            # Note: torchtune models may need specific setup
            try:
                self.model.backbone.gradient_checkpointing = True
            except:
                pass
    
    def _setup_optimizer(self, num_training_steps: int):
        """Setup optimizer and scheduler."""
        # Only train LoRA parameters
        lora_params = []
        for name, layer in self.lora_layers.items():
            lora_params.extend([layer.lora_A, layer.lora_B])
        
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Linear warmup then constant
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        
        self.log(f"Optimizer configured with {len(lora_params)} trainable parameters")
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        # Clip gradients
        total_norm = 0.0
        for layer in self.lora_layers.values():
            for param in [layer.lora_A, layer.lora_B]:
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > self.config.max_grad_norm:
            scale = self.config.max_grad_norm / total_norm
            for layer in self.lora_layers.values():
                for param in [layer.lora_A, layer.lora_B]:
                    if param.grad is not None:
                        param.grad.data *= scale
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1
    
    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute training loss on a batch.
        
        For HeartMuLa, we train on next-token prediction for audio codebooks.
        """
        audio_tokens = batch['audio_tokens'].to(self.device)  # [B, seq_len, num_codebooks]
        attention_mask = batch['attention_mask'].to(self.device)  # [B, seq_len]
        
        if audio_tokens.dim() != 3:
            raise ValueError(f"Expected 3D audio_tokens, got shape {audio_tokens.shape}")
        
        B, seq_len, num_codebooks = audio_tokens.shape
        
        # Shift for next-token prediction
        input_tokens = audio_tokens[:, :-1]  # [B, seq_len-1, num_codebooks]
        target_tokens = audio_tokens[:, 1:, 0]  # [B, seq_len-1] - codebook 0 only
        target_mask = attention_mask[:, 1:]
        
        # Embed audio tokens using HeartMuLa's audio_embeddings
        # audio_embeddings uses combined index: codebook_idx * audio_vocab_size + token_id
        audio_vocab_size = self.model.config.audio_vocab_size
        num_model_codebooks = self.model.config.audio_num_codebooks
        
        # Sum embeddings across all codebooks (similar to how model does it)
        embeddings = None
        # Only use as many codebooks as the model supports
        limit_codebooks = min(num_codebooks, num_model_codebooks)
        
        for cb_idx in range(limit_codebooks):
            cb_tokens = input_tokens[:, :, cb_idx]  # [B, seq_len-1]
            # Offset by codebook index
            offset_tokens = cb_idx * audio_vocab_size + cb_tokens
            cb_emb = self.model.audio_embeddings(offset_tokens.long())
            if embeddings is None:
                embeddings = cb_emb
            else:
                embeddings = embeddings + cb_emb
        
        # Forward through backbone
        hidden = self.model.backbone(embeddings, input_pos=None)
        
        # Handle if backbone returns tuple (hidden, cache)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        
        # Predict codebook 0
        logits = self.model.codebook0_head(hidden)  # [B, seq_len-1, vocab_size]
        
        # Compute loss (cross-entropy)
        # Compute loss (cross-entropy)
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            target_tokens.contiguous().view(-1).long(),
            reduction='none'
        )
        
        # Apply mask
        loss = (loss * target_mask.contiguous().view(-1).float()).sum() / target_mask.sum()
        
        return loss
    
    def train(
        self,
        train_dataset,
        output_dir: Path,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """
        Run training loop.
        
        Args:
            train_dataset: PyTorch Dataset of training samples
            output_dir: Directory to save checkpoints
            progress_callback: Called with (epoch, step, loss)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and inject LoRA
        self._load_model()
        self._inject_lora()
        
        # Setup data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )
        
        # FIX: Dynamically adjust gradient accumulation to ensure at least 1 step per epoch
        # This prevents the bug where optimizer.step() never triggers with small datasets
        effective_accum_steps = min(
            self.config.gradient_accumulation_steps,
            max(1, len(train_loader))  # Never larger than batch count, minimum 1
        )
        
        if effective_accum_steps != self.config.gradient_accumulation_steps:
            self.log(f"Adjusted gradient_accumulation_steps from {self.config.gradient_accumulation_steps} "
                     f"to {effective_accum_steps} (dataset has {len(train_loader)} batches)")
        
        num_training_steps = (
            len(train_loader) // effective_accum_steps
        ) * self.config.epochs
        
        # Ensure at least 1 training step
        num_training_steps = max(1, num_training_steps)
        
        self._setup_optimizer(num_training_steps)
        
        self.log(f"Starting training: {self.config.epochs} epochs, "
                 f"{len(train_loader)} batches/epoch, "
                 f"accumulation={effective_accum_steps}")
        
        self.model.train()
        accumulated_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            self.log(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    loss = self._compute_loss(batch)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    
                    accumulated_loss += loss.item()
                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    epoch_steps += 1
                    
                    # Gradient accumulation step
                    if (batch_idx + 1) % effective_accum_steps == 0:
                        self._optimizer_step()
                        
                        # Log
                        if self.global_step % self.config.log_steps == 0:
                            avg_loss = accumulated_loss / min(self.global_step, self.config.log_steps)
                            lr = self.scheduler.get_last_lr()[0]
                            self.log(f"  Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                            accumulated_loss = 0.0
                            
                            if progress_callback:
                                progress_callback(epoch, self.global_step, avg_loss, num_training_steps)
                        
                        # Save checkpoint
                        if self.global_step % self.config.save_steps == 0:
                            self._save_checkpoint(output_dir / f"checkpoint-{self.global_step}")
                
                except Exception as e:
                    self.log(f"Error in batch {batch_idx}: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                    continue
            
            # FIX: Flush remaining accumulated gradients at epoch end
            # This ensures we don't lose gradients from the last incomplete accumulation batch
            remaining_batches = (batch_idx + 1) % effective_accum_steps
            if remaining_batches != 0:
                self.log(f"  Flushing {remaining_batches} remaining batches at epoch end")
                self._optimizer_step()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            self.log(f"Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}, global_step={self.global_step}")
            
            # Always report progress at end of epoch
            if progress_callback:
                progress_callback(epoch, self.global_step, avg_epoch_loss, num_training_steps)
            
            # Save epoch checkpoint
            self._save_checkpoint(output_dir / f"checkpoint-epoch-{epoch + 1}")
        
        # Save final checkpoint
        self._save_checkpoint(output_dir / "checkpoint-final")
        
        # FIX: Training validation - warn if training appears to have failed
        if self.global_step == 0:
            self.log("WARNING: global_step is 0 - no optimizer steps occurred! Training failed.")
        else:
            self.log(f"Training complete! Total steps: {self.global_step}")
    
    def _save_checkpoint(self, path: Path):
        """Save LoRA checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        lora_state = {}
        for name, layer in self.lora_layers.items():
            lora_state[name] = layer.get_lora_state_dict()
        
        torch.save(lora_state, path / "lora_weights.pt")
        
        # Save config
        config_dict = {
            'lora': {
                'rank': self.config.lora.rank,
                'alpha': self.config.lora.alpha,
                'target_modules': self.config.lora.target_modules
            },
            'training': {
                'epochs': self.config.epochs,
                'learning_rate': self.config.learning_rate,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch
            }
        }
        
        with open(path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.log(f"Saved checkpoint: {path}")
    
    @classmethod
    def load_lora_checkpoint(
        cls,
        model,
        checkpoint_path: Path,
        device: str = "cpu"
    ):
        """
        Load LoRA weights into a model.
        
        Args:
            model: HeartMuLa model
            checkpoint_path: Path to checkpoint directory
            device: Device to load to
            
        Returns:
            Model with LoRA weights merged
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load config
        with open(checkpoint_path / "config.json", 'r') as f:
            config = json.load(f)
        
        lora_config = LoRAConfig(**config['lora'])
        
        # Load weights
        lora_state = torch.load(
            checkpoint_path / "lora_weights.pt",
            map_location=device,
            weights_only=True
        )
        
        logger.info(f"Loading {len(lora_state)} LoRA layers from {checkpoint_path}")
        loaded_count = 0
        failed_layers = []
        
        # Inject LoRA layers and load weights
        for name, state in lora_state.items():
            try:
                # Find and replace module
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                attr_name = parts[-1]
                original_layer = getattr(parent, attr_name)
                
                if isinstance(original_layer, nn.Linear):
                    # Get target dtype and device from original layer
                    target_dtype = original_layer.weight.dtype
                    target_device = original_layer.weight.device
                    
                    lora_layer = LoRALinear(
                        original_layer,
                        rank=state['rank'],
                        alpha=state['alpha']
                    )
                    # Load weights with correct dtype and device
                    lora_layer.lora_A.data = state['lora_A'].to(device=target_device, dtype=target_dtype)
                    lora_layer.lora_B.data = state['lora_B'].to(device=target_device, dtype=target_dtype)
                    setattr(parent, attr_name, lora_layer)
                    loaded_count += 1
                    
                    if loaded_count == 1:
                        logger.info(f"First LoRA layer loaded: {name}, dtype={target_dtype}, device={target_device}")
                else:
                    logger.warning(f"Layer {name} is not nn.Linear, got {type(original_layer).__name__}")
                    failed_layers.append(name)
            except AttributeError as e:
                logger.error(f"Failed to find module path '{name}': {e}")
                failed_layers.append(name)
            except Exception as e:
                logger.error(f"Failed to load LoRA layer '{name}': {e}")
                import traceback
                logger.error(traceback.format_exc())
                failed_layers.append(name)
        
        logger.info(f"Successfully loaded {loaded_count}/{len(lora_state)} LoRA layers")
        if failed_layers:
            logger.warning(f"Failed to load {len(failed_layers)} layers: {failed_layers[:5]}...")
        
        return model
