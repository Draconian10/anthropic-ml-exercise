"""
Improved Transformer Model Implementation
Addresses all issues from code review.
Author: Yash
"""

import math
from typing import Optional, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_amp: bool = True
    log_interval: int = 100
    patience: int = 5
    checkpoint_dir: str = "checkpoints"


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding - CRITICAL FIX."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerModel(nn.Module):
    """Improved Transformer with all fixes applied."""

    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1,
                 max_len: int = 512, pad_token_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)  # FIX: Added

        # FIX: batch_first=True for correct shape handling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.embedding.weight  # Weight tying

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None,
                causal: bool = False) -> torch.Tensor:
        # FIX: Create padding mask if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = (x == self.pad_token_id)

        # FIX: Create causal mask if needed
        src_mask = None
        if causal:
            src_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()

        # FIX: Added positional encoding
        x = self.pos_encoding(self.embedding(x) * self.embed_scale)
        x = self.transformer(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.fc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Trainer:
    """Complete training pipeline with all best practices."""

    def __init__(self, model: nn.Module, train_loader, val_loader, config: TrainingConfig, device: torch.device):
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.config, self.device = config, device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_loader) * config.epochs)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
        self.scaler = GradScaler() if config.use_amp else None
        self.writer = SummaryWriter()

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            inputs = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)

            self.optimizer.zero_grad()  # CRITICAL FIX

            if self.config.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                if torch.isnan(loss):  # FIX: Error handling
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)  # FIX
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            self.global_step += 1

            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()  # CRITICAL FIX
        total_loss = 0.0

        for batch in self.val_loader:
            outputs = self.model(batch['input_ids'].to(self.device))
            targets = batch['targets'].to(self.device)
            total_loss += self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1)).item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(), 'best_val_loss': self.best_val_loss}
        torch.save(checkpoint, f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
        if is_best:
            torch.save(checkpoint, f"{self.config.checkpoint_dir}/best_model.pt")

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.writer.close()


if __name__ == "__main__":
    model = TransformerModel(vocab_size=10000, d_model=512, nhead=8, num_layers=6)
    print(f"Model created with {model.count_parameters():,} parameters")