"""
Transformer-based Sentiment Classifier with Attention Visualization
Author: Yash
"""

import math
from typing import Optional, Dict, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the Transformer Sentiment Classifier."""
    vocab_size: int = 30522
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_length: int = 256
    dropout: float = 0.1
    num_classes: int = 2
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with attention weight extraction."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, store_attention: bool = False) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        if store_attention:
            self.attention_weights = attention.detach()

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(context)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer with Pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                store_attention: bool = False) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attention(normed, normed, normed, mask, store_attention))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerSentimentClassifier(nn.Module):
    """Transformer-based Sentiment Classifier with Attention Visualization."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.d_model // 2, config.num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                store_attention: bool = False) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            mask = (input_ids != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)
        else:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

        x = self.positional_encoding(self.embedding(input_ids))
        for layer in self.encoder_layers:
            x = layer(x, mask, store_attention)

        hidden_states = self.final_norm(x)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        return {"logits": self.classifier(pooled), "hidden_states": hidden_states}

    def get_attention_weights(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        return self.encoder_layers[layer_idx].self_attention.attention_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
