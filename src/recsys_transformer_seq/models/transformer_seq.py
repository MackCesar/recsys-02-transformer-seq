from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerSeqModel(nn.Module):
    """Transformer encoder for next-item prediction.

    Given an input sequence of item indices, predicts the next item over the item vocabulary.
    """

    def __init__(
        self,
        num_items: int,
        max_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model

        # Item embedding (0 is reserved for padding)
        self.item_embedding = nn.Embedding(num_items, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, num_items)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_seq: (batch, seq_len) of item indices.

        Returns:
            logits: (batch, num_items) scores over all items.
        """
        # (B, L, D)
        x = self.item_embedding(input_seq)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.dropout(x)

        # Use representation at the last position as sequence summary
        last_hidden = x[:, -1, :]  # (B, D)
        logits = self.output_layer(last_hidden)  # (B, num_items)
        return logits