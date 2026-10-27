"""
models.py
---------
Three deep learning architectures for RF modulation classification:

  1. CNN1D       — 1D convolutions on raw I/Q time series
  2. CNN2D       — 2D convolutions on STFT spectrogram images
  3. RFTransformer — lightweight Transformer encoder on I/Q sequence

All models output raw logits of shape (B, num_classes).
Use nn.CrossEntropyLoss which expects logits, not softmax probabilities.
"""

import torch
import torch.nn as nn
from src.dataset import NUM_CLASSES


# ======================================================================
# 1.  CNN1D  —  operates on raw I/Q (B, 2, 128)
# ======================================================================

class CNN1D(nn.Module):
    """
    Four-block 1D-CNN with residual-style depth.
    Input  : (B, 2, 128)   — 2 channels (I and Q), 128 time steps
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(2,   64,  kernel_size=3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  64,  kernel_size=3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(2),                                                      # → 64×64

            # Block 2
            nn.Conv1d(64,  128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),                                                      # → 128×32

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                                              # → 256×1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# 2.  CNN2D  —  operates on STFT spectrogram (B, 1, F, T)
# ======================================================================

class CNN2D(nn.Module):
    """
    Three-block 2D-CNN on log-magnitude STFT spectrogram.
    Input  : (B, 1, F, T)  — 1 channel, frequency × time
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                                        # → 128×4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# 3.  RFTransformer  —  lightweight Transformer on I/Q sequence
# ======================================================================

class PositionalEncoding(nn.Module):
    """Learnable positional encoding (better than sinusoidal for short sequences)."""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class RFTransformer(nn.Module):
    """
    Lightweight Transformer encoder for I/Q classification.
    Input  : (B, 2, 128)  — raw I/Q
    Output : (B, num_classes)

    Pipeline:
        Permute → (B, 128, 2)
        Linear projection → (B, 128, d_model)
        + Positional Encoding
        → TransformerEncoder (num_layers blocks of multi-head attention + FFN)
        → Global Average Pool → (B, d_model)
        → Classifier head → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        d_model: int     = 64,
        nhead: int       = 4,
        num_layers: int  = 2,
        dim_ff: int      = 128,
        dropout: float   = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(2, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_ff,
            dropout        = dropout,
            batch_first    = True,   # (B, seq, d_model)
            norm_first     = True,   # Pre-LN — more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 128)
        x = x.permute(0, 2, 1)          # (B, 128, 2)
        x = self.input_proj(x)           # (B, 128, d_model)
        x = self.pos_enc(x)              # + positional encoding
        x = self.transformer(x)          # (B, 128, d_model)
        x = self.norm(x)
        x = x.mean(dim=1)               # global average pool → (B, d_model)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# Factory function
# ======================================================================

def build_model(name: str, **kwargs) -> nn.Module:
    """
    Build a model by name.

    Args:
        name : 'cnn1d' | 'cnn2d' | 'transformer'
        **kwargs: passed to the model constructor

    Returns:
        nn.Module
    """
    registry = {
        "cnn1d":       CNN1D,
        "cnn2d":       CNN2D,
        "transformer": RFTransformer,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")

    model = registry[name](**kwargs)
    print(f"[Model] {name.upper()} — {model.count_params():,} trainable parameters")
    return model
