"""
classifier.py — Stage 2 binary MLP classifier for aug_aware_antispoofing.

Accepts 256-dim L2-normalized embeddings produced by a frozen Stage 1
encoder + projection head. Trained with BCEBinaryLoss from losses.py.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Stage 2 binary deepfake classifier.

    Architecture: 256 → LayerNorm → 64 → LeakyReLU → Dropout → 1 (logit)

    LayerNorm is used instead of BatchNorm1d so that the model is safe to
    evaluate at batch_size=1 (EER computation typically scores utterances
    individually) and correct at any batch size in eval mode.

    The encoder and projection head remain frozen; only this MLP is trained.
    Loss: BCEBinaryLoss (from losses.py).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, 256) L2-normalized embeddings.
        Returns: (B,) logits (unbounded; apply sigmoid for probabilities).
        """
        return self.net(z).squeeze(-1)
