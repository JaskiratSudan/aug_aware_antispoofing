"""
losses.py — Loss functions for aug_aware_antispoofing.

Re-exports existing losses from loss.py (copied locally) and adds
NTXentMultiPositiveLoss for self-supervised multi-view training.
"""

import torch
import torch.nn as nn

from loss import (  # noqa: F401 — re-exported for callers
    BCEBinaryLoss,
    SupConBinaryLoss,
    SupConMultiClassLoss,
    compute_pos_weight_from_dataset,
)

__all__ = [
    "SupConBinaryLoss",
    "SupConMultiClassLoss",
    "BCEBinaryLoss",
    "compute_pos_weight_from_dataset",
    "NTXentMultiPositiveLoss",
]


class NTXentMultiPositiveLoss(nn.Module):
    """
    NT-Xent with M positives per anchor (self-supervised — no labels).

    Implements SupCon Eq. 2: average of per-positive log-probs, not log of
    summed numerator.

    Input z: (N, n_views, D) L2-normalized embeddings.
      N       = utterances in batch
      n_views = total augmented views per utterance (no +1 offset — all views
                are treated symmetrically; there is no separate clean anchor)
      D       = embedding dimension (256)

    For each sample i (in flattened N*n_views tensor):
      Positives P(i) = all other views of the same utterance  |P(i)| = n_views - 1
      Negatives      = all views of all other utterances

    Loss per sample:
      L_i = -(1/|P(i)|) * Σ_{j∈P(i)} [sim(z_i,z_j)/τ - log Σ_{k≠i} exp(sim(z_i,z_k)/τ)]
    Total loss = mean over all valid samples.

    Temperature default 0.2 matches the SupCon scale used in the existing pipeline.
    (0.07 is tuned for large memory banks of ~65K negatives; for batch sizes of
    ~32×5=160 samples the denominator is too small at 0.07 and training is unstable.)
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.tau = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (N, n_views, D) — L2-normalized embeddings.
        Returns scalar loss.
        """
        N, n_views, D = z.shape
        total = N * n_views
        z_flat = z.view(total, D)  # (total, D)

        # Cosine similarity matrix — z is already L2-normalized so mm gives cosines
        sim = torch.mm(z_flat, z_flat.t()) / self.tau  # (total, total)

        # Utterance assignment: sample i belongs to utterance i // n_views
        utt_idx = torch.arange(N, device=z.device).repeat_interleave(n_views)  # (total,)

        # Positive mask: same utterance, different sample index
        pos_mask = utt_idx.unsqueeze(0) == utt_idx.unsqueeze(1)  # (total, total)
        eye = torch.eye(total, dtype=torch.bool, device=z.device)
        pos_mask = pos_mask & ~eye  # exclude self

        # Log-denominator: logsumexp over all k ≠ i
        log_denom = torch.logsumexp(
            sim.masked_fill(eye, float("-inf")), dim=1, keepdim=True
        )  # (total, 1)

        # Per-pair log probs; average over positives (SupCon Eq. 2)
        log_probs = sim - log_denom                              # (total, total)
        n_pos = pos_mask.float().sum(dim=1)                      # (total,)
        pos_sum = (log_probs * pos_mask.float()).sum(dim=1)      # (total,)

        valid = n_pos > 0
        return -(pos_sum[valid] / n_pos[valid]).mean()
