import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SupConBinaryLoss(nn.Module):
    """
    Supervised Contrastive (binary) with alpha-blend:
      main_loss = (1 - alpha) * SupCon_full + alpha * SupCon_mined
    where SupCon_mined uses the top-K hardest negatives per anchor.

    Optionally adds a uniformity regularizer on the hypersphere:
      L_total = main_loss + lambda_uni * L_uni(z)

    L_uni(z) = log E_{i<j}[ exp( -t * ||z_i - z_j||^2 ) ]
    (Wang & Isola, "Understanding Contrastive Representation Learning
     through Alignment and Uniformity on the Hypersphere")
    """
    def __init__(
        self,
        temperature: float = 0.2,
        similarity: str = "geodesic",
        uniformity_weight: float = 0.0,
        uniformity_t: float = 2.0,
    ):
        super().__init__()
        self.tau = temperature
        self.similarity = similarity.lower()
        self.lambda_uni = float(uniformity_weight)
        self.uni_t = float(uniformity_t)
        if self.similarity not in ("cosine", "geodesic"):
            raise ValueError(f"Unknown similarity: {similarity}")

    # ---------- SupCon pieces (unchanged) ----------

    def _supcon_full(self, sim_row: torch.Tensor, pos_mask_row: torch.Tensor) -> torch.Tensor:
        """
        One-anchor vanilla SupCon over all non-self samples.
        sim_row: (B,) similarity to everyone (self has been -inf).
        pos_mask_row: (B,) bool for positives (self already False).
        """
        # logits for all non-self
        logits_all = sim_row / self.tau                               # (B,)
        # if no positives for this anchor, skip
        if not pos_mask_row.any():
            return None
        pos_logits = logits_all[pos_mask_row]                         # (P,)
        log_prob = pos_logits - torch.logsumexp(logits_all, dim=0)    # (P,)
        return -log_prob.mean()

    def _supcon_mined_topk(self, sim_row: torch.Tensor, pos_mask_row: torch.Tensor,
                           neg_mask_row: torch.Tensor, topk_neg: int) -> torch.Tensor:
        """
        One-anchor mined SupCon:
          - positives: all positives
          - negatives: top-K most similar negatives
        """
        if not pos_mask_row.any() or not neg_mask_row.any():
            return None

        pos_sims = sim_row[pos_mask_row]                              # (P,)
        neg_sims = sim_row[neg_mask_row]                              # (N,)
        # top-K hardest negatives
        k = min(topk_neg, neg_sims.numel())
        if k < 1:
            return None
        neg_topk, _ = torch.sort(neg_sims, descending=True)
        neg_topk = neg_topk[:k]                                       # (k,)

        denom = torch.cat([pos_sims, neg_topk], dim=0)                # (P+k,)
        logits = denom / self.tau
        log_probs = logits - torch.logsumexp(logits, dim=0)
        return -log_probs[:pos_sims.numel()].mean()

    # ---------- Uniformity regularizer ----------

    def _uniformity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Wang & Isola uniformity loss on the hypersphere.

        z: (B, D) row-wise L2-normalized embeddings.
        L_uni = log E_{i<j} [ exp( -t * ||z_i - z_j||^2 ) ]
        """
        B = z.size(0)
        if B < 2:
            # Degenerate tiny-batch case
            return torch.tensor(0.0, device=z.device, requires_grad=False)

        # pairwise squared Euclidean distances over the batch
        # torch.pdist gives distances for all i<j pairs
        sq_pdist = torch.pdist(z, p=2).pow(2)  # (B*(B-1)/2,)
        # uniformity: log mean(exp(-t * dist^2))
        return torch.log(torch.exp(-self.uni_t * sq_pdist).mean() + 1e-8)

    # ---------- Forward ----------
    def _pairwise_similarity(self, z: torch.Tensor) -> torch.Tensor:
        if self.similarity == "cosine":
            return z @ z.t()

        dot = z @ z.t()
        eps = 1e-7
        dot = dot.clamp(-1.0 + eps, 1.0 - eps)

        theta = torch.acos(dot)               # [0, pi]
        sim01 = 1.0 - theta / math.pi         # [0, 1]  
        sim = 2.0 * sim01 - 1.0               # [-1, 1] (MATCH cosine range)
        return sim


    def forward(self,
                z: torch.Tensor,        # (B, D) L2-normalized embeddings
                labels: torch.Tensor,   # (B,) in {0,1}
                topk_neg: int = 32,
                alpha: float = 0.0) -> torch.Tensor:

        device = z.device
        B = z.size(0)

        sim = self._pairwise_similarity(z)                             # (B,B)
        eye = torch.eye(B, device=device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float("-inf"))

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & (~eye)                   # (B,B)
        neg_mask = (~pos_mask) & (~eye)

        losses_full = []
        losses_mined = []
        for i in range(B):
            lf = self._supcon_full(sim[i], pos_mask[i])
            lm = self._supcon_mined_topk(sim[i], pos_mask[i], neg_mask[i], topk_neg)
            if lf is not None:
                losses_full.append(lf)
            if lm is not None:
                losses_mined.append(lm)

        # main SupCon + hard-mining loss
        if len(losses_full) == 0:
            main_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss_full = torch.stack(losses_full).mean()
            if len(losses_mined) == 0:
                loss_mined = loss_full
            else:
                loss_mined = torch.stack(losses_mined).mean()
            main_loss = (1.0 - alpha) * loss_full + alpha * loss_mined

        # add uniformity regularizer if enabled
        if self.lambda_uni > 0.0 and B > 1:
            uni_loss = self._uniformity_loss(z)
            main_loss = main_loss + self.lambda_uni * uni_loss

        return main_loss


class SupConMultiClassLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al.) for multi-class labels.
    Treats each distinct label ID (0..C-1) as its own class.
    For each anchor:
      - positives: same-class samples in the batch (excluding self)
      - negatives: all other samples in the batch
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self,
                z: torch.Tensor,        # (B, D) L2-normalized embeddings
                labels: torch.Tensor    # (B,) long, values in {0..C-1}
                ) -> torch.Tensor:
        device = z.device
        B = z.size(0)
        assert labels.dim() == 1 and labels.size(0) == B, "labels must be shape (B,)"

        # Cosine sim (since z normalized, dot product)
        sim = torch.matmul(z, z.t()) / self.tau  # (B, B)

        # Mask out self-comparisons
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, float('-inf'))

        labels = labels.view(-1, 1)  # (B, 1)
        eq = (labels == labels.t())  # (B, B)
        pos_mask = eq & (~self_mask)  # same class, not self

        losses = []

        for i in range(B):
            pos_idx = torch.nonzero(pos_mask[i], as_tuple=False).squeeze(-1)
            if pos_idx.numel() == 0:
                # No positive of same class in batch -> skip anchor
                continue

            # Positives for anchor i
            pos_logits = sim[i, pos_idx]  # (P,)

            # Denominator: all non-self logits (both pos+neg)
            all_logits = sim[i, ~self_mask[i]]  # (B-1,)

            # log p(pos | all)
            log_prob = pos_logits - torch.logsumexp(all_logits, dim=0)  # (P,)
            loss_i = -log_prob.mean()
            losses.append(loss_i)

        if len(losses) == 0:
            # Degenerate tiny-batch case
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses, dim=0).mean()

# Baseline Loss
class BCEBinaryLoss(nn.Module):
    """
    BCE baseline for binary deepfake detection.

    Expects:
      logits: (B,) raw logits (NO sigmoid)
      labels: (B,) in {0,1} (int/long/float ok; will be cast to float)

    Optional pos_weight (neg/pos) to handle imbalance:
      BCEWithLogitsLoss(pos_weight=neg/pos)
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        if pos_weight is None:
            self.crit = nn.BCEWithLogitsLoss()
        else:
            # pos_weight must be a tensor on the correct device at runtime
            self.register_buffer("_pos_weight_tensor", torch.tensor([float(pos_weight)], dtype=torch.float32))
            self.crit = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight_tensor)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        # ensure buffer is on same device (important if you load on cpu then move model)
        if hasattr(self, "_pos_weight_tensor"):
            self._pos_weight_tensor = self._pos_weight_tensor.to(logits.device)
        return self.crit(logits, labels)


def compute_pos_weight_from_dataset(dataset) -> float:
    """
    dataset.data is used in your code already.
    Assumes item[1] is label with 1=bonafide, 0=spoof (same as your sampler).
    Returns neg/pos.
    """
    pos = 0
    neg = 0
    for it in dataset.data:
        y = int(it[1])
        if y == 1:
            pos += 1
        else:
            neg += 1
    if pos == 0 or neg == 0:
        return 1.0
    return float(neg) / float(pos)