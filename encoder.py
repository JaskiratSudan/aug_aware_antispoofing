# encoder.py
# Wav2Vec2-Large frontend that returns a 4D tensor shaped for your CompressionModule:
# (B, K, F, T) where:
#   B = batch, K = number of selected transformer layers,
#   F = feature dim, T = time frames

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2Encoder(nn.Module):
    """
    Wraps a pretrained Wav2Vec2 model and exposes stacked hidden states
    as (B, K, F, T) to match the expected input of CompressionModule.
    """
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-large-960h",
                #  take_last_k_layers: int = 8,
                 freeze_encoder: bool = True):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        # self.take_last_k_layers = take_last_k_layers

        if freeze_encoder:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        else:
            self.model.gradient_checkpointing_enable()

    @torch.no_grad()
    def _forward_frozen(self, waveforms: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward used when the encoder is frozen (default).
        """
        out = self.model(
            waveforms, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )
        return out.hidden_states

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveforms: (B, T_samples) float32 mono at 16kHz, padded with zeros.
            attention_mask: (B, T_samples) 1 for real samples, 0 for padding. If None, inferred.

        Returns:
            hs_4d: (B, K, F, T) stacked hidden states (last K transformer layers),
                   permuted to match CompressionModule input.
        """
        if attention_mask is None:
            attention_mask = (waveforms != 0.0).long()

        # Handle training vs eval depending on whether encoder is frozen
        if all(not p.requires_grad for p in self.model.parameters()):
            hidden_states = self._forward_frozen(waveforms, attention_mask)
        else:
            out = self.model(
                waveforms, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True
            )
            hidden_states = out.hidden_states

        # hidden_states is a tuple of tensors: each (B, T_feat, D)
        # We take the last K (post-encoder) hidden states.
        selected = hidden_states[:]  # list of K tensors (B, T, D)
        # Stack to (K, B, T, D) -> then to (B, K, T, D) -> then permute to (B, K, D, T)
        hs = torch.stack(selected, dim=0).transpose(0, 1)       # (B, K, T, D)
        hs_4d = hs.permute(0, 1, 3, 2).contiguous()             # (B, K, F=D, T)

        return hs_4d
