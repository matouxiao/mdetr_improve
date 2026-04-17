# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
2D relative position bias for encoder self-attention on flattened image tokens.
Used when --position_embedding relative: encoder self-attn uses sine PE on Q/K (same as sine)
and adds this bias to attention logits (hybrid); decoder cross-attn unchanged.
"""
import torch
from torch import nn


class Relative2DPositionBias(nn.Module):
    """Learnable bias B[i,j] = f(row_i - row_j, col_i - col_j) for image tokens in a H×W grid."""

    def __init__(self, max_offset: int = 49):
        super().__init__()
        self.K = max_offset
        self.bias_table = nn.Parameter(torch.zeros(2 * self.K + 1, 2 * self.K + 1))

    def grid_bias(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Returns (H*W, H*W) bias for row-major flattened grid."""
        HW = H * W
        idx = torch.arange(HW, device=device)
        y = idx // W
        x = idx % W
        rel_y = y[:, None] - y[None, :]
        rel_x = x[:, None] - x[None, :]
        dr = rel_y.clamp(-self.K, self.K) + self.K
        dc = rel_x.clamp(-self.K, self.K) + self.K
        return self.bias_table[dr.long(), dc.long()].to(dtype)

    def build_full_bias(
        self,
        h: int,
        w: int,
        seq_len: int,
        has_cls: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build (seq_len, seq_len) additive mask for encoder self-attention.
        Image tokens occupy either [:hw] or [1:1+hw] if CLS is prepended; rest (text) stays 0.
        """
        hw = h * w
        bias = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        block = self.grid_bias(h, w, device, dtype)
        if has_cls:
            if 1 + hw > seq_len:
                raise ValueError("CLS + image tokens exceed sequence length")
            bias[1 : 1 + hw, 1 : 1 + hw] = block
        else:
            if hw > seq_len:
                raise ValueError("image tokens exceed sequence length")
            bias[:hw, :hw] = block
        return bias
