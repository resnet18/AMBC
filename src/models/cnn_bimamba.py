"""
src/models/cnn_bimamba.py
============================
Original CNN+BiMamba architecture from SongX-1's reproduction.

Architecture preserved as-is for AMBC Standard Protocol audit.
Under fixed-length 4096 batching (no padding), the backward-Mamba
padding contamination issue (torch.flip over padded tensors) is
structurally avoided, so no architectural modification is needed.

Components:
- ConvBNAct, MultiScaleResBlock
- BiMambaBlock (bidirectional causal Mamba)
- StaticMLP
- CNNBiMambaClassifier
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception as e:
    Mamba = None
    import warnings
    warnings.warn(f"mamba_ssm not installed: {e}. BiMambaBlock will fail at init.")


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        branch = channels // 4
        self.b1 = ConvBNAct(channels, branch, 3)
        self.b2 = ConvBNAct(channels, branch, 7)
        self.b3 = ConvBNAct(channels, branch, 15)
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvBNAct(channels, branch, 1),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.out = nn.Sequential(
            nn.Conv1d(branch * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = self.proj(x)
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.pool(x)], dim=1)
        y = self.out(y)
        return F.gelu(y + res)


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba using two causal Mamba instances.
    
    NOTE: Under variable-length batching with padding, the backward
    branch ingests padded positions as valid history due to torch.flip
    over the padded batch tensor. This is a structural limitation of
    causal SSMs (mamba_ssm.Mamba lacks key-padding-mask support).
    
    Under AMBC Standard Protocol (fixed 4096, no padding), this issue
    is structurally avoided because all sequences are equal-length and
    the mask is all-True. We preserve the original implementation.
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is not installed. Install with: pip install mamba-ssm")
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T) bool
        m = mask.unsqueeze(-1).to(x.dtype)

        y1 = self.fwd(x)
        xr = torch.flip(x, dims=[1])
        y2 = self.bwd(xr)
        y2 = torch.flip(y2, dims=[1])

        y = 0.5 * (y1 + y2)
        y = y * m
        x = x + self.dropout(y)
        x = self.norm(x)
        x = x + self.ffn(x)
        x = x * m
        return x


class StaticMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        if in_dim == 0:
            self.net = None
            self.out_dim = 0
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
                nn.GELU(),
            )
            self.out_dim = out_dim

    def forward(self, x):
        if self.net is None:
            return x.new_zeros((x.shape[0], 0))
        return self.net(x)


class CNNBiMambaClassifier(nn.Module):
    def __init__(self,
                 seq_in_dim: int = 23,
                 static_dim: int = 0,
                 cnn_dim: int = 128,
                 stem_kernel: int = 7,
                 stem_stride: int = 2,
                 mamba_layers: int = 2,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 static_hidden: int = 128,
                 static_out: int = 64,
                 num_classes: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(seq_in_dim, cnn_dim, stem_kernel, stride=stem_stride),
            ConvBNAct(cnn_dim, cnn_dim, 3, stride=1),
        )
        self.ms1 = MultiScaleResBlock(cnn_dim, dropout=dropout)
        self.down = ConvBNAct(cnn_dim, cnn_dim, 3, stride=2)
        self.ms2 = MultiScaleResBlock(cnn_dim, dropout=dropout)

        self.mamba_blocks = nn.ModuleList([
            BiMambaBlock(cnn_dim, mamba_d_state, mamba_d_conv, mamba_expand, dropout)
            for _ in range(mamba_layers)
        ])

        self.static_branch = StaticMLP(static_dim, static_hidden, static_out, dropout)
        fusion_dim = cnn_dim * 2 + self.static_branch.out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, mask, static):
        # x: (B, T, C), mask: (B, T)
        x = x.transpose(1, 2)  # (B, C, T)

        x = self.stem(x)
        # FIX: downsample mask to match CNN output length after stem
        if mask.shape[1] != x.shape[-1]:
            idx = torch.linspace(0, mask.shape[1] - 1, steps=x.shape[-1], device=mask.device).long()
            mask = mask[:, idx]

        x = self.ms1(x)
        x = self.down(x)
        # FIX: downsample mask again after down block
        if mask.shape[1] != x.shape[-1]:
            idx = torch.linspace(0, mask.shape[1] - 1, steps=x.shape[-1], device=mask.device).long()
            mask = mask[:, idx]

        x = self.ms2(x)
        x = x.transpose(1, 2)  # (B, T', D)

        for block in self.mamba_blocks:
            x = block(x, mask)

        m = mask.unsqueeze(-1).to(x.dtype)
        denom = m.sum(dim=1).clamp_min(1.0)
        mean_pool = (x * m).sum(dim=1) / denom

        neg_inf = torch.full_like(x, -1e9)
        max_pool = torch.where(mask.unsqueeze(-1), x, neg_inf).amax(dim=1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        static_feat = self.static_branch(static)
        fused = torch.cat([mean_pool, max_pool, static_feat], dim=1)
        return self.head(fused)