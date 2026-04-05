"""Vision Transformer (ViT) with optional InnerNet FFN.

Tests whether InnerNet FFN improvement transfers from language to vision.
Uses the same TransformerBlock/FFN modules as the language model.
"""
import math
import torch
import torch.nn as nn
from .transformer import (TransformerBlock, StandardFFN, InnerNetFFN,
                           SwiGLUFFN, MultiHeadAttention)


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings + CLS token + position encoding."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # [B, C, H, W] -> [B, num_patches, patch_dim]
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, -1)
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        return x


class InnerNetViT(nn.Module):
    """ViT with InnerNet FFN activation (GLU-style, same as Transformer LM)."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128,
                 n_heads=4, d_ff=512, n_layers=4, num_classes=10,
                 inner_hidden=32, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                InnerNetFFN(d_model, d_ff, inner_hidden, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.patch_embed(x))
        for block in self.blocks:
            x = block(x)  # no causal mask for ViT
        x = self.ln_f(x[:, 0])  # CLS token
        return self.head(x)


class StandardViT(nn.Module):
    """ViT with standard GELU FFN (baseline)."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128,
                 n_heads=4, d_ff=512, n_layers=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                StandardFFN(d_model, d_ff, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.patch_embed(x))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x[:, 0])
        return self.head(x)


class SwiGLUViT(nn.Module):
    """ViT with SwiGLU FFN (comparison)."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128,
                 n_heads=4, d_ff=512, n_layers=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                SwiGLUFFN(d_model, d_ff, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.patch_embed(x))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x[:, 0])
        return self.head(x)
