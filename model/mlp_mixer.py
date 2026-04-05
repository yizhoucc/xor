"""MLP-Mixer models with optional InnerNet activation.

MLP-Mixer (Tolstikhin et al., 2021) uses NO attention — only MLPs.
This makes it a clean testbed for InnerNet: any improvement must come
from the learned 2-arg activation itself, not from interaction with
existing multiplicative gates.

Architecture:
  Image → Patch Embedding → N × (Token-Mixing MLP + Channel-Mixing MLP) → Classification

InnerNet variant replaces GELU in both mixing MLPs with GLU-style InnerNet.
"""
import torch
import torch.nn as nn


class InnerNetMixerActivation(nn.Module):
    """Small InnerNet used as activation in MLP-Mixer."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class GELUMixingMLP(nn.Module):
    """Standard mixing MLP: Linear → GELU → Linear."""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class InnerNetMixingMLP(nn.Module):
    """Mixing MLP with GLU-style InnerNet activation.

    Two projections create value and gate, InnerNet combines them.
    """
    def __init__(self, dim, hidden_dim, inner_net, dropout=0.0):
        super().__init__()
        self.w1a = nn.Linear(dim, hidden_dim)  # value
        self.w1b = nn.Linear(dim, hidden_dim)  # gate
        self.inner_net = inner_net  # shared InnerNet
        self.w2 = nn.Linear(hidden_dim, dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        a = self.w1a(x)
        b = self.w1b(x)
        pairs = torch.stack([a, b], dim=-1)
        shape = pairs.shape[:-1]
        activated = self.inner_net(pairs.reshape(-1, 2)).view(*shape)
        return self.drop2(self.w2(self.drop1(activated)))


class MixerBlock(nn.Module):
    """Single Mixer block: LN → Token-Mix → residual, LN → Channel-Mix → residual."""
    def __init__(self, num_patches, d_model, token_mlp, channel_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.token_mix = token_mlp
        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mix = channel_mlp

    def forward(self, x):
        # x: [B, num_patches, d_model]
        # Token mixing: transpose to [B, d_model, num_patches], mix, transpose back
        x = x + self.token_mix(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        # Channel mixing
        x = x + self.channel_mix(self.ln2(x))
        return x


class MLPMixerBase(nn.Module):
    """Base MLP-Mixer for image classification."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128,
                 num_layers=4, num_classes=10):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size

        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _extract_patches(self, x):
        # x: [B, C, H, W] → [B, num_patches, patch_dim]
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, self.patch_dim)
        return x


class InnerNetMLPMixer(MLPMixerBase):
    """MLP-Mixer with InnerNet activation (no multiplicative interactions)."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128,
                 token_hidden=64, channel_hidden=512, num_layers=4,
                 num_classes=10, inner_hidden=32, dropout=0.1):
        super().__init__(image_size, patch_size, in_channels, d_model, num_layers, num_classes)

        # Shared InnerNet for all mixing MLPs
        self.inner_net = InnerNetMixerActivation(hidden_dim=inner_hidden)

        self.blocks = nn.ModuleList([
            MixerBlock(
                self.num_patches, d_model,
                token_mlp=InnerNetMixingMLP(self.num_patches, token_hidden, self.inner_net, dropout),
                channel_mlp=InnerNetMixingMLP(d_model, channel_hidden, self.inner_net, dropout),
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.patch_embed(self._extract_patches(x))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)


class StandardMLPMixer(MLPMixerBase):
    """Standard MLP-Mixer with GELU activation (baseline)."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128,
                 token_hidden=64, channel_hidden=512, num_layers=4,
                 num_classes=10, dropout=0.1):
        super().__init__(image_size, patch_size, in_channels, d_model, num_layers, num_classes)

        self.blocks = nn.ModuleList([
            MixerBlock(
                self.num_patches, d_model,
                token_mlp=GELUMixingMLP(self.num_patches, token_hidden, dropout),
                channel_mlp=GELUMixingMLP(d_model, channel_hidden, dropout),
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.patch_embed(self._extract_patches(x))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)
        return self.head(x)
