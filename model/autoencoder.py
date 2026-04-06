"""Autoencoder models with optional InnerNet activation.

Tests InnerNet on unsupervised reconstruction (MNIST).
BaselineAE uses ReLU; InnerNetAE uses learned 2-arg activation.
"""
import torch
import torch.nn as nn


class BaselineAE(nn.Module):
    """MLP Autoencoder with ReLU.

    Architecture: encoder (784→256→64→latent) → decoder (latent→64→256→784)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = config.model.input_dim
        latent_dim = getattr(config.model, 'latent_dim', 32)
        self.num_classes = 1  # signals regression/reconstruction

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid(),
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        z = self.encoder(x_flat)
        recon = self.decoder(z)
        # Normalize input to [0,1] for MSE with sigmoid output
        x_norm = (x_flat + 1) / 2  # from [-1,1] to [0,1]
        loss = self.loss_func(recon, x_norm)
        return recon, loss, []


class InnerNetAEActivation(nn.Module):
    """2-arg activation for autoencoder. Pairs adjacent units."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, D) → pair adjacent → (B, D//2, 2) → InnerNet → (B, D//2)
        B, D = x.shape
        x = x.view(B, D // 2, 2)
        x = self.net(x.view(-1, 2)).view(B, D // 2)
        return x


class InnerNetAE(nn.Module):
    """MLP Autoencoder with InnerNet activation.

    Encoder uses 2× width then InnerNet halves it.
    Architecture: encoder (784→512→128→latent) with InnerNet → decoder (latent→64→256→784)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = config.model.input_dim
        latent_dim = getattr(config.model, 'latent_dim', 32)
        inner_hidden = getattr(config.model, 'inner_hidden', 32)
        self.num_classes = 1

        self.inner_act = InnerNetAEActivation(inner_hidden)

        # Encoder: 2× width → InnerNet halves
        self.enc_fc1 = nn.Linear(input_dim, 512)
        self.enc_ln1 = nn.LayerNorm(512)
        # After InnerNet: 256
        self.enc_fc2 = nn.Linear(256, 128)
        self.enc_ln2 = nn.LayerNorm(128)
        # After InnerNet: 64
        self.enc_fc3 = nn.Linear(64, latent_dim)

        # Decoder uses ReLU (symmetric to baseline)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid(),
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        # Encoder with InnerNet
        h = self.inner_act(self.enc_ln1(self.enc_fc1(x_flat)))
        h = self.inner_act(self.enc_ln2(self.enc_fc2(h)))
        z = self.enc_fc3(h)

        recon = self.decoder(z)
        x_norm = (x_flat + 1) / 2
        loss = self.loss_func(recon, x_norm)
        return recon, loss, []
