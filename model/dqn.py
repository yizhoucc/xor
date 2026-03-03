"""DQN models with InnerNet activation for RL experiments.

InnerNetDQN replaces ReLU with a learned 2-arg InnerNet activation.
BaselineDQN uses standard ReLU for comparison.

Width matching: InnerNet halves width (pairs of 2 → 1), so InnerNetDQN uses
2× hidden_dim to match BaselineDQN's effective width after InnerNet.
"""
import torch
import torch.nn as nn


class InnerNetDQNActivation(nn.Module):
    """Small InnerNet used as activation function inside DQN."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class InnerNetDQN(nn.Module):
    """DQN Q-network with InnerNet activation.

    hidden_dim should be 2× the desired effective width (default 256 → 128 after InnerNet).
    Architecture: Linear(state, hidden_dim) → ReLU → reshape(hidden_dim/2, 2) → InnerNet
                  → hidden_dim/2 → Linear(hidden_dim/2, 64) → ReLU → Linear(64, actions)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, inner_hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.inner_net = InnerNetDQNActivation(hidden_dim=inner_hidden)
        effective_dim = hidden_dim // 2
        self.fc2 = nn.Linear(effective_dim, 64)
        self.head = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x_pairs = x.view(x.size(0), -1, 2)
        B, P, _ = x_pairs.shape
        acts = self.inner_net(x_pairs.view(-1, 2)).view(B, P)
        x = self.relu(self.fc2(acts))
        return self.head(x)


class BaselineDQN(nn.Module):
    """Standard DQN Q-network with ReLU activation.

    Architecture: Linear(state, hidden_dim) → ReLU → Linear(hidden_dim, 64) → ReLU → Linear(64, actions)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)
