"""PPO actor-critic models with optional InnerNet activation.

InnerNetPPO replaces ReLU with a learned 2-arg InnerNet activation.
BaselinePPO uses standard ReLU. SwiGLUPPO uses SwiGLU gating.
"""
import torch
import torch.nn as nn
import numpy as np


class BaselinePPO(nn.Module):
    """PPO Actor-Critic with ReLU activation."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def act(self, state):
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        probs, values = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)


class InnerNetPPOActivation(nn.Module):
    """Small InnerNet used as activation function inside PPO."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, D) → pair adjacent → (B, D//2, 2) → net → (B, D//2)
        B, D = x.shape
        x = x.view(B, D // 2, 2)
        return self.net(x.view(-1, 2)).view(B, D // 2)


class InnerNetPPO(nn.Module):
    """PPO Actor-Critic with InnerNet activation.

    hidden_dim should be 2× desired effective width (default 128 → 64 after InnerNet).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, inner_hidden=32):
        super().__init__()
        self.inner_net = InnerNetPPOActivation(inner_hidden)
        eff = hidden_dim // 2

        # Actor
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_ln1 = nn.LayerNorm(hidden_dim)
        self.actor_fc2 = nn.Linear(eff, hidden_dim)
        self.actor_ln2 = nn.LayerNorm(hidden_dim)
        self.actor_head = nn.Linear(eff, action_dim)

        # Critic
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_ln1 = nn.LayerNorm(hidden_dim)
        self.critic_fc2 = nn.Linear(eff, hidden_dim)
        self.critic_ln2 = nn.LayerNorm(hidden_dim)
        self.critic_head = nn.Linear(eff, 1)

    def _actor_forward(self, x):
        x = self.inner_net(self.actor_ln1(self.actor_fc1(x)))
        x = self.inner_net(self.actor_ln2(self.actor_fc2(x)))
        return torch.softmax(self.actor_head(x), dim=-1)

    def _critic_forward(self, x):
        x = self.inner_net(self.critic_ln1(self.critic_fc1(x)))
        x = self.inner_net(self.critic_ln2(self.critic_fc2(x)))
        return self.critic_head(x)

    def forward(self, x):
        return self._actor_forward(x), self._critic_forward(x)

    def act(self, state):
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        probs, values = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)


class SwiGLUPPO(nn.Module):
    """PPO Actor-Critic with SwiGLU activation."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        eff = hidden_dim // 2

        # Actor
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_gate1 = nn.Linear(eff, eff)
        self.actor_val1 = nn.Linear(eff, eff)
        self.actor_fc2 = nn.Linear(eff, hidden_dim)
        self.actor_gate2 = nn.Linear(eff, eff)
        self.actor_val2 = nn.Linear(eff, eff)
        self.actor_head = nn.Linear(eff, action_dim)

        # Critic
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_gate1 = nn.Linear(eff, eff)
        self.critic_val1 = nn.Linear(eff, eff)
        self.critic_fc2 = nn.Linear(eff, hidden_dim)
        self.critic_gate2 = nn.Linear(eff, eff)
        self.critic_val2 = nn.Linear(eff, eff)
        self.critic_head = nn.Linear(eff, 1)

    def _swiglu(self, x, gate_layer, val_layer):
        # Split into pairs
        x = x.view(x.size(0), -1, 2)
        a, b = x[..., 0], x[..., 1]
        return torch.sigmoid(gate_layer(a)) * a * val_layer(b)

    def _actor_forward(self, x):
        x = torch.relu(self.actor_fc1(x))
        x = self._swiglu(x, self.actor_gate1, self.actor_val1)
        x = torch.relu(self.actor_fc2(x))
        x = self._swiglu(x, self.actor_gate2, self.actor_val2)
        return torch.softmax(self.actor_head(x), dim=-1)

    def _critic_forward(self, x):
        x = torch.relu(self.critic_fc1(x))
        x = self._swiglu(x, self.critic_gate1, self.critic_val1)
        x = torch.relu(self.critic_fc2(x))
        x = self._swiglu(x, self.critic_gate2, self.critic_val2)
        return self.critic_head(x)

    def forward(self, x):
        return self._actor_forward(x), self._critic_forward(x)

    def act(self, state):
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        probs, values = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)
