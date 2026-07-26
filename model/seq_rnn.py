"""RNN models for Sequential MNIST — testing whether InnerNet can discover gates.

4 models:
  - SeqRNN:          Standard RNN + tanh (baseline)
  - SeqLSTM:         Standard LSTM (upper bound)
  - SeqInnerNetRNN:  Plan A — RNN with 1 InnerNet replacing tanh (h and x separate)
  - SeqGatedRNN:     Plan B — RNN with 2 InnerNets + cell state (can learn gates)
"""
import math
import torch
import torch.nn as nn


class InnerNetActivation(nn.Module):
    """Small 2-input MLP: f(a, b) -> scalar, applied element-wise."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pairs):
        # pairs: (..., 2) -> (..., 1)
        return self.net(pairs)


# ============================================================
# Baseline: Standard RNN + tanh
# ============================================================

class SeqRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_classes=10):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))


# ============================================================
# Baseline: Standard LSTM (upper bound)
# ============================================================

class SeqLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))


# ============================================================
# Plan A: InnerNet RNN — 1 InnerNet, two separate projections
# ============================================================

class InnerNetRNNCell(nn.Module):
    """RNN cell where tanh is replaced by InnerNet(W_h @ h, W_x @ x).

    The two inputs are kept separate so InnerNet can learn gating:
    e.g., f(a,b) ≈ σ(a)·b would be an input gate.
    """
    def __init__(self, input_size, hidden_size, inner_hidden=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln = nn.LayerNorm(hidden_size)
        self.inner_net = InnerNetActivation(hidden_dim=inner_hidden)

    def forward(self, x_t, h_prev):
        a = self.W_h(h_prev)
        b = self.W_x(x_t)
        a = self.ln(a)
        b = self.ln(b)
        pairs = torch.stack([a, b], dim=-1)  # [B, H, 2]
        h_t = self.inner_net(pairs.view(-1, 2)).view(x_t.size(0), self.hidden_size)
        return h_t


class SeqInnerNetRNN(nn.Module):
    """Plan A: Simple RNN with InnerNet activation."""
    def __init__(self, input_size=1, hidden_size=128, num_classes=10, inner_hidden=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = InnerNetRNNCell(input_size, hidden_size, inner_hidden)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, S, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(S):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)


# ============================================================
# Plan B: Gated RNN — 2 InnerNets + cell state
# ============================================================

class GatedInnerNetRNNCell(nn.Module):
    """RNN cell with cell state + 2 InnerNets at key positions.

    Architecture:
      a = W_h @ h_{t-1}   (recurrent signal)
      b = W_x @ x_t        (input signal)

      InnerNet1 BEFORE cell state: learns input/forget gate
        cell_update = InnerNet1(a, b)
        c_t = c_{t-1} + cell_update

      InnerNet2 AFTER cell state: learns output gate
        h_t = InnerNet2(proj(c_t), a)

    If InnerNet1 learns σ(a)·tanh(b) → input gate + candidate
    If InnerNet2 learns σ(a)·tanh(c) → output gate
    Then this is effectively an LSTM discovered from scratch.
    """
    def __init__(self, input_size, hidden_size, inner_hidden=32, cell_tanh=False,
                 ortho_init=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_tanh = cell_tanh
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_a = nn.LayerNorm(hidden_size)
        self.ln_b = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

        self.inner_net1 = InnerNetActivation(hidden_dim=inner_hidden)
        self.inner_net2 = InnerNetActivation(hidden_dim=inner_hidden)

        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)

        if ortho_init:
            # Orthogonal recurrent matrices keep the spectral radius near 1,
            # preventing the BPTT gradient explosion over 784 steps that makes
            # some seeds NaN at epoch 1. Small-gain InnerNet output layers keep
            # the initial cell_update / h_t tiny so the cell state grows slowly.
            nn.init.orthogonal_(self.W_h.weight)
            nn.init.orthogonal_(self.W_c.weight)
            for net in (self.inner_net1, self.inner_net2):
                last = net.net[-1]
                nn.init.normal_(last.weight, std=0.01)
                nn.init.zeros_(last.bias)

    def forward(self, x_t, h_prev, c_prev):
        a = self.ln_a(self.W_h(h_prev))
        b = self.ln_b(self.W_x(x_t))

        # InnerNet1: before cell state — can learn input/forget gate
        pairs1 = torch.stack([a, b], dim=-1)
        cell_update = self.inner_net1(pairs1.view(-1, 2)).view(x_t.size(0), self.hidden_size)
        # tanh-bounded update prevents unbounded cell state growth over long sequences
        if self.cell_tanh:
            cell_update = torch.tanh(cell_update)
        c_t = c_prev + cell_update

        # InnerNet2: after cell state — can learn output gate
        c_proj = self.ln_c(self.W_c(c_t))
        pairs2 = torch.stack([c_proj, a], dim=-1)
        h_t = self.inner_net2(pairs2.view(-1, 2)).view(x_t.size(0), self.hidden_size)

        return h_t, c_t


class SeqGatedRNN(nn.Module):
    """Plan B: RNN with cell state + 2 InnerNets (can discover gates)."""
    def __init__(self, input_size=1, hidden_size=128, num_classes=10, inner_hidden=32,
                 cell_tanh=False, ortho_init=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GatedInnerNetRNNCell(input_size, hidden_size, inner_hidden, cell_tanh,
                                         ortho_init)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, S, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(S):
            h, c = self.cell(x[:, t, :], h, c)
        return self.fc(h)


# ============================================================
# Plan B (constrained): force the output gate into inner_net2
# ============================================================

class MinGatedInnerNetRNNCell(nn.Module):
    """Constrained gated cell for identifiability of the OUTPUT gate.

    Same as GatedInnerNetRNNCell except the learnable ``W_c`` projection before
    inner_net2 is removed: inner_net2 reads the (LayerNorm-normalised) cell state
    directly. In the unconstrained cell, W_c can absorb/rotate the cell-to-output
    mapping, so the gating role need not live in inner_net2 — which is why the
    learned inner_net2 surface is seed-inconsistent. Removing that linear degree
    of freedom forces inner_net2 to be the (only) nonlinear map from cell state
    to hidden output, testing whether it then reproducibly converges on an
    output-gate shape sigma(a) * f(c). ln_c is kept (non-learnable-mixing) for
    stability over the 784-step additive cell state.
    """
    def __init__(self, input_size, hidden_size, inner_hidden=32, ortho_init=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_a = nn.LayerNorm(hidden_size)
        self.ln_b = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        self.inner_net1 = InnerNetActivation(hidden_dim=inner_hidden)
        self.inner_net2 = InnerNetActivation(hidden_dim=inner_hidden)
        if ortho_init:
            nn.init.orthogonal_(self.W_h.weight)
            for net in (self.inner_net1, self.inner_net2):
                last = net.net[-1]
                nn.init.normal_(last.weight, std=0.01)
                nn.init.zeros_(last.bias)

    def forward(self, x_t, h_prev, c_prev):
        a = self.ln_a(self.W_h(h_prev))
        b = self.ln_b(self.W_x(x_t))
        pairs1 = torch.stack([a, b], dim=-1)
        cell_update = self.inner_net1(pairs1.view(-1, 2)).view(x_t.size(0), self.hidden_size)
        c_t = c_prev + cell_update
        # No W_c: inner_net2 reads the normalised cell state directly.
        c_norm = self.ln_c(c_t)
        pairs2 = torch.stack([c_norm, a], dim=-1)
        h_t = self.inner_net2(pairs2.view(-1, 2)).view(x_t.size(0), self.hidden_size)
        return h_t, c_t


class SeqMinGatedRNN(nn.Module):
    """Constrained Plan B: gated cell with the W_c output projection removed."""
    def __init__(self, input_size=1, hidden_size=128, num_classes=10, inner_hidden=32,
                 ortho_init=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = MinGatedInnerNetRNNCell(input_size, hidden_size, inner_hidden, ortho_init)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, S, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(S):
            h, c = self.cell(x[:, t, :], h, c)
        return self.fc(h)


# ============================================================
# Standard GRU baseline
# ============================================================

class SeqGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))
