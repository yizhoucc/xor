"""LSTM models with InnerNet activation for language modeling experiments.

InnerNetLSTMCell replaces the cell candidate's tanh with a learned 2-arg InnerNet,
using separate projections from input and hidden state as the two arguments.
"""
import math
import torch
import torch.nn as nn


class InnerNetLSTMActivation(nn.Module):
    """Small InnerNet used inside LSTM cell."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class InnerNetLSTMCell(nn.Module):
    """LSTM cell with InnerNet activation for cell candidate.

    Standard LSTM gates (i, f, o) use sigmoid as usual.
    Cell candidate g uses InnerNet(input_proj, hidden_proj) instead of tanh(combined).

    The two InnerNet arguments have clear semantic meaning:
      arg1 = W_cx @ x  (current input projection, "what's coming in now")
      arg2 = W_ch @ h  (hidden state projection, "what memory says")
    This mirrors the biological basal (feedforward) vs apical (feedback) dendrite analogy.
    """
    def __init__(self, input_size, hidden_size, inner_hidden=32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 3 gates (i, f, o) use standard linear + sigmoid
        self.gate_linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        # Cell candidate: two SEPARATE projections for InnerNet's 2 args
        self.cell_input_proj = nn.Linear(input_size, hidden_size)    # from current input x
        self.cell_hidden_proj = nn.Linear(hidden_size, hidden_size)  # from memory h
        self.inner_net = InnerNetLSTMActivation(hidden_dim=inner_hidden)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)

        # Standard gates
        gates = self.gate_linear(combined)
        i_gate, f_gate, o_gate = gates.chunk(3, dim=1)
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)

        # Cell candidate via InnerNet with separate projections
        # arg1: "what the current input says" (feedforward / basal)
        # arg2: "what the memory says" (feedback / apical)
        arg1 = self.cell_input_proj(x)          # [B, hidden_size]
        arg2 = self.cell_hidden_proj(h_prev)    # [B, hidden_size]
        cell_pairs = torch.stack([arg1, arg2], dim=-1)  # [B, hidden_size, 2]
        g_flat = self.inner_net(cell_pairs.view(-1, 2))
        g = g_flat.view(x.size(0), self.hidden_size)

        # State update
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ClassicInnerNetLSTMCell(nn.Module):
    """LSTM cell with 'classic' InnerNet: no semantic pairing.

    Like MLP/AE approach: compute W @ [x, h] → 2*hidden_size, then reshape
    adjacent dims into pairs and apply InnerNet. No distinction between
    input and hidden state — just adjacent-dimension pairing.

    This serves as an ablation to test whether semantic pairing matters.
    """
    def __init__(self, input_size, hidden_size, inner_hidden=32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 3 gates (i, f, o) — same as standard
        self.gate_linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        # Cell candidate: single combined projection → 2× width → InnerNet halves
        self.cell_linear = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.cell_ln = nn.LayerNorm(2 * hidden_size)
        self.inner_net = InnerNetLSTMActivation(hidden_dim=inner_hidden)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)

        # Standard gates
        gates = self.gate_linear(combined)
        i_gate, f_gate, o_gate = gates.chunk(3, dim=1)
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)

        # Cell candidate: classic approach — combined → 2×hidden → pair → InnerNet → hidden
        cell_out = self.cell_ln(self.cell_linear(combined))  # [B, 2*hidden_size]
        cell_pairs = cell_out.view(x.size(0), self.hidden_size, 2)  # [B, hidden_size, 2]
        g = self.inner_net(cell_pairs.view(-1, 2)).view(x.size(0), self.hidden_size)

        # State update
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BoundedInnerNetLSTMCell(nn.Module):
    """LSTM cell with bounded InnerNet: tanh(InnerNet(input_proj, hidden_proj)).

    Same as InnerNetLSTMCell but wraps InnerNet output with tanh to bound g ∈ [-1, 1],
    matching the original LSTM's tanh on cell candidate.
    """
    def __init__(self, input_size, hidden_size, inner_hidden=32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.cell_input_proj = nn.Linear(input_size, hidden_size)
        self.cell_hidden_proj = nn.Linear(hidden_size, hidden_size)
        self.inner_net = InnerNetLSTMActivation(hidden_dim=inner_hidden)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)

        gates = self.gate_linear(combined)
        i_gate, f_gate, o_gate = gates.chunk(3, dim=1)
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)

        arg1 = self.cell_input_proj(x)
        arg2 = self.cell_hidden_proj(h_prev)
        cell_pairs = torch.stack([arg1, arg2], dim=-1)
        g_raw = self.inner_net(cell_pairs.view(-1, 2))
        g = torch.tanh(g_raw.view(x.size(0), self.hidden_size))  # bound to [-1, 1]

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BoundedInnerNetLSTMModel(nn.Module):
    """Language model using LSTM with bounded InnerNet cell."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, inner_hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = BoundedInnerNetLSTMCell(embed_dim, hidden_dim, inner_hidden)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embeds = self.embedding(x)
        B, S, _ = embeds.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        for t in range(S):
            h, c = self.cell(embeds[:, t, :], (h, c))
        return self.fc(h)


class BoundedClassicInnerNetLSTMCell(nn.Module):
    """LSTM cell with bounded classic InnerNet: tanh(InnerNet(pair(W@[x,h]))).

    Combines classic (no semantic pairing) with tanh bounding.
    """
    def __init__(self, input_size, hidden_size, inner_hidden=32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.cell_linear = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.cell_ln = nn.LayerNorm(2 * hidden_size)
        self.inner_net = InnerNetLSTMActivation(hidden_dim=inner_hidden)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)

        gates = self.gate_linear(combined)
        i_gate, f_gate, o_gate = gates.chunk(3, dim=1)
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)

        cell_out = self.cell_ln(self.cell_linear(combined))
        cell_pairs = cell_out.view(x.size(0), self.hidden_size, 2)
        g_raw = self.inner_net(cell_pairs.view(-1, 2))
        g = torch.tanh(g_raw.view(x.size(0), self.hidden_size))

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BoundedClassicInnerNetLSTMModel(nn.Module):
    """Language model using LSTM with bounded classic InnerNet cell."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, inner_hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = BoundedClassicInnerNetLSTMCell(embed_dim, hidden_dim, inner_hidden)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embeds = self.embedding(x)
        B, S, _ = embeds.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        for t in range(S):
            h, c = self.cell(embeds[:, t, :], (h, c))
        return self.fc(h)


class InnerNetLSTMModel(nn.Module):
    """Language model using LSTM with InnerNet cell (semantic pairing, unbounded)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, inner_hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = InnerNetLSTMCell(embed_dim, hidden_dim, inner_hidden)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embeds = self.embedding(x)
        B, S, _ = embeds.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        for t in range(S):
            h, c = self.cell(embeds[:, t, :], (h, c))
        return self.fc(h)


class ClassicInnerNetLSTMModel(nn.Module):
    """Language model using LSTM with classic InnerNet cell (no semantic pairing)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, inner_hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = ClassicInnerNetLSTMCell(embed_dim, hidden_dim, inner_hidden)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embeds = self.embedding(x)
        B, S, _ = embeds.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        for t in range(S):
            h, c = self.cell(embeds[:, t, :], (h, c))
        return self.fc(h)


class StandardLSTMModel(nn.Module):
    """Standard LSTM language model for baseline comparison."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.lstm(embeds)
        return self.fc(out[:, -1, :])
