"""Transformer models with InnerNet activation for language modeling experiments.

InnerNetTransformer replaces GELU in FFN with a GLU-style InnerNet:
  Standard FFN:  W2 · GELU(W1 · x + b1) + b2
  InnerNet FFN:  W2 · InnerNet(W1a · x, W1b · x) + b2

The two projections have distinct semantic roles (value vs gate),
similar to SwiGLU but with a learned gating function.
"""
import math
import torch
import torch.nn as nn


class InnerNetFFNActivation(nn.Module):
    """Small InnerNet used as activation in Transformer FFN."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class InnerNetFFN(nn.Module):
    """FFN block with GLU-style InnerNet activation.

    Two separate projections d→d_ff create value and gate signals.
    InnerNet combines each (value[i], gate[i]) pair into a scalar.
    """
    def __init__(self, d_model, d_ff, inner_hidden=32, dropout=0.1):
        super().__init__()
        self.w1a = nn.Linear(d_model, d_ff)  # value projection
        self.w1b = nn.Linear(d_model, d_ff)  # gate projection
        self.inner_net = InnerNetFFNActivation(hidden_dim=inner_hidden)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, S, d_model]
        a = self.w1a(x)  # [B, S, d_ff] — value
        b = self.w1b(x)  # [B, S, d_ff] — gate
        # Pair element-wise: stack along last dim → [B, S, d_ff, 2]
        pairs = torch.stack([a, b], dim=-1)
        shape = pairs.shape[:-1]  # [B, S, d_ff]
        activated = self.inner_net(pairs.reshape(-1, 2)).view(*shape)
        return self.w2(self.dropout(activated))


class StandardFFN(nn.Module):
    """Standard FFN block with GELU activation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class SwiGLUFFN(nn.Module):
    """FFN block with SwiGLU activation: Swish(W1a·x) ⊙ W1b·x.

    Same dual-projection structure as InnerNetFFN, but with a fixed
    gating function (Swish element-wise multiply) instead of learned InnerNet.
    This serves as a controlled comparison: does InnerNet learn something
    beyond what SwiGLU already provides?
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1a = nn.Linear(d_model, d_ff)  # gate projection
        self.w1b = nn.Linear(d_model, d_ff)  # value projection
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.w1a(x))  # Swish = SiLU
        value = self.w1b(x)
        return self.w2(self.dropout(gate * value))


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with causal mask."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → Attn → residual, LN → FFN → residual."""
    def __init__(self, d_model, n_heads, ffn_module, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ffn_module
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop1(self.attn(self.ln1(x), mask))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class InnerNetTransformer(nn.Module):
    """Decoder-only Transformer with InnerNet FFN for language modeling."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, inner_hidden=32, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                InnerNetFFN(d_model, d_ff, inner_hidden, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        # Share embedding and output weights
        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, S = x.shape
        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)
        x = self.pos_enc(self.embedding(x) * math.sqrt(self.d_model))
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x[:, -1, :])  # predict next token from last position


class StandardTransformer(nn.Module):
    """Decoder-only Transformer with GELU FFN for baseline comparison."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                StandardFFN(d_model, d_ff, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, S = x.shape
        mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)
        x = self.pos_enc(self.embedding(x) * math.sqrt(self.d_model))
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x[:, -1, :])


class SwiGLUTransformer(nn.Module):
    """Decoder-only Transformer with SwiGLU FFN for comparison.

    Same architecture as InnerNetTransformer but uses SwiGLU (a fixed
    multiplicative gating function) instead of a learned InnerNet.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                SwiGLUFFN(d_model, d_ff, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, S = x.shape
        mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)
        x = self.pos_enc(self.embedding(x) * math.sqrt(self.d_model))
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x[:, -1, :])
