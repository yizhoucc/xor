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
    The inner_net is shared across all layers (passed in from outside).
    """
    def __init__(self, d_model, d_ff, inner_net, dropout=0.1):
        super().__init__()
        self.w1a = nn.Linear(d_model, d_ff)  # value projection
        self.w1b = nn.Linear(d_model, d_ff)  # gate projection
        self.inner_net = inner_net  # shared across layers
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


class SiLUInnerNetFFNActivation(nn.Module):
    """InnerNet with SiLU instead of ReLU internally. Smoother inductive bias."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class SiLUInnerNetFFN(nn.Module):
    """FFN with SiLU-InnerNet: classic pairing (single proj → 2× → pair adjacent).

    Tests whether SiLU inductive bias helps InnerNet learn SwiGLU-like functions.
    The inner_net is shared across all layers.
    """
    def __init__(self, d_model, d_ff, inner_net, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff * 2)  # single proj, 2× width
        self.inner_net = inner_net  # shared across layers
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_ff = d_ff

    def forward(self, x):
        h = self.w1(x)  # [B, S, 2*d_ff]
        # Pair adjacent dims
        pairs = h.view(*h.shape[:-1], self.d_ff, 2)  # [B, S, d_ff, 2]
        shape = pairs.shape[:-1]
        activated = self.inner_net(pairs.reshape(-1, 2)).view(*shape)
        return self.w2(self.dropout(activated))


class ClassicInnerNetFFN(nn.Module):
    """FFN with Classic InnerNet: single projection → 2× width → pair adjacent.

    Mirrors the LSTM finding that classic (adjacent) pairing > semantic pairing.
    The inner_net is shared across all layers.
    """
    def __init__(self, d_model, d_ff, inner_net, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff * 2)  # single proj, 2× width
        self.inner_net = inner_net  # shared across layers
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_ff = d_ff

    def forward(self, x):
        h = self.w1(x)  # [B, S, 2*d_ff]
        pairs = h.view(*h.shape[:-1], self.d_ff, 2)  # [B, S, d_ff, 2]
        shape = pairs.shape[:-1]
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


class BilinearGLUFFN(nn.Module):
    """Bilinear GLU FFN: (W1a·x) ⊙ (W1b·x) — pure multiplicative gate, no activation.

    Identical dual-projection structure and parameter names (w1a/w1b/w2) as
    SwiGLUFFN and InnerNetFFN, but the gate is plain a·b instead of silu(a)·b.
    Used as a NON-SwiGLU warm-start base: if an InnerNet dropped into a network
    trained with this a·b gate still converges to silu(a)·b, the SwiGLU form is
    a network-independent attractor, not an artifact of a SwiGLU-shaped host.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1a = nn.Linear(d_model, d_ff)
        self.w1b = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(self.w1a(x) * self.w1b(x)))


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


class InnerNetAttention(nn.Module):
    """Multi-head attention with InnerNet replacing softmax.

    Standard attention: weights = softmax(QK^T / sqrt(d_k))
    InnerNet attention: weights = normalize(InnerNet(score, mean_score))

    For each attention score s_ij, InnerNet takes two inputs:
      1. s_ij: the raw attention score (Q_i · K_j / sqrt(d_k))
      2. mean_j(s_i): the mean score for that query (context signal)
    InnerNet outputs a positive weight (via abs), then L1-normalized.
    """
    def __init__(self, d_model, n_heads, inner_net, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.inner_net = inner_net  # shared InnerNet

    def forward(self, x, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, S, S]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0.0)

        # Compute mean score per query as context signal
        if mask is not None:
            # Count valid positions per query for correct mean
            valid = mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
            mean_scores = scores.sum(dim=-1, keepdim=True) / valid  # [B, H, S, 1]
        else:
            mean_scores = scores.mean(dim=-1, keepdim=True)

        mean_expanded = mean_scores.expand_as(scores)  # [B, H, S, S]

        # Stack (score, mean_score) → InnerNet
        pairs = torch.stack([scores, mean_expanded], dim=-1)  # [B, H, S, S, 2]
        shape = pairs.shape[:-1]  # [B, H, S, S]
        raw_weights = self.inner_net(pairs.reshape(-1, 2)).view(*shape)

        # Make positive and apply mask
        raw_weights = raw_weights.abs()
        if mask is not None:
            raw_weights = raw_weights.masked_fill(mask == 0, 0.0)

        # L1 normalize (like softmax but learned)
        attn = raw_weights / (raw_weights.sum(dim=-1, keepdim=True) + 1e-8)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → Attn → residual, LN → FFN → residual."""
    def __init__(self, d_model, n_heads, ffn_module, dropout=0.1, attn_module=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn_module if attn_module is not None else MultiHeadAttention(d_model, n_heads, dropout)
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


class SiLUInnerNetTransformer(nn.Module):
    """Decoder-only Transformer with SiLU-InnerNet FFN (classic pairing, smooth bias)."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, inner_hidden=32, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        shared_inner = SiLUInnerNetFFNActivation(hidden_dim=inner_hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                SiLUInnerNetFFN(d_model, d_ff, shared_inner, dropout),
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


class ClassicInnerNetTransformer(nn.Module):
    """Decoder-only Transformer with Classic InnerNet FFN (adjacent pairing).

    LSTM ablation showed classic > semantic pairing. This tests
    the same hypothesis in Transformers.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, inner_hidden=32, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        shared_inner = InnerNetFFNActivation(hidden_dim=inner_hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                ClassicInnerNetFFN(d_model, d_ff, shared_inner, dropout),
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


class InnerNetTransformer(nn.Module):
    """Decoder-only Transformer with InnerNet FFN for language modeling."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, inner_hidden=32, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        shared_inner = InnerNetFFNActivation(hidden_dim=inner_hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                InnerNetFFN(d_model, d_ff, shared_inner, dropout),
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


class BilinearGLUTransformer(nn.Module):
    """Decoder-only Transformer with a Bilinear GLU FFN (gate = a·b, no activation).

    Same architecture as SwiGLUTransformer; used as a non-SwiGLU warm-start base
    for the InnerNet init-independence / network-independence probe.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                BilinearGLUFFN(d_model, d_ff, dropout),
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


def _eval_distilled(a, b, coeffs):
    """Evaluate a fixed closed-form operator g(a, b) from distilled coefficients.

    coeffs maps term names to scalars. Supported terms:
      '1','a','b','a^2','a*b','b^2','a^3','a^2*b','a*b^2','b^3'  (polynomial)
      'silu(a)*b','silu(b)*a'                                    (SwiGLU family)
    Pure elementwise ops — no inner MLP — so inference is ~as cheap as SwiGLU.
    """
    out = None

    def add(term, val):
        nonlocal out
        c = coeffs.get(term)
        if c is None or c == 0.0:
            return
        contrib = c if val is None else c * val
        out = contrib if out is None else out + contrib

    add('1', None)
    add('a', a)
    add('b', b)
    add('a^2', a * a)
    add('a*b', a * b)
    add('b^2', b * b)
    add('a^3', a * a * a)
    add('a^2*b', a * a * b)
    add('a*b^2', a * b * b)
    add('b^3', b * b * b)
    if 'silu(a)*b' in coeffs:
        add('silu(a)*b', torch.nn.functional.silu(a) * b)
    if 'silu(b)*a' in coeffs:
        add('silu(b)*a', torch.nn.functional.silu(b) * a)
    if out is None:
        out = torch.zeros_like(a)
    return out


class DistilledFFN(nn.Module):
    """FFN with a FIXED closed-form operator distilled from a trained InnerNet.

    Same dual-projection structure as InnerNetFFN/SwiGLUFFN, but the learned
    inner network is replaced by a fixed g(a, b) recovered via least squares
    (scripts/distill_innernet.py). This is the DEPLOY step of the
    discover -> distill -> deploy loop: InnerNet's per-element MLP is gone, so
    the operator runs at SwiGLU-like speed while preserving the discovered
    interaction.
    """
    def __init__(self, d_model, d_ff, coeffs, dropout=0.1):
        super().__init__()
        self.w1a = nn.Linear(d_model, d_ff)  # value projection
        self.w1b = nn.Linear(d_model, d_ff)  # gate projection
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.coeffs = dict(coeffs)

    def forward(self, x):
        a = self.w1a(x)
        b = self.w1b(x)
        activated = _eval_distilled(a, b, self.coeffs)
        return self.w2(self.dropout(activated))


class DistilledTransformer(nn.Module):
    """Decoder-only Transformer with a fixed distilled FFN operator.

    Drop-in counterpart to InnerNetTransformer / SwiGLUTransformer, used to
    show that the distilled operator matches InnerNet quality at SwiGLU speed.
    """
    def __init__(self, vocab_size, coeffs, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                DistilledFFN(d_model, d_ff, coeffs, dropout),
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


class InnerNetAttnTransformer(nn.Module):
    """Decoder-only Transformer with InnerNet replacing softmax in attention.

    FFN uses standard GELU to isolate the effect of InnerNet in attention.
    The InnerNet is shared across all layers and heads.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=64, inner_hidden=32, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.attn_inner_net = InnerNetFFNActivation(hidden_dim=inner_hidden)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads,
                StandardFFN(d_model, d_ff, dropout),
                dropout,
                attn_module=InnerNetAttention(d_model, n_heads, self.attn_inner_net, dropout)
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
