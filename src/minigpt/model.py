"""Decoder-only pre-LayerNorm Transformer with explicit causal attention."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from minigpt.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Project Q/K/V, split heads, and attend only to the current token and its past."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.attention = config.attention
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.block_size, config.block_size, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return a contextualized (batch, time, embedding) tensor."""
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            item.view(batch, time, self.n_head, self.head_dim).transpose(1, 2) for item in (q, k, v)
        )
        if self.attention == "sdpa":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~self.causal_mask[:time, :time], float("-inf"))
            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, time, width)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Expand each token's channels, apply GELU, and project back to the residual width."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the same feed-forward network independently at every position."""
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Use two pre-normalized residual branches: attention, then a tokenwise MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """Add attention and feed-forward updates to the residual stream."""
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Predict next-character logits with learned positions and tied input/output weights."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (
            nn.Embedding(config.block_size, config.n_embd) if config.positional_embeddings else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight
        residual_std = config.init_std / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.proj.weight, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Return (B,T,V) logits and optional mean cross entropy; reject oversized context."""
        if idx.ndim != 2 or not 0 < idx.size(1) <= self.config.block_size:
            raise ValueError("idx must have shape (B, T), with 1 <= T <= block_size")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(idx.size(1), device=idx.device)
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = (
            F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if targets is not None
            else None
        )
        return logits, loss


def parameter_formula(config: ModelConfig) -> int:
    """Count tied embeddings, optional positions, block weights/biases, and final norm.

    For width E, MLP ratio R, layers L, vocabulary V, context T, bias switch b,
    and position switch p: V*E + p*T*E + L*((4+2R)*E² + 2E + b*(R+7)*E)
    + E + b*E. The 2E per block are LayerNorm scales, independent of biases.
    """
    e, r, b = config.n_embd, config.mlp_ratio, int(config.bias)
    embeddings = config.vocab_size * e + int(config.positional_embeddings) * config.block_size * e
    per_block = (4 + 2 * r) * e * e + 2 * e + b * (r + 7) * e
    return embeddings + config.n_layer * per_block + e * (1 + b)
