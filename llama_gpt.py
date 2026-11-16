"""LLaMA style decoder only transformer (RoPE, RMSNorm, SwiGLU, grouped query attention, KV cache).

The same code as notebook 04, in a module so that train.py, generate.py and the tests can import it.
Config flags also allow the GPT-2 style variant: pos="learned", norm="layer", mlp="gelu", n_kv_head=n_head.
"""
import math
from dataclasses import dataclass, replace, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        out = (xf - xf.mean(-1, keepdim=True)) / torch.sqrt(xf.var(-1, keepdim=True, unbiased=False) + self.eps)
        return out.type_as(x) * self.weight

def rope_cos_sin(positions, head_dim, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    angles = positions.float()[:, None] * inv_freq[None, :]          # (T, head_dim / 2)
    emb = torch.cat([angles, angles], dim=-1)                          # same angle for both members of a pair
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).type_as(x)


@dataclass
class Config:
    block_size: int = 128
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 128
    pos: str = "rope"          # "rope" or "learned"
    norm: str = "rms"          # "rms" or "layer"
    mlp: str = "swiglu"        # "swiglu" or "gelu"
    dropout: float = 0.0
    rope_theta: float = 10000.0

def make_norm(cfg):
    return RMSNorm(cfg.n_embd) if cfg.norm == "rms" else LayerNorm(cfg.n_embd)

def swiglu_hidden(n_embd, multiple_of=32):
    return multiple_of * ((int(8 * n_embd / 3) + multiple_of - 1) // multiple_of)     # about 8/3 of the width

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0 and cfg.n_head % cfg.n_kv_head == 0
        self.cfg = cfg
        self.hd = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.wq = nn.Linear(cfg.n_embd, cfg.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.hd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.hd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, start_pos=0, cache=None, use_cache=False):
        B, T, _ = x.shape
        cfg = self.cfg
        q = self.wq(x).view(B, T, cfg.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, cfg.n_kv_head, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, cfg.n_kv_head, self.hd).transpose(1, 2)
        if cfg.pos == "rope":
            cos, sin = rope_cos_sin(torch.arange(start_pos, start_pos + T, device=x.device), self.hd, cfg.rope_theta)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v) if use_cache else None
        S = k.size(2)
        if self.n_rep > 1:                                   # each key/value head serves n_rep query heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        p = cfg.dropout if self.training else 0.0
        if T == S:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)             # one new token sees everything cached
        else:
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        y = y.transpose(1, 2).contiguous().view(B, T, cfg.n_head * self.hd)
        return self.drop(self.wo(y)), new_cache

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = swiglu_hidden(cfg.n_embd)
        self.w1 = nn.Linear(cfg.n_embd, h, bias=False)       # gate
        self.w3 = nn.Linear(cfg.n_embd, h, bias=False)       # value
        self.w2 = nn.Linear(h, cfg.n_embd, bias=False)       # back down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class GeluMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.w2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.gelu(self.w1(x))))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = make_norm(cfg), Attention(cfg)
        self.n2, self.mlp = make_norm(cfg), (SwiGLU(cfg) if cfg.mlp == "swiglu" else GeluMLP(cfg))

    def forward(self, x, start_pos=0, cache=None, use_cache=False):
        a, new_cache = self.attn(self.n1(x), start_pos, cache, use_cache)
        x = x + a
        return x + self.mlp(self.n2(x)), new_cache

def sample_next(logits, temperature=1.0, top_k=None):
    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(F.softmax(logits, dim=-1), 1)

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0, use_cache=False):
        B, T = idx.shape
        x = self.wte(idx)
        if self.wpe is not None:
            assert start_pos + T <= self.cfg.block_size, "learned positions cannot go past block_size"
            x = x + self.wpe(torch.arange(start_pos, start_pos + T, device=idx.device))
        x = self.drop(x)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, start_pos, None if kv_caches is None else kv_caches[i], use_cache)
            new_caches.append(c)
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss, (new_caches if use_cache else None)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        was_training = self.training
        self.eval()
        bs = self.cfg.block_size
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _, _ = self(idx[:, -bs:])
                idx = torch.cat([idx, sample_next(logits[:, -1], temperature, top_k)], dim=1)
        else:
            if self.wpe is not None:
                assert idx.size(1) + max_new_tokens <= bs, "with learned positions the whole sequence must fit in block_size"
            ctx = idx[:, -bs:]
            logits, _, caches = self(ctx, use_cache=True)
            pos = ctx.size(1)
            for _ in range(max_new_tokens):
                nxt = sample_next(logits[:, -1], temperature, top_k)
                idx = torch.cat([idx, nxt], dim=1)
                if caches[0][0].size(2) >= bs:                 # sliding window: keep the newest keys only
                    caches = [(k[:, :, -(bs - 1):], v[:, :, -(bs - 1):]) for k, v in caches]
                logits, _, caches = self(nxt, kv_caches=caches, start_pos=pos, use_cache=True)
                pos += 1
        self.train(was_training)
        return idx


class CharData:
    """Character level data: vocabulary, train/val split (90/10) and batch sampling."""

    def __init__(self, text, chars=None):
        self.chars = chars if chars is not None else "".join(sorted(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(ids))
        self.train, self.val = ids[:n], ids[n:]

    def encode(self, s):
        missing = sorted({c for c in s if c not in self.stoi})
        if missing:
            raise ValueError(f"characters not in the vocabulary: {missing}")
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split, batch_size, block_size, generator=None, device=None):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
        return (x.to(device), y.to(device)) if device else (x, y)


def get_lr(it, lr, min_lr, warmup_iters, max_iters):
    """Linear warmup, then cosine decay to min_lr."""
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup_iters) / max(max_iters - warmup_iters, 1)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)
