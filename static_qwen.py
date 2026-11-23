"""Static-shape, single-graph rewrite of the Qwen3.5 / Qwen3.8 text decoder (hybrid Gated DeltaNet + gated attention)
for AWS Neuron (Inferentia2 / Trainium), where neuronx-cc rejects torch.linalg.solve_triangular (NCC_EVRF001).

The two phases are each one straight-line graph with fixed shapes and no data-dependent control flow:

  prefill(input_ids[B, T], lengths[B])            -> logits[B, V] at the last real token, State
  decode(token[B], positions[B], State)           -> logits[B, V], State

T is a bucket (a multiple of CHUNK); prompts are right-padded. State is a flat tuple of tensors with fixed shapes:
  kv cache per full-attention layer [B, kv_heads, MAX_LEN, head_dim] (x2), conv state per GDN layer
  [B, conv_dim, K-1], recurrent state per GDN layer [B, v_heads, k_dim, v_dim] (fp32).

Key changes versus transformers' modeling_qwen3_5.py:
  * the per-chunk unit-lower-triangular solve becomes `unit_lower_inverse`: log2(CHUNK) doubling steps of two dense
    CHUNK x CHUNK matmuls each, using constant masks (no solve, no row loop, no large matrix powers);
  * padded positions get beta = 0 and g = 0, so they neither write nor decay the recurrent state;
  * caches are explicit inputs/outputs updated with one-hot blends instead of in-place indexing / dynamic slices;
  * the final conv state is gathered at each row's real length, so batches with mixed prompt lengths are exact.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

CHUNK = 64


# ----------------------------------------------------------------------------------------------------- config
@dataclass
class Cfg:
    vocab: int
    hidden: int
    inter: int
    layers: int
    layer_types: list
    eps: float
    # full attention
    heads: int
    kv_heads: int
    head_dim: int
    rot_dim: int
    rope_theta: float
    # gated delta net
    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv_k: int
    tie: bool

    @classmethod
    def from_hf(cls, path):
        c = json.load(open(os.path.join(path, "config.json")))
        tie = c.get("tie_word_embeddings", False)
        t = c.get("text_config", c)
        tie = t.get("tie_word_embeddings", tie)
        rp = t.get("rope_parameters") or {"rope_theta": t.get("rope_theta"), "partial_rotary_factor": t.get("partial_rotary_factor", 1.0)}
        return cls(vocab=t["vocab_size"], hidden=t["hidden_size"], inter=t["intermediate_size"],
                   layers=t["num_hidden_layers"], layer_types=t["layer_types"], eps=t["rms_norm_eps"],
                   heads=t["num_attention_heads"], kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
                   rot_dim=int(t["head_dim"] * rp.get("partial_rotary_factor", 1.0)), rope_theta=rp["rope_theta"],
                   k_heads=t["linear_num_key_heads"], v_heads=t["linear_num_value_heads"],
                   k_dim=t["linear_key_head_dim"], v_dim=t["linear_value_head_dim"], conv_k=t["linear_conv_kernel_dim"],
                   tie=tie)

    @property
    def conv_dim(self):
        return 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim


# ---------------------------------------------------------------------------------------- triangular inverse
def _offdiag_masks(c: int) -> torch.Tensor:
    """masks[l][i, j] = 1 where (i, j) is in the lower-left b x b block of a diagonal 2b x 2b block, b = 2**l."""
    idx = torch.arange(c)
    out, b = [], 1
    while b < c:
        same = (idx[:, None] // (2 * b)) == (idx[None, :] // (2 * b))
        lower_half = ((idx[:, None] // b) % 2 == 1) & ((idx[None, :] // b) % 2 == 0)
        out.append((same & lower_half).float())
        b *= 2
    return torch.stack(out)


def unit_lower_inverse(strict_lower: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """(I + L)^-1 for strictly lower-triangular L [..., C, C], with matmuls only.

    Let D_b be the block-diagonal part of T = I + L with b x b blocks and X_b = D_b^-1 (X_1 = I). Then
    D_2b = D_b + O_b, where O_b holds the lower-left sub-blocks inside each 2b block, and since O_b X_b O_b = 0,
        X_2b = X_b - X_b O_b X_b
    exactly. After log2(C) steps X = T^-1. Every intermediate entry is an entry of a true inverse block, so no
    large powers of L are formed (unlike the Neumann product (I - L)(I + L^2)(I + L^4)..., which overflows fp32
    on real checkpoints).
    """
    c = strict_lower.shape[-1]
    x = torch.eye(c, dtype=strict_lower.dtype, device=strict_lower.device).expand_as(strict_lower)
    for l in range(masks.shape[0]):
        o = strict_lower * masks[l]
        x = x - x @ o @ x
    return x


# ----------------------------------------------------------------------------------------------- primitives
def rms_norm(x, w, eps):  # Qwen3_5RMSNorm: zero-centred weight, (x * (1 + w)) in fp32
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


def gated_rms_norm(x, z, w, eps):  # Qwen3_5RMSNormGated: plain weight, norm before silu gate
    xf = x.float()
    y = (w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype))
    return (y * F.silu(z.float())).to(x.dtype)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rope(x, cos, sin, rot):  # x [B, H, T, D]; cos/sin [B, T, rot]
    cos, sin = cos[:, None], sin[:, None]
    xr, xp = x[..., :rot], x[..., rot:]
    h = rot // 2
    rotated = torch.cat((-xr[..., h:], xr[..., :h]), -1)
    return torch.cat((xr * cos + rotated * sin, xp), -1)


def chunk_gated_delta(q, k, v, g, beta, state, masks):
    """Chunked gated delta rule, static shapes. q,k [B,H,T,Dk]; v [B,H,T,Dv]; g,beta [B,H,T]; state [B,H,Dk,Dv].
    All fp32; T must be a multiple of the chunk size; q,k already l2-normalised; q already scaled.

    Batch and heads are folded into one leading dim so no tensor exceeds 4-D: NxDI's Qwen3.5 contrib reports that the
    5-D PyTorch chunk path crashes neuronx-cc codegen (NCC_INLA001)."""
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    c = masks.shape[-1]
    n = t // c
    bh = b * h
    q, k = q.reshape(bh, n, c, dk), k.reshape(bh, n, c, dk)
    v, beta, g = v.reshape(bh, n, c, dv), beta.reshape(bh, n, c), g.reshape(bh, n, c)
    state = state.reshape(bh, dk, dv)
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    # inclusive prefix sum as a matmul with a constant upper-triangular ones matrix (Tensor engine, no scan op)
    upper = torch.triu(torch.ones(c, c, dtype=g.dtype, device=g.device))
    cum = g @ upper
    lower = torch.tril(torch.ones(c, c, dtype=torch.bool, device=q.device))
    # exp of masked differences; the where() keeps exp() away from positive arguments
    diff = cum[..., :, None] - cum[..., None, :]
    decay = torch.where(lower, diff, torch.full_like(diff, -1e30)).exp()
    strict = torch.tril(k_beta @ k.transpose(-1, -2) * decay, -1)
    inv = unit_lower_inverse(strict, masks)
    u = inv @ v_beta                                   # "new values"
    w = inv @ (k_beta * cum.exp()[..., None])          # decayed keys reading the old state
    attn = (q @ k.transpose(-1, -2)) * decay           # intra-chunk, lower triangular incl. diagonal
    q_dec = q * cum.exp()[..., None]
    k_dec = k * (cum[..., -1:] - cum).exp()[..., None]
    chunk_decay = cum[..., -1].exp()[..., None, None]  # [bh, n, 1, 1]
    outs = []
    for i in range(n):  # static unroll over chunks (n = bucket / CHUNK)
        v_new = u[:, i] - w[:, i] @ state
        outs.append(q_dec[:, i] @ state + attn[:, i] @ v_new)
        state = state * chunk_decay[:, i] + k_dec[:, i].transpose(-1, -2) @ v_new
    return torch.cat(outs, 1).reshape(b, h, t, dv), state.reshape(b, h, dk, dv)


def recurrent_gated_delta(q, k, v, g, beta, state):
    """One decode step. q,k [B,H,Dk]; v [B,H,Dv]; g,beta [B,H]; state [B,H,Dk,Dv]. fp32."""
    state = state * g.exp()[..., None, None]
    kv_mem = (state * k[..., None]).sum(-2)
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None] * delta[..., None, :]
    return (state * q[..., None]).sum(-2), state


# --------------------------------------------------------------------------------------------------- model
class StaticQwen35(nn.Module):
    def __init__(self, cfg: Cfg, max_len: int, chunk: int = CHUNK):
        super().__init__()
        self.cfg, self.max_len, self.chunk = cfg, max_len, chunk
        self.register_buffer("masks", _offdiag_masks(chunk), persistent=False)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rot_dim, 2, dtype=torch.float) / cfg.rot_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.full_idx = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
        self.lin_idx = [i for i, t in enumerate(cfg.layer_types) if t != "full_attention"]
        self.w = {}  # name -> tensor, filled by load(); kept as a dict so tracing sees constants/params uniformly

    # ---- weights
    @classmethod
    def from_pretrained(cls, path, max_len, dtype=torch.float32, chunk=CHUNK):
        cfg = Cfg.from_hf(path)
        m = cls(cfg, max_len, chunk)
        params = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            from safetensors import safe_open
            with safe_open(f, "pt") as fh:
                for key in fh.keys():
                    name = key
                    for p in ("model.language_model.", "model."):
                        if name.startswith(p):
                            name = name[len(p):]
                            break
                    if name.startswith(("visual", "mtp")) or key.startswith(("mtp", "model.visual")):
                        continue
                    t = fh.get_tensor(key)
                    keep_fp32 = name.endswith(("A_log", "dt_bias")) or "norm" in name
                    params[name.replace(".", "__")] = nn.Parameter(t.to(torch.float32 if keep_fp32 else dtype), requires_grad=False)
        if "lm_head__weight" not in params:
            params["lm_head__weight"] = params["embed_tokens__weight"]
        m.p = nn.ParameterDict(params)
        return m

    def W(self, name):
        return self.p[name.replace(".", "__")]

    def mm(self, x, name):
        w = self.W(name)
        return F.linear(x.to(w.dtype), w)

    # ---- state
    def empty_state(self, b, device="cpu", dtype=torch.float32):
        c = self.cfg
        st = []
        for _ in self.full_idx:
            st += [torch.zeros(b, c.kv_heads, self.max_len, c.head_dim, dtype=dtype, device=device)] * 2
        for _ in self.lin_idx:
            st += [torch.zeros(b, c.conv_dim, c.conv_k - 1, dtype=dtype, device=device),
                   torch.zeros(b, c.v_heads, c.k_dim, c.v_dim, dtype=torch.float32, device=device)]
        return tuple(st)

    def _rope_tables(self, pos):  # pos [B, T] -> cos/sin [B, T, rot] (text-only mrope == plain rope)
        f = pos[..., None].float() * self.inv_freq
        f = torch.cat((f, f), -1)
        return f.cos(), f.sin()

    # ---- blocks
    def _mlp(self, x, i):
        pre = f"layers.{i}.mlp."
        return self.mm(F.silu(self.mm(x, pre + "gate_proj.weight")) * self.mm(x, pre + "up_proj.weight"), pre + "down_proj.weight")

    def _qkv_attn(self, x, i, cos, sin):
        c = self.cfg
        b, t, _ = x.shape
        pre = f"layers.{i}.self_attn."
        qg = self.mm(x, pre + "q_proj.weight").view(b, t, c.heads, 2 * c.head_dim)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:].reshape(b, t, -1)
        q = rms_norm(q, self.W(pre + "q_norm.weight"), c.eps).transpose(1, 2)
        k = rms_norm(self.mm(x, pre + "k_proj.weight").view(b, t, c.kv_heads, c.head_dim), self.W(pre + "k_norm.weight"), c.eps).transpose(1, 2)
        v = self.mm(x, pre + "v_proj.weight").view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        return rope(q, cos, sin, c.rot_dim), rope(k, cos, sin, c.rot_dim), v, gate

    def _attend(self, q, k, v, mask, gate, i):  # mask [B, 1, Tq, Tk] additive
        c = self.cfg
        rep = c.heads // c.kv_heads
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        s = (q @ k.transpose(-1, -2)).float() * c.head_dim ** -0.5 + mask
        o = (s.softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        return self.mm(o * torch.sigmoid(gate), f"layers.{i}.self_attn.o_proj.weight")

    def _gdn_inputs(self, x, i):
        c = self.cfg
        pre = f"layers.{i}.linear_attn."
        qkv = self.mm(x, pre + "in_proj_qkv.weight")                       # [B, T, conv_dim]
        z = self.mm(x, pre + "in_proj_z.weight")
        beta = self.mm(x, pre + "in_proj_b.weight").float().sigmoid()
        g = -self.W(pre + "A_log").exp() * F.softplus(self.mm(x, pre + "in_proj_a.weight").float() + self.W(pre + "dt_bias"))
        return qkv, z, beta, g

    def _gdn_split(self, mixed, b, t):
        c = self.cfg
        kd = c.k_heads * c.k_dim
        q = mixed[..., :kd].reshape(b, t, c.k_heads, c.k_dim)
        k = mixed[..., kd:2 * kd].reshape(b, t, c.k_heads, c.k_dim)
        v = mixed[..., 2 * kd:].reshape(b, t, c.v_heads, c.v_dim)
        rep = c.v_heads // c.k_heads
        if rep > 1:
            q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
        q, k = l2norm(q.float()), l2norm(k.float())
        return q * c.k_dim ** -0.5, k, v.float()

    def _gdn_out(self, core, z, i, b, t):
        c = self.cfg
        pre = f"layers.{i}.linear_attn."
        core = core.to(z.dtype).reshape(-1, c.v_dim)
        y = gated_rms_norm(core, z.reshape(-1, c.v_dim), self.W(pre + "norm.weight"), c.eps).reshape(b, t, -1)
        return self.mm(y, pre + "out_proj.weight")

    # ---- phases
    def prefill(self, input_ids, lengths):
        c = self.cfg
        b, t = input_ids.shape
        assert t % self.chunk == 0 and t <= self.max_len
        dev = input_ids.device
        pos = torch.arange(t, device=dev).expand(b, t)
        valid = (pos < lengths[:, None])                                   # [B, T]
        cos, sin = self._rope_tables(pos)
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=dev))
        mask = torch.where(causal, 0.0, -1e30)[None, None]
        x = self.W("embed_tokens.weight")[input_ids]
        wdt = x.dtype
        st = []
        kpad = (0, 0, 0, self.max_len - t)
        # gather index for the final conv state: last K-1 real inputs (left zero-padded for short prompts)
        conv_idx = (lengths[:, None] + torch.arange(c.conv_k - 1, device=dev))  # into [K-1 zeros | T inputs]
        for i in range(c.layers):
            h = rms_norm(x, self.W(f"layers.{i}.input_layernorm.weight"), c.eps)
            if i in self.full_idx:
                q, k, v, gate = self._qkv_attn(h, i, cos, sin)
                h = self._attend(q, k, v, mask, gate, i)
                st += [F.pad(k, kpad), F.pad(v, kpad)]
            else:
                pre = f"layers.{i}.linear_attn."
                qkv, z, beta, g = self._gdn_inputs(h, i)
                qkv = qkv * valid[..., None].to(qkv.dtype)
                xc = F.pad(qkv.transpose(1, 2), (c.conv_k - 1, 0))           # [B, conv_dim, K-1+T]
                conv_state = torch.gather(xc, 2, conv_idx[:, None, :].expand(b, c.conv_dim, -1))
                mixed = F.silu(F.conv1d(xc, self.W(pre + "conv1d.weight").to(xc.dtype), groups=c.conv_dim)).transpose(1, 2)
                q, k, v = self._gdn_split(mixed, b, t)
                beta = beta * valid[..., None]                                      # padded tokens: no write ...
                g = g * valid[..., None]                                            # ... and no decay
                s0 = torch.zeros(b, c.v_heads, c.k_dim, c.v_dim, device=dev)
                core, s = chunk_gated_delta(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                            g.transpose(1, 2), beta.transpose(1, 2), s0, self.masks)
                h = self._gdn_out(core.transpose(1, 2), z, i, b, t)
                st += [conv_state, s]
            x = x + h.to(wdt)
            x = x + self._mlp(rms_norm(x, self.W(f"layers.{i}.post_attention_layernorm.weight"), c.eps), i).to(wdt)
        last = torch.gather(x, 1, (lengths - 1)[:, None, None].expand(b, 1, c.hidden))[:, 0]
        logits = self.mm(rms_norm(last, self.W("norm.weight"), c.eps), "lm_head.weight").float()
        return logits, self._order(st)

    def _order(self, st):
        # prefill appends per layer in model order; re-group to [kv pairs..., (conv, rec) pairs...]
        kv, lin, j = [], [], 0
        for i in range(self.cfg.layers):
            if i in self.full_idx:
                kv += st[j:j + 2]
            else:
                lin += st[j:j + 2]
            j += 2
        return tuple(kv + lin)

    def decode(self, token, positions, state):
        c = self.cfg
        b = token.shape[0]
        dev = token.device
        cos, sin = self._rope_tables(positions[:, None])
        slots = torch.arange(self.max_len, device=dev)
        onehot = (slots[None, :] == positions[:, None])                    # [B, MAX]
        mask = torch.where(slots[None, :] <= positions[:, None], 0.0, -1e30)[:, None, None, :]
        oh = onehot[:, None, :, None]
        x = self.W("embed_tokens.weight")[token][:, None]
        wdt = x.dtype
        nkv = 2 * len(self.full_idx)
        kv, lin = list(state[:nkv]), list(state[nkv:])
        fi = li = 0
        for i in range(c.layers):
            h = rms_norm(x, self.W(f"layers.{i}.input_layernorm.weight"), c.eps)
            if i in self.full_idx:
                q, k, v, gate = self._qkv_attn(h, i, cos, sin)
                kc = torch.where(oh, k.to(kv[2 * fi].dtype), kv[2 * fi])
                vc = torch.where(oh, v.to(kv[2 * fi + 1].dtype), kv[2 * fi + 1])
                kv[2 * fi], kv[2 * fi + 1] = kc, vc
                h = self._attend(q, kc, vc, mask, gate, i)
                fi += 1
            else:
                pre = f"layers.{i}.linear_attn."
                qkv, z, beta, g = self._gdn_inputs(h, i)
                window = torch.cat((lin[2 * li], qkv.transpose(1, 2).to(lin[2 * li].dtype)), -1)   # [B, conv_dim, K]
                lin[2 * li] = window[..., 1:]
                w = self.W(pre + "conv1d.weight")[:, 0].to(window.dtype)      # [conv_dim, K]
                mixed = F.silu((window * w).sum(-1))[:, None]                 # [B, 1, conv_dim]
                q, k, v = self._gdn_split(mixed, b, 1)
                core, s = recurrent_gated_delta(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], lin[2 * li + 1])
                lin[2 * li + 1] = s
                h = self._gdn_out(core[:, None], z, i, b, 1)
                li += 1
            x = x + h.to(wdt)
            x = x + self._mlp(rms_norm(x, self.W(f"layers.{i}.post_attention_layernorm.weight"), c.eps), i).to(wdt)
        logits = self.mm(rms_norm(x[:, 0], self.W("norm.weight"), c.eps), "lm_head.weight").float()
        return logits, tuple(kv + lin)
