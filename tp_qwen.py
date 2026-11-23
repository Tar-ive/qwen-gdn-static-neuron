"""Tensor-parallel version of the static Qwen3.5 / Qwen3.8 graphs (Megatron-style, weights sliced by head).

Each rank holds a slice of every projection, chosen so that all per-head computation stays local:
  full attention   q_proj rows of query heads [r*hq, (r+1)*hq), k/v_proj rows of kv head (r*hq)//rep_q
                   (kv heads are replicated when tp > kv_heads), o_proj the matching columns  -> all-reduce
  Gated DeltaNet   in_proj_qkv / conv1d rows of key heads [r*hk, ...) for q and k and of value heads [r*hv, ...) for v,
                   in_proj_z/b/a, A_log, dt_bias the value-head rows, out_proj the matching columns  -> all-reduce
  MLP              gate/up rows, down columns  -> all-reduce
  lm_head          vocab rows  -> all-gather of the logits
  embeddings and norms are replicated.
The recurrent and conv state of a GDN layer and the KV cache of an attention layer are therefore rank-local; the only
collectives are three all-reduces per layer pair and one all-gather.

For a fixed interface the state is packed into three tensors (rank <= 4):
  kv   [2 * n_full * B, MAX_LEN, head_dim]          (local kv heads == 1)
  conv [n_lin * B, conv_dim_local, K - 1]
  rec  [n_lin * B * v_heads_local, k_dim, v_dim]    fp32

The collectives are injectable (`set_collectives`) so the same code runs under NxD on device and under a threaded CPU
emulator in tests.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

import static_qwen as sq

_COLL = {}


def set_collectives(all_reduce, all_gather_last):
    _COLL["ar"], _COLL["ag"] = all_reduce, all_gather_last


def use_nxd_collectives():
    from neuronx_distributed.parallel_layers.mappings import (gather_from_tensor_model_parallel_region,
                                                              reduce_from_tensor_model_parallel_region)
    set_collectives(reduce_from_tensor_model_parallel_region, gather_from_tensor_model_parallel_region)


def local_cfg(cfg: sq.Cfg, tp: int) -> sq.Cfg:
    assert cfg.heads % tp == 0 and cfg.k_heads % tp == 0 and cfg.v_heads % tp == 0 and cfg.inter % tp == 0
    assert cfg.vocab % tp == 0
    assert tp % cfg.kv_heads == 0 or cfg.kv_heads % tp == 0
    return replace(cfg, heads=cfg.heads // tp, kv_heads=max(1, cfg.kv_heads // tp), k_heads=cfg.k_heads // tp,
                   v_heads=cfg.v_heads // tp, inter=cfg.inter // tp)


# ------------------------------------------------------------------------------------------- weight slicing
def shard_plan(name: str, cfg: sq.Cfg, tp: int, r: int):
    """How rank r slices full tensor `name` (HF names without the model prefix): list of (dim, start, stop) pieces
    concatenated along dim, or None for a replicated tensor."""
    hd = cfg.head_dim
    if ".self_attn." in name:
        hq = cfg.heads // tp
        if name.endswith("q_proj.weight"):
            return [(0, r * hq * 2 * hd, (r + 1) * hq * 2 * hd)]           # [q_h; gate_h] rows per head
        if name.endswith(("k_proj.weight", "v_proj.weight")):
            kvl = max(1, cfg.kv_heads // tp)
            first = (r * hq) // (cfg.heads // cfg.kv_heads)                 # kv head of this rank's first q head
            return [(0, first * hd, (first + kvl) * hd)]
        if name.endswith("o_proj.weight"):
            return [(1, r * hq * hd, (r + 1) * hq * hd)]
        return None                                                         # q_norm, k_norm
    if ".linear_attn." in name:
        hk, hv = cfg.k_heads // tp, cfg.v_heads // tp
        kd, vd = cfg.k_dim, cfg.v_dim
        K = cfg.k_heads * kd
        qkv = [(0, r * hk * kd, (r + 1) * hk * kd), (0, K + r * hk * kd, K + (r + 1) * hk * kd),
               (0, 2 * K + r * hv * vd, 2 * K + (r + 1) * hv * vd)]
        if name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
            return qkv
        if name.endswith("in_proj_z.weight"):
            return [(0, r * hv * vd, (r + 1) * hv * vd)]
        if name.endswith(("in_proj_b.weight", "in_proj_a.weight", "A_log", "dt_bias")):
            return [(0, r * hv, (r + 1) * hv)]
        if name.endswith("out_proj.weight"):
            return [(1, r * hv * vd, (r + 1) * hv * vd)]
        return None                                                         # gated norm weight
    if ".mlp." in name:
        n = cfg.inter // tp
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            return [(0, r * n, (r + 1) * n)]
        if name.endswith("down_proj.weight"):
            return [(1, r * n, (r + 1) * n)]
    if name == "lm_head.weight":
        n = cfg.vocab // tp
        return [(0, r * n, (r + 1) * n)]
    return None                                                             # embed_tokens, norms


def _take(get, shape, pieces):
    """get(index_tuple) reads a sub-tensor (safetensors slice or a tensor)."""
    if pieces is None:
        return get(tuple(slice(None) for _ in shape))
    parts = []
    for dim, a, b in pieces:
        idx = [slice(None)] * len(shape)
        idx[dim] = slice(a, b)
        parts.append(get(tuple(idx)))
    return torch.cat(parts, pieces[0][0]) if len(parts) > 1 else parts[0]


def _hf_names(path):
    """name -> (file, key) for the text decoder, with the language-model prefix stripped."""
    out = {}
    from safetensors import safe_open
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, "pt") as fh:
            for key in fh.keys():
                if key.startswith(("mtp", "model.visual", "visual")):
                    continue
                name = key
                for p in ("model.language_model.", "model."):
                    if name.startswith(p):
                        name = name[len(p):]
                        break
                if name.startswith(("visual", "mtp")):
                    continue
                out[name] = (f, key)
    return out


def load_rank_weights(path, cfg, tp, r, dtype=torch.bfloat16, prefix=""):
    """State dict for rank r, keyed like StaticQwen35TP's parameters (with `prefix`)."""
    from safetensors import safe_open
    names = _hf_names(path)
    if "lm_head.weight" not in names:
        names["lm_head.weight"] = names["embed_tokens.weight"]
    by_file = {}
    for n, (f, k) in names.items():
        by_file.setdefault(f, []).append((n, k))
    out = {}
    for f, items in by_file.items():
        with safe_open(f, "pt") as fh:
            for n, k in items:
                sl = fh.get_slice(k)
                shape = sl.get_shape()
                t = _take(lambda idx: sl[idx], shape, shard_plan(n, cfg, tp, r))
                keep_fp32 = n.endswith(("A_log", "dt_bias")) or "norm" in n
                out[prefix + "p." + n.replace(".", "__")] = t.to(torch.float32 if keep_fp32 else dtype).contiguous()
    return out


# ------------------------------------------------------------------------------------------------- model
class StaticQwen35TP(sq.StaticQwen35):
    """Rank-local static model. Same phases as StaticQwen35, plus packed-state wrappers."""

    def __init__(self, full_cfg: sq.Cfg, tp: int, max_len: int, chunk: int = sq.CHUNK):
        super().__init__(local_cfg(full_cfg, tp), max_len, chunk)
        self.full_cfg, self.tp = full_cfg, tp

    @classmethod
    def empty(cls, path, tp, max_len, dtype=torch.bfloat16):
        """Rank-shaped model with zero weights (for tracing; weights are supplied per rank at load time)."""
        full = sq.Cfg.from_hf(path)
        m = cls(full, tp, max_len)
        names = _hf_names(path)
        if "lm_head.weight" not in names:
            names["lm_head.weight"] = names["embed_tokens.weight"]
        from safetensors import safe_open
        params = {}
        for n, (f, k) in names.items():
            with safe_open(f, "pt") as fh:
                shape = fh.get_slice(k).get_shape()
            pieces = shard_plan(n, full, tp, 0)
            shp = list(shape)
            if pieces is not None:
                shp[pieces[0][0]] = sum(b - a for _, a, b in pieces)
            keep_fp32 = n.endswith(("A_log", "dt_bias")) or "norm" in n
            params[n.replace(".", "__")] = nn.Parameter(torch.zeros(shp, dtype=torch.float32 if keep_fp32 else dtype),
                                                        requires_grad=False)
        m.p = nn.ParameterDict(params)
        return m

    @classmethod
    def from_full(cls, path, tp, r, max_len, dtype=torch.float32):
        """Rank r with real weights (CPU tests)."""
        m = cls.empty(path, tp, max_len, dtype)
        sd = load_rank_weights(path, m.full_cfg, tp, r, dtype)
        with torch.no_grad():
            for k, v in sd.items():
                m.p[k[2:]].copy_(v)
        return m

    def mm(self, x, name):
        y = super().mm(x, name)
        if name.endswith(("o_proj.weight", "out_proj.weight", "down_proj.weight")):
            y = _COLL["ar"](y)
        elif name == "lm_head.weight":
            y = _COLL["ag"](y)
        return y

    # ---- packed state
    def pack(self, st):
        b = st[0].shape[0]
        nkv = 2 * len(self.full_idx)
        kv = torch.cat([t.reshape(b, self.max_len, -1) for t in st[:nkv]], 0)          # local kv heads == 1
        conv = torch.cat(list(st[nkv::2]), 0)
        rec = torch.cat([t.reshape(-1, self.cfg.k_dim, self.cfg.v_dim) for t in st[nkv + 1::2]], 0)
        return kv, conv, rec

    def unpack(self, kv, conv, rec, b):
        c = self.cfg
        st = [t.reshape(b, c.kv_heads, self.max_len, c.head_dim) for t in kv.split(b, 0)]
        convs = list(conv.split(b, 0))
        recs = [t.reshape(b, c.v_heads, c.k_dim, c.v_dim) for t in rec.split(b * c.v_heads, 0)]
        for cv, rc in zip(convs, recs):
            st += [cv, rc]
        return tuple(st)

    def prefill_packed(self, ids, lengths):
        logits, st = self.prefill(ids, lengths)
        return (logits, *self.pack(st))

    def decode_packed(self, tok, pos, kv, conv, rec):
        logits, st = self.decode(tok, pos, self.unpack(kv, conv, rec, tok.shape[0]))
        return (logits, *self.pack(st))


class Phase(nn.Module):
    """Both phases with one signature, forward(x, n), and the state held in buffers updated in place, so that NxD keeps
    it resident on the device (aliased INPUT_STATE) between calls and shares it between prefill and decode by name:
        prefill: x = ids[B, T], n = lengths[B]   -> writes kv/conv/rec, returns logits
        decode:  x = tok[B],    n = positions[B] -> reads and updates kv/conv/rec, returns logits
    Buffers are named per batch size (kv_b{B}, ...) so graphs for several batch sizes can coexist."""

    def __init__(self, m, batch):
        super().__init__()
        self.m, self.b = m, batch
        kv, conv, rec = m.pack(m.empty_state(batch, dtype=m.W("embed_tokens.weight").dtype))
        self.register_buffer(f"kv_b{batch}", kv)
        self.register_buffer(f"conv_b{batch}", conv)
        self.register_buffer(f"rec_b{batch}", rec)

    def state(self):
        return getattr(self, f"kv_b{self.b}"), getattr(self, f"conv_b{self.b}"), getattr(self, f"rec_b{self.b}")

    def forward(self, x, n):
        kv, conv, rec = self.state()
        if x.dim() == 2:
            out = self.m.prefill_packed(x, n)
        else:
            out = self.m.decode_packed(x, n, kv, conv, rec)
        kv.copy_(out[1])
        conv.copy_(out[2])
        rec.copy_(out[3])
        return out[0]
