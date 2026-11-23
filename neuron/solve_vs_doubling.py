"""Follow-up to run_inf2.py: on SDK 2.28 torch-neuronx, torch.linalg.solve_triangular *does* compile (the negative
control passed). This measures what the two formulations cost and how accurate they are on the device.

  A. HLO: which op reaches neuronx-cc for solve_triangular (is a triangular-solve op present, or was it expanded?)
  B. kernel level, on the hardest real unit-lower systems of the checkpoint: solve_triangular vs the doubling inverse vs
     the transformers export fallback (63-step row loop): compile time, warm device latency, error vs fp64
  C. whole model: prefill [2x256] compiled with solve_triangular in place of the doubling inverse: compile time, warm
     latency, KL vs the fp32 CPU reference (compare with run_inf2.py results for the doubling version)

usage: python neuron/solve_vs_doubling.py CKPT OUT [--bucket 256] [--skip model]
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import static_qwen as sq  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out")
ap.add_argument("--bucket", type=int, default=256)
ap.add_argument("--cc", default="--auto-cast=none --model-type=transformer")
ap.add_argument("--skip", default="")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
skip = set(filter(None, args.skip.split(",")))
CC = args.cc.split()
C = sq.CHUNK
R = {"args": vars(args), "started": time.strftime("%FT%TZ", time.gmtime())}


def save():
    json.dump(R, open(os.path.join(args.out, "solve_vs_doubling.json"), "w"), indent=1, default=str)


import torch_neuronx  # noqa: E402

R["torch_neuronx"] = torch_neuronx.__version__


def trace(mod, inp, name):
    t0 = time.time()
    m = torch_neuronx.trace(mod, inp, compiler_args=CC, compiler_workdir=os.path.join(args.out, "neuronx-cc", name))
    return m, time.time() - t0


def timed(m, inp, warm=3, reps=20):
    for _ in range(warm):
        out = m(*inp)
    t0 = time.time()
    for _ in range(reps):
        out = m(*inp)
    return out, (time.time() - t0) / reps * 1e3


def solve_inverse(strict, masks=None):
    eye = torch.eye(strict.shape[-1], dtype=strict.dtype).expand_as(strict)
    return torch.linalg.solve_triangular(eye + strict, eye, upper=False, unitriangular=True)


def rowloop_inverse(strict, masks=None):
    """transformers' torch fallback (forward substitution on -L, one row at a time, in-place slices)."""
    a = -strict.clone()
    for i in range(1, a.shape[-1]):
        row = a[..., i, :i].clone()
        sub = a[..., :i, :i].clone()
        a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return a + torch.eye(a.shape[-1], dtype=a.dtype)


class Fn(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.register_buffer("masks", sq._offdiag_masks(C))

    def forward(self, L):
        return self.f(L, self.masks)


# ---------------------------------------------------------------- A. HLO of solve_triangular through torch-xla
try:
    import torch_xla.core.xla_model as xm  # noqa: E402
    import torch_xla  # noqa: E402
    dev = xm.xla_device()
    L = torch.tril(torch.randn(4, C, C) * 0.1, -1).to(dev)
    hlo = {}
    for name, f in (("solve_triangular", solve_inverse), ("doubling", sq.unit_lower_inverse), ("rowloop", rowloop_inverse)):
        out = f(L, sq._offdiag_masks(C).to(dev))
        txt = torch_xla._XLAC._get_xla_tensors_hlo([out])
        ops = sorted({ln.split("=")[1].strip().split("(")[0].split(" ")[-1] for ln in txt.splitlines() if " = " in ln and "(" in ln})
        hlo[name] = {"ops": ops, "has_triangular_solve": "triangular-solve" in txt, "lines": len(txt.splitlines())}
        open(os.path.join(args.out, f"hlo_{name}.txt"), "w").write(txt)
    R["hlo"] = hlo
except Exception as e:  # noqa: BLE001
    R["hlo"] = {"error": "".join(traceback.format_exception_only(type(e), e))[-800:]}
print(json.dumps(R["hlo"], indent=1), flush=True)
save()

# ---------------------------------------------------------------- real systems
from tokenizers import Tokenizer  # noqa: E402

tk = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))
prompts = ["The capital of France is",
           "Explain, step by step, why the sky looks blue during the day but red at sunset. "
           "Use the physics of Rayleigh scattering and mention the wavelength dependence explicitly. " * 5]
ids = [torch.tensor(tk.encode(p, add_special_tokens=False).ids) for p in prompts]
lengths = torch.tensor([len(x) for x in ids])
B, T = len(ids), args.bucket
batch = torch.full((B, T), tk.token_to_id("<|endoftext|>"), dtype=torch.long)
for i, x in enumerate(ids):
    batch[i, :len(x)] = x
captured = []
_orig = sq.unit_lower_inverse


def _cap(strict, masks):
    captured.append(strict.detach().clone())
    return _orig(strict, masks)


ref = sq.StaticQwen35.from_pretrained(args.ckpt, max_len=T + 64).eval()
sq.unit_lower_inverse = _cap
with torch.no_grad():
    ref_logits, _ = ref.prefill(batch, lengths)
sq.unit_lower_inverse = _orig
Ls = torch.cat([x.reshape(-1, C, C) for x in captured])
Ls = Ls[Ls.flatten(1).abs().amax(1) > 0]
N = 256
order = Ls.norm(dim=(-1, -2)).argsort(descending=True)
sel = Ls[order[:N]].contiguous()          # traced with these
sel2 = Ls[order[N:2 * N]].contiguous()    # and evaluated also on these (disjoint): catches outputs frozen at trace time
eye64 = torch.eye(C, dtype=torch.float64).expand(N, C, C)


def rel_err(out, x):
    gold = torch.linalg.solve_triangular(eye64 + x.double(), eye64, upper=False, unitriangular=True)
    scale = gold.abs().amax(dim=(-1, -2)).clamp_min(1)
    return (out.double() - gold).abs().amax(dim=(-1, -2)) / scale

# ---------------------------------------------------------------- B. kernel level
R["kernel"] = {"systems": N, "L_fro_max": float(sel.norm(dim=(-1, -2)).max())}
for name, f in (("doubling", sq.unit_lower_inverse), ("solve_triangular", solve_inverse), ("rowloop", rowloop_inverse)):
    print(f"\nB. {name}", flush=True)
    try:
        m, ct = trace(Fn(f), (sel,), f"k_{name}")
        out, ms = timed(m, (sel,))
        err = rel_err(out, sel)
        err2 = rel_err(m(sel2), sel2)
        R["kernel"][name] = {"compiled": True, "compile_s": ct, "warm_ms_per_call": ms,
                             "rel_err_max": float(err.max()), "rel_err_median": float(err.median()),
                             "unseen_inputs_rel_err_max": float(err2.max()), "unseen_inputs_rel_err_median": float(err2.median()),
                             "neff_bytes": (os.path.getsize(nf) if os.path.exists(nf := os.path.join(args.out, "neuronx-cc", f"k_{name}", "graph.neff")) else None)}
    except Exception as e:  # noqa: BLE001
        R["kernel"][name] = {"compiled": False, "error": "".join(traceback.format_exception_only(type(e), e))[-1200:]}
    print(json.dumps(R["kernel"][name], indent=1), flush=True)
    save()


# ---------------------------------------------------------------- C. whole-model prefill with solve_triangular
def kl(a, b):
    pa, pb = b.float().log_softmax(-1), a.float().log_softmax(-1)
    return (pa.exp() * (pa - pb)).sum(-1)


class Prefill(torch.nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, ids_, lens):
        return s.m.prefill(ids_, lens)[0]   # logits only: isolates the prefill compute


prompts2 = ["Write a haiku about the ocean at night.",
            "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 6]
ids2 = [torch.tensor(tk.encode(p, add_special_tokens=False).ids) for p in prompts2]
lengths2 = torch.tensor([len(x) for x in ids2])
batch2 = torch.full((B, T), tk.token_to_id("<|endoftext|>"), dtype=torch.long)
for i, x in enumerate(ids2):
    batch2[i, :len(x)] = x
with torch.no_grad():
    ref_logits2, _ = ref.prefill(batch2, lengths2)
R["prompts2_lengths"] = lengths2.tolist()

if "model" not in skip:
    model = sq.StaticQwen35.from_pretrained(args.ckpt, max_len=T + 64, dtype=torch.bfloat16).eval()
    R["prefill"] = {}
    for name, f in (("doubling", _orig), ("solve_triangular", solve_inverse)):
        print(f"\nC. whole-model prefill with {name}", flush=True)
        sq.unit_lower_inverse = f
        try:
            m, ct = trace(Prefill(model), (batch, lengths), f"prefill_{name}")
            out, ms = timed(m, (batch, lengths), warm=2, reps=10)
            k = kl(out, ref_logits)
            out2 = m(batch2, lengths2)
            k2 = kl(out2, ref_logits2)
            R["prefill"][name] = {"compiled": True, "compile_s": ct, "warm_ms": ms, "kl": k.tolist(),
                                  "argmax_eq": (out.argmax(-1) == ref_logits.argmax(-1)).tolist(),
                                  "unseen_batch_kl": k2.tolist(),
                                  "unseen_batch_argmax_eq": (out2.argmax(-1) == ref_logits2.argmax(-1)).tolist()}
        except Exception as e:  # noqa: BLE001
            R["prefill"][name] = {"compiled": False, "error": "".join(traceback.format_exception_only(type(e), e))[-1200:]}
        sq.unit_lower_inverse = _orig
        print(json.dumps(R["prefill"][name], indent=1), flush=True)
        save()

R["finished"] = time.strftime("%FT%TZ", time.gmtime())
save()
print("DONE")
