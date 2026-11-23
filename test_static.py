"""CPU checks for static_qwen.py against transformers' Qwen3.5 reference, on a real checkpoint.

  1. unit_lower_inverse vs fp64 solve_triangular on the real UT systems of every GDN layer (and the Neumann product
     for contrast);
  2. prefill logits for a mixed-length, right-padded batch vs HF run unpadded per prompt;
  3. greedy decode tokens/logits vs HF generate with its cache;
  4. torch.export of both phases: no triangular_solve, no data-dependent ops, fixed shapes.

usage: python test_static.py [ckpt_dir] [--steps N]
"""
import argparse
import json
import sys

import torch
import torch.nn.functional as F

import static_qwen as sq

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="ckpt/Qwen3.5-0.8B")
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--bucket", type=int, default=256)
ap.add_argument("--json", default=None)
args = ap.parse_args()
torch.manual_seed(0)
report = {"ckpt": args.ckpt}

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.ckpt)
prompts = ["The capital of France is",
           "Explain, step by step, why the sky looks blue during the day but red at sunset. "
           "Use the physics of Rayleigh scattering and mention the wavelength dependence explicitly. " * 5]
ids = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]
lengths = torch.tensor([len(x) for x in ids])
B, T = len(ids), args.bucket
assert lengths.max() + args.steps <= T * 2
batch = torch.full((B, T), tok.pad_token_id or 0, dtype=torch.long)
for i, x in enumerate(ids):
    batch[i, :len(x)] = x
print("prompt lengths", lengths.tolist(), "bucket", T)

# ------------------------------------------------------------------ 1. inverse on real UT systems
captured = []
orig = sq.unit_lower_inverse


def capture(strict, masks):
    captured.append(strict.detach().clone())
    return orig(strict, masks)


sq.unit_lower_inverse = capture
model = sq.StaticQwen35.from_pretrained(args.ckpt, max_len=2 * T).eval()
with torch.no_grad():
    logits_s, state = model.prefill(batch, lengths)
sq.unit_lower_inverse = orig


def neumann(strict):  # (I + L)^-1 = (I - L)(I + L^2)(I + L^4)... for nilpotent L
    c = strict.shape[-1]
    eye = torch.eye(c, dtype=strict.dtype)
    out, p = eye - strict, strict @ strict
    k = 2
    while k < c:
        out = out @ (eye + p)
        p = p @ p
        k *= 2
    return out


inv_rows = []
for li, L in enumerate(captured):
    T64 = torch.eye(L.shape[-1], dtype=torch.float64) + L.double()
    ref = torch.linalg.solve_triangular(T64, torch.eye(L.shape[-1], dtype=torch.float64).expand_as(T64), upper=False, unitriangular=True)
    ours = orig(L, model.masks).double()
    neu = neumann(L).double()
    scale = ref.abs().amax(dim=(-1, -2)).clamp_min(1)
    e_ours = ((ours - ref).abs().amax(dim=(-1, -2)) / scale).max().item()
    e_neu = ((neu - ref).abs().amax(dim=(-1, -2)) / scale).max().item()
    fro = L.norm(dim=(-1, -2))
    inv_rows.append(dict(layer=model.lin_idx[li], L_fro_max=fro.max().item(), L_fro_median=fro.median().item(),
                         inv_max=ref.abs().max().item(), err_ours=e_ours, err_neumann=e_neu))
print("\n1. inverse (fp32 vs fp64 solve_triangular, max relative entry error)")
print(f"{'layer':>5} {'|L|F max':>9} {'|L|F med':>9} {'max|inv|':>10} {'doubling':>10} {'neumann':>10}")
for r in inv_rows:
    print(f"{r['layer']:5d} {r['L_fro_max']:9.2f} {r['L_fro_median']:9.2f} {r['inv_max']:10.3g} {r['err_ours']:10.2e} {r['err_neumann']:10.2e}")
report["inverse"] = inv_rows

# ------------------------------------------------------------------ 2. prefill vs HF
_cfg = json.load(open(f"{args.ckpt}/config.json"))
_Auto = AutoModelForImageTextToText if "text_config" in _cfg else AutoModelForCausalLM
hf = _Auto.from_pretrained(args.ckpt, dtype=torch.float32, attn_implementation="eager").eval()
hf_logits, hf_gen = [], []
with torch.no_grad():
    for x in ids:
        out = hf(input_ids=x[None])
        hf_logits.append(out.logits[0, -1].float())
        g = hf.generate(input_ids=x[None], max_new_tokens=args.steps, do_sample=False, output_logits=True,
                        return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        hf_gen.append((g.sequences[0, len(x):], torch.stack([l[0] for l in g.logits]).float()))


def cmp(a, b):
    pa, pb = a.log_softmax(-1), b.log_softmax(-1)
    return dict(max_abs=(a - b).abs().max().item(), kl=F.kl_div(pa, pb, log_target=True, reduction="sum").item(),
                argmax_eq=bool(a.argmax() == b.argmax()))


print("\n2. prefill last-token logits vs HF (unpadded)")
report["prefill"] = []
for i in range(B):
    r = cmp(logits_s[i], hf_logits[i])
    report["prefill"].append(r)
    print(f"  prompt {i} (len {lengths[i]}): max|dlogit| {r['max_abs']:.2e}  KL {r['kl']:.2e}  argmax equal {r['argmax_eq']}")

# ------------------------------------------------------------------ 3. greedy decode vs HF generate
print(f"\n3. greedy decode {args.steps} steps vs HF generate")
toks, logs = [logits_s.argmax(-1)], [logits_s]
pos = lengths.clone()
with torch.no_grad():
    for s in range(args.steps - 1):
        lg, state = model.decode(toks[-1], pos, state)
        toks.append(lg.argmax(-1)); logs.append(lg); pos = pos + 1
toks, logs = torch.stack(toks, 1), torch.stack(logs, 1)
report["decode"] = []
for i in range(B):
    ref_t, ref_l = hf_gen[i]
    n = min(len(ref_t), toks.shape[1])
    match = int((toks[i, :n] == ref_t[:n]).long().cumprod(0).sum())
    kl = max(cmp(logs[i, j], ref_l[j])["kl"] for j in range(n))
    report["decode"].append(dict(tokens=n, identical_prefix=match, max_step_kl=kl))
    print(f"  prompt {i}: identical greedy prefix {match}/{n}, max per-step KL {kl:.2e}")
    print("    ours:", repr(tok.decode(toks[i, :n])))
    print("    hf:  ", repr(tok.decode(ref_t[:n])))

# ------------------------------------------------------------------ 4. export: static graph, no solve
print("\n4. torch.export op scan")


class Prefill(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, ids, lens): return s.m.prefill(ids, lens)


class Decode(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, tok_, pos_, *st): return s.m.decode(tok_, pos_, st)


bad = ("triangular", "solve", "nonzero", "item", "_local_scalar_dense", "masked_select", "unique", "linalg")
report["export"] = {}
with torch.no_grad():
    for name, mod, inp in (("prefill", Prefill(model), (batch, lengths)),
                           ("decode", Decode(model), (toks[:, -1], pos, *state))):
        ep = torch.export.export(mod, inp, strict=False)
        ops = sorted({str(n.target) for n in ep.graph.nodes if n.op == "call_function"})
        offending = [o for o in ops if any(b in o for b in bad)]
        max_rank = max(len(n.meta["val"].shape) for n in ep.graph.nodes
                       if n.op == "call_function" and isinstance(n.meta.get("val"), torch.Tensor))
        report["export"][name] = dict(nodes=len(ep.graph.nodes), distinct_ops=len(ops), offending=offending, max_rank=max_rank)
        print(f"  {name}: {len(ep.graph.nodes)} nodes, {len(ops)} distinct ops, max tensor rank {max_rank}, offending: {offending or 'none'}")
        if name == "prefill":
            print("    ops:", ", ".join(o.replace("aten.", "") for o in ops))

if args.json:
    json.dump(report, open(args.json, "w"), indent=1)
ok = all(r["argmax_eq"] for r in report["prefill"]) and all(not v["offending"] for v in report["export"].values())
sys.exit(0 if ok else 1)
