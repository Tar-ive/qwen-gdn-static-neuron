"""On-device check of the static Qwen3.5 / Qwen3.8 graphs on AWS Inferentia2 (Neuron SDK 2.28, torch-neuronx).

Steps, each recorded in results.json:
  0. negative control: trace a module that calls torch.linalg.solve_triangular and record the compiler error;
  1. the matmul-only inverse alone, compiled for Neuron, on the real unit-lower systems captured from layer inputs of the
     checkpoint, against fp64 solve_triangular on CPU;
  2. compile `prefill` (B x bucket) and `decode` (B x 1) as whole-model graphs with torch_neuronx.trace;
  3. run greedy decoding on the device and compare with the fp32 CPU reference (the same static model, which matches
     transformers to 4e-5 in test_static.py): prefill KL, teacher-forced per-step KL and argmax agreement, free-running
     greedy prefix match;
  4. latency: prefill and decode step wall time (decode includes the host round-trip of the state tensors).

usage: python neuron/run_inf2.py CKPT_DIR OUT_DIR [--bucket 256] [--steps 32] [--dtype bf16]
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import static_qwen as sq  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out")
ap.add_argument("--bucket", type=int, default=256)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--cc", default="--auto-cast=none --model-type=transformer",
                help="neuronx-cc flags; auto-cast none keeps the fp32 GDN recurrence in fp32")
ap.add_argument("--skip", default="", help="comma list of steps to skip: control,inverse,model")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
skip = set(filter(None, args.skip.split(",")))
CC = args.cc.split()
R = {"args": vars(args), "host": platform.node(), "started": time.strftime("%FT%TZ", time.gmtime())}


def save():
    json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=1, default=str)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


import torch_neuronx  # noqa: E402

R["env"] = {"torch": torch.__version__, "torch_neuronx": torch_neuronx.__version__,
            "neuronx_cc": sh("neuronx-cc --version 2>&1 | head -3"), "neuron_ls": sh("neuron-ls 2>&1 | head -20"),
            "instance_type": sh("curl -s -m 2 -H \"X-aws-ec2-metadata-token: $(curl -s -m 2 -X PUT "
                                "http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')\" "
                                "http://169.254.169.254/latest/meta-data/instance-type")}
print(json.dumps(R["env"], indent=1), flush=True)
save()


def trace(mod, inp, name):
    wd = os.path.join(args.out, "neuronx-cc", name)
    t0 = time.time()
    m = torch_neuronx.trace(mod, inp, compiler_args=CC, compiler_workdir=wd)
    return m, time.time() - t0


# ------------------------------------------------------------------------- 0. negative control
class Solve(torch.nn.Module):
    def forward(self, L):
        eye = torch.eye(L.shape[-1]).expand_as(L)
        return torch.linalg.solve_triangular(eye + L, eye, upper=False, unitriangular=True)


class Inverse(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.register_buffer("masks", sq._offdiag_masks(c))

    def forward(self, L):
        return sq.unit_lower_inverse(L, self.masks)


C = sq.CHUNK
if "control" not in skip:
    print("\n0. negative control: torch.linalg.solve_triangular", flush=True)
    L0 = torch.tril(torch.randn(8, C, C) * 0.1, -1)
    try:
        trace(Solve(), (L0,), "control_solve")
        R["control"] = {"compiled": True}
    except Exception as e:  # noqa: BLE001
        msg = "".join(traceback.format_exception_only(type(e), e))
        logs = sh(f"grep -rhoE 'NCC_[A-Z0-9]+[^\\n]{{0,200}}' {args.out}/neuronx-cc/control_solve 2>/dev/null | sort -u | head -5")
        R["control"] = {"compiled": False, "error": msg[-1500:], "ncc_codes": logs}
    print(json.dumps(R["control"], indent=1)[:2000], flush=True)
    save()

# ------------------------------------------------------------------------- reference model + real UT systems
from tokenizers import Tokenizer  # noqa: E402  (plain tokenizers: independent of the transformers version on the AMI)


class Tok:
    def __init__(self, d):
        self.t = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.pad_token_id = self.t.token_to_id("<|endoftext|>")

    def __call__(self, text, return_tensors=None):
        class _O:
            pass
        o = _O()
        o.input_ids = torch.tensor([self.t.encode(text, add_special_tokens=False).ids])
        return o

    def decode(self, ids):
        return self.t.decode(ids.tolist())


tok = Tok(args.ckpt)
prompts = ["The capital of France is",
           "Explain, step by step, why the sky looks blue during the day but red at sunset. "
           "Use the physics of Rayleigh scattering and mention the wavelength dependence explicitly. " * 5]
ids = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]
lengths = torch.tensor([len(x) for x in ids])
B, T = len(ids), args.bucket
MAXLEN = T + args.steps + 32
MAXLEN = (MAXLEN + 63) // 64 * 64
batch = torch.full((B, T), tok.pad_token_id or 0, dtype=torch.long)
for i, x in enumerate(ids):
    batch[i, :len(x)] = x
R["prompts"] = {"lengths": lengths.tolist(), "bucket": T, "max_len": MAXLEN}

captured = []
_orig = sq.unit_lower_inverse


def _cap(strict, masks):
    captured.append(strict.detach().clone())
    return _orig(strict, masks)


print("\nCPU fp32 reference", flush=True)
t0 = time.time()
ref = sq.StaticQwen35.from_pretrained(args.ckpt, max_len=MAXLEN).eval()
sq.unit_lower_inverse = _cap
with torch.no_grad():
    ref_logits, ref_state = ref.prefill(batch, lengths)
sq.unit_lower_inverse = _orig
R["cpu_ref_prefill_s"] = time.time() - t0
save()

# ------------------------------------------------------------------------- 1. inverse alone on device
if "inverse" not in skip:
    print("\n1. matmul-only inverse on Neuron vs fp64 solve_triangular", flush=True)
    Ls = torch.cat([L.reshape(-1, C, C) for L in captured])           # all chunks, all heads, all GDN layers
    nz = Ls.flatten(1).abs().amax(1) > 0                              # drop all-padding chunks
    Ls = Ls[nz]
    N = 256
    order = Ls.norm(dim=(-1, -2)).argsort(descending=True)
    sel = Ls[order[:N]].contiguous()                                  # the hardest (largest ||L||_F) systems
    inv_mod, ct = trace(Inverse(C), (sel,), "inverse")
    for _ in range(3):
        inv_mod(sel)                                                  # warm-up: first call loads the NEFF
    t0 = time.time()
    for _ in range(10):
        out = inv_mod(sel)
    dev_s = (time.time() - t0) / 10
    eye = torch.eye(C, dtype=torch.float64).expand(N, C, C)
    gold = torch.linalg.solve_triangular(eye + sel.double(), eye, upper=False, unitriangular=True)
    scale = gold.abs().amax(dim=(-1, -2)).clamp_min(1)
    err = ((out.double() - gold).abs().amax(dim=(-1, -2)) / scale)
    cpu = _orig(sel, sq._offdiag_masks(C)).double()
    err_cpu = ((cpu - gold).abs().amax(dim=(-1, -2)) / scale)
    R["inverse"] = {"systems": N, "of_nonpadding": int(nz.sum()), "L_fro_max": float(sel.norm(dim=(-1, -2)).max()),
                    "L_fro_min_selected": float(sel.norm(dim=(-1, -2)).min()),
                    "device_rel_err_max": float(err.max()), "device_rel_err_median": float(err.median()),
                    "cpu_fp32_rel_err_max": float(err_cpu.max()), "compile_s": ct,
                    "device_call_ms_256_systems": dev_s * 1e3}
    print(json.dumps(R["inverse"], indent=1), flush=True)
    save()

if "model" in skip:
    sys.exit(0)

# ------------------------------------------------------------------------- 2. compile both phases
dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
print(f"\n2. compile prefill [{B}x{T}] and decode [{B}x1], weights {args.dtype}", flush=True)
model = sq.StaticQwen35.from_pretrained(args.ckpt, max_len=MAXLEN, dtype=dt).eval()


class Prefill(torch.nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, ids_, lens):
        return s.m.prefill(ids_, lens)


class Decode(torch.nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, tok_, pos_, *st):
        return s.m.decode(tok_, pos_, st)


with torch.no_grad():
    cpu_lp, cpu_st = model.prefill(batch, lengths)          # same dtype on CPU: gives example decode inputs
pre_mod, pre_ct = trace(Prefill(model), (batch, lengths), "prefill")
R["compile"] = {"prefill_s": pre_ct}
save()
print(f"prefill compiled in {pre_ct:.0f}s", flush=True)
dec_mod, dec_ct = trace(Decode(model), (cpu_lp.argmax(-1), lengths.clone(), *cpu_st), "decode")
R["compile"]["decode_s"] = dec_ct
R["neff_bytes"] = sh(f"find {args.out}/neuronx-cc -name '*.neff' -exec ls -l {{}} \\; | awk '{{print $5, $9}}'")
save()
print(f"decode compiled in {dec_ct:.0f}s", flush=True)
torch.jit.save(pre_mod, os.path.join(args.out, "prefill.pt"))
torch.jit.save(dec_mod, os.path.join(args.out, "decode.pt"))


# ------------------------------------------------------------------------- 3. run + compare
def kl(a, b):  # KL(ref || dev) over the vocab, per row
    pa, pb = b.float().log_softmax(-1), a.float().log_softmax(-1)
    return (pa.exp() * (pa - pb)).sum(-1)


print("\n3. greedy decode on device", flush=True)
for _ in range(2):
    d_logits, d_state = pre_mod(batch, lengths)  # warm-up
t0 = time.time()
reps = 5
for _ in range(reps):
    d_logits, d_state = pre_mod(batch, lengths)
pre_ms = (time.time() - t0) / reps * 1e3
toks, logs, step_ms = [d_logits.argmax(-1)], [d_logits], []
pos = lengths.clone()
st = d_state
for s in range(args.steps - 1):
    t0 = time.time()
    lg, st = dec_mod(toks[-1], pos, *st)
    step_ms.append((time.time() - t0) * 1e3)
    toks.append(lg.argmax(-1))
    logs.append(lg)
    pos = pos + 1
dtoks, dlogs = torch.stack(toks, 1), torch.stack(logs, 1)

# reference: teacher-forced on the device's tokens (same contexts), and free-running greedy
with torch.no_grad():
    tf_logs, st_r, p = [ref_logits], ref_state, lengths.clone()
    for s in range(args.steps - 1):
        lg, st_r = ref.decode(dtoks[:, s], p, st_r)
        tf_logs.append(lg)
        p = p + 1
    tf_logs = torch.stack(tf_logs, 1)
    fr, st_r, p = [ref_logits.argmax(-1)], ref_state, lengths.clone()
    for s in range(args.steps - 1):
        lg, st_r = ref.decode(fr[-1], p, st_r)
        fr.append(lg.argmax(-1))
        p = p + 1
    fr = torch.stack(fr, 1)

rows = []
for i in range(B):
    k = kl(dlogs[i], tf_logs[i])
    agree = (dlogs[i].argmax(-1) == tf_logs[i].argmax(-1)).float().mean().item()
    prefix = int((dtoks[i] == fr[i]).long().cumprod(0).sum())
    rows.append({"prompt_len": int(lengths[i]), "prefill_kl": float(k[0]), "max_step_kl": float(k.max()),
                 "mean_step_kl": float(k.mean()), "teacher_forced_argmax_agree": agree,
                 "greedy_identical_prefix": prefix, "steps": args.steps,
                 "prefill_max_abs_dlogit": float((dlogs[i, 0].float() - tf_logs[i, 0]).abs().max()),
                 "device_text": tok.decode(dtoks[i]), "cpu_fp32_text": tok.decode(fr[i])})
    print(json.dumps(rows[-1], indent=1), flush=True)
R["accuracy"] = rows
step_ms_sorted = sorted(step_ms)
R["latency"] = {"prefill_ms": pre_ms, "prefill_tokens": int(lengths.sum()), "bucket": T, "batch": B,
                "decode_step_ms_median": step_ms_sorted[len(step_ms) // 2], "decode_step_ms_p90": step_ms_sorted[int(len(step_ms) * 0.9)],
                "decode_tok_per_s_batch": B * 1e3 / step_ms_sorted[len(step_ms) // 2],
                "note": "decode wall time includes copying all state tensors host<->device every step"}
state_bytes = sum(t.numel() * t.element_size() for t in st)
R["latency"]["state_bytes_per_step_each_way"] = state_bytes
R["finished"] = time.strftime("%FT%TZ", time.gmtime())
save()
print(json.dumps(R["latency"], indent=1))
print("DONE")
