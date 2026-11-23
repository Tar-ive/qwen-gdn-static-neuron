"""Tensor-parallel on-device run of the static graphs (e.g. Qwen3.8-27B at TP=8 on inf2.24xlarge), same methodology
as run_inf2.py:
  1. trace the rank-0 prefill and decode graphs with NxD's SPMD functional API (mocked 8-rank parallel state) and
     compile each once with neuronx-cc;
  2. load hand-sharded per-rank weights (tp_qwen.shard_plan) into an NxDModel and move it to the NeuronCores;
  3. greedy decode on device, compared with the fp32 host reference (the unsharded static model): prefill KL,
     teacher-forced per-step KL / argmax, free-running greedy prefix; plus a batch the graphs were not traced with;
  4. latency: prefill, decode step (state stays on device between steps when the runtime allows it).

usage: python neuron/run_tp.py CKPT OUT [--tp 8] [--bucket 256] [--steps 32]
"""
import argparse
import concurrent.futures as cf
import json
import os
import platform
import subprocess
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import static_qwen as sq  # noqa: E402
import tp_qwen as tq  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out")
ap.add_argument("--tp", type=int, default=8)
ap.add_argument("--bucket", type=int, default=256)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--cc", default="--auto-cast=none --model-type=transformer -O1")
ap.add_argument("--skip-ref", action="store_true")
ap.add_argument("--decode-batches", default="", help="extra decode batch sizes to compile and time, e.g. 8,32")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
TP = args.tp
R = {"args": vars(args), "host": platform.node(), "started": time.strftime("%FT%TZ", time.gmtime())}
T0 = time.time()


def save():
    R["elapsed_s"] = time.time() - T0
    json.dump(R, open(os.path.join(args.out, "results_tp.json"), "w"), indent=1, default=str)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


import torch_neuronx  # noqa: E402
import neuronx_distributed  # noqa: E402
from neuronx_distributed.trace.functions import compile as nxd_compile, trace as nxd_trace  # noqa: E402
from neuronx_distributed.trace.nxd_model import NxDModel  # noqa: E402
from neuronx_distributed.trace.parallel_context import NxDParallelState  # noqa: E402

R["env"] = {"torch": torch.__version__, "torch_neuronx": torch_neuronx.__version__,
            "neuronx_distributed": getattr(neuronx_distributed, "__version__", "?"),
            "neuronx_cc": sh("neuronx-cc --version 2>&1 | head -1"), "neuron_ls": sh("neuron-ls 2>&1 | head -30"),
            "cpus": os.cpu_count(), "mem": sh("free -g | head -2")}
save()

# ------------------------------------------------------------------------- inputs (same prompts as run_inf2.py)
from tokenizers import Tokenizer  # noqa: E402

tk = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))
PAD = tk.token_to_id("<|endoftext|>")


def make_batch(prompts, T):
    ids = [torch.tensor(tk.encode(p, add_special_tokens=False).ids) for p in prompts]
    lengths = torch.tensor([len(x) for x in ids])
    b = torch.full((len(ids), T), PAD, dtype=torch.long)
    for i, x in enumerate(ids):
        b[i, :len(x)] = x
    return b, lengths


T = args.bucket
batch, lengths = make_batch(["The capital of France is",
                             "Explain, step by step, why the sky looks blue during the day but red at sunset. "
                             "Use the physics of Rayleigh scattering and mention the wavelength dependence explicitly. " * 5], T)
batch2, lengths2 = make_batch(["Write a haiku about the ocean at night.",
                               "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n" * 6], T)
B = batch.shape[0]
MAXLEN = (T + args.steps + 32 + 63) // 64 * 64
R["prompts"] = {"lengths": lengths.tolist(), "lengths_unseen": lengths2.tolist(), "bucket": T, "max_len": MAXLEN}
full_cfg = sq.Cfg.from_hf(args.ckpt)
R["model"] = {"layers": full_cfg.layers, "hidden": full_cfg.hidden, "gdn_layers": full_cfg.layer_types.count("linear_attention")}

# ------------------------------------------------------------------------- 1. trace + compile (rank 0, SPMD)
from neuronx_distributed.trace.nxd_model.base_nxd_model import StateInitializer  # noqa: E402
from torch_neuronx.proto import metaneff_pb2  # noqa: E402
from torch_neuronx.xla_impl.trace import get_torch_dtype  # noqa: E402


def state_tensors(tr):
    out = {}
    for t in tr.metaneff.input_tensors:
        if t.type is metaneff_pb2.MetaTensor.Type.INPUT_STATE:
            out[str(t.checkpoint_key).replace("b'", "").replace("'", "")] = (list(t.shape), get_torch_dtype(t.data_type))
    return out


log("trace")
extra_b = [int(x) for x in args.decode_batches.split(",") if x]
traces = {}
with NxDParallelState(world_size=TP, tensor_model_parallel_size=TP):
    tq.use_nxd_collectives()
    m = tq.StaticQwen35TP.empty(args.ckpt, TP, MAXLEN).eval()
    t0 = time.time()
    traces["prefill"] = nxd_trace(tq.Phase(m, B), args=(batch, lengths))
    traces["decode"] = nxd_trace(tq.Phase(m, B), args=(lengths.clone(), lengths.clone()))
    for bb in extra_b:
        traces[f"decode_b{bb}"] = nxd_trace(tq.Phase(m, bb), args=(torch.zeros(bb, dtype=torch.long), torch.full((bb,), 100)))
    R["trace_s"] = time.time() - t0
    buffers = {"m." + k: v.clone() for k, v in m.named_buffers()}   # constant buffers are weights to NxDModel too
del m
states = {}
for k, tr in traces.items():
    st = state_tensors(tr)
    R.setdefault("states", {})[k] = {n: v[0] for n, v in st.items()}
    states.update(st)
save()
if not states:
    raise SystemExit("no INPUT_STATE tensors in the traced graphs: in-place buffer updates were not aliased")
log(f"device-resident state tensors: {sorted(states)}")
log("compile")


def comp(name, tr):
    t0 = time.time()
    c = nxd_compile(tr.hlo, tr.metaneff, compiler_workdir=os.path.join(args.out, "neuronx-cc"), compiler_args=args.cc, key=name)
    return name, c, time.time() - t0


R["compile"] = {}
comps = {}
with cf.ThreadPoolExecutor(len(traces)) as ex:
    for f in cf.as_completed([ex.submit(comp, k, tr) for k, tr in traces.items()]):
        name, c, s_ = f.result()
        comps[name] = c
        R["compile"][name + "_s"] = s_
        log(f"{name} compiled in {s_:.0f}s")
        save()
R["neff_bytes"] = sh(f"find {args.out}/neuronx-cc -name '*.neff' -exec ls -l {{}} \\; | awk '{{print $5, $9}}'")
save()

# ------------------------------------------------------------------------- 2. weights + load
log("load per-rank weights")
t0 = time.time()
with cf.ThreadPoolExecutor(TP) as ex:
    weights = list(ex.map(lambda r: tq.load_rank_weights(args.ckpt, full_cfg, TP, r, prefix="m."), range(TP)))
for w in weights:
    w.update({k: v.clone() for k, v in buffers.items()})
R["weights_s"] = time.time() - t0
R["weights_bytes_per_rank"] = sum(v.numel() * v.element_size() for v in weights[0].values())
save()
nxd = NxDModel(world_size=TP, state_initializer=StateInitializer({k: v[0] for k, v in states.items()},
                                                                  {k: v[1] for k, v in states.items()}, TP))
for k in traces:
    nxd.add(k, traces[k], comps[k])
t0 = time.time()
nxd.set_weights(weights)
nxd.to_neuron()
R["to_neuron_s"] = time.time() - t0
del weights
save()
log("loaded on neuron")


def run(x, n):
    return nxd(x, n).float()   # default mode: inputs replicated, rank-0 output (logits are all-gathered)


# ------------------------------------------------------------------------- 3. run
for _ in range(2):
    d_logits = run(batch, lengths)
t0 = time.time()
for _ in range(5):
    d_logits = run(batch, lengths)
pre_ms = (time.time() - t0) / 5 * 1e3
log(f"prefill {pre_ms:.1f} ms")
d_logits = run(batch, lengths)              # fresh state for the decode run
toks, logs, step_ms = [d_logits.argmax(-1)], [d_logits], []
pos = lengths.clone()
for s in range(args.steps - 1):
    t0 = time.time()
    lg = run(toks[-1], pos)
    step_ms.append((time.time() - t0) * 1e3)
    toks.append(lg.argmax(-1))
    logs.append(lg)
    pos = pos + 1
dtoks, dlogs = torch.stack(toks, 1), torch.stack(logs, 1)
d2_logits = run(batch2, lengths2)
ss = sorted(step_ms)
R["latency"] = {"prefill_ms": pre_ms, "batch": B, "bucket": T, "prefill_tokens": int(lengths.sum()),
                "decode_step_ms_median": ss[len(ss) // 2], "decode_step_ms_p90": ss[int(len(ss) * 0.9)],
                "decode_tok_per_s_batch": B * 1e3 / ss[len(ss) // 2], "decode_state": "device-resident (aliased)"}
R["device_text"] = [tk.decode(dtoks[i].tolist()) for i in range(B)]
save()
log(json.dumps(R["latency"]))

# batch sweep: decode step time at larger batch from a zero state (static graph: time does not depend on values)
R["batch_sweep"] = {str(B): {"step_ms_median": ss[len(ss) // 2], "tok_per_s": B * 1e3 / ss[len(ss) // 2]}}
for bb in extra_b:
    try:
        tokb, posb, times = torch.zeros(bb, dtype=torch.long), torch.full((bb,), 100), []
        for s in range(14):
            t0 = time.time()
            run(tokb, posb)
            times.append((time.time() - t0) * 1e3)
            posb = posb + 1
        tt = sorted(times[2:])
        R["batch_sweep"][str(bb)] = {"step_ms_median": tt[len(tt) // 2], "tok_per_s": bb * 1e3 / tt[len(tt) // 2]}
    except Exception:  # noqa: BLE001
        R["batch_sweep"][str(bb)] = {"error": traceback.format_exc()[-1500:]}
    log(f"batch {bb}: {R['batch_sweep'][str(bb)]}")
    save()
torch.save({"dtoks": dtoks, "dlogs": dlogs, "d2_logits": d2_logits}, os.path.join(args.out, "device_outputs.pt"))
del nxd

if args.skip_ref:
    R["finished"] = time.strftime("%FT%TZ", time.gmtime())
    save()
    sys.exit(0)

# ------------------------------------------------------------------------- 4. fp32 host reference
log("fp32 host reference")
torch.set_num_threads(os.cpu_count())
t0 = time.time()
ref = sq.StaticQwen35.from_pretrained(args.ckpt, max_len=MAXLEN).eval()
R["ref_load_s"] = time.time() - t0


def kl(a, b):  # KL(ref || dev)
    pa, pb = b.float().log_softmax(-1), a.float().log_softmax(-1)
    return (pa.exp() * (pa - pb)).sum(-1)


with torch.no_grad():
    t0 = time.time()
    ref_logits, ref_state = ref.prefill(batch, lengths)
    R["ref_prefill_s"] = time.time() - t0
    ref2_logits, _ = ref.prefill(batch2, lengths2)
    tf, st_r, p = [ref_logits], ref_state, lengths.clone()
    for s in range(args.steps - 1):
        lg, st_r = ref.decode(dtoks[:, s], p, st_r)
        tf.append(lg)
        p = p + 1
    tf = torch.stack(tf, 1)
    fr, st_r, p = [ref_logits.argmax(-1)], ref_state, lengths.clone()
    for s in range(args.steps - 1):
        lg, st_r = ref.decode(fr[-1], p, st_r)
        fr.append(lg.argmax(-1))
        p = p + 1
    fr = torch.stack(fr, 1)

rows = []
for i in range(B):
    k = kl(dlogs[i], tf[i])
    rows.append({"prompt_len": int(lengths[i]), "prefill_kl": float(k[0]), "max_step_kl": float(k.max()),
                 "mean_step_kl": float(k.mean()),
                 "teacher_forced_argmax_agree": float((dlogs[i].argmax(-1) == tf[i].argmax(-1)).float().mean()),
                 "greedy_identical_prefix": int((dtoks[i] == fr[i]).long().cumprod(0).sum()), "steps": args.steps,
                 "device_text": tk.decode(dtoks[i].tolist()), "cpu_fp32_text": tk.decode(fr[i].tolist())})
R["accuracy"] = rows
k2 = kl(d2_logits, ref2_logits)
R["unseen_batch"] = {"lengths": lengths2.tolist(), "prefill_kl": k2.tolist(),
                     "argmax_eq": (d2_logits.argmax(-1) == ref2_logits.argmax(-1)).tolist()}
R["finished"] = time.strftime("%FT%TZ", time.gmtime())
save()
log(json.dumps(rows, indent=1))
log(json.dumps(R["unseen_batch"]))
print("DONE")
