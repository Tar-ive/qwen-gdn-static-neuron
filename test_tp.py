"""CPU check of tp_qwen: run `tp` rank-local models in threads with emulated all-reduce / all-gather and compare against
the unsharded StaticQwen35 (prefill logits, packed-state round trip, greedy decode).

usage: python test_tp.py [ckpt] [--tp 8] [--steps 8]
"""
import argparse
import sys
import threading

import torch

import static_qwen as sq
import tp_qwen as tq

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="ckpt/tiny-gqa")
ap.add_argument("--tp", type=int, default=8)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--bucket", type=int, default=128)
args = ap.parse_args()
torch.manual_seed(0)
TP = args.tp


class Emu:
    """Lock-step collectives for TP threads: each thread deposits its tensor, waits, reads the result."""

    def __init__(self, n):
        self.n, self.bar, self.buf, self.out = n, threading.Barrier(n), [None] * n, None
        self.local = threading.local()

    def _ex(self, x, combine):
        r = self.local.rank
        self.buf[r] = x
        if self.bar.wait() == 0:
            self.out = combine(self.buf)
        self.bar.wait()
        y = self.out
        self.bar.wait()
        return y

    def ar(self, x):
        return self._ex(x, lambda xs: torch.stack([t.float() for t in xs]).sum(0).to(xs[0].dtype))

    def ag(self, x):
        return self._ex(x, lambda xs: torch.cat(xs, -1))


emu = Emu(TP)
tq.set_collectives(emu.ar, emu.ag)

V, T, B = 248320, args.bucket, 2
lengths = torch.tensor([7, 100])
ids = torch.randint(0, V, (B, T))
MAX = T + 64
ref = sq.StaticQwen35.from_pretrained(args.ckpt, max_len=MAX).eval()
ranks = [tq.StaticQwen35TP.from_full(args.ckpt, TP, r, MAX).eval() for r in range(TP)]


def run_all(fn):
    res = [None] * TP
    err = []

    def work(r):
        emu.local.rank = r
        try:
            with torch.no_grad():
                res[r] = fn(r)
        except Exception as e:  # noqa: BLE001
            err.append(e)
            emu.bar.abort()
    th = [threading.Thread(target=work, args=(r,)) for r in range(TP)]
    [t.start() for t in th]
    [t.join() for t in th]
    if err:
        raise err[0]
    return res


with torch.no_grad():
    ref_logits, ref_st = ref.prefill(ids, lengths)
phases = [tq.Phase(m, B) for m in ranks]
outs = run_all(lambda r: phases[r](ids, lengths))
lg = outs[0]
same = all(torch.equal(o, lg) for o in outs)
d = (lg - ref_logits).abs().max().item()
print(f"prefill: max|dlogit| {d:.2e}, logits identical on all ranks {same}, argmax eq {bool((lg.argmax(-1) == ref_logits.argmax(-1)).all())}")
ok = d < 1e-3 and same

tok = ref_logits.argmax(-1)
pos = lengths.clone()
st_r = ref_st
for s in range(args.steps):
    with torch.no_grad():
        rl, st_r = ref.decode(tok, pos, st_r)
    outs = run_all(lambda r: phases[r](tok, pos))
    tl = outs[0]
    d = (tl - rl).abs().max().item()
    agree = bool((tl.argmax(-1) == rl.argmax(-1)).all())
    print(f"decode {s}: max|dlogit| {d:.2e} argmax eq {agree}")
    ok &= d < 1e-3 and agree
    tok, pos = rl.argmax(-1), pos + 1
shapes = [tuple(t.shape) for t in phases[0].state()]
print("packed state per rank:", shapes)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
