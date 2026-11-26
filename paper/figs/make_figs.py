"""Figures for static_gdn_inferentia2.tex, from the released results files.
uv run --with matplotlib python paper/figs/make_figs.py  (from repo root)"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# ---- Figure 1: inverse accuracy -------------------------------------------------------------
inv = json.load(open(ROOT / "results/results_0p8b.json"))["inverse"]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
x = [r["L_fro_max"] for r in inv]
a.scatter(x, [r["err_neumann"] for r in inv], s=22, color=ORANGE, edgecolor="white", linewidth=0.8, label="Neumann product", zorder=3)
a.scatter(x, [r["err_ours"] for r in inv], s=22, color=BLUE, edgecolor="white", linewidth=0.8, marker="s", label="block doubling (ours)", zorder=3)
a.axhline(1, color=GRAY, lw=0.8, ls="--")
a.text(43, 1.8, "error = 1", ha="right", va="bottom", color=INK2, fontsize=7.5)
a.set_yscale("log"); a.set_xlabel(r"$\max\,\Vert L\Vert_F$ of the layer's UT systems"); a.set_ylabel("max entry error vs. fp64")
a.set_title("(a) CPU fp32, 18 GDN layers of Qwen3.5-0.8B", fontsize=8.5, loc="left", color=INK)
a.legend(loc="upper left", fontsize=7.5, handletextpad=0.3)

dev = json.load(open(ROOT / "results/20261123T194739Z-inf2.8xlarge/solve_vs_doubling.json"))
names = ["solve_triangular", "row loop", "doubling (ours)"]
k = dev["kernel"]; keys = ["solve_triangular", "rowloop", "doubling"]
traced = [k[n]["rel_err_max"] for n in keys]; unseen = [k[n]["unseen_inputs_rel_err_max"] for n in keys]
import numpy as np
ix = np.arange(3); w = 0.36
b.bar(ix - w/2 - 0.01, traced, w, color=GRAY, label="traced input", zorder=3)
b.bar(ix + w/2 + 0.01, unseen, w, color=[ORANGE, BLUE, BLUE], label="unseen inputs", zorder=3)
for i, v in enumerate(unseen):
    b.text(i + w/2, v * 1.6, f"{v:.2g}" if v > 1e-3 else f"{v:.1e}".replace("e-0", "e-"), ha="center", va="bottom", fontsize=7, color=INK)
b.set_yscale("log"); b.set_ylim(1e-8, 30); b.set_xticks(ix, names, fontsize=7.5)
b.set_ylabel("max entry error vs. fp64")
b.set_title("(b) Inferentia2, 256 traced / 256 unseen systems", fontsize=8.5, loc="left", color=INK)
from matplotlib.patches import Patch
b.legend(handles=[Patch(color=GRAY, label="traced input"), Patch(color=BLUE, label="unseen inputs")], loc="upper right", fontsize=7.5)
b.text(w/2 + 0.24, 0.02, "frozen\nconstant", ha="left", va="center", fontsize=7, color=INK2)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig_inverse.pdf")

# ---- Figure 2: 27B throughput and cost ------------------------------------------------------
tp = {2: 38.2, 8: 52.1, 32: 78.1, 64: 128.0}   # ms/step, results/20261126T191354Z-trn1.32xlarge
B = list(tp); toks = [bb / (tp[bb] / 1e3) for bb in B]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5))
a.plot(B, toks, color=BLUE, lw=2, marker="o", ms=5, mec="white", mew=1, zorder=3)
for bb, t, ms in zip(B, toks, tp.values()):
    a.annotate(f"{ms:.0f} ms/step", (bb, t), textcoords="offset points", xytext=(6, -10) if bb != 64 else (-6, 2),
               ha="left" if bb != 64 else "right", fontsize=7, color=INK2)
a.set_xscale("log", base=2); a.set_xticks(B, [str(v) for v in B]); a.set_ylim(0, 620)
a.set_xlabel("decode batch size"); a.set_ylabel("output tokens/s per copy")
a.set_title("(a) Qwen3.8-27B, TP=8 on trn1.32xlarge", fontsize=8.5, loc="left", color=INK)

def usd(price, t): return price / 4 / (t * 3600 / 1e6)
Bc = [8, 32, 64]; tc = [bb / (tp[bb] / 1e3) for bb in Bc]
b.axhspan(1.30, 3.00, color="#eeede9", zorder=0)
b.text(8.6, 2.0, "typical hosted API", fontsize=7, color=INK2, va="center")
b.axhline(1.05, color=GRAY, lw=1, ls="--"); b.text(80*0.97, 1.16, "cheapest API \\$1.05", ha="right", fontsize=7, color=INK2, va="center")
b.plot(Bc, [usd(21.50, t) for t in tc], color=ORANGE, lw=2, marker="o", ms=5, mec="white", mew=1, label="on-demand \\$21.50/h", zorder=3)
b.plot(Bc, [usd(2.15, t) for t in tc], color=BLUE, lw=2, marker="o", ms=5, mec="white", mew=1, label="Spot \\$2.15/h", zorder=3)
b.annotate("\\$0.30", (64, usd(2.15, tc[-1])), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, color=INK)
b.annotate("\\$2.99", (64, usd(21.50, tc[-1])), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=INK)
b.set_xscale("log", base=2); b.set_yscale("log"); b.set_xticks(Bc, [str(v) for v in Bc]); b.set_xlim(6.5, 80)
b.set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"]); b.set_ylim(0.18, 16)
b.set_xlabel("decode batch size"); b.set_ylabel("\\$ per million output tokens")
b.set_title("(b) cost, four copies per instance", fontsize=8.5, loc="left", color=INK)
b.legend(loc="upper right", fontsize=7)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig_27b.pdf")
print("ok")
