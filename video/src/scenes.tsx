import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { C, MONO, PIXEL, SANS, clamp, enter, reveal } from "./style";
import { Curve, Glass, Globe, Icon, Kicker, Pill, Stage } from "./ui";
import facts from "./facts.json";
import type { SceneProps } from "./Explainer";

const F = facts as Record<string, string | boolean>;
const s = (k: string) => String(F[k]);
const Hero: React.FC<{ children: React.ReactNode; size?: number; style?: React.CSSProperties }> = ({ children, size = 220, style }) => (
  <div style={{ fontFamily: PIXEL, fontSize: size, lineHeight: 0.95, color: C.ink, letterSpacing: "-0.01em", ...style }}>{children}</div>
);

/* ---------------------------------------------------------- layer-card wall (intro + tp backdrop) */
const LayerCard: React.FC<{ i: number }> = ({ i }) => {
  const attn = i % 4 === 3;
  const rows = attn
    ? ["q · 24 heads", "k/v · 4 heads", "gated output", "rope · 64 dims", "kv cache ↑"]
    : ["conv1d · k = 4", "β, g gates", "state 128×128", "chunk 64", "out_proj"];
  return (
    <div style={{ width: 250, height: 310, borderRadius: 22, background: "#141416", border: `1.5px solid ${C.faint}`, padding: 20, fontFamily: MONO, color: C.dim, fontSize: 17 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <div style={{ width: 38, height: 38, borderRadius: 10, background: attn ? C.attn : C.gdn, opacity: 0.85 }} />
        <div>
          <div style={{ fontFamily: SANS, color: C.ink, fontSize: 21 }}>Layer {i}</div>
          <div style={{ fontSize: 14 }}>{attn ? "full attention" : "Gated DeltaNet"}</div>
        </div>
      </div>
      {rows.map((r) => <div key={r} style={{ marginTop: 16, display: "flex", gap: 10 }}><span style={{ opacity: 0.5 }}>▢</span>{r}</div>)}
    </div>
  );
};
const Wall: React.FC<{ drift?: number }> = ({ drift = 0 }) => (
  <AbsoluteFill style={{ perspective: 1800, overflow: "hidden" }}>
    <div style={{ position: "absolute", left: -500, top: -620, display: "grid", gridTemplateColumns: "repeat(11, 250px)", gap: 26,
      transform: `rotateX(38deg) rotateZ(-14deg) translateY(${drift}px)` }}>
      {Array.from({ length: 66 }).map((_, i) => <LayerCard key={i} i={i % 64} />)}
    </div>
  </AbsoluteFill>
);

/* ---------------------------------------------------------- 1. intro */
const Intro: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const dark = f >= cue(1) - 4;
  if (!dark)
    return (
      <AbsoluteFill style={{ background: C.light, alignItems: "center", justifyContent: "center", fontFamily: SANS }}>
        <div style={{ display: "flex", alignItems: "center", gap: 22, fontSize: 84, color: "#111", ...reveal(f, 2) }}>
          Introducing
          <div style={{ width: 70, height: 70, borderRadius: 18, background: "#111", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, padding: 14 }}>
            {[C.gdn, C.gdn, C.gdn, C.attn].map((c, i) => <div key={i} style={{ background: c, borderRadius: 3 }} />)}
          </div>
          <span style={{ fontWeight: 600 }}>Qwen3.8-27B</span>
        </div>
      </AbsoluteFill>
    );
  const g = f - cue(1);
  return (
    <Stage>
      <AbsoluteFill style={{ opacity: 0.55 }}><Wall drift={-g * 0.8} /></AbsoluteFill>
      <AbsoluteFill style={{ background: "radial-gradient(ellipse at center, rgba(9,9,10,0.85) 0%, rgba(9,9,10,0.2) 60%)" }} />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 24 }}>
        <div style={{ display: "flex", gap: 18, alignItems: "center", fontSize: 44, ...reveal(f, cue(1)) }}>
          Introducing <span style={{ fontWeight: 600 }}>Qwen3.8-27B</span>
          <span style={{ color: C.dim, fontFamily: MONO }}>&lt;&gt;</span>
          <span style={{ fontWeight: 600 }}>NeuronCore</span>
        </div>
        <Hero size={190} style={reveal(f, cue(1) + 8, 22)}>Exact on Trainium</Hero>
      </AbsoluteFill>
    </Stage>
  );
};

/* ---------------------------------------------------------- 2. hybrid model */
const Hybrid: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const up = enter(f, cue(0) + 10, 24);
  const tokens = Math.round(interpolate(f, [cue(0) + 40, cue(1) + 30], [1, 4096], clamp));
  return (
    <Stage>
      <Globe opacity={1 - up * 0.6} />
      <div style={{ position: "absolute", left: 960, top: 540 - up * 300, transform: "translate(-50%, -50%)", ...reveal(f, 0) }}>
        <Glass style={{ borderRadius: 999, padding: "18px 22px 18px 30px", display: "flex", alignItems: "center", gap: 20 }}>
          <span style={{ fontFamily: MONO, fontSize: 30 }}>serve</span>
          <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "10px 22px 10px 10px", borderRadius: 999, background: "rgba(255,255,255,0.07)", border: `1.5px solid ${C.edge}` }}>
            <div style={{ width: 40, height: 40, borderRadius: 999, background: `conic-gradient(${C.gdn} 0 75%, ${C.attn} 0)` }} />
            <span style={{ fontSize: 30 }}>Qwen3.8-27B</span>
          </div>
          <Icon size={44}>↵</Icon>
        </Glass>
      </div>
      <div style={{ position: "absolute", left: 160, right: 160, top: 400, opacity: up }}>
        <Kicker>64 layers · 3 of 4 are recurrent</Kicker>
        <div style={{ display: "flex", gap: 8, marginTop: 26 }}>
          {Array.from({ length: 64 }).map((_, i) => {
            const k = enter(f, cue(0) + 14 + i * 0.7, 10);
            return <div key={i} style={{ flex: 1, height: 64, borderRadius: 6, background: i % 4 === 3 ? C.attn : C.gdn, opacity: 0.15 + 0.75 * k }} />;
          })}
        </div>
        <div style={{ display: "flex", gap: 40, marginTop: 50 }}>
          <Glass style={{ flex: 1, padding: 32 }}>
            <Kicker color={C.gdn}>48 × Gated DeltaNet</Kicker>
            <div style={{ display: "flex", alignItems: "center", gap: 28, marginTop: 24 }}>
              <div style={{ width: 96, height: 96, borderRadius: 14, background: C.gdn, opacity: 0.9 }} />
              <div style={{ fontFamily: MONO, fontSize: 28, color: C.dim }}>128 × 128 state<br />same at {tokens.toLocaleString()} tokens</div>
            </div>
          </Glass>
          <Glass style={{ flex: 1, padding: 32 }}>
            <Kicker color={C.attn}>16 × full attention</Kicker>
            <div style={{ height: 36, borderRadius: 10, background: "rgba(255,255,255,0.06)", marginTop: 36 }}>
              <div style={{ height: "100%", width: `${(tokens / 4096) * 100}%`, borderRadius: 10, background: C.attn }} />
            </div>
            <div style={{ fontFamily: MONO, fontSize: 28, color: C.dim, marginTop: 16 }}>kv cache · {tokens.toLocaleString()} tokens</div>
          </Glass>
        </div>
        <div style={{ ...reveal(f, cue(1)), marginTop: 46, display: "flex", justifyContent: "center" }}>
          <Pill>cheap to serve · if the chip can run it</Pill>
        </div>
      </div>
    </Stage>
  );
};

/* ---------------------------------------------------------- 3. the silent bug ("which one is real?") */
const wave = (x0: number, y0: number, x1: number, y1: number) => {
  const n = 7, pts: string[] = [];
  for (let i = 0; i <= 140; i++) {
    const t = i / 140, amp = 46 * Math.sin(Math.PI * t) * (1 - t * 0.4);
    pts.push(`${x0 + (x1 - x0) * t},${y0 + (y1 - y0) * t + amp * Math.sin(t * n * 2 * Math.PI)}`);
  }
  return "M" + pts.join(" L");
};
const Bug: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const dead = enter(f, cue(3), 14);
  const good = enter(f, cue(4), 20);
  const ring = 1 + 0.06 * Math.sin(f / 6);
  const RX = 1500, RY = 470;
  return (
    <Stage behind={<Globe />}>
      <div style={{ position: "absolute", left: 120, top: 150 }}>
        <Kicker style={reveal(f, 0)}>2 inverses compiled · which one is real?</Kicker>
        <Glass glow={dead < 0.5} style={{ marginTop: 26, padding: "34px 44px", width: 900, opacity: 1 - dead * 0.45, ...reveal(f, 4) }}>
          <div style={{ fontFamily: PIXEL, fontSize: 92, position: "relative", display: "inline-block" }}>
            solve_triangular
            <div style={{ position: "absolute", left: -6, right: -6, top: "52%", height: 6, background: C.ink, transform: `scaleX(${dead})`, transformOrigin: "left" }} />
          </div>
          <div style={{ marginTop: 22, display: "flex", gap: 16 }}>
            {dead < 0.5
              ? <Pill solid>✓ compiles · no errors</Pill>
              : <Pill dashed style={{ color: C.bad }}>✕ frozen constant · err 0.56 – 1.44</Pill>}
            <Pill style={{ ...reveal(f, cue(2)) }}>trace input · exact</Pill>
          </div>
        </Glass>
        <Glass glow={good > 0.5} style={{ marginTop: 40, padding: "34px 44px", width: 900, ...reveal(f, cue(4), 22) }}>
          <div style={{ fontFamily: PIXEL, fontSize: 92 }}>6 × matmul</div>
          <div style={{ marginTop: 22, display: "flex", gap: 16 }}>
            <Pill solid>✓ exact · err 1.6e-7 on unseen inputs</Pill>
          </div>
        </Glass>
      </div>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <Curve d={wave(1010, 330, RX - 70, RY)} k={enter(f, cue(1), 30) * (1 - good)} opacity={0.9 - dead * 0.6} width={3} />
        <Curve d={wave(1010, 680, RX - 70, RY)} k={good} opacity={0.95} width={3} />
        <Curve d={`M ${RX - 40} ${RY + 60} Q ${RX - 200} ${RY + 260} ${RX - 120} ${RY + 330}`} k={enter(f, cue(1) + 20, 20)} dashed />
        {[150, 115, 80].map((r, i) => <circle key={r} cx={RX} cy={RY} r={r * (i === 0 ? ring : 1)} fill="none" stroke="white" strokeOpacity={0.12 + i * 0.12} strokeWidth={2} />)}
        <circle cx={RX} cy={RY} r={56} fill="#111" stroke="white" strokeOpacity={0.8} strokeWidth={2.5} />
      </svg>
      <div style={{ position: "absolute", left: RX - 24, top: RY - 30, fontFamily: PIXEL, fontSize: 52 }}>N</div>
      <div style={{ position: "absolute", left: RX - 140, top: RY + 175 }}><Kicker color={C.ink}>neuroncore-v2</Kicker></div>
      <div style={{ position: "absolute", left: RX - 260, top: RY + 340, ...reveal(f, cue(1) + 30) }}>
        <Pill dashed>cpu · runs once · baked into neff</Pill>
      </div>
    </Stage>
  );
};

/* ---------------------------------------------------------- 4. static graphs (profile-card layout) */
const Row: React.FC<{ icon: string; text: string; f: number; at: number; hi?: boolean }> = ({ icon, text, f, at, hi }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 20, padding: "16px 0", ...reveal(f, at, 14, 8) }}>
    <Icon size={46} color={hi ? C.gdn : C.ink}>{icon}</Icon>
    <span style={{ fontSize: 30, color: hi ? C.gdn : C.ink }}>{text}</span>
  </div>
);
const Static: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  return (
    <Stage behind={<Globe />}>
      <div style={{ position: "absolute", left: 120, top: 250 }}>
        <Hero size={210} style={reveal(f, 0, 20)}>1 graph</Hero>
        <Kicker style={{ marginTop: 30, ...reveal(f, 8) }}>per phase · prefill + decode · no data-dependent shapes</Kicker>
      </div>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <Curve d="M 820 330 C 1000 330, 980 470, 1140 470" k={enter(f, 10, 24)} />
      </svg>
      <Glass style={{ position: "absolute", left: 1140, top: 140, width: 660, padding: "34px 40px", ...reveal(f, 6) }}>
        <div style={{ display: "flex", alignItems: "center", gap: 22, paddingBottom: 22, borderBottom: `1.5px solid ${C.faint}` }}>
          <div style={{ width: 76, height: 76, borderRadius: 999, background: `conic-gradient(${C.gdn} 0 75%, ${C.attn} 0)` }} />
          <div><div style={{ fontSize: 40 }}>static_qwen.py</div><div style={{ color: C.dim, fontSize: 26 }}>prefill · decode</div></div>
        </div>
        <Row f={f} at={cue(0)} icon="▭" text="ids [B, 256] · lengths [B]" />
        <Row f={f} at={cue(0) + 6} icon="△" text="prefix sums → triu matmul" />
        <Row f={f} at={cue(0) + 12} icon="⌗" text="tensor rank ≤ 4" />
        <Row f={f} at={cue(1)} icon="β" text="padding: β = 0, g = 0 → no-op" hi />
        <Row f={f} at={cue(2)} icon="▣" text="state resident on chip · aliased" hi />
        <Row f={f} at={cue(2) + 8} icon="×" text="6-step exact inverse" />
      </Glass>
    </Stage>
  );
};

/* ---------------------------------------------------------- 5. tensor parallel */
const TP: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const SX = 960, SY = 560;
  const pos = Array.from({ length: 8 }, (_, i) => ({ x: i < 4 ? 140 : 1420, y: 230 + (i % 4) * 175 }));
  const flow = f > cue(1) ? ((f - cue(1)) % 40) / 40 : -1;
  return (
    <Stage behind={<Wall />}>
      <div style={{ position: "absolute", top: 90, width: "100%", textAlign: "center", ...reveal(f, 0) }}>
        <Kicker>tp = 8 · 54 GB bf16 → 9 GB per core</Kicker>
      </div>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        {pos.map((p, i) => {
          const x0 = p.x < 960 ? p.x + 360 : p.x;
          const d = `M ${x0} ${p.y + 60} C ${(x0 + SX) / 2} ${p.y + 60}, ${(x0 + SX) / 2} ${SY}, ${SX + (p.x < 960 ? -70 : 70)} ${SY}`;
          return <g key={i}><Curve d={d} k={enter(f, cue(0) + 20 + i * 3, 24)} opacity={0.45} />
            {flow >= 0 && <path d={d} fill="none" stroke={C.gdn} strokeWidth={4} pathLength={1} strokeDasharray={`0.08 1`} strokeDashoffset={-flow} />}</g>;
        })}
        {[130, 100].map((r) => <circle key={r} cx={SX} cy={SY} r={r} fill="none" stroke="white" strokeOpacity={0.15} strokeWidth={2} />)}
        <circle cx={SX} cy={SY} r={70} fill="#111" stroke="white" strokeOpacity={0.8} strokeWidth={2.5} />
      </svg>
      <div style={{ position: "absolute", left: SX - 30, top: SY - 48, fontFamily: PIXEL, fontSize: 84 }}>Σ</div>
      <div style={{ position: "absolute", left: SX - 160, top: SY + 150, width: 320, textAlign: "center", ...reveal(f, cue(1)) }}>
        <Pill solid>2 all-reduces / layer</Pill>
      </div>
      <div style={{ position: "absolute", left: SX - 160, top: SY + 225, width: 320, textAlign: "center", ...reveal(f, cue(2)) }}>
        <Kicker>nothing else crosses</Kicker>
      </div>
      {pos.map((p, i) => (
        <Glass key={i} style={{ position: "absolute", left: p.x, top: p.y, width: 360, padding: "20px 26px", ...reveal(f, cue(0) + i * 3) }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 30 }}>core {i}</span>
            <div style={{ width: 34, height: 34, borderRadius: 8, background: C.gdn, opacity: 0.85 }} />
          </div>
          <div style={{ fontFamily: MONO, fontSize: 19, color: C.dim, marginTop: 8 }}>3q · 2k/6v heads · ⅛ MLP</div>
        </Glass>
      ))}
    </Stage>
  );
};

/* ---------------------------------------------------------- 6. result */
const Result: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const n = Math.round(interpolate(f, [cue(1), cue(1) + 24], [0, Number(s("greedy"))], clamp));
  const second = f >= cue(2);
  return (
    <Stage behind={<Globe />}>
      <div style={{ position: "absolute", left: 120, top: 260 }}>
        <Hero size={second ? 190 : 250} style={reveal(f, second ? cue(2) : cue(1), 18)}>{second ? `${s("top_tok_s")} tok/s` : `${n}/${s("steps")}`}</Hero>
        <Kicker style={{ marginTop: 34, ...reveal(f, cue(1) + 6) }}>
          {second ? `batch ${s("top_batch")} · one model copy · 8 neuroncores` : `greedy tokens identical to fp32 · kl ≤ ${s("kl")}`}
        </Kicker>
      </div>
      <Glass style={{ position: "absolute", left: 1150, top: 150, width: 650, padding: "34px 40px", ...reveal(f, 0) }}>
        <div style={{ paddingBottom: 20, borderBottom: `1.5px solid ${C.faint}` }}>
          <div style={{ fontSize: 40 }}>{s("model")}</div>
          <div style={{ color: C.dim, fontSize: 26 }}>{s("instance")} · bf16 · tp 8</div>
        </div>
        <Row f={f} at={8} icon="⇥" text={`prefill 2×256 · ${s("prefill_ms")} ms`} />
        <Row f={f} at={14} icon="↻" text={`decode step, batch 2 · ${s("decode_ms")} ms`} />
        <Row f={f} at={cue(1)} icon="=" text={`${s("greedy")}/${s("steps")} greedy · vs fp32 cpu`} hi />
        <Row f={f} at={cue(1) + 8} icon="?" text="unseen batch · same argmax" hi />
        <Row f={f} at={cue(2)} icon="▤" text={`batch ${s("top_batch")} · ${s("top_tok_s")} tok/s`} />
      </Glass>
      {F.pending === true && (
        <div style={{ position: "absolute", left: 120, top: 70 }}><Kicker color={C.bad}>placeholder · 0.8b rehearsal numbers · 27b run in progress</Kicker></div>
      )}
    </Stage>
  );
};

/* ---------------------------------------------------------- 7. cost ($0.07 layout) */
const Cost: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const apis = [["OpenRouter · cheapest", "$1.05/M"], ["Chutes", "$1.25/M"], ["typical provider", "$1.30–3.00/M"]];
  const od = enter(f, cue(2), 16);
  return (
    <Stage behind={<Wall />}>
      <div style={{ position: "absolute", left: 120, top: 200 }}>
        <Hero size={300} style={reveal(f, 0, 20)}>${s("usd_per_m")}</Hero>
        <Kicker style={{ marginTop: 30, ...reveal(f, 8) }}>per million tokens · trn1 spot ${s("spot_hr")}/hr</Kicker>
        <div style={{ marginTop: 60 }}>
          <Kicker style={{ fontSize: 18, ...reveal(f, cue(1)) }}>instead of an api at</Kicker>
          {apis.map(([name, p], i) => {
            const k = enter(f, cue(1) + 8 + i * 8, 12);
            return (
              <div key={name} style={{ display: "flex", gap: 30, fontSize: 34, marginTop: 14, color: C.dim, opacity: k, position: "relative", width: 700 }}>
                <span style={{ width: 420 }}>{name}</span><span style={{ fontFamily: MONO }}>{p}</span>
                <div style={{ position: "absolute", left: 0, top: "52%", height: 2.5, width: `${100 * enter(f, cue(1) + 14 + i * 8, 10)}%`, background: C.dim }} />
              </div>
            );
          })}
        </div>
      </div>
      <div style={{ position: "absolute", left: 1180, top: 300, opacity: od, filter: `blur(${(1 - od) * 10}px)` }}>
        <Glass style={{ padding: "30px 38px", width: 600 }}>
          <Kicker>same run · on-demand $21.50/hr</Kicker>
          <div style={{ fontFamily: PIXEL, fontSize: 110, marginTop: 14, color: C.bad }}>${s("ondemand_usd_per_m")}</div>
          <Kicker>per million tokens · ~3× the cheapest api</Kicker>
        </Glass>
      </div>
      <div style={{ position: "absolute", left: 120, top: 90 }}><Kicker>measured: {s("top_tok_s")} tok/s per copy at batch {s("top_batch")} · 4 copies per instance projected</Kicker></div>
    </Stage>
  );
};

/* ---------------------------------------------------------- 8. outro (light) */
const DOTS = Array.from({ length: 46 }, (_, i) => ({ x: (i * 397) % 1840 + 40, y: (i * 233) % 1000 + 40, c: [C.gdn, C.attn, "#111", "#60A5FA", "#F59E0B"][i % 5] }));
const Outro: React.FC<SceneProps> = ({ props }) => {
  const f = useCurrentFrame();
  const cmd = `$ git clone ${props.repoUrl}`;
  const typed = Math.round(interpolate(f, [24, 70], [0, cmd.length], clamp));
  return (
    <AbsoluteFill style={{ background: "#FFFFFF", fontFamily: SANS, color: "#111", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
      {DOTS.map((d, i) => {
        if (Math.abs(d.x - 960) < 520 && Math.abs(d.y - 520) < 220) return null;
        return <div key={i} style={{ position: "absolute", left: d.x, top: d.y + Math.sin((f + i * 9) / 30) * 6, width: 26, height: 26, borderRadius: 7, background: d.c, opacity: 0.85 * enter(f, i * 0.6, 14) }} />;
      })}
      <div style={{ display: "flex", alignItems: "center", gap: 18, fontSize: 52, ...reveal(f, 0) }}>
        <div style={{ width: 54, height: 54, borderRadius: 14, background: "#111", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5, padding: 11 }}>
          {[C.gdn, C.gdn, C.gdn, C.attn].map((c, i) => <div key={i} style={{ background: c, borderRadius: 2 }} />)}
        </div>
        qwen-gdn-static-neuron
      </div>
      <div style={{ fontFamily: PIXEL, fontSize: 92, marginTop: 30, textAlign: "center", ...reveal(f, 8) }}>Exact hybrid models<br />on Inferentia2 + Trainium1</div>
      <div style={{ marginTop: 50, background: "#111", color: "#E5E5E5", fontFamily: MONO, fontSize: 30, padding: "18px 34px", borderRadius: 12, minWidth: 900, ...reveal(f, 16) }}>
        {cmd.slice(0, typed)}<span style={{ opacity: Math.floor(f / 15) % 2 ? 1 : 0 }}>▌</span>
      </div>
    </AbsoluteFill>
  );
};

export const SCENES: Record<string, React.FC<SceneProps>> = { intro: Intro, hybrid: Hybrid, bug: Bug, static: Static, tp: TP, result: Result, cost: Cost, outro: Outro };
export const LIGHT = new Set(["outro"]);
