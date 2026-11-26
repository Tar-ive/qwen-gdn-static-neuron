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

/* ---------------------------------------------------------- 2. hybrid: growing memory vs fixed memory */
const Title: React.FC<{ children: React.ReactNode; size?: number; style?: React.CSSProperties }> = ({ children, size = 52, style }) => (
  <div style={{ fontFamily: SANS, fontSize: size, fontWeight: 500, letterSpacing: "-0.02em", color: C.ink, ...style }}>{children}</div>
);
const Hybrid: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const words = Math.round(interpolate(f, [12, cue(2) + 20], [1, 2000], clamp));
  const blocks = Math.min(120, Math.floor(words / 16) + 1);
  const pulse = 1 + 0.04 * Math.sin(f / 7);
  return (
    <Stage behind={<Globe />}>
      <div style={{ position: "absolute", left: 120, top: 110, ...reveal(f, 0) }}>
        <Kicker>inside qwen3.8 · 48 of its 64 layers</Kicker>
      </div>
      <div style={{ position: "absolute", left: 120, right: 120, top: 190, display: "flex", gap: 48 }}>
        <Glass style={{ flex: 1, padding: "40px 46px", height: 560, ...reveal(f, 4) }}>
          <Kicker color={C.attn}>normal attention</Kicker>
          <Title style={{ marginTop: 14 }}>Memory grows<br />with every word</Title>
          <Kicker style={{ marginTop: 12, fontSize: 18 }}>kv cache · O(n) per layer</Kicker>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 34, width: 700 }}>
            {Array.from({ length: blocks }).map((_, i) => <div key={i} style={{ width: 22, height: 22, borderRadius: 5, background: C.attn, opacity: 0.85 }} />)}
          </div>
        </Glass>
        <div style={{ flex: 1, position: "relative" }}>
        <div style={{ position: "absolute", inset: 0, borderRadius: 30, border: `1.5px dashed ${C.faint}`, opacity: 1 - enter(f, cue(1), 12) }} />
        <Glass glow={f > cue(1)} style={{ padding: "40px 46px", height: 560, ...reveal(f, cue(1)) }}>
          <Kicker color={C.gdn}>gated deltanet</Kicker>
          <Title style={{ marginTop: 14 }}>Small memory<br />that never grows</Title>
          <Kicker style={{ marginTop: 12, fontSize: 18 }}>128 × 128 state per head · O(1)</Kicker>
          <div style={{ display: "flex", alignItems: "center", gap: 36, marginTop: 50 }}>
            <div style={{ width: 150, height: 150, borderRadius: 22, background: C.gdn, transform: `scale(${pulse})` }} />
            <div style={{ fontFamily: MONO, fontSize: 28, color: C.dim }}>same size<br />after {words.toLocaleString()} words</div>
          </div>
        </Glass>
        </div>
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 810, display: "flex", justifyContent: "center", ...reveal(f, cue(2)) }}>
        <Pill solid>→ should be cheap to run</Pill>
      </div>
    </Stage>
  );
};

/* ---------------------------------------------------------- 3. blocker: the chips couldn't run it -> you rent an API */
const Blocker: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const out = enter(f, cue(1), 16);
  return (
    <Stage behind={<Wall />}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 34, opacity: 1 - out, filter: `blur(${out * 10}px)` }}>
        <Glass style={{ padding: "44px 64px", textAlign: "center", ...reveal(f, 0) }}>
          <Kicker>aws ai chips</Kicker>
          <Hero size={120} style={{ marginTop: 16 }}>Inferentia · Trainium</Hero>
        </Glass>
        <div style={reveal(f, 18)}><Pill dashed style={{ color: C.bad, fontSize: 30 }}>✕ can't run these layers</Pill></div>
        <div style={reveal(f, 34)}><Kicker style={{ fontSize: 18 }}>existing ports use nki kernels on trainium2 · unsupported on inf2 / trn1 since sdk 2.29</Kicker></div>
      </AbsoluteFill>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", opacity: out, filter: `blur(${(1 - out) * 10}px)` }}>
        <Kicker>so today you rent it from an api</Kicker>
        <Hero size={300} style={{ marginTop: 20 }}>$1.05</Hero>
        <Kicker style={{ marginTop: 20 }}>per million tokens · cheapest provider</Kicker>
      </AbsoluteFill>
    </Stage>
  );
};

/* ---------------------------------------------------------- 4. the silent bug: solve_triangular frozen at trace time */
const Verdict: React.FC<{ label: string; detail: string; ok: boolean; f: number; at: number; strike?: number }> = ({ label, detail, ok, f, at, strike = 0 }) => (
  <Glass glow={ok} style={{ padding: "26px 36px", width: 940, ...reveal(f, at, 18) }}>
    <Kicker>{label}</Kicker>
    <div style={{ fontFamily: PIXEL, fontSize: 64, marginTop: 10, position: "relative", display: "inline-block", color: ok ? C.ink : C.bad }}>
      {detail}
      <span style={{ position: "absolute", left: -6, right: -6, top: "52%", height: 5, background: C.bad, transform: `scaleX(${strike})`, transformOrigin: "left" }} />
    </div>
    <div style={{ marginTop: 14 }}>
      {ok ? <Pill solid>✓ exact</Pill> : <Pill dashed style={{ color: C.bad }}>✕ wrong · median relative error 0.56, max 1.44</Pill>}
    </div>
  </Glass>
);
const Bug: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const RX = 1500, RY = 300;
  const ring = 1 + 0.05 * Math.sin(f / 6);
  const step = (i: number) => enter(f, cue(1) + 8 + i * 14, 14);
  const pipe = ["solve_triangular", "no device kernel", "runs on CPU at trace time", "result stored as a constant"];
  return (
    <Stage behind={<Globe />}>
      <div style={{ position: "absolute", left: 120, top: 90, ...reveal(f, 0) }}><Kicker>the obvious port · torch.linalg.solve_triangular · neuron sdk 2.28</Kicker></div>
      <div style={{ position: "absolute", left: 120, top: 150, ...reveal(f, 6) }}>
        <div style={{ background: "#0E0E10", border: `1.5px solid ${C.edge}`, borderRadius: 16, padding: "20px 28px", fontFamily: MONO, fontSize: 28, width: 940 }}>
          <span style={{ color: C.dim }}>$ </span>torch_neuronx.trace(model, inputs)<br />
          <span style={{ color: C.gdn, opacity: enter(f, 26, 8) }}>✓ compiled · 0 errors · 0 warnings</span>
        </div>
      </div>
      <div style={{ position: "absolute", left: 120, top: 340, display: "flex", flexDirection: "column", gap: 10 }}>
        {pipe.map((t, i) => (
          <div key={t} style={{ display: "flex", alignItems: "center", gap: 16, opacity: step(i), filter: `blur(${(1 - step(i)) * 8}px)` }}>
            <span style={{ fontFamily: MONO, color: C.dim, fontSize: 24, width: 30 }}>{i === 0 ? "" : "↳"}</span>
            <span style={{ fontFamily: i === 0 ? MONO : SANS, fontSize: 32, color: i === 3 ? C.ink : C.dim }}>{t}</span>
          </div>
        ))}
        <div style={{ marginTop: 8, background: "#0E0E10", border: `1.5px dashed ${C.edge}`, borderRadius: 14, padding: "14px 24px", fontFamily: MONO, fontSize: 24, color: C.dim, ...reveal(f, cue(1) + 70) }}>
          HLO&nbsp;&nbsp;<span style={{ color: C.ink }}>ROOT %out = %p0</span>&nbsp;&nbsp;// identity: the input is never read
        </div>
      </div>
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity: enter(f, cue(1), 16) }}>
        {[150, 118].map((r, i) => <circle key={r} cx={RX} cy={RY} r={r * (i === 0 ? ring : 1)} fill="none" stroke="white" strokeOpacity={0.14 + i * 0.1} strokeWidth={2} />)}
        <circle cx={RX} cy={RY} r={84} fill="#111" stroke="white" strokeOpacity={0.85} strokeWidth={2.5} />
      </svg>
      <div style={{ position: "absolute", left: RX - 38, top: RY - 50, fontSize: 76, ...reveal(f, cue(1)) }}>❄</div>
      <div style={{ position: "absolute", left: RX - 220, width: 440, top: RY + 175, textAlign: "center", ...reveal(f, cue(1) + 10) }}>
        <Kicker color={C.ink}>L⁻¹ frozen into the neff</Kicker>
      </div>
      <div style={{ position: "absolute", left: 120, top: 660, transform: "scale(0.8)", transformOrigin: "left top" }}><Verdict label="the input it was traced with" detail="max error 0" ok f={f} at={cue(2)} /></div>
      <div style={{ position: "absolute", left: 1000, top: 660, transform: "scale(0.8)", transformOrigin: "left top" }}><Verdict label="any other input" detail="error 56%" ok={false} f={f} at={cue(3)} /></div>
      <div style={{ position: "absolute", left: RX - 320, width: 640, top: RY + 225, textAlign: "center", ...reveal(f, cue(3) + 30) }}><Kicker color={C.bad} style={{ fontSize: 19 }}>inside the full model: compiles, then segfaults</Kicker></div>
    </Stage>
  );
};

/* ---------------------------------------------------------- 5. the fix: exact block-doubling inverse */
const Fix: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const chips = ["fixed shapes · rank ≤ 4", "padding β = g = 0 → state unchanged", "prefix sums as matmuls", "state stays on the device", "one graph per phase"];
  return (
    <Stage behind={<Globe />}>
      <div style={{ position: "absolute", left: 120, top: 110 }}>
        <Kicker style={reveal(f, 0)}>the fix · exact block-doubling inverse</Kicker>
        <Hero size={150} style={{ marginTop: 22, ...reveal(f, 4, 20) }}>6 × matmul</Hero>
        <div style={{ fontFamily: MONO, fontSize: 52, marginTop: 30, ...reveal(f, 20) }}>X ← X − X·O<sub style={{ fontSize: 30 }}>b</sub>·X</div>
        <div style={{ display: "flex", gap: 10, marginTop: 22 }}>
          {Array.from({ length: 6 }).map((_, i) => {
            const k = enter(f, 30 + i * 8, 8);
            return <div key={i} style={{ padding: "8px 18px", borderRadius: 999, border: `1.5px solid ${C.gdn}`, background: k > 0.5 ? C.gdn : "transparent", color: k > 0.5 ? "#0A0A0A" : C.gdn, fontFamily: MONO, fontSize: 20 }}>b = {2 ** i}</div>;
          })}
        </div>
        <Kicker style={{ marginTop: 16, ...reveal(f, 40) }}>64×64 unit-lower inverse · 12 products per chunk</Kicker>
        <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 44 }}>
          <div style={reveal(f, cue(1))}><Pill solid style={{ fontSize: 26 }}>✓ max error 1.6e-7 on 256 systems it never saw</Pill></div>
          <div style={reveal(f, cue(1) + 14)}><Pill style={{ fontSize: 26, color: C.ink }}>2.9× faster than the row loop · 22× smaller hlo</Pill></div>
        </div>
      </div>
      <div style={{ position: "absolute", left: 1250, top: 180, display: "flex", flexDirection: "column", gap: 16 }}>
        {chips.map((c, i) => (
          <Glass key={c} style={{ padding: "20px 28px", width: 610, whiteSpace: "nowrap", display: "flex", alignItems: "center", gap: 18, ...reveal(f, cue(2) + i * 7) }}>
            <div style={{ width: 24, height: 24, borderRadius: 6, background: C.gdn, flex: "none" }} />
            <span style={{ fontSize: 28 }}>{c}</span>
          </Glass>
        ))}
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

export const SCENES: Record<string, React.FC<SceneProps>> = { intro: Intro, hybrid: Hybrid, blocker: Blocker, bug: Bug, fix: Fix, tp: TP, result: Result, cost: Cost, outro: Outro };
export const LIGHT = new Set(["outro"]);
