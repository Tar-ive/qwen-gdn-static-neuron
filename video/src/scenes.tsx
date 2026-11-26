import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { C, MONO, enter, fadeUp, FPS } from "./style";
import type { SceneProps } from "./Explainer";

const Stage: React.FC<{ children: React.ReactNode; kicker?: string }> = ({ children, kicker }) => (
  <AbsoluteFill style={{ padding: "70px 120px 190px", display: "flex", flexDirection: "column" }}>
    {kicker && (
      <div style={{ fontFamily: MONO, fontSize: 24, letterSpacing: 3, color: C.muted, textTransform: "uppercase", marginBottom: 28 }}>
        {kicker}
      </div>
    )}
    {children}
  </AbsoluteFill>
);

const Panel: React.FC<{ style?: React.CSSProperties; children: React.ReactNode; color?: string }> = ({ style, children, color }) => (
  <div style={{ background: C.panel, border: `2px solid ${color ?? C.line}`, borderRadius: 18, padding: "26px 32px", ...style }}>{children}</div>
);

const Big: React.FC<{ children: React.ReactNode; color?: string; size?: number; style?: React.CSSProperties }> = ({ children, color, size = 64, style }) => (
  <div style={{ fontSize: size, fontWeight: 700, color: color ?? C.text, lineHeight: 1.1, ...style }}>{children}</div>
);

const Small: React.FC<{ children: React.ReactNode; color?: string; style?: React.CSSProperties }> = ({ children, color, style }) => (
  <div style={{ fontSize: 30, color: color ?? C.muted, lineHeight: 1.35, ...style }}>{children}</div>
);

/* ---------------------------------------------------------------- 1. hook */
const Hook: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const tokens = Math.round(interpolate(f, [cue(3), cue(3) + 8 * FPS], [1, 4096], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const grow = interpolate(f, [cue(3), cue(3) + 8 * FPS], [0.03, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <Stage kicker="Hybrid decoder">
      <div style={fadeUp(f, 0)}><Big size={110}>Qwen3.8-27B</Big></div>
      <div style={fadeUp(f, cue(1))}><Small style={{ fontSize: 40, marginTop: 12 }}>27.8 B parameters · 64 layers</Small></div>
      <div style={{ display: "flex", gap: 7, marginTop: 50 }}>
        {Array.from({ length: 64 }).map((_, i) => {
          const k = enter(f, cue(2) + i * 0.6, 8);
          const attn = i % 4 === 3;
          return <div key={i} style={{ width: 18, height: 70, borderRadius: 4, background: attn ? C.attn : C.gdn, opacity: k, transform: `scaleY(${k})` }} />;
        })}
      </div>
      <div style={{ display: "flex", gap: 40, marginTop: 18, ...fadeUp(f, cue(2) + 20) }}>
        <Small color={C.gdn}>■ 48 Gated DeltaNet</Small>
        <Small color={C.attn}>■ 16 full attention</Small>
      </div>
      <div style={{ display: "flex", gap: 40, marginTop: 44, ...fadeUp(f, cue(3)) }}>
        <Panel style={{ flex: 1 }} color={C.attn}>
          <Small color={C.attn}>attention: KV cache grows with every token</Small>
          <div style={{ marginTop: 18, height: 34, background: C.line, borderRadius: 8 }}>
            <div style={{ height: "100%", width: `${grow * 100}%`, background: C.attn, borderRadius: 8 }} />
          </div>
          <Small style={{ marginTop: 10, fontFamily: MONO }}>{tokens.toLocaleString()} tokens</Small>
        </Panel>
        <Panel style={{ flex: 1 }} color={C.gdn}>
          <Small color={C.gdn}>DeltaNet: fixed 128 × 128 memory per head</Small>
          <div style={{ display: "flex", alignItems: "center", gap: 24, marginTop: 14 }}>
            <div style={{ width: 82, height: 82, background: C.gdn, borderRadius: 8, opacity: 0.85 }} />
            <Small style={{ fontFamily: MONO }}>same size at {tokens.toLocaleString()} tokens</Small>
          </div>
        </Panel>
      </div>
    </Stage>
  );
};

/* ---------------------------------------------------------------- 2. stakes */
const Chip: React.FC<{ size?: number; label?: string }> = ({ size = 300, label }) => (
  <div style={{ position: "relative", width: size, height: size }}>
    {Array.from({ length: 7 }).map((_, i) =>
      ["top", "bottom", "left", "right"].map((side) => (
        <div key={side + i} style={{
          position: "absolute", background: C.muted, opacity: 0.6,
          ...(side === "top" || side === "bottom"
            ? { width: 10, height: 26, left: size * 0.16 + i * size * 0.11, [side]: -24 }
            : { height: 10, width: 26, top: size * 0.16 + i * size * 0.11, [side]: -24 }),
        }} />
      )),
    )}
    <div style={{ position: "absolute", inset: 0, background: C.panel, border: `3px solid ${C.muted}`, borderRadius: 22, display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center", fontSize: 30, color: C.text, padding: 20 }}>
      {label}
    </div>
  </div>
);

const Stakes: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  return (
    <Stage kicker="Why it matters">
      <div style={fadeUp(f, 0)}><Big>Cheap to serve… in theory.</Big></div>
      <div style={{ display: "flex", alignItems: "center", gap: 90, marginTop: 70 }}>
        <div style={fadeUp(f, cue(1))}><Chip label={"Inferentia2 · Trainium1\nNeuronCore-v2"} /></div>
        <div style={{ display: "flex", flexDirection: "column", gap: 26 }}>
          <div style={fadeUp(f, cue(1) + 25)}><Panel><Big size={44}>fixed-shape graphs only</Big></Panel></div>
          <div style={fadeUp(f, cue(1) + 60)}><Panel color={C.bad}><Big size={44} color={C.bad}>✕ Gated DeltaNet: no support</Big></Panel></div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 30, marginTop: 70, ...fadeUp(f, cue(2)) }}>
        <Big size={96} color={C.cost}>$1.05</Big>
        <Small style={{ fontSize: 36 }}>per million output tokens · cheapest API today</Small>
      </div>
    </Stage>
  );
};

/* ---------------------------------------------------------------- 3. the silent bug */
const Bug: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  return (
    <Stage kicker="The silent failure">
      <div style={{ display: "flex", alignItems: "baseline", gap: 30, ...fadeUp(f, 0) }}>
        <Big style={{ fontFamily: MONO }} size={60}>torch.linalg.solve_triangular</Big>
        <Small color={C.gdn} style={{ fontSize: 34 }}>compiles ✓</Small>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 36, marginTop: 56 }}>
        <div style={fadeUp(f, cue(1))}><Panel style={{ width: 300, textAlign: "center" }} color={C.bad}><Big size={38}>chip</Big><Small color={C.bad}>no solve op</Small></Panel></div>
        <div style={{ fontSize: 60, color: C.muted, ...fadeUp(f, cue(2)) }}>→</div>
        <div style={fadeUp(f, cue(2) + 10)}><Panel style={{ width: 300, textAlign: "center" }}><Big size={38}>CPU</Big><Small>runs it once</Small></Panel></div>
        <div style={{ fontSize: 60, color: C.muted, ...fadeUp(f, cue(2) + 30) }}>→</div>
        <div style={fadeUp(f, cue(2) + 45)}><Panel style={{ width: 500, textAlign: "center" }} color={C.bad}><Big size={38} color={C.bad}>❄ frozen constant</Big><Small style={{ fontFamily: MONO, fontSize: 24 }}>HLO: identity on a parameter</Small></Panel></div>
      </div>
      <div style={{ display: "flex", gap: 60, marginTop: 56, alignItems: "center" }}>
        <div style={{ ...fadeUp(f, cue(3)), transform: `${fadeUp(f, cue(3)).transform} rotate(-4deg)` }}>
          <div style={{ background: "#f4f1ea", padding: "16px 16px 50px", borderRadius: 6, width: 230 }}>
            <div style={{ background: "#1d2530", height: 150, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: MONO, fontSize: 40, color: C.gdn }}>0.7310</div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
          <div style={fadeUp(f, cue(4))}><Big size={44} color={C.gdn}>✓ tracing input: exact</Big></div>
          <div style={fadeUp(f, cue(5))}><Big size={44} color={C.bad}>✕ any other input: error 0.56 – 1.44</Big></div>
          <div style={fadeUp(f, cue(5) + 20)}><Small>inside the full model: compiles, then segfaults</Small></div>
        </div>
      </div>
    </Stage>
  );
};

/* ---------------------------------------------------------------- 4. the fix */
const Fix: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  return (
    <Stage kicker="The fix">
      <div style={{ display: "flex", gap: 30, ...fadeUp(f, 0) }}>
        <Panel color={C.gdn}><Big size={44}>prefill graph</Big></Panel>
        <Panel color={C.gdn}><Big size={44}>decode graph</Big></Panel>
        <Small style={{ alignSelf: "center" }}>one static graph per phase</Small>
      </div>
      <div style={{ marginTop: 56, ...fadeUp(f, cue(1)) }}>
        <Big size={64} style={{ fontFamily: MONO }}>X ← X − X·O<sub>b</sub>·X</Big>
        <div style={{ display: "flex", gap: 16, marginTop: 24 }}>
          {Array.from({ length: 6 }).map((_, i) => {
            const k = enter(f, cue(1) + 20 + i * 9, 8);
            return <div key={i} style={{ width: 120, height: 46, borderRadius: 23, border: `2px solid ${C.gdn}`, background: k > 0.5 ? C.gdn : "transparent", color: k > 0.5 ? C.bg : C.gdn, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: MONO, fontSize: 24 }}>step {i + 1}</div>;
          })}
        </div>
        <Small style={{ marginTop: 18 }}>exact inverse from matmuls only · max error 1.6e-7 on inputs it never saw · 2.9× faster than the row loop</Small>
      </div>
      <div style={{ display: "flex", gap: 30, marginTop: 50 }}>
        <div style={fadeUp(f, cue(2))}><Panel><Big size={38}>padding: β = 0, g = 0</Big><Small>state provably unchanged</Small></Panel></div>
        <div style={fadeUp(f, cue(3))}><Panel><Big size={38}>no data-dependent shapes</Big><Small>rank ≤ 4 · prefix sums as matmuls</Small></Panel></div>
      </div>
    </Stage>
  );
};

/* ---------------------------------------------------------------- 5. tensor parallel */
const TP: React.FC<SceneProps> = ({ cue }) => {
  const f = useCurrentFrame();
  const pulse = (start: number) => interpolate((f - start) % 45, [0, 10, 30], [0, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const on = f > cue(2) + 20 ? pulse(cue(2) + 20) : 0;
  return (
    <Stage kicker="Scaling to 27 B">
      <div style={{ display: "flex", gap: 50, alignItems: "center", ...fadeUp(f, 0) }}>
        <div style={{ width: 700 }}>
          <Small>bf16 weights</Small>
          <div style={{ height: 40, width: 700, background: C.bad, borderRadius: 8, marginTop: 8 }} />
          <Small style={{ marginTop: 6, fontFamily: MONO }}>54 GB</Small>
        </div>
        <div style={{ width: 410 }}>
          <Small>one chip</Small>
          <div style={{ height: 40, width: 410, background: C.line, borderRadius: 8, marginTop: 8 }} />
          <Small style={{ marginTop: 6, fontFamily: MONO }}>32 GB</Small>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 24, marginTop: 56 }}>
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} style={fadeUp(f, cue(1) + i * 4)}>
            <Panel style={{ padding: "18px 22px" }} color={on > 0.3 ? C.gdn : C.line}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <Big size={32}>core {i}</Big>
                <div style={{ width: 44, height: 44, borderRadius: 6, background: C.gdn, opacity: enter(f, cue(2), 15) * 0.9 }} />
              </div>
              <Small style={{ fontSize: 24, marginTop: 6 }}>3 q heads · 2 k / 6 v GDN heads · ⅛ MLP</Small>
            </Panel>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 40, marginTop: 36, alignItems: "center", ...fadeUp(f, cue(2) + 20) }}>
        <Big size={44} color={C.gdn}>Σ</Big>
        <Small style={{ fontSize: 34 }}>memory stays on each core · 2 all-reduces per layer · nothing else crosses</Small>
      </div>
    </Stage>
  );
};

/* ---------------------------------------------------------------- 6. result */
const DEVICE = " Paris. The capital of France is Paris. The capital of France is Paris.".split(" ");
const Result: React.FC<SceneProps> = ({ cue, props }) => {
  const f = useCurrentFrame();
  const n = Math.round(interpolate(f, [cue(1), cue(1) + 30], [0, 32], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const shown = Math.round(interpolate(f, [10, cue(1)], [0, DEVICE.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const bar = (ms: number) => (ms / 21.7) * 640;
  return (
    <Stage kicker="On the chip">
      <div style={{ display: "flex", gap: 30 }}>
        {["NeuronCores (bf16)", "CPU reference (fp32)"].map((h) => (
          <Panel key={h} style={{ flex: 1 }}>
            <Small>{h}</Small>
            <div style={{ fontFamily: MONO, fontSize: 30, marginTop: 10, minHeight: 90 }}>
              {DEVICE.slice(0, shown).map((w, i) => <span key={i} style={{ color: C.gdn }}>{w} </span>)}
            </div>
          </Panel>
        ))}
      </div>
      <div style={{ display: "flex", gap: 70, alignItems: "center", marginTop: 44 }}>
        <div style={fadeUp(f, cue(1))}><Big size={150} color={C.gdn}>{n}/32</Big><Small style={{ whiteSpace: "nowrap" }}>greedy tokens identical<br />KL ≤ 7.5e-4 vs fp32</Small></div>
        <div style={{ display: "flex", flexDirection: "column", gap: 20, ...fadeUp(f, cue(2)) }}>
          <Small>decode step</Small>
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ height: 44, width: bar(21.7) * enter(f, cue(2), 20), background: C.muted, borderRadius: 8 }} />
            <Small style={{ fontFamily: MONO, whiteSpace: "nowrap" }}>21.7 ms · 1 core, state via host</Small>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ height: 44, width: bar(6.3) * enter(f, cue(2) + 15, 20), background: C.gdn, borderRadius: 8 }} />
            <Small style={{ fontFamily: MONO, whiteSpace: "nowrap" }} color={C.gdn}>6.3 ms · 2 cores, state on chip</Small>
          </div>
        </div>
      </div>
      <Small style={{ fontSize: 22, marginTop: 30, ...fadeUp(f, cue(1)) }}>{props.resultNote}</Small>
    </Stage>
  );
};

/* ---------------------------------------------------------------- 7. cost */
const Cost: React.FC<SceneProps> = ({ cue, props }) => {
  const f = useCurrentFrame();
  // $/M tokens for one model copy at $2.15/hr / 4 copies, vs tok/s
  const perCopy = 2.15 / 4;
  const X0 = 40, X1 = 400, Y1 = 4; // tok/s range, $/M range
  const gw = 900, gh = 380;
  const px = (x: number) => ((x - X0) / (X1 - X0)) * gw;
  const py = (y: number) => gh - (Math.min(y, Y1) / Y1) * gh;
  const pts = Array.from({ length: 80 }, (_, i) => X0 + ((X1 - X0) * i) / 79).map((x) => `${px(x)},${py((perCopy / (3600 * x)) * 1e6)}`);
  const draw = enter(f, cue(2), 40);
  const be = (perCopy / 1.05 / 3600) * 1e6;
  return (
    <Stage kicker="Cost">
      <div style={{ display: "flex", gap: 60 }}>
        <div style={{ width: 640, ...fadeUp(f, cue(1)) }}>
          <Panel color={C.cost}>
            <Big size={40}>trn1.32xlarge · Spot</Big>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginTop: 20 }}>
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} style={{ height: 110, borderRadius: 10, border: `2px solid ${C.gdn}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24, color: C.gdn, opacity: enter(f, cue(1) + 15 + i * 8, 10) }}>27B #{i + 1}</div>
              ))}
            </div>
            <Big size={70} color={C.cost} style={{ marginTop: 26 }}>$2.15/hr</Big>
            <Small>= $0.54 per hour per model copy</Small>
          </Panel>
        </div>
        <div style={fadeUp(f, cue(2))}>
          <Small>$ per million tokens, one copy</Small>
          <svg width={gw + 40} height={gh + 70} style={{ marginTop: 10, overflow: "visible" }}>
            <line x1={0} y1={gh} x2={gw} y2={gh} stroke={C.line} strokeWidth={2} />
            <line x1={0} y1={py(1.05)} x2={gw} y2={py(1.05)} stroke={C.cost} strokeWidth={3} strokeDasharray="10 8" />
            <text x={gw - 4} y={py(1.05) - 12} fill={C.cost} fontSize={26} textAnchor="end">cheapest API $1.05/M</text>
            <polyline points={pts.slice(0, Math.max(2, Math.round(pts.length * draw))).join(" ")} fill="none" stroke={C.gdn} strokeWidth={5} />
            <circle cx={px(be)} cy={py(1.05)} r={11 * enter(f, cue(2) + 40, 10)} fill={C.gdn} />
            <text x={10} y={py(1.05) + 48} fill={C.gdn} fontSize={28} opacity={enter(f, cue(2) + 40, 10)}>break-even ≈ {Math.round(be)} tok/s</text>
            {[50, 100, 200, 300, 400].map((x) => (
              <text key={x} x={px(x)} y={gh + 36} fill={C.muted} fontSize={22} textAnchor="middle">{x}</text>
            ))}
            <text x={gw / 2} y={gh + 66} fill={C.muted} fontSize={22} textAnchor="middle">tokens / s per copy</text>
          </svg>
        </div>
      </div>
      <Small style={{ fontSize: 22, marginTop: 20 }}>{props.costNote}</Small>
    </Stage>
  );
};

/* ---------------------------------------------------------------- 8. CTA */
const CTA: React.FC<SceneProps> = ({ props }) => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", paddingBottom: 120 }}>
      <div style={fadeUp(f, 0)}><Big size={84}>Qwen3.8-27B on NeuronCores</Big></div>
      <div style={fadeUp(f, 12)}><Small style={{ fontSize: 36, marginTop: 20, textAlign: "center" }}>code · paper · one-command Spot launcher</Small></div>
      <div style={fadeUp(f, 24)}><Big size={50} color={C.gdn} style={{ fontFamily: MONO, marginTop: 50 }}>{props.repoUrl}</Big></div>
    </AbsoluteFill>
  );
};

export const SCENES: Record<string, React.FC<SceneProps>> = { hook: Hook, stakes: Stakes, bug: Bug, fix: Fix, tp: TP, result: Result, cost: Cost, cta: CTA };
