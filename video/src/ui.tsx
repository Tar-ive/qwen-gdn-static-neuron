import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { C, MONO, SANS, clamp } from "./style";

export const Kicker: React.FC<{ children: React.ReactNode; style?: React.CSSProperties; color?: string }> = ({ children, style, color }) => (
  <div style={{ fontFamily: MONO, fontSize: 22, letterSpacing: "0.2em", textTransform: "uppercase", color: color ?? C.dim, ...style }}>{children}</div>
);

export const Glass: React.FC<{ children: React.ReactNode; style?: React.CSSProperties; glow?: boolean }> = ({ children, style, glow }) => (
  <div style={{
    background: C.card, border: `1.5px solid ${glow ? "rgba(255,255,255,0.75)" : C.edge}`, borderRadius: 30,
    boxShadow: glow ? "0 0 60px rgba(255,255,255,0.12), 0 30px 80px rgba(0,0,0,0.6)" : "0 30px 80px rgba(0,0,0,0.6)",
    ...style,
  }}>{children}</div>
);

export const Pill: React.FC<{ children: React.ReactNode; solid?: boolean; dashed?: boolean; style?: React.CSSProperties }> = ({ children, solid, dashed, style }) => (
  <div style={{
    display: "inline-flex", alignItems: "center", gap: 12, whiteSpace: "nowrap", padding: "12px 24px", borderRadius: 999,
    fontFamily: MONO, fontSize: 24, letterSpacing: "0.06em", textTransform: "uppercase",
    background: solid ? C.ink : "rgba(20,20,22,0.9)", color: solid ? "#111" : C.dim,
    border: dashed ? `1.5px dashed ${C.edge}` : solid ? "none" : `1.5px solid ${C.edge}`, ...style,
  }}>{children}</div>
);

export const Icon: React.FC<{ children: React.ReactNode; size?: number; color?: string }> = ({ children, size = 40, color }) => (
  <div style={{ width: size, height: size, borderRadius: size * 0.28, border: `1.5px solid ${C.edge}`, background: "rgba(255,255,255,0.04)",
    display: "flex", alignItems: "center", justifyContent: "center", fontSize: size * 0.5, color: color ?? C.ink, fontFamily: MONO, flex: "none" }}>{children}</div>
);

/** dark stage with soft vignette; optional dim, blurred layer behind (depth like the reference) */
export const Stage: React.FC<{ children: React.ReactNode; behind?: React.ReactNode }> = ({ children, behind }) => {
  const f = useCurrentFrame();
  const s = interpolate(f, [0, 600], [1, 1.05], clamp);
  return (
    <AbsoluteFill style={{ background: `radial-gradient(ellipse at 50% 45%, #17171A 0%, ${C.bg} 70%)`, fontFamily: SANS, color: C.ink }}>
      {behind && <AbsoluteFill style={{ opacity: 0.22, filter: "blur(5px)", transform: `scale(${s * 1.04})` }}>{behind}</AbsoluteFill>}
      <AbsoluteFill style={{ transform: `scale(${s})` }}>{children}</AbsoluteFill>
    </AbsoluteFill>
  );
};

/** thin white bezier connector, drawn on from 0..1 */
export const Curve: React.FC<{ d: string; k: number; dashed?: boolean; opacity?: number; width?: number }> = ({ d, k, dashed, opacity = 0.7, width = 2 }) => (
  <path d={d} fill="none" stroke="white" strokeOpacity={opacity} strokeWidth={width} pathLength={1}
    strokeDasharray={dashed ? "0.006 0.008" : `${k} 1`} style={dashed ? { opacity: k } : undefined} />
);

/** deterministic dot-sphere ("network globe") */
const PTS = Array.from({ length: 420 }, (_, i) => {
  const y = 1 - (i / 419) * 2, r = Math.sqrt(1 - y * y), th = i * 2.399963;
  return [Math.cos(th) * r, y, Math.sin(th) * r];
});
export const Globe: React.FC<{ rx?: number; ry?: number; cx?: number; cy?: number; speed?: number; opacity?: number }> = ({ rx = 620, ry = 330, cx = 960, cy = 540, speed = 0.004, opacity = 1 }) => {
  const f = useCurrentFrame();
  const a = f * speed;
  const pts = PTS.map(([x, y, z]) => {
    const X = x * Math.cos(a) + z * Math.sin(a), Z = -x * Math.sin(a) + z * Math.cos(a);
    return { x: cx + X * rx, y: cy + y * ry * 0.98 + Z * 40, z: Z };
  });
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, opacity }}>
      {pts.map((p, i) => {
        const q = pts[(i + 34) % pts.length];
        const front = (p.z + 1) / 2;
        return (
          <g key={i}>
            {i % 3 === 0 && Math.hypot(p.x - q.x, p.y - q.y) < 140 && <line x1={p.x} y1={p.y} x2={q.x} y2={q.y} stroke="white" strokeOpacity={0.05 + 0.06 * front} />}
            {i % 11 === 0
              ? <rect x={p.x - 6} y={p.y - 6} width={12} height={12} rx={3} fill="none" stroke="white" strokeOpacity={0.15 + 0.3 * front} />
              : <circle cx={p.x} cy={p.y} r={1.6} fill="white" fillOpacity={0.12 + 0.35 * front} />}
          </g>
        );
      })}
    </svg>
  );
};
