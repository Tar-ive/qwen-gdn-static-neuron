import { interpolate, Easing } from "remotion";

export const FPS = 30;
export const W = 1920;
export const H = 1080;

// color = meaning, never reassigned
export const C = {
  bg: "#0B0F14",
  panel: "#121922",
  line: "#243042",
  text: "#E6EDF3",
  muted: "#8B98A5",
  gdn: "#2DD4BF", // Gated DeltaNet / on-chip / correct
  attn: "#A78BFA", // full attention
  bad: "#F87171", // wrong / frozen / does not fit
  cost: "#FBBF24", // money
};
export const SANS = "'SF Pro Display', 'Helvetica Neue', Helvetica, Arial, sans-serif";
export const MONO = "'SF Mono', Menlo, Consolas, monospace";

export const EASE = Easing.bezier(0.22, 1, 0.36, 1);

/** 0..1 progress of an entrance starting at frame `start` lasting `dur` frames */
export const enter = (frame: number, start: number, dur = 15) =>
  interpolate(frame, [start, start + dur], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: EASE });

export const fadeUp = (frame: number, start: number, dur = 15, dist = 24): React.CSSProperties => {
  const k = enter(frame, start, dur);
  return { opacity: k, transform: `translateY(${(1 - k) * dist}px)` };
};
