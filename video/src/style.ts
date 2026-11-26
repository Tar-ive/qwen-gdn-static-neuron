import { interpolate, Easing, staticFile, continueRender, delayRender } from "remotion";
import { loadFont } from "@remotion/fonts";

export const FPS = 30;
export const W = 1920;
export const H = 1080;

// near-black stage, white ink; one accent for "real/valid", one for "frozen/invalid"
export const C = {
  bg: "#09090A",
  light: "#F1F0EB",
  ink: "#F5F5F4",
  dim: "#8A8A8E",
  faint: "rgba(255,255,255,0.10)",
  edge: "rgba(255,255,255,0.16)",
  card: "rgba(28,28,31,0.88)",
  gdn: "#5EEAD4",
  attn: "#C4B5FD",
  bad: "#F87171",
};
export const PIXEL = "'Geist Pixel', 'Geist Mono', monospace";
export const SANS = "'Geist', 'Helvetica Neue', Arial, sans-serif";
export const MONO = "'Geist Mono', Menlo, monospace";

const h = delayRender("fonts");
Promise.all([
  loadFont({ family: "Geist Pixel", url: staticFile("fonts/GeistPixel-Square.woff2") }),
  loadFont({ family: "Geist", url: staticFile("fonts/Geist-Variable.woff2"), weight: "100 900" }),
  loadFont({ family: "Geist Mono", url: staticFile("fonts/GeistMono-Variable.woff2"), weight: "100 900" }),
]).then(() => continueRender(h));

export const EASE = Easing.bezier(0.16, 1, 0.3, 1);
export const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
export const enter = (f: number, start: number, dur = 18) => interpolate(f, [start, start + dur], [0, 1], { ...clamp, easing: EASE });
/** blur-in entrance used everywhere (the reference's soft focus pull) */
export const reveal = (f: number, start: number, dur = 18, dy = 16): React.CSSProperties => {
  const k = enter(f, start, dur);
  return { opacity: k, transform: `translateY(${(1 - k) * dy}px)`, filter: `blur(${(1 - k) * 10}px)` };
};
