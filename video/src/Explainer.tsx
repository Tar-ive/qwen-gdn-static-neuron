import React from "react";
import { AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame, interpolate } from "remotion";
import timing from "./timing.json";
import { C, FPS, MONO, enter } from "./style";
import { SCENES, LIGHT } from "./scenes";

type Cue = { text: string; start: number; end: number };
type SceneTiming = { id: string; seconds: number; cues: Cue[] };
const T = timing as SceneTiming[];

const TAIL = 0.5; // s of breathing room after each scene's narration
const END_HOLD = 2.5; // s on the end card
const sceneFrames = (i: number) =>
  Math.ceil((T[i].seconds + (i === T.length - 1 ? END_HOLD : TAIL)) * FPS);
export const totalFrames = () => T.reduce((a, _, i) => a + sceneFrames(i), 0);

export type SceneProps = { cue: (i: number) => number; dur: number; props: Record<string, string> };

/** split a sentence into caption chunks of <= 2 lines x 42 chars, timed by character share */
function chunks(c: Cue): Cue[] {
  const words = c.text.split(" ");
  const n = Math.ceil(c.text.length / 80); // even split into n chunks of <= ~80 chars (2 lines)
  const target = c.text.length / n;
  const out: string[] = [];
  let cur = "";
  for (const w of words) {
    if (cur && (cur + " " + w).length > target + 6 && out.length < n - 1) {
      out.push(cur);
      cur = w;
    } else cur = cur ? cur + " " + w : w;
  }
  if (cur) out.push(cur);
  const total = out.reduce((a, s) => a + s.length, 0);
  let t = c.start;
  return out.map((s) => {
    const d = ((c.end - c.start) * s.length) / total;
    const r = { text: s, start: t, end: t + d };
    t += d;
    return r;
  });
}

const Captions: React.FC<{ cues: Cue[]; light: boolean }> = ({ cues, light }) => {
  const f = useCurrentFrame();
  const t = f / FPS;
  const cur = cues.flatMap(chunks).find((c) => t >= c.start && t < c.end + 0.15);
  if (!cur) return null;
  const k = enter(f, cur.start * FPS, 6);
  return (
    <div style={{ position: "absolute", bottom: 54, width: "100%", display: "flex", justifyContent: "center" }}>
      <div style={{
        maxWidth: 1300, padding: "10px 22px", borderRadius: 12, fontFamily: MONO, fontSize: 30, lineHeight: 1.35, textAlign: "center",
        color: light ? "#222" : C.ink, background: light ? "rgba(255,255,255,0.85)" : "rgba(9,9,10,0.72)", opacity: k,
      }}>
        {cur.text}
      </div>
    </div>
  );
};

/** scene in/out: opacity + focus pull, like the reference's soft cuts */
const Fade: React.FC<{ dur: number; children: React.ReactNode }> = ({ dur, children }) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, 9, dur - 9, dur], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return <AbsoluteFill style={{ opacity: o, filter: `blur(${(1 - o) * 14}px)` }}>{children}</AbsoluteFill>;
};

export const Explainer: React.FC<Record<string, string>> = (props) => {
  let from = 0;
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      {T.map((s, i) => {
        const dur = sceneFrames(i);
        const Scene = SCENES[s.id];
        const start = from;
        from += dur;
        const cue = (k: number) => Math.round((s.cues[Math.min(k, s.cues.length - 1)]?.start ?? 0) * FPS);
        return (
          <Sequence key={s.id} from={start} durationInFrames={dur} name={s.id}>
            <Audio src={staticFile(`vo/${s.id}.wav`)} />
            <Fade dur={dur}>
              <Scene cue={cue} dur={dur} props={props} />
            </Fade>
            <Captions cues={s.cues} light={LIGHT.has(s.id)} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
