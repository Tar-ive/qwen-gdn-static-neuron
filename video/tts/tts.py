"""Narration: one WAV per scene with Kokoro-82M, plus timing.json (seconds per scene) for the Remotion composition.

usage: uv run --python 3.12 --with kokoro --with soundfile --with pip python tts/tts.py
"""
import json
import re
import pathlib

import numpy as np
import soundfile as sf
from kokoro import KPipeline

HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "public" / "vo"
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads((HERE / "script.json").read_text())
pipe = KPipeline(lang_code="a")
SR = 24000
timing = []
for sc in cfg["scenes"]:
    parts, cues, t = [], [], 0.0
    for sent in re.split(r"(?<=[.!?])\s+", sc["vo"].strip()):           # one cue per sentence, timed from its audio
        a = np.concatenate([np.asarray(x) for _, _, x in pipe(sent, voice=cfg["voice"], speed=cfg["speed"])])
        loud = np.flatnonzero(np.abs(a) > 0.01)                             # trim Kokoro's lead/trail silence
        a = a[max(0, loud[0] - int(0.03 * SR)): loud[-1] + int(0.06 * SR)]
        gap = np.zeros(int(0.22 * SR), dtype=a.dtype)
        cues.append({"text": sent, "start": round(t, 3), "end": round(t + len(a) / SR, 3)})
        parts += [a, gap]
        t += (len(a) + len(gap)) / SR
    audio = np.concatenate(parts)
    sf.write(OUT / f"{sc['id']}.wav", audio, SR)
    timing.append({"id": sc["id"], "seconds": round(len(audio) / SR, 3), "cues": cues})
    print(f"{sc['id']:8s} {len(audio) / SR:5.1f}s  {len(sc['vo'].split())} words")
(HERE.parent / "src" / "timing.json").write_text(json.dumps(timing, indent=1))
print("total", round(sum(t["seconds"] for t in timing), 1), "s")
