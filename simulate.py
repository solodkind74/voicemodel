"""Replay recordings through the exact live-stream Detector to tune hop/smoothing/threshold offline.

Builds a stream: [1-2 s of background noise] + clip + [1-2 s noise] + ..., then counts which clips
triggered the right word, and false triggers on _unknown_ words (real Speech Commands test speakers).

uv run python simulate.py --gap 0.2 0.6          # words spoken close together
uv run python simulate.py --grid                 # try many detector settings
"""
import argparse
import itertools
import random

import numpy as np

from common import COMMANDS, SC_DIR, SR, fit_1s, load_wav
from listen import MINE, Detector, load
from train import load_noise


def make_stream(clips, noise, rng, gap_s=(1.0, 2.0)):
    parts, spans = [], []
    t = 0
    for c in clips:
        gap = noise[(s := rng.randrange(len(noise) - 3 * SR)): s + rng.randrange(int(gap_s[0] * SR), int(gap_s[1] * SR))] * 0.05
        parts += [gap, c]
        spans.append((t + len(gap), t + len(gap) + len(c)))
        t += len(gap) + len(c)
    parts.append(noise[:2 * SR] * 0.05)
    return np.concatenate(parts).astype(np.float32), spans


def run(model, audio, spans, labels, thresh, hop, smooth):
    det = Detector(model, thresh, hop, smooth)
    n = int(hop * SR)
    hits = []
    for i in range(0, len(audio) - n, n):
        h = det.feed(audio[i:i + n])
        if h:
            hits.append((i + n, h[0]))
    # credit each detection to the word overlapping its 1 s window the most
    got = [None] * len(spans)
    for t, w in hits:
        ov = [max(0, min(b, t) - max(a, t - SR)) for a, b in spans]
        k = int(np.argmax(ov))
        if ov[k] > 0 and got[k] is None:
            got[k] = w
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hubert")
    ap.add_argument("--n-real", type=int, default=6, help="real Speech Commands clips per command")
    ap.add_argument("--gap", type=float, nargs=2, default=(1.0, 2.0), help="silence between words, min max (s)")
    ap.add_argument("--grid", action="store_true", help="try many detector settings (default: just the live defaults)")
    args = ap.parse_args()
    rng = random.Random(0)
    model = load(args.run)
    _, noise = load_noise()
    noise = noise.numpy()

    test = (SC_DIR / "testing_list.txt").read_text().split()
    clips, labels = [], []
    for f in sorted(MINE.glob("jarvis/*.wav")):  # your recordings: keep full length, no trimming
        clips.append(load_wav(f).astype(np.float32) / 32768); labels.append("jarvis")
    for w in COMMANDS:
        for rel in rng.sample([r for r in test if r.startswith(w + "/")], args.n_real):
            clips.append(load_wav(SC_DIR / rel).astype(np.float32) / 32768); labels.append(w)
    unk = [r for r in test if r.split("/")[0] in ("marvin", "house", "bird", "sheila", "learn", "follow")]
    for rel in rng.sample(unk, 30):
        clips.append(load_wav(SC_DIR / rel).astype(np.float32) / 32768); labels.append("_unknown_")
    order = list(range(len(clips))); rng.shuffle(order)
    clips, labels = [clips[i] for i in order], [labels[i] for i in order]
    # Speech Commands clips are 1 s with silence padding; trim it so short gaps really mean close words
    clips = [c[i[0]:i[-1] + 1] if len(i := np.where(np.abs(c) > 0.1 * np.abs(c).max())[0]) else c for c in clips]
    audio, spans = make_stream(clips, noise, rng, args.gap)
    print(f"stream {len(audio) / SR:.0f}s, {len(clips)} words\n")

    print(f"{'hop':>5} {'smooth':>6} {'thresh':>6} | {'jarvis':>7} {'commands':>9} {'false trig':>10}")
    grid = itertools.product((0.25, 0.1), (3, 2, 1), (0.8, 0.6)) if args.grid else [(0.25, 3, 0.8), (0.1, 3, 0.8)]
    for hop, smooth, thresh in grid:
        got = run(model, audio, spans, labels, thresh, hop, smooth)
        j = [g == l for g, l in zip(got, labels) if l == "jarvis"]
        c = [g == l for g, l in zip(got, labels) if l in COMMANDS]
        fa = [g is not None for g, l in zip(got, labels) if l == "_unknown_"]
        print(f"{hop:5.2f} {smooth:6d} {thresh:6.1f} | {sum(j):3d}/{len(j):<3d} {sum(c):4d}/{len(c):<4d} "
              f"{sum(fa):5d}/{len(fa):<4d}")


if __name__ == "__main__":
    main()
