"""Live mic demo: streams 1 s windows every 250 ms and prints detected words.

uv run python listen.py                      # runs/hubert (best); --run runs/v2 for the small CNN
uv run python listen.py --thresh 0.9 --run runs/v1_mac
uv run python listen.py --test jarvis -n 10  # record 10 prompted clips of your voice, score them
uv run python listen.py --score              # re-score all saved clips (data/mine/<word>/) with --run
"""
import argparse
import collections
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from common import CLIP, DATA, LABELS, SR, HubertKWS, KWSNet, device, fit_1s, load_wav, trim_silence

MINE = DATA / "mine"


def load(run):
    hub = "hubert" in run
    m = HubertKWS() if hub else KWSNet()
    m.load_state_dict(torch.load(f"{run}/model.pt", map_location="cpu"), strict=not hub)
    return m.eval().to(device() if hub else "cpu")  # HuBERT (95M params) wants the GPU


@torch.no_grad()
def probs(model, a):
    dev = next(model.parameters()).device
    return model(torch.from_numpy(a.astype(np.float32))[None].to(dev)).softmax(-1)[0].cpu()


def stream(model, thresh, hop=0.25):
    buf = np.zeros(CLIP, np.float32)
    hist = collections.deque(maxlen=3)  # smooth over 3 consecutive windows
    last = 0.0
    q = collections.deque()
    print("listening… say: " + ", ".join(LABELS[:-2]) + "  (ctrl-c to quit)")
    with sd.InputStream(samplerate=SR, channels=1, blocksize=int(SR * hop),
                        callback=lambda d, *_: q.append(d[:, 0].copy())):
        while True:
            while not q:
                time.sleep(0.01)
            blk = q.popleft()
            buf = np.concatenate([buf[len(blk):], blk])
            hist.append(probs(model, buf))
            p = torch.stack(list(hist)).mean(0)
            k = int(p.argmax())
            if LABELS[k][0] != "_" and p[k] > thresh and time.time() - last > 1.0:
                last = time.time()
                print(f"  >> {LABELS[k]:<8} {p[k]:.2f}")


def classify(model, a_int16):
    p = probs(model, fit_1s(trim_silence(a_int16)).astype(np.float32) / 32768)
    top = p.topk(3)
    return LABELS[int(top.indices[0])], ", ".join(f"{LABELS[j]} {v:.2f}" for v, j in zip(top.values, top.indices))


def test(model, word, n):
    """Record n clips of `word`; each is saved to data/mine/<word>/ for later --score / training eval."""
    d = MINE / word
    d.mkdir(parents=True, exist_ok=True)
    hits = 0
    for i in range(n):
        input(f"[{i + 1}/{n}] press Enter, then say '{word}'")
        a = sd.rec(int(1.5 * SR), samplerate=SR, channels=1, dtype="int16")
        sd.wait()
        sf.write(d / f"{time.strftime('%Y%m%d-%H%M%S')}-{i}.wav", a[:, 0], SR)
        top, desc = classify(model, a[:, 0])
        print("   ", desc)
        hits += top == word
    print(f"{word}: {hits}/{n} correct (clips saved to {d})")


def score(model):
    for d in sorted(p for p in MINE.iterdir() if p.is_dir()):
        res = [classify(model, load_wav(f)) for f in sorted(d.glob("*.wav"))]
        hits = sum(top == d.name for top, _ in res)
        wrong = collections.Counter(top for top, _ in res if top != d.name)
        print(f"{d.name:<10} {hits}/{len(res)} correct   mistaken for: {dict(wrong)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hubert")
    ap.add_argument("--thresh", type=float, default=0.8)
    ap.add_argument("--test", metavar="WORD")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    m = load(a.run)
    score(m) if a.score else test(m, a.test, a.n) if a.test else stream(m, a.thresh)
