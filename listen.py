"""Live mic demo: streams 1 s windows every 250 ms and prints detected words.

uv run python listen.py                      # runs/main/model.pt
uv run python listen.py --thresh 0.9 --run runs/no_hard_neg
uv run python listen.py --test jarvis -n 10  # record 10 prompted clips of your voice, score them
"""
import argparse
import collections
import time

import numpy as np
import sounddevice as sd
import torch

from common import CLIP, LABELS, SR, KWSNet, fit_1s, trim_silence


def load(run):
    m = KWSNet()
    m.load_state_dict(torch.load(f"{run}/model.pt", map_location="cpu"))
    return m.eval()


@torch.no_grad()
def probs(model, a):
    return model(torch.from_numpy(a.astype(np.float32))[None]).softmax(-1)[0]


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


def test(model, word, n):
    hits = 0
    for i in range(n):
        input(f"[{i + 1}/{n}] press Enter, then say '{word}'")
        a = sd.rec(int(1.5 * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        a = fit_1s((trim_silence((a[:, 0] * 32767).astype(np.int16)))).astype(np.float32) / 32768
        p = probs(model, a)
        top = p.topk(3)
        print("   ", ", ".join(f"{LABELS[j]} {v:.2f}" for v, j in zip(top.values, top.indices)))
        hits += LABELS[int(top.indices[0])] == word
    print(f"{word}: {hits}/{n} correct")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--thresh", type=float, default=0.8)
    ap.add_argument("--test", metavar="WORD")
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    m = load(a.run)
    test(m, a.test, a.n) if a.test else stream(m, a.thresh)
