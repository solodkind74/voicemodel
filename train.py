"""Train the keyword-spotting CNN on Speech Commands (real) + TTS wake word (synthetic).

uv run python train.py                          # full recipe -> runs/main
uv run python train.py --no-hard-neg --out runs/no_hard_neg   # ablation
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from common import (CLIP, COMMANDS, L2I, LABELS, SC_DIR, SYN_DIR, KWSNet, device, fit_1s,
                    load_wav, trim_silence)

CACHE = SC_DIR.parent / "cache_{}.npz"


# ---------------------------------------------------------------- data
SYNTH = {"mac": (SYN_DIR, 8), "piper": (SYN_DIR.parent / "synth_piper", 60)}
MINE = SC_DIR.parent / "mine"


def build_index(synth=("mac", "piper"), seed=0):
    """Returns list of (path, label, split, source)."""
    rng = random.Random(seed)
    val = set((SC_DIR / "validation_list.txt").read_text().split())
    test = set((SC_DIR / "testing_list.txt").read_text().split())
    items = []
    for d in sorted(p for p in SC_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")):
        label = d.name if d.name in COMMANDS else "_unknown_"
        for f in sorted(d.glob("*.wav")):
            rel = f"{d.name}/{f.name}"
            split = "val" if rel in val else "test" if rel in test else "train"
            items.append((str(f), label, split, "real"))

    # cap real _unknown_ so it doesn't swamp everything (keep all "marvin": closest to "jarvis")
    caps = {"train": 9000, "val": 1000, "test": 1000}
    unk = {s: [i for i in items if i[1] == "_unknown_" and i[2] == s] for s in caps}
    keep = set()
    for s, lst in unk.items():
        marvin = [i for i in lst if "/marvin/" in i[0]]
        rest = [i for i in lst if "/marvin/" not in i[0]]
        keep.update(marvin + rng.sample(rest, caps[s] - len(marvin)))
    items = [i for i in items if i[1] != "_unknown_" or i in keep]

    # synthetic: split by *voice* so val/test speakers are never seen in training
    for src in synth:
        root, n_val = SYNTH[src]
        vj = json.loads((root / "voices.json").read_text())
        tag = lambda v: v.replace(" ", "_").replace("(", "").replace(")", "")
        test_v = {tag(v) for v in vj["test"]}
        train_v = sorted(tag(v) for v in vj["all"] if tag(v) not in test_v)
        val_v = set(random.Random(seed + 1).sample(train_v, n_val))
        for kind in ("pos", "neg", "cmd"):
            for f in sorted((root / kind).glob("*.wav")):
                v, word = f.name.split("__")[:2]
                label = {"pos": "jarvis", "neg": "_unknown_", "cmd": word}[kind]
                split = "test" if v in test_v else "val" if v in val_v else "train"
                items.append((str(f), label, split, src))
    return items


def load_mine():
    """Your own recordings from `listen.py --test WORD` (data/mine/<word>/*.wav): the real-world check."""
    fs = sorted(MINE.glob("*/*.wav"))
    if not fs:
        return None
    x = torch.from_numpy(np.stack([fit_1s(trim_silence(load_wav(f))) for f in fs]).astype(np.float32) / 32768)
    return x, torch.tensor([L2I[f.parent.name] for f in fs]), [f.parent.name for f in fs]


def load_all(items, synth):
    cache = Path(str(CACHE).format("_".join(synth)))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) == [i[0] for i in items]:
            return z["audio"]
    with ThreadPoolExecutor(8) as ex:
        audio = np.stack(list(ex.map(lambda i: fit_1s(load_wav(i[0])), items)))
    np.savez(cache, paths=np.array([i[0] for i in items]), audio=audio)
    return audio


def load_noise():
    """Background noise files -> (train_part, eval_part) float tensors; 80/20 split per file."""
    tr, ev = [], []
    for f in sorted((SC_DIR / "_background_noise_").glob("*.wav")):
        a = load_wav(f).astype(np.float32) / 32768
        k = int(len(a) * 0.8)
        tr.append(a[:k]); ev.append(a[k:])
    return torch.from_numpy(np.concatenate(tr)), torch.from_numpy(np.concatenate(ev))


def noise_crops(noise, n, gen=None):
    starts = torch.randint(0, len(noise) - CLIP, (n,), generator=gen)
    return noise[starts[:, None] + torch.arange(CLIP)]


def make_split(items, audio, split, noise, n_silence, seed):
    idx = [k for k, i in enumerate(items) if i[2] == split]
    x = torch.from_numpy(audio[idx].astype(np.float32) / 32768)
    y = torch.tensor([L2I[items[k][1]] for k in idx])
    src = [items[k][3] for k in idx]
    g = torch.Generator().manual_seed(seed)
    sil = noise_crops(noise, n_silence, g) * torch.rand(n_silence, 1, generator=g)
    x = torch.cat([x, sil])
    y = torch.cat([y, torch.full((n_silence,), L2I["_silence_"])])
    return x, y, src + ["noise"] * n_silence


# ---------------------------------------------------------------- augmentation (on GPU)
def reverb(x):
    B, dev = len(x), x.device
    L = 6400  # 0.4 s tail
    t = torch.arange(L, device=dev) / 16000
    rt60 = torch.empty(B, 1, device=dev).uniform_(0.15, 0.8)
    tail = torch.randn(B, L, device=dev) * torch.exp(-6.9 * t / rt60)
    h = tail / tail.norm(dim=1, keepdim=True) * torch.empty(B, 1, device=dev).uniform_(0.3, 1.5)
    h[:, 0] = 1.0
    n = CLIP + L
    y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h, n), n)[:, :CLIP]
    return y * (x.abs().amax(1, keepdim=True) / y.abs().amax(1, keepdim=True).clamp_min(1e-6))


def shift(x, s):
    """Shift each row by s samples (B, 1), zero-filling."""
    pos = torch.arange(CLIP, device=x.device)[None] - s
    return torch.where((pos >= 0) & (pos < CLIP), x.gather(1, pos.clamp(0, CLIP - 1)), 0.0)


def add_context(x, p=0.7):
    """Streaming realism: paste the edge of another clip's word into the window, as when words are
    spoken back to back. A partner shifted by 0.55-0.85 s leaves only its first/last 50-350 ms visible."""
    B, dev = len(x), x.device
    partner = x[torch.randperm(B, device=dev)]
    mag = torch.randint(int(0.55 * CLIP), int(0.85 * CLIP), (B, 1), device=dev)
    sign = torch.where(torch.rand(B, 1, device=dev) < 0.5, -1, 1)
    frag = shift(partner, mag * sign) * torch.empty(B, 1, device=dev).uniform_(0.3, 1.0)
    return x + frag * (torch.rand(B, 1, device=dev) < p)


def augment(x, noise, is_synth):
    B, dev = len(x), x.device
    # random time shift +-150 ms
    x = shift(x, torch.randint(-2400, 2401, (B, 1), device=dev))
    # TTS audio is studio-clean: randomly muffle it (lowpass) to look more like a laptop mic
    # (FFT-domain soft rolloff; a biquad's sample-by-sample loop is very slow on GPU)
    if is_synth.any():
        X = torch.fft.rfft(x)
        f = torch.linspace(0, 8000, X.shape[1], device=dev)
        cut = torch.empty(B, 1, device=dev).uniform_(2500, 7000)
        mask = torch.where(is_synth[:, None], 1 / (1 + (f / cut) ** 4), 1.0)
        x = torch.fft.irfft(X * mask, n=CLIP)
    # room reverb on 50%: random synthetic impulse response (direct path + decaying noise tail)
    x = torch.where(torch.rand(B, 1, device=dev) < 0.5, reverb(x), x)
    # background noise at random SNR (0-25 dB) for 80% of samples
    n = noise_crops(noise, B).to(dev)
    sp = x.pow(2).mean(1, keepdim=True).clamp_min(1e-8)
    npow = n.pow(2).mean(1, keepdim=True).clamp_min(1e-8)
    snr = torch.empty(B, 1, device=dev).uniform_(0, 25)
    scale = torch.sqrt(sp / (npow * 10 ** (snr / 10)))
    x = x + n * scale * (torch.rand(B, 1, device=dev) < 0.8)
    # random gain -20..+6 dB
    x = x * 10 ** (torch.empty(B, 1, device=dev).uniform_(-20, 6) / 20)
    return x.clamp(-1, 1)


# ---------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, x, bs=1024):
    model.eval()
    dev = next(model.parameters()).device
    return torch.cat([model(x[i:i + bs].to(dev)).softmax(-1).cpu() for i in range(0, len(x), bs)])


def report(probs, y, src):
    pred = probs.argmax(-1)
    src = np.array(src)
    out = {"acc": (pred == y).float().mean().item()}
    for l in LABELS:
        m = y == L2I[l]
        out[f"acc/{l}"] = (pred[m] == y[m]).float().mean().item()
    W = L2I["jarvis"]
    real_non = torch.from_numpy(np.isin(src, ["real", "noise"])) & (y != W)
    out["jarvis_false_accept/real"] = (pred[real_non] == W).float().mean().item()
    for s in ("mac", "piper"):
        m = torch.from_numpy(src == s)
        if m.any():
            out[f"jarvis_recall/{s}"] = (pred[m & (y == W)] == W).float().mean().item()
            out[f"jarvis_false_accept/{s}"] = (pred[m & (y != W)] == W).float().mean().item()
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--no-hard-neg", action="store_true", help="drop TTS negatives from train")
    ap.add_argument("--synth", default="mac,piper", help="synthetic sources: mac,piper")
    ap.add_argument("--out", default="runs/main")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    synth = tuple(args.synth.split(","))
    items = build_index(synth)
    audio = load_all(items, synth)
    noise_tr, noise_ev = load_noise()
    xtr, ytr, str_ = make_split(items, audio, "train", noise_tr, 3000, 1)
    xva, yva, sva = make_split(items, audio, "val", noise_ev, 400, 2)
    keep = torch.ones(len(ytr), dtype=torch.bool)
    if args.no_hard_neg:
        keep = ~((torch.tensor([s in SYNTH for s in str_])) & (ytr == L2I["_unknown_"]))
    xtr, ytr = xtr[keep], ytr[keep]
    str_ = [s for s, k in zip(str_, keep) if k]
    synth_tr = torch.tensor([s in SYNTH for s in str_])
    print(f"loaded in {time.time() - t0:.0f}s | train {len(ytr)} val {len(yva)}")
    print("train counts:", {l: int((ytr == i).sum()) for i, l in enumerate(LABELS)})

    # class-balanced sampling (jarvis has far fewer clips than real words)
    counts = torch.bincount(ytr, minlength=len(LABELS)).float()
    cw = 1 / counts
    cw[L2I["_unknown_"]] *= 2  # unknown is the most varied class
    sample_w = cw[ytr]

    dev = device()
    model = KWSNet().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k on {dev}")
    specaug = torch.nn.Sequential(T.FrequencyMasking(7, iid_masks=True),
                                  T.TimeMasking(20, iid_masks=True)).to(dev)
    steps = len(ytr) // args.bs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps)
    noise_dev = noise_tr.to(dev)

    best = -1
    for ep in range(args.epochs):
        model.train()
        t = time.time(); tot = 0.0
        order = torch.multinomial(sample_w, steps * args.bs, replacement=True)
        for s in range(steps):
            b = order[s * args.bs:(s + 1) * args.bs]
            x = augment(xtr[b].to(dev), noise_dev, synth_tr[b].to(dev))
            loss = F.cross_entropy(model(x, specaug), ytr[b].to(dev), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        r = report(predict(model, xva), yva, sva)
        print(f"ep {ep + 1:2d} loss {tot / steps:.3f} | val acc {r['acc']:.3f} jarvis " + " ".join(
            f"{k.split('_')[-1]} {v:.3f}" for k, v in r.items() if k.startswith("jarvis"))
            + f" | {time.time() - t:.0f}s")
        if r["acc"] > best:
            best = r["acc"]
            torch.save(model.state_dict(), out / "model.pt")

    model.load_state_dict(torch.load(out / "model.pt"))
    xte, yte, ste = make_split(items, audio, "test", noise_ev, 400, 3)
    r = report(predict(model, xte), yte, ste)
    mine = load_mine()
    if mine:
        p = predict(model, mine[0]).argmax(-1)
        for w in sorted(set(mine[2])):
            m = torch.tensor([v == w for v in mine[2]])
            r[f"mine/{w}"] = (p[m] == mine[1][m]).float().mean().item()
    (out / "metrics.json").write_text(json.dumps(r, indent=1))
    print("TEST:", json.dumps({k: round(v, 4) for k, v in r.items()}, indent=1))


if __name__ == "__main__":
    main()
