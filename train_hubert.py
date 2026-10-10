"""Transfer learning: frozen HuBERT features + small head, same data/splits as train.py.

Why: the from-scratch CNN learns TTS-specific acoustics, so a wake word learned only from TTS
doesn't transfer to real voices. A model pretrained on lots of real speech maps TTS and human
"jarvis" to nearby phonetic features.

Embeddings are precomputed once (train clips get one random augmentation), then the head
trains in seconds. uv run python train_hubert.py --out runs/hubert
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import L2I, LABELS, HubertKWS, device
from train import (add_context, augment, build_index, load_all, load_mine, load_noise, make_split, report)

EMB = Path("data/emb_hubert.pt")


@torch.no_grad()
def embed_all(model, x, noise=None, synth=None, context=False, bs=256):
    dev = device()
    out = []
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(dev)
        if noise is not None:
            if context:  # partner clips come from this batch (shuffled), so augment before noise/reverb
                b = add_context(b)
            b = augment(b, noise, synth[i:i + bs].to(dev))
        out.append(model.embed(b).half().cpu())
        print(f"\r  {i + len(b)}/{len(x)}", end="", flush=True)
    print()
    return torch.cat(out)


def cap(y, src, per_class, seed):
    """Subsample to at most per_class clips per label (keeps embedding time ~10 min)."""
    rng = random.Random(seed)
    keep = []
    for c in range(len(LABELS)):
        idx = [i for i in range(len(y)) if y[i] == c]
        keep += rng.sample(idx, min(per_class, len(idx)))
    return torch.tensor(sorted(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--context", action="store_true", help="neighbor-word fragments (streaming realism)")
    ap.add_argument("--copies", type=int, default=1, help="augmented copies of the train set")
    ap.add_argument("--out", default="runs/hubert")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = device()
    model = HubertKWS().to(dev)

    synth = ("mac", "piper")
    emb_train = Path(f"data/emb_hubert_train{'_ctx' if args.context else ''}_x{args.copies}.pt")
    d = torch.load(EMB) if EMB.exists() else {}
    if "val" not in d or not emb_train.exists():
        items = build_index(synth)
        audio = load_all(items, synth)
        noise_tr, noise_ev = load_noise()
    if "val" not in d:  # eval sets: embedded once, shared by all settings, never augmented
        for split, n_sil, seed, per in (("val", 400, 2, 400), ("test", 400, 3, 600)):
            x, y, src = make_split(items, audio, split, noise_ev, n_sil, seed)
            k = cap(y, src, per, seed)
            print(split, len(k))
            d[split] = (embed_all(model, x[k]), y[k], [src[i] for i in k])
        torch.save(d, EMB)
    mine = load_mine()  # re-embedded each run: you may have recorded more
    d["mine"] = (embed_all(model, mine[0]), mine[1], mine[2])
    if emb_train.exists():
        d["train"] = torch.load(emb_train)
    else:
        x, y, src = make_split(items, audio, "train", noise_tr, 3000, 1)
        k = cap(y, src, args.per_class, 1)
        x, y, src = x[k], y[k], [src[i] for i in k]
        is_syn = torch.tensor([s in synth for s in src])
        embs = []
        for c in range(args.copies):
            print(f"train copy {c + 1}/{args.copies}: {len(y)}")
            perm = torch.randperm(len(y))  # shuffle so in-batch context partners are random
            e = embed_all(model, x[perm], noise_tr.to(dev), is_syn[perm], args.context)
            embs.append(e[perm.argsort()])
        d["train"] = (torch.cat(embs), y.repeat(args.copies), src * args.copies)
        torch.save(d["train"], emb_train)

    etr, ytr, _ = d["train"]
    etr, ytr = etr.float().to(dev), ytr.to(dev)
    params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    bs, best = 256, -1
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(ytr), bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(model.classify(etr[b]), ytr[b], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            r = {}
            for s in ("val", "mine"):
                e, y, src = d[s]
                r[s] = report(model.classify(e.float().to(dev)).softmax(-1).cpu(), y, src)
        if ep % 10 == 9:
            print(f"ep {ep + 1} loss {loss.item():.3f} val acc {r['val']['acc']:.3f} "
                  f"jarvis piper {r['val'].get('jarvis_recall/piper', 0):.3f} | mine/jarvis {r['mine']['acc/jarvis']:.2f}")
        if r["val"]["acc"] > best:  # select on val only; "mine" is never used for decisions
            best = r["val"]["acc"]
            torch.save(model.state_dict(), out / "model.pt")

    model.load_state_dict(torch.load(out / "model.pt"), strict=False)
    model.eval()
    res = {}
    with torch.no_grad():
        for s in ("test", "mine"):
            e, y, src = d[s]
            res[s] = report(model.classify(e.float().to(dev)).softmax(-1).cpu(), y, src)
    r = {**res["test"], "mine/jarvis": res["mine"]["acc/jarvis"]}
    (out / "metrics.json").write_text(json.dumps(r, indent=1))
    print("TEST:", json.dumps({k: round(v, 4) for k, v in r.items() if v == v}, indent=1))


if __name__ == "__main__":
    main()
