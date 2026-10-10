"""Generate synthetic audio with Piper (open-source neural TTS), LibriTTS-R model: 904 speakers
modeled on real human recordings, far more varied than macOS `say`.

pos: wake word (label jarvis) | neg: confusable words (label _unknown_)
cmd: the 10 commands (label = the word). TTS audio then appears in *every* class, so the
model can't use "sounds synthetic" as a shortcut for the wake word.

Writes data/synth_piper/{pos,neg,cmd}/<spk>__<word>__<i>.wav + voices.json (held-out test speakers).
"""
import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

from common import COMMANDS, DATA, SR, WAKE, trim_silence
from gen_synth import NEG_WORDS

OUT = DATA / "synth_piper"
MODEL = DATA / "piper" / "en_US-libritts_r-medium.onnx"
POS_TEXTS = [WAKE, WAKE.capitalize() + "!", WAKE.capitalize() + "?", WAKE.capitalize() + "."]
NEG = sorted(set(NEG_WORDS + ["marvin", "jarvik", "garvey", "harvey", "davis", "mavis", "jervis",
                              "carver", "starving", "jars of", "drive us", "are this", "nervous"]))

_voice = None


def init():
    global _voice
    from piper import PiperVoice
    torch.set_num_threads(1)
    _voice = PiperVoice.load(str(MODEL))


def synth(job):
    from piper import SynthesisConfig
    spk, text, out, seed = job
    if out.exists():
        return
    r = random.Random(seed)
    cfg = SynthesisConfig(speaker_id=spk, length_scale=r.uniform(0.75, 1.5),
                          noise_scale=r.uniform(0.4, 1.0), noise_w_scale=r.uniform(0.5, 1.2))
    a = np.concatenate([c.audio_int16_array for c in _voice.synthesize(text, cfg)])
    sr = _voice.config.sample_rate
    a = AF.resample(torch.from_numpy(a.astype(np.float32)), sr, SR).numpy()
    a = trim_silence(np.clip(a, -32768, 32767).astype(np.int16))
    if 0.15 * SR < len(a) <= 1.2 * SR:
        sf.write(out, a, SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-per-spk", type=int, default=4)
    ap.add_argument("--neg-per-spk", type=int, default=8)
    ap.add_argument("--cmd-per-spk", type=int, default=3)
    ap.add_argument("--test-spk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    n_spk = 904
    rng = random.Random(0)
    test = sorted(rng.sample(range(n_spk), args.test_spk))
    for d in ("pos", "neg", "cmd"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    (OUT / "voices.json").write_text(json.dumps({"all": [f"s{i}" for i in range(n_spk)],
                                                 "test": [f"s{i}" for i in test]}))
    jobs = []
    for s in range(n_spk):
        for i in range(args.pos_per_spk):
            jobs.append((s, POS_TEXTS[i % len(POS_TEXTS)], OUT / "pos" / f"s{s}__{WAKE}__{i}.wav"))
        for i, w in enumerate(rng.sample(NEG, args.neg_per_spk)):
            jobs.append((s, w, OUT / "neg" / f"s{s}__{w.replace(' ', '_')}__{i}.wav"))
        for i, w in enumerate(rng.sample(COMMANDS, args.cmd_per_spk)):
            jobs.append((s, w, OUT / "cmd" / f"s{s}__{w}__{i}.wav"))
    jobs = [(*j, k) for k, j in enumerate(jobs)]

    with Pool(args.workers, initializer=init) as pool:
        list(tqdm(pool.imap_unordered(synth, jobs, chunksize=16), total=len(jobs)))
    for d in ("pos", "neg", "cmd"):
        print(d, len(list((OUT / d).glob("*.wav"))), "clips")


if __name__ == "__main__":
    main()
