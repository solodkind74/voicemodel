"""Generate synthetic training audio with macOS `say` (free, offline, ~100 voices).

Positives: the wake word in every voice x rate x intonation.
Hard negatives: confusable / random words in the same voices, labeled _unknown_.
Without them the model can learn "sounds like TTS => wake word" instead of the word itself.

Writes data/synth/{pos,neg}/<voice>__<rate>__<i>.wav and data/synth/voices.json
(which voices are held out for testing, so test = unseen speakers).
"""
import argparse
import json
import random
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from common import SR, SYN_DIR, WAKE, trim_silence

NOVELTY = {"Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Wobble",
           "Good News", "Jester", "Organ", "Superstar", "Trinoids", "Whisper", "Zarvox"}
# Non-English voices reading an English word = free accent diversity.
LOCALES = ("en_", "de_", "fr_", "es_", "it_", "nl_", "pt_", "sv_", "nb_", "da_", "pl_", "ru_", "fi_")
RATES = [110, 140, 170, 200, 230, 260]
POS_TEXTS = [WAKE, WAKE.capitalize() + "!", WAKE.capitalize() + "?"]
NEG_WORDS = ["jars", "jarvi", "harvest", "service", "nervous", "travis", "carvings", "marvel",
             "garbage", "jarred", "javis", "artist", "purpose", "charlie", "office", "nervous",
             "chavez", "jasmine", "darkness", "carpet", "parties", "hello", "computer", "alexa",
             "music", "weather", "seven", "banana", "kitchen", "really", "starfish", "jovial"]


def voices():
    out = []
    for line in subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z0-9]+)\s+#", line)
        if m and m[1] not in NOVELTY and m[2].startswith(LOCALES):
            out.append(m[1])
    return out


def synth(job):
    voice, rate, text, out = job
    if out.exists():
        return
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        r = subprocess.run(["say", "-v", voice, "-r", str(rate), f"--data-format=LEI16@{SR}",
                            "-o", tmp.name, text], capture_output=True)
        if r.returncode:
            return
        a, _ = sf.read(tmp.name, dtype="int16")
    if len(a) == 0 or np.abs(a).max() < 300:
        return
    a = trim_silence(a)
    if 0.15 * SR < len(a) <= 1.2 * SR:  # drop failed/garbled or too-long renders
        sf.write(out, a, SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-voices", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    vs = voices()
    rng = random.Random(0)
    test = sorted(rng.sample(vs, args.test_voices))
    print(f"{len(vs)} voices, {len(test)} held out for test: {test}")
    for d in ("pos", "neg"):
        (SYN_DIR / d).mkdir(parents=True, exist_ok=True)
    (SYN_DIR / "voices.json").write_text(json.dumps({"all": vs, "test": test}, indent=1))

    jobs = []
    for v in vs:
        tag = v.replace(" ", "_").replace("(", "").replace(")", "")
        for rate in RATES:
            for i, t in enumerate(POS_TEXTS):
                jobs.append((v, rate, t, SYN_DIR / "pos" / f"{tag}__{rate}__{i}.wav"))
        for i, w in enumerate(sorted(set(NEG_WORDS))):
            jobs.append((v, rng.choice(RATES), w, SYN_DIR / "neg" / f"{tag}__{w}__{i}.wav"))

    with ThreadPoolExecutor(args.workers) as ex:
        list(tqdm(ex.map(synth, jobs), total=len(jobs)))
    for d in ("pos", "neg"):
        print(d, len(list((SYN_DIR / d).glob("*.wav"))), "clips")


if __name__ == "__main__":
    main()
