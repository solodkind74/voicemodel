# voicemodel — train a voice-command + custom wake-word recognizer

Small CNN (0.57M params) on log-mel spectrograms, PyTorch on the M5 GPU (MPS). ~10 min train, $0.

- **Real data:** [Google Speech Commands v2](https://arxiv.org/abs/1804.03209) (CC-BY 4.0, 105k 1-s clips, ~2.6k speakers):
  10 commands `yes no up down left right on off stop go` + `_unknown_` (other 25 words) + `_silence_` (background noise).
- **Synthetic data:** a custom wake word **`jarvis`** that has no real recordings anywhere, generated with
  macOS `say`: 108 voices (incl. German/French/Russian/… voices reading English = accents) × 6 speeds × 3 intonations.
  Plus TTS **hard negatives** (`travis`, `harvest`, `service`, `jars`, …) labeled `_unknown_`.

```sh
uv sync
curl -L http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz | tar -xz -C data/speech_commands  # 2.3 GB (mkdir first)
uv run python gen_synth.py                          # ~16 min -> data/synth/{pos,neg}
uv run python train.py                              # -> runs/main (25 epochs, ~10 min)
uv run python train.py --no-hard-neg --out runs/no_hard_neg   # ablation
uv run python listen.py                             # live mic: talk to it
uv run python listen.py --test jarvis -n 10         # record 10 clips of YOUR voice and score them
```

Splits are by speaker: Speech Commands' official lists, and for TTS whole voices (12) are held out for test.
Training uses on-GPU augmentation: time shift, background noise at 0–25 dB SNR, gain, lowpass on TTS
(to make studio-clean audio look like a laptop mic), and SpecAugment.

## Results (test set)

| | main | no TTS hard negatives |
|---|---|---|
| Overall accuracy (13 classes) | **95.8%** | 93.0% |
| 10 commands (avg) | ~97.5% | ~97.5% |
| `jarvis` recall (12 unseen TTS voices) | 91.7% | 100% |
| `jarvis` false accepts, real speech (incl. "marvin") | 0.0% | 0.0% |
| `jarvis` false accepts, TTS sound-alike words | **4.0%** | **56.5%** |

**Lesson:** without hard negatives the model learned a shortcut, "sounds like a TTS voice → jarvis".
That gives a perfect recall score while firing on most synthetic words. The recall number looked better, but
the model was worse. Always test on negatives drawn from the same distribution as your positives.

Remaining errors make sense: false accepts are mostly `jarvi` / `javis` (near-homophones), and misses come
from two voices, `Xander` (Dutch) and `Daniel` (British, non-rhotic "jah-vis").

## Ideas to try next
- **Sim-to-real check:** `listen.py --test jarvis` on your own voice. The gap between TTS and real voices is the
  real question. Fixes: more varied TTS (Piper/Kokoro open-source voices), room reverb augmentation,
  or adding 20 recordings of yourself.
- **Your own word:** change `WAKE` in `common.py` + `POS_TEXTS`/`NEG_WORDS` in `gen_synth.py`, then rerun.
- **Silence is the weakest class (81%):** loud noise crops look like unknown words. Try more silence data or a lower-energy gate.
- **Deploy:** export to ONNX / CoreML and run on a phone or a Raspberry Pi. 0.57M params is tiny.
- **Next level:** fine-tune `openai/whisper-tiny` for speech-to-text on a dataset like LibriSpeech/Common Voice.
