# voicemodel — train a voice-command + custom wake-word recognizer

PyTorch on the M5 GPU (MPS), $0. Two models: a small CNN from scratch (0.57M params) and a frozen pretrained HuBERT + tiny head.

- **Real data:** [Google Speech Commands v2](https://arxiv.org/abs/1804.03209) (CC-BY 4.0, 105k 1-s clips, ~2.6k speakers):
  10 commands `yes no up down left right on off stop go` + `_unknown_` (other 25 words) + `_silence_` (background noise).
- **Synthetic data:** a custom wake word **`jarvis`** that has no real recordings anywhere, generated with
  macOS `say`: 108 voices (incl. German/French/Russian/… voices reading English = accents) × 6 speeds × 3 intonations.
  Then [Piper](https://github.com/rhasspy/piper) open-source neural TTS, LibriTTS-R model: **904 speakers**.
  Plus TTS **hard negatives** (`travis`, `harvest`, `service`, `jars`, …) labeled `_unknown_`.

```sh
uv sync
mkdir -p data/speech_commands && curl -L http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz | tar -xz -C data/speech_commands
uv run python gen_synth.py                 # macOS `say`: 108 voices, ~16 min
mkdir -p data/piper && cd data/piper && for f in en_US-libritts_r-medium.onnx{,.json}; do \
  curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/$f; done && cd ../..
uv run python gen_piper.py                 # Piper neural TTS: 904 speakers, ~7 min
uv run python train.py --out runs/v2       # small CNN from scratch, ~12 min
uv run python train_hubert.py              # frozen HuBERT + head, ~12 min (mostly embedding)
uv run python listen.py                    # live mic (runs/hubert)
uv run python listen.py --test jarvis -n 15   # record + score YOUR voice -> data/mine/ (eval only)
uv run python listen.py --score --run runs/v2  # re-score saved clips with any model
```

Splits are by speaker: Speech Commands' official lists; for TTS whole voices are held out (12 mac, 100 Piper).
Augmentation: time shift, background noise at 0–25 dB SNR, gain, room reverb (synthetic impulse responses),
lowpass on TTS, SpecAugment.

## Results

| | v1: CNN, mac TTS | v2: CNN, mac+Piper TTS, reverb | **HuBERT + head** |
|---|---|---|---|
| Test accuracy (13 classes) | 95.8% | 95.8% | **97.2%** |
| `jarvis` recall, unseen TTS speakers | 92% (mac) | 100% (Piper) | 100% |
| `jarvis` false accepts, real speech (incl. "marvin") | 0% | 0% | 0% (0/4352) |
| **`jarvis` on my real voice (20 clips)** | **0/20** | **3/20** | **20/20** |

### Lesson 1: hard negatives (v1 ablation)
Without TTS sound-alikes labeled `_unknown_`, false accepts on TTS words went from 4% to 56%. The model learned
"sounds like TTS → jarvis". The recall number looked better while the model was worse.

### Lesson 2: sim-to-real gap, fixed with transfer learning
v1/v2 hit 92–100% on held-out *synthetic* speakers but 0–15% on a real person. Test-set accuracy measured on
synthetic data said nothing about real use. More synthetic variety (Piper, 904 speakers) and reverb barely helped:
a CNN trained from scratch learns the acoustics of TTS. Its only real-voice examples were the Speech Commands
words, so "real human + unfamiliar word" meant `_unknown_`.

A frozen **HuBERT-base** (self-supervised on 960 h of real speech, 95M params, free) turns audio into phonetic
features where TTS and human "jarvis" land close together. A 0.2M-param head trained on the *same* data gets
20/20 on a real voice with confidence ~0.9. This is why openWakeWord can train on TTS alone: it uses a pretrained
speech embedding. Only the head is trained (layer mixing weights + 2-layer MLP); it learned to weight
layers 7–9 most, the "phonetic" middle of the network.

Caveat: "my voice" is one speaker and 20 clips. It's a sanity check, not a benchmark.

## Ideas to try next
- **Record negatives too** (`--test stop`, `--test _unknown_` saying random words) to measure false accepts on your voice.
- **Fine-tune the top HuBERT layers** instead of freezing everything, or try `WAVLM_BASE_PLUS` / `WAV2VEC2_BASE`.
- **Smaller deployable model:** distill HuBERT+head into the 0.57M CNN (teacher labels on real speech).
- **Your own word:** change `WAKE` in `common.py` + `POS_TEXTS`/`NEG_WORDS` in `gen_synth.py`, then rerun.
- **Deploy:** export to ONNX / CoreML and run on a phone or a Raspberry Pi. 0.57M params is tiny.
- **Next level:** fine-tune `openai/whisper-tiny` for speech-to-text on a dataset like LibriSpeech/Common Voice.
