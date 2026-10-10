"""Shared bits: labels, audio loading, features, model."""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

SR = 16000
CLIP = SR  # 1 second
DATA = Path(__file__).parent / "data"
SC_DIR = DATA / "speech_commands"
SYN_DIR = DATA / "synth"

WAKE = "jarvis"  # custom word: no real recordings exist, trained from TTS only
COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = COMMANDS + [WAKE, "_unknown_", "_silence_"]
L2I = {l: i for i, l in enumerate(LABELS)}


def load_wav(path) -> np.ndarray:
    a, sr = sf.read(path, dtype="int16")
    assert sr == SR, (path, sr)
    return a


def fit_1s(a: np.ndarray, center=True) -> np.ndarray:
    """Pad/crop to exactly 1 s. Long clips keep the 1 s window with the most energy
    (centering on the loudest *point* would put the stressed syllable mid-window and cut the word's tail)."""
    if len(a) >= CLIP:
        if not center:
            return a[:CLIP]
        c = np.concatenate([[0], np.cumsum(a.astype(np.float64) ** 2)])
        start = int(np.argmax(c[CLIP:] - c[:-CLIP]))
        return a[start:start + CLIP]
    pad = CLIP - len(a)
    left = pad // 2 if center else 0
    return np.pad(a, (left, pad - left))


def trim_silence(a: np.ndarray, thresh=0.02) -> np.ndarray:
    x = np.abs(a.astype(np.float32)) / 32768
    idx = np.where(x > thresh * x.max())[0]
    return a[idx[0]: idx[-1] + 1] if len(idx) else a


class Features(nn.Module):
    """Waveform (B, 16000) float in [-1,1] -> log-mel (B, 1, 40, 101)."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=480, win_length=480, hop_length=160, n_mels=40, f_min=20, f_max=7600)

    def forward(self, x):
        return torch.log(self.mel(x) + 1e-6).unsqueeze(1)


def block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
    )


class KWSNet(nn.Module):
    """Small CNN on log-mels (~0.57M params). Per-utterance normalization lives inside."""

    def __init__(self, n_classes=len(LABELS), width=32):
        super().__init__()
        self.feat = Features()
        self.net = nn.Sequential(
            # stem halves the time axis (101 -> 51 frames): ~2.4x faster training
            nn.Conv2d(1, width, 3, (1, 2), 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
            block(width, width * 2, 2),
            block(width * 2, width * 4, 2),
            block(width * 4, width * 4, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(width * 4, n_classes),
        )

    def features(self, wav):
        f = self.feat(wav)
        return (f - f.mean(dim=(2, 3), keepdim=True)) / (f.std(dim=(2, 3), keepdim=True) + 1e-5)

    def forward(self, wav, specaug=None):
        f = self.features(wav)
        if specaug is not None:
            f = specaug(f)
        return self.net(f)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class HubertKWS(nn.Module):
    """Frozen pretrained HuBERT-base (self-supervised on 960 h of real speech) + small trained head.
    Head sees each transformer layer's time-averaged output, mixed with learned layer weights."""

    def __init__(self, n_classes=len(LABELS), n_layers=12, dim=768):
        super().__init__()
        self.backbone = None  # loaded lazily, never trained
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, 256), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(256, n_classes))

    @torch.no_grad()
    def embed(self, wav):
        """(B, 16000) -> (B, 12, 768): per-layer mean over time."""
        if self.backbone is None:
            self.backbone = torchaudio.pipelines.HUBERT_BASE.get_model().eval().to(wav.device)
        feats, _ = self.backbone.extract_features(wav)
        return torch.stack([f.mean(1) for f in feats], 1)

    def classify(self, emb):
        return self.head((emb * self.layer_w.softmax(0)[:, None]).sum(1))

    def forward(self, wav):
        return self.classify(self.embed(wav))

    def state_dict(self, *a, **k):  # don't save the 360 MB frozen backbone
        return {n: v for n, v in super().state_dict(*a, **k).items() if not n.startswith("backbone.")}
