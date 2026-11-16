"""Decoding, resampling and segmentation.

Everything downstream assumes mono float32 at :attr:`AudioConfig.sample_rate`
(16 kHz by default). 16 kHz is a deliberate choice rather than a shortcut: it keeps
the whole 0-8 kHz band where the energy of the MFCC filterbank and the vast majority
of usable spectral peaks live, while costing about a third of the decode and FFT work
of 44.1 kHz. Anything above 8 kHz is the first thing a low-bitrate encoder discards,
so it is not a band a robust fingerprint should rely on anyway.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import AudioConfig


class DecodeError(RuntimeError):
    """Raised when a file cannot be decoded into usable audio."""


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def decode(source: Path | str | bytes, sample_rate: int) -> np.ndarray:
    """Decode an audio file (or raw bytes) to mono float32 at ``sample_rate``.

    libsndfile 1.2 reads MP3 directly, so the FMA corpus needs no external decoder.
    Resampling uses ``soxr`` through librosa, whose default 'soxr_hq' setting is well
    above the quality floor that would matter for MFCC or peak extraction.
    """
    try:
        if isinstance(source, bytes):
            data, sr = sf.read(io.BytesIO(source), dtype="float32", always_2d=False)
        else:
            data, sr = sf.read(str(source), dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile raises a variety of errors
        raise DecodeError(str(exc)) from exc

    data = _to_mono(np.asarray(data, dtype=np.float32))
    if data.size == 0:
        raise DecodeError("decoded to zero samples")
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if sr != sample_rate:
        # Imported lazily: librosa pulls in numba, which is slow to import and is not
        # needed by callers that only want a native-rate decode.
        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        data = np.asarray(data, dtype=np.float32)

    return data


def segment_starts(n_samples: int, cfg: AudioConfig) -> list[int]:
    """Start offsets (in samples) of every full-length segment.

    Only whole windows are emitted. A trailing partial window is dropped instead of
    zero-padded: a padded segment's MFCC statistics are computed over a mixture of
    audio and silence, which shifts the pooled mean and makes the vector a poor
    representative of anything.

    A 30 s track at 5 s / 2.5 s gives floor((30 - 5) / 2.5) + 1 = 11 segments.
    """
    seg = cfg.segment_samples
    hop = cfg.hop_samples
    if n_samples < seg:
        return []
    return list(range(0, n_samples - seg + 1, hop))


def segments(y: np.ndarray, cfg: AudioConfig) -> list[tuple[float, np.ndarray]]:
    """Yield ``(start_seconds, samples)`` for each full-length segment."""
    out = []
    for start in segment_starts(y.size, cfg):
        out.append((start / cfg.sample_rate, y[start:start + cfg.segment_samples]))
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Add ``noise`` to ``signal`` at the requested signal-to-noise ratio.

    The noise is tiled or randomly cropped to the signal length, then scaled so that
    10*log10(P_signal / P_noise) equals ``snr_db``. The result is peak-limited to
    avoid clipping artefacts that would otherwise be an uncontrolled second
    degradation on top of the intended one.
    """
    rng = rng or np.random.default_rng()
    n = signal.size
    if noise.size == 0:
        return signal.copy()
    if noise.size < n:
        reps = int(np.ceil(n / noise.size))
        noise = np.tile(noise, reps)
    if noise.size > n:
        offset = int(rng.integers(0, noise.size - n + 1))
        noise = noise[offset:offset + n]

    s_rms = rms(signal)
    n_rms = rms(noise)
    target_noise_rms = s_rms / (10.0 ** (snr_db / 20.0))
    mixed = signal + noise * (target_noise_rms / n_rms)

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)
