"""Degradations applied to query clips.

The point of the evaluation is not "can the system find a track given a bit-identical
copy of part of it" -- any hash table does that. It is whether retrieval survives the
things that happen to audio between a recording and a query: it gets cropped, picked up
over background noise, and squeezed through a lossy codec.

Three degradations are applied, each chosen because it attacks the two paths
differently:

**Cropping** removes the context a pooled summary relies on and breaks alignment with
the indexed segment grid. A 5 s query that starts 1.3 s into a track overlaps no
indexed segment exactly.

**Additive noise** at a controlled SNR. The noise is real environmental audio from
ESC-50 rather than white noise, because white noise is both unrealistic and unusually
easy: it spreads energy evenly and so perturbs every spectral peak and every mel band by
a similar amount, whereas a passing car or a crowd is concentrated in time and
frequency and can obliterate a specific band completely.

**Lossy re-encoding** at low bitrate. MP3 at 32-64 kbit/s on a 16 kHz mono signal
discards high-frequency detail and quantises what remains, which is a structurally
different attack from additive noise: it does not add energy, it removes it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import decode, mix_at_snr

FFMPEG = shutil.which("ffmpeg")


def crop(y: np.ndarray, sample_rate: int, start_seconds: float,
         length_seconds: float) -> np.ndarray:
    a = int(round(start_seconds * sample_rate))
    b = a + int(round(length_seconds * sample_rate))
    return y[a:b].copy()


def add_noise(clip: np.ndarray, noise_path: Path, sample_rate: int, snr_db: float,
              rng: np.random.Generator) -> np.ndarray:
    noise = decode(noise_path, sample_rate)
    return mix_at_snr(clip, noise, snr_db, rng)


def reencode_mp3(clip: np.ndarray, sample_rate: int, bitrate_kbps: int) -> np.ndarray:
    """Round-trip a clip through an MP3 encoder at the given bitrate.

    Writing to disk and shelling out to ffmpeg is slower than an in-process encoder
    would be, but it means the degradation is produced by a real, standard encoder
    rather than an approximation of one. The encoder's own delay is removed by trimming
    the decoded result back to the original length.
    """
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot run the codec conditions")

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.mp3"
        sf.write(src, clip, sample_rate, subtype="PCM_16")
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(src), "-codec:a", "libmp3lame",
             "-b:a", f"{bitrate_kbps}k", "-ar", str(sample_rate), "-ac", "1",
             str(dst)],
            check=True,
        )
        out = decode(dst, sample_rate)

    # MP3 encoding prepends encoder delay and pads the tail; align by length.
    if out.size > clip.size:
        offset = (out.size - clip.size) // 2
        out = out[offset:offset + clip.size]
    elif out.size < clip.size:
        out = np.pad(out, (0, clip.size - out.size))
    return out.astype(np.float32)


def list_noise_files(noise_root: Path, limit: int | None = None) -> list[Path]:
    files = sorted(noise_root.glob("*.wav"))
    return files[:limit] if limit else files
