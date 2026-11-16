"""Central configuration.

Every tunable lives here as a frozen dataclass so that a configuration object can be
pickled and handed to multiprocessing workers (Windows uses the *spawn* start method,
so worker arguments must survive a round-trip through ``pickle``).

Defaults can be overridden with ``AUDIOSIM_*`` environment variables, which is what the
PowerShell driver scripts in ``scripts/`` do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"AUDIOSIM_{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DBConfig:
    """Connection parameters for the local PostgreSQL 16 cluster."""

    host: str = _env("DB_HOST", "localhost")
    port: int = _env_int("DB_PORT", 5433)
    user: str = _env("DB_USER", "postgres")
    password: str = _env("DB_PASSWORD", "audiosim_local")
    dbname: str = _env("DB_NAME", "audiosim")

    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={self.dbname}"
        )


@dataclass(frozen=True)
class AudioConfig:
    """Decoding and segmentation.

    A 5 s window with a 2.5 s hop means consecutive segments overlap by half their
    length, so a query that straddles a segment boundary in the reference track still
    lands almost entirely inside *some* indexed segment. That is the whole point of
    indexing per segment rather than per track.
    """

    sample_rate: int = _env_int("SAMPLE_RATE", 16_000)
    segment_seconds: float = 5.0
    hop_seconds: float = 2.5
    # Tracks shorter than this are rejected rather than zero-padded.
    min_track_seconds: float = 6.0

    @property
    def segment_samples(self) -> int:
        return int(round(self.segment_seconds * self.sample_rate))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_seconds * self.sample_rate))


@dataclass(frozen=True)
class EmbeddingConfig:
    """Path A: MFCC summary statistics pooled over each segment.

    ``n_mfcc`` cepstral coefficients give ``2 * n_mfcc`` numbers from mean/std pooling,
    doubled again when first-order deltas are included. Dropping the mean of
    coefficient 0 (see features/embedding.py for why) removes one, so the stored vector
    has ``4 * n_mfcc - 1`` = 159 dimensions with the defaults below.

    ``hop_length`` is 250 rather than a power of two on purpose: a 2.5 s segment hop is
    40 000 samples at 16 kHz and 40 000 / 250 = 160 frames exactly, so segment
    boundaries fall on frame boundaries and each segment's frames can be sliced
    directly out of the track-level MFCC matrix.
    """

    n_mfcc: int = _env_int("N_MFCC", 40)
    n_fft: int = 1024          # 64 ms analysis window at 16 kHz
    hop_length: int = 250      # 15.625 ms frame hop -> 316 frames per 5 s segment
    n_mels: int = 96
    fmin: float = 20.0
    fmax: float = 7800.0       # just below Nyquist at 16 kHz
    include_delta: bool = True
    # MFCC coefficient 0 tracks frame log-energy, i.e. loudness. It is excluded from
    # the pooled mean so that gain changes do not dominate the cosine distance; its
    # standard deviation is kept because dynamics are genuinely discriminative.
    drop_c0_mean: bool = True
    # Per-dimension standardisation fitted on a corpus sample (see ingest/fit_scaler.py).
    standardise: bool = True

    @property
    def dim(self) -> int:
        blocks = 4 if self.include_delta else 2
        return blocks * self.n_mfcc - (1 if self.drop_c0_mean else 0)


@dataclass(frozen=True)
class LandmarkConfig:
    """Path B: spectral peak pairs hashed into integers.

    ``max_peaks_per_second`` is the main lever on both database size and robustness:
    more peaks means more chances that a few survive noise, but also a linear blow-up
    in the number of stored rows.
    """

    n_fft: int = 1024
    hop_length: int = 256      # 16 ms per frame -> 62.5 frames/s
    # Local-maximum neighbourhood, in (frequency bins, frames).
    peak_neighbourhood_f: int = 21
    peak_neighbourhood_t: int = 21
    # Peaks quieter than (max_dB - peak_floor_db) are treated as silence.
    peak_floor_db: float = 70.0
    max_peaks_per_second: int = 16
    # Target zone: pair an anchor with later peaks inside this time/frequency window.
    fanout: int = 5
    dt_min: int = 2            # >= 32 ms after the anchor
    dt_max: int = 96           # <= ~1.5 s after the anchor
    df_max: int = 160          # bins
    # Bit layout of the 64-bit hash: f1 | f2 | dt.
    freq_quant_div: int = 2    # 512 STFT bins -> 256 quantised levels (8 bits each)
    dt_bits: int = 7           # dt clipped to 0..127

    @property
    def freq_levels(self) -> int:
        return (self.n_fft // 2) // self.freq_quant_div

    @property
    def freq_bits(self) -> int:
        return max(1, (self.freq_levels - 1).bit_length())


@dataclass(frozen=True)
class IngestConfig:
    workers: int = _env_int("WORKERS", 8)
    # Rows claimed per worker transaction by the SKIP LOCKED poller.
    claim_batch: int = _env_int("CLAIM_BATCH", 16)
    # Storing the undecoded bytes in audio_staging is realistic but expensive; the
    # loader can be told to record the path only.
    store_blob: bool = True


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT
    data_root: Path = Path(_env("DATA_ROOT", r"C:\Users\User\audiosim_data"))

    @property
    def fma_audio(self) -> Path:
        return self.data_root / "fma_small"

    @property
    def fma_metadata(self) -> Path:
        return self.data_root / "fma_metadata"

    @property
    def noise_root(self) -> Path:
        return self.data_root / "ESC-50-master" / "audio"

    @property
    def queries(self) -> Path:
        return self.project_root / "queries"

    @property
    def results(self) -> Path:
        return self.project_root / "results"

    @property
    def figures(self) -> Path:
        return self.project_root / "results" / "figures"


@dataclass(frozen=True)
class Config:
    db: DBConfig = field(default_factory=DBConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    landmark: LandmarkConfig = field(default_factory=LandmarkConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    paths: Paths = field(default_factory=Paths)

    def with_embedding(self, **kwargs) -> "Config":
        return replace(self, embedding=replace(self.embedding, **kwargs))


def load_config() -> Config:
    return Config()
