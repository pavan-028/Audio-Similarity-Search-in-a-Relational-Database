"""Path A -- dense segment embeddings from pooled MFCC statistics.

Pipeline for one audio buffer:

1. one MFCC matrix for the *whole* buffer (``n_mfcc`` x frames);
2. first-order deltas of that matrix;
3. for each 5 s segment, slice out its frames and pool them into a fixed vector of
   means and standard deviations;
4. standardise each dimension using statistics fitted on a corpus sample;
5. L2-normalise, so cosine distance is the metric pgvector indexes.

Step 1 is computed once per buffer rather than once per segment. With a 2.5 s hop the
segments overlap by half, so per-segment extraction would do roughly twice the FFT
work for identical output.

FEATURE LAYOUT (n_mfcc = 40 -> 159 dimensions)
----------------------------------------------
    [  0 ..  38]  mean of MFCC c1..c39          (39)
    [ 39 ..  78]  std  of MFCC c0..c39          (40)
    [ 79 .. 118]  mean of delta-MFCC c0..c39    (40)
    [119 .. 158]  std  of delta-MFCC c0..c39    (40)

The mean of c0 is deliberately absent. c0 is proportional to frame log-energy, so its
segment mean is essentially "how loud was this segment" -- a quantity that carries
little about *which* track this is and that moves under any gain change. Its standard
deviation is kept, because how much the loudness fluctuates within five seconds is
genuinely characteristic. The mean of *delta*-c0 is also kept and needs no apology: a
constant gain adds a constant to log-energy, and that constant cancels in a first
difference, so delta-c0 is gain-invariant by construction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..config import AudioConfig, EmbeddingConfig


def frames_per_segment(audio: AudioConfig, emb: EmbeddingConfig) -> int:
    """Number of non-centred STFT frames that fit entirely inside one segment."""
    return 1 + (audio.segment_samples - emb.n_fft) // emb.hop_length


def frames_per_hop(audio: AudioConfig, emb: EmbeddingConfig) -> int:
    """Frame stride between consecutive segments.

    The default parameters are chosen so this is exact: a 2.5 s hop is 40 000 samples
    at 16 kHz, and 40 000 / 250 = 160 frames with no remainder. Exact alignment means
    a segment's frames can be sliced straight out of the track-level matrix instead of
    being recomputed or rounded to the nearest frame.
    """
    hop_samples = audio.hop_samples
    if hop_samples % emb.hop_length != 0:
        raise ValueError(
            f"hop_seconds*sample_rate ({hop_samples}) must be an exact multiple of "
            f"EmbeddingConfig.hop_length ({emb.hop_length}) for frame-aligned slicing"
        )
    return hop_samples // emb.hop_length


def mfcc_and_delta(y: np.ndarray, audio: AudioConfig,
                   emb: EmbeddingConfig) -> tuple[np.ndarray, np.ndarray]:
    """MFCC matrix and its first-order delta for a whole buffer."""
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=audio.sample_rate,
            n_mfcc=emb.n_mfcc,
            n_fft=emb.n_fft,
            hop_length=emb.hop_length,
            n_mels=emb.n_mels,
            fmin=emb.fmin,
            fmax=emb.fmax,
            center=False,
        )
        if emb.include_delta:
            # width must be odd and <= number of frames; 9 frames ~ 140 ms.
            width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
            if width >= 3:
                delta = librosa.feature.delta(mfcc, width=width)
            else:
                delta = np.zeros_like(mfcc)
        else:
            delta = np.zeros((0, mfcc.shape[1]), dtype=mfcc.dtype)

    return np.asarray(mfcc, dtype=np.float32), np.asarray(delta, dtype=np.float32)


def pool(mfcc_slice: np.ndarray, delta_slice: np.ndarray,
         emb: EmbeddingConfig) -> np.ndarray:
    """Pool one segment's frames into the raw (un-standardised) feature vector."""
    parts: list[np.ndarray] = []
    m_mean = mfcc_slice.mean(axis=1)
    parts.append(m_mean[1:] if emb.drop_c0_mean else m_mean)
    parts.append(mfcc_slice.std(axis=1))
    if emb.include_delta:
        parts.append(delta_slice.mean(axis=1))
        parts.append(delta_slice.std(axis=1))
    return np.concatenate(parts).astype(np.float32)


def raw_segment_features(y: np.ndarray, audio: AudioConfig,
                         emb: EmbeddingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Raw pooled features for every full segment of ``y``.

    Returns ``(starts_seconds, features)`` where ``features`` has shape
    ``(n_segments, emb.dim)``. An empty result means the buffer was shorter than one
    segment.
    """
    n_seg_frames = frames_per_segment(audio, emb)
    stride = frames_per_hop(audio, emb)

    if y.size < audio.segment_samples:
        return np.zeros(0, dtype=np.float32), np.zeros((0, emb.dim), dtype=np.float32)

    mfcc, delta = mfcc_and_delta(y, audio, emb)
    total_frames = mfcc.shape[1]

    starts: list[float] = []
    feats: list[np.ndarray] = []
    seg_index = 0
    while True:
        f0 = seg_index * stride
        f1 = f0 + n_seg_frames
        start_sample = seg_index * audio.hop_samples
        if f1 > total_frames or start_sample + audio.segment_samples > y.size:
            break
        feats.append(pool(mfcc[:, f0:f1], delta[:, f0:f1] if delta.size else delta, emb))
        starts.append(start_sample / audio.sample_rate)
        seg_index += 1

    if not feats:
        return np.zeros(0, dtype=np.float32), np.zeros((0, emb.dim), dtype=np.float32)
    return (np.asarray(starts, dtype=np.float32),
            np.vstack(feats).astype(np.float32))


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Standardiser:
    """Per-dimension z-scoring followed by L2 normalisation.

    Why this matters: the raw pooled dimensions live on wildly different scales. MFCC
    standard deviations are order 10-100, delta means are order 0.01. Under cosine
    distance the large-scale dimensions would decide nearly every comparison and the
    rest would be noise. Z-scoring puts every dimension on equal footing first; the
    ablation in the report quantifies what this is worth.

    ``mean`` and ``std`` are fitted once on a sample of the corpus and then stored in
    the ``feature_models`` table, because the identical transform has to be applied to
    every query.
    """

    mean: np.ndarray
    std: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply to a single vector (1-D) or a batch (2-D). Returns float32."""
        single = x.ndim == 1
        X = np.atleast_2d(np.asarray(x, dtype=np.float64))
        X = (X - self.mean) / self.std
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X = (X / norms).astype(np.float32)
        return X[0] if single else X

    @staticmethod
    def fit(X: np.ndarray) -> "Standardiser":
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # A dimension that never varies carries no information; guarding the divisor
        # at 1.0 maps it to a constant 0 instead of producing inf/NaN.
        std[std < 1e-8] = 1.0
        return Standardiser(mean=mean, std=std)


class IdentityStandardiser(Standardiser):
    """Used by the no-standardisation ablation: L2-normalise the raw vector only."""

    def __init__(self, dim: int):
        super().__init__(mean=np.zeros(dim), std=np.ones(dim))


def embed_buffer(y: np.ndarray, audio: AudioConfig, emb: EmbeddingConfig,
                 scaler: Standardiser) -> tuple[np.ndarray, np.ndarray]:
    """Full Path A extraction: raw pooling, standardisation, L2 normalisation."""
    starts, raw = raw_segment_features(y, audio, emb)
    if raw.shape[0] == 0:
        return starts, raw
    return starts, scaler.transform(raw)
