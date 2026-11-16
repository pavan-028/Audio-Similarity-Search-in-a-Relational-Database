"""Path B -- Shazam-style landmark fingerprints.

The classic construction (Wang, "An Industrial-Strength Audio Search Algorithm",
ISMIR 2003): find prominent local maxima in the magnitude spectrogram, pair each
anchor peak with a handful of peaks in a target zone just after it, and hash the
triple ``(f_anchor, f_target, dt)`` into an integer. A query is matched by looking its
hashes up and testing whether many of them agree on a single time offset.

Why it behaves so differently from Path A
-----------------------------------------
A pooled MFCC vector is a *summary*: it degrades smoothly and gracefully, which makes
it good at "sounds like" but blurry about "is exactly". A landmark hash is a *sparse
exact* description. Peaks are chosen to be the loudest things in their neighbourhood,
so moderate additive noise and lossy re-encoding tend to leave most of them where they
were; and because matching requires many hashes to agree on one offset, a handful of
accidental collisions cannot produce a false match. The failure mode is abrupt rather
than gradual: once noise is loud enough to reorder the local maxima, the agreeing
offset disappears and the match is simply lost.

Hash layout (64-bit integer, 23 bits used with the default configuration)
-------------------------------------------------------------------------
    bits 15-22 : anchor frequency, quantised to 256 levels  (8 bits)
    bits  7-14 : target frequency, quantised to 256 levels  (8 bits)
    bits  0- 6 : dt in frames, clipped to 0..127            (7 bits)

Quantising frequency by a factor of two (512 STFT bins -> 256 levels, ~31 Hz per
level) is a robustness/selectivity trade-off: finer bins make hashes more selective but
more fragile, because a peak that drifts by one bin under noise then produces a
different hash.
"""

from __future__ import annotations

import numpy as np

from ..config import AudioConfig, LandmarkConfig


def spectrogram_db(y: np.ndarray, audio: AudioConfig,
                   lm: LandmarkConfig) -> np.ndarray:
    """Log-magnitude STFT, shape ``(n_fft // 2 + 1, frames)``."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=lm.n_fft, hop_length=lm.hop_length, center=False))
    return librosa.amplitude_to_db(S, ref=np.max, top_db=None).astype(np.float32)


def find_peaks(Sdb: np.ndarray, audio: AudioConfig,
               lm: LandmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Locate spectral peaks. Returns ``(times_frames, freqs_bins)`` sorted by time.

    Three filters are applied in order:

    1. **local maximum** over a ``(peak_neighbourhood_f, peak_neighbourhood_t)``
       window, which is what makes a peak "prominent" rather than merely loud;
    2. **absolute floor** at ``max_dB - peak_floor_db``, which discards maxima found
       inside silence or near-silence, where the "peak" is just the loudest noise;
    3. **density cap** of ``max_peaks_per_second``, keeping the strongest peaks in each
       one-second block. Without this the peak rate varies by more than an order of
       magnitude between sparse and dense material, which would make both the row count
       and the match score wildly uneven across the corpus.
    """
    from scipy.ndimage import maximum_filter

    if Sdb.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    footprint_shape = (lm.peak_neighbourhood_f, lm.peak_neighbourhood_t)
    local_max = maximum_filter(Sdb, size=footprint_shape, mode="constant",
                               cval=-np.inf)
    threshold = float(Sdb.max()) - lm.peak_floor_db
    mask = (Sdb == local_max) & (Sdb > threshold)

    freqs, times = np.nonzero(mask)
    if freqs.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    mags = Sdb[freqs, times]

    # Density cap, applied per one-second block of frames.
    frames_per_second = audio.sample_rate / lm.hop_length
    block = np.floor(times / frames_per_second).astype(np.int64)
    keep = np.zeros(freqs.size, dtype=bool)
    # Sorting by (block, -magnitude) groups each block together with its loudest peaks
    # first, so the per-block top-k is a single pass.
    order = np.lexsort((-mags, block))
    blocks_sorted = block[order]
    boundaries = np.flatnonzero(np.diff(blocks_sorted)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, blocks_sorted.size]):
        keep[order[start:min(stop, start + lm.max_peaks_per_second)]] = True

    freqs, times = freqs[keep], times[keep]
    order = np.lexsort((freqs, times))          # time-major, then frequency
    return times[order].astype(np.int32), freqs[order].astype(np.int32)


def pack_hash(f1: np.ndarray, f2: np.ndarray, dt: np.ndarray,
              lm: LandmarkConfig) -> np.ndarray:
    """Pack quantised ``(f1, f2, dt)`` triples into int64 hashes."""
    q1 = (f1 // lm.freq_quant_div).astype(np.int64)
    q2 = (f2 // lm.freq_quant_div).astype(np.int64)
    dtq = np.clip(dt, 0, (1 << lm.dt_bits) - 1).astype(np.int64)
    fb = lm.freq_bits
    return (q1 << (fb + lm.dt_bits)) | (q2 << lm.dt_bits) | dtq


def pair_peaks(times: np.ndarray, freqs: np.ndarray,
               lm: LandmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Form anchor/target pairs and hash them.

    Returns ``(hashes, anchor_times)``. ``anchor_times`` is the frame index of the
    anchor peak, which is the offset the alignment vote is computed against.

    The target zone is ``dt in [dt_min, dt_max]`` and ``|df| <= df_max``, and at most
    ``fanout`` pairs are emitted per anchor. Because ``times`` is sorted, the candidate
    window for each anchor is a contiguous slice found by binary search rather than a
    scan over all later peaks.
    """
    n = times.size
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)

    f1_list: list[np.ndarray] = []
    f2_list: list[np.ndarray] = []
    dt_list: list[np.ndarray] = []
    t1_list: list[np.ndarray] = []

    lo_idx = np.searchsorted(times, times + lm.dt_min, side="left")
    hi_idx = np.searchsorted(times, times + lm.dt_max, side="right")

    for i in range(n):
        lo, hi = int(lo_idx[i]), int(hi_idx[i])
        if hi <= lo:
            continue
        cand_t = times[lo:hi]
        cand_f = freqs[lo:hi]
        ok = np.abs(cand_f.astype(np.int32) - int(freqs[i])) <= lm.df_max
        if not ok.any():
            continue
        sel_t = cand_t[ok][: lm.fanout]
        sel_f = cand_f[ok][: lm.fanout]
        k = sel_t.size
        f1_list.append(np.full(k, freqs[i], dtype=np.int32))
        f2_list.append(sel_f.astype(np.int32))
        dt_list.append((sel_t - times[i]).astype(np.int32))
        t1_list.append(np.full(k, times[i], dtype=np.int32))

    if not f1_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)

    f1 = np.concatenate(f1_list)
    f2 = np.concatenate(f2_list)
    dt = np.concatenate(dt_list)
    t1 = np.concatenate(t1_list)
    return pack_hash(f1, f2, dt, lm), t1


def fingerprint_buffer(y: np.ndarray, audio: AudioConfig,
                       lm: LandmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Full Path B extraction. Returns ``(hashes, anchor_frame_offsets)``."""
    Sdb = spectrogram_db(y, audio, lm)
    times, freqs = find_peaks(Sdb, audio, lm)
    return pair_peaks(times, freqs, lm)


def frames_to_seconds(frames: np.ndarray, audio: AudioConfig,
                      lm: LandmarkConfig) -> np.ndarray:
    return frames.astype(np.float64) * lm.hop_length / audio.sample_rate
