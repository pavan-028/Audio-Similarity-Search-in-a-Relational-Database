"""Generate the evaluation query set and register it in the database.

Grid
----
For each of the ~200 held-out tracks:

* clip lengths 5 s, 10 s and 15 s, each at a random (non-grid-aligned) start;
* noise conditions: clean, and ESC-50 noise at 20, 10 and 0 dB SNR;
* additionally, 10 s clips re-encoded as MP3 at 64 and 32 kbit/s.

That gives 200 x 3 x 4 = 2 400 crop/noise queries plus 200 x 2 = 400 codec queries.

Every random choice -- crop offset, which noise file, where in the noise file -- is
drawn from a generator seeded per query id, so the whole set regenerates bit-for-bit
and the conditions compared in the report differ only in the intended variable.

The clip start is deliberately *not* snapped to the 2.5 s segment grid. Aligning it
would quietly hand Path A a much easier problem, because each query segment would then
correspond exactly to one indexed segment.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..audio import DecodeError, decode
from ..config import Config, load_config
from ..db import connect
from . import perturb

CLIP_LENGTHS = (5.0, 10.0, 15.0)
SNR_LEVELS: tuple[float | None, ...] = (None, 20.0, 10.0, 0.0)
CODEC_BITRATES = (64, 32)
CODEC_CLIP_LENGTH = 10.0
GLOBAL_SEED = 20260211


@dataclass
class QuerySpec:
    query_id: str
    track_id: int
    clip_start: float
    clip_seconds: float
    snr_db: float | None
    noise_file: str | None
    codec: str | None
    path: Path


def _rng_for(query_id: str) -> np.random.Generator:
    """A generator seeded by the query id, so each query is independently reproducible."""
    digest = hashlib.sha256(f"{GLOBAL_SEED}:{query_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def plan(cfg: Config, track_rows: list[tuple[int, float]],
         noise_files: list[Path]) -> list[QuerySpec]:
    specs: list[QuerySpec] = []
    out_dir = cfg.paths.queries

    def make(track_id: int, duration: float, length: float,
             snr: float | None, codec: str | None) -> QuerySpec | None:
        tag = "clean" if snr is None else f"snr{int(snr):02d}"
        if codec:
            tag = codec.replace("@", "_").replace("kbit/s", "k")
        query_id = f"t{track_id}_l{int(length)}_{tag}"
        rng = _rng_for(query_id)

        headroom = duration - length
        if headroom <= 0.5:
            return None
        start = float(rng.uniform(0.25, headroom - 0.25))
        noise = None
        if snr is not None and noise_files:
            noise = str(noise_files[int(rng.integers(0, len(noise_files)))])
        return QuerySpec(query_id, track_id, start, length, snr, noise, codec,
                         out_dir / f"{query_id}.wav")

    for track_id, duration in track_rows:
        for length in CLIP_LENGTHS:
            for snr in SNR_LEVELS:
                spec = make(track_id, duration, length, snr, None)
                if spec:
                    specs.append(spec)
        for bitrate in CODEC_BITRATES:
            spec = make(track_id, duration, CODEC_CLIP_LENGTH, None,
                        f"mp3@{bitrate}k")
            if spec:
                specs.append(spec)
    return specs


def render(cfg: Config, specs: list[QuerySpec], overwrite: bool = False) -> int:
    """Produce the WAV files. Source tracks are decoded once each."""
    cfg.paths.queries.mkdir(parents=True, exist_ok=True)

    by_track: dict[int, list[QuerySpec]] = {}
    for spec in specs:
        by_track.setdefault(spec.track_id, []).append(spec)

    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT track_id, source_path FROM tracks WHERE track_id = ANY(%s)",
            (list(by_track),),
        )
        source_paths = {int(t): p for t, p in cur.fetchall()}

    written = 0
    for track_id, group in tqdm(by_track.items(), desc="rendering queries",
                                unit="track"):
        pending = [s for s in group if overwrite or not s.path.exists()]
        if not pending:
            continue
        try:
            y = decode(source_paths[track_id], cfg.audio.sample_rate)
        except (DecodeError, KeyError):
            continue

        for spec in pending:
            rng = _rng_for(spec.query_id)
            clip = perturb.crop(y, cfg.audio.sample_rate, spec.clip_start,
                                spec.clip_seconds)
            if clip.size < int(spec.clip_seconds * cfg.audio.sample_rate * 0.98):
                continue
            if spec.snr_db is not None and spec.noise_file:
                clip = perturb.add_noise(clip, Path(spec.noise_file),
                                         cfg.audio.sample_rate, spec.snr_db, rng)
            if spec.codec:
                bitrate = int(spec.codec.split("@")[1].rstrip("k"))
                clip = perturb.reencode_mp3(clip, cfg.audio.sample_rate, bitrate)
            sf.write(spec.path, clip, cfg.audio.sample_rate, subtype="PCM_16")
            written += 1
    return written


def register(cfg: Config, specs: list[QuerySpec]) -> int:
    rows = [
        (s.query_id, s.track_id, s.clip_start, s.clip_seconds, s.snr_db,
         s.noise_file, s.codec, str(s.path))
        for s in specs if s.path.exists()
    ]
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO eval_queries (query_id, track_id, clip_start, clip_seconds,
                                      snr_db, noise_file, codec, path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (query_id) DO UPDATE
               SET clip_start = EXCLUDED.clip_start,
                   clip_seconds = EXCLUDED.clip_seconds,
                   snr_db = EXCLUDED.snr_db,
                   noise_file = EXCLUDED.noise_file,
                   codec = EXCLUDED.codec,
                   path = EXCLUDED.path
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=int, default=None,
                        help="limit the number of query-source tracks used")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-codec", action="store_true",
                        help="skip the MP3 conditions (useful without ffmpeg)")
    args = parser.parse_args(argv)

    cfg = load_config()
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT track_id, duration FROM tracks
             WHERE is_query_source ORDER BY ingest_order
            """
        )
        track_rows = [(int(t), float(d)) for t, d in cur.fetchall()]
    if not track_rows:
        print("no query-source tracks -- run finalize first")
        return 2
    if args.tracks:
        track_rows = track_rows[: args.tracks]

    noise_files = perturb.list_noise_files(cfg.paths.noise_root)
    print(f"{len(track_rows)} query-source tracks, {len(noise_files)} noise files")
    if perturb.FFMPEG is None:
        print("ffmpeg not found: codec conditions will be skipped")

    specs = plan(cfg, track_rows, noise_files)
    if args.no_codec or perturb.FFMPEG is None:
        specs = [s for s in specs if s.codec is None]
    print(f"planned {len(specs)} queries")

    written = render(cfg, specs, overwrite=args.overwrite)
    registered = register(cfg, specs)
    print(f"wrote {written} new wav files; {registered} rows in eval_queries")

    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT clip_seconds, snr_db, codec, count(*)
            FROM eval_queries GROUP BY 1, 2, 3 ORDER BY 1, 2 NULLS FIRST, 3
            """
        )
        print("\n  length  snr    codec      n")
        for length, snr, codec, n in cur.fetchall():
            print(f"  {length:5.0f}s  {str(snr):<5}  {str(codec):<9}  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
