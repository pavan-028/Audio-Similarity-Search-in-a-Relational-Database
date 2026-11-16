"""Pre-extract the features of every evaluation query, once.

A benchmark sweep replays the same queries against dozens of index configurations. If
each run decoded the WAV and recomputed MFCCs, the great majority of the wall clock
would be spent in librosa, and -- worse -- tiny numerical differences or a change in
the stored scaler between runs could shift results in ways that have nothing to do with
the index being measured.

So extraction happens once and the resulting vectors and hash lists are cached on disk.
Benchmarks then measure only what differs between configurations: the database search.
The extraction cost is not swept under the carpet -- it is measured here, per query, and
reported alongside search latency as the end-to-end figure in the report.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..audio import decode
from ..config import Config, load_config
from ..db import connect
from ..features.embedding import Standardiser, embed_buffer
from ..features.landmarks import fingerprint_buffer
from ..ingest import fit_scaler


@dataclass
class CachedQuery:
    query_id: str
    track_id: int
    vectors: np.ndarray        # (n_segments, dim), standardised and L2-normalised
    hashes: np.ndarray         # int64
    offsets: np.ndarray        # int32, anchor frame index within the clip
    embed_extract_ms: float
    hash_extract_ms: float


class QueryCache:
    """Disk-backed store of pre-extracted query features."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict[str, CachedQuery] = {}

    # -- construction --------------------------------------------------------

    @staticmethod
    def build(cfg: Config, scaler: Standardiser, scaler_name: str,
              limit: int | None = None) -> "QueryCache":
        with connect(cfg) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT query_id, track_id, path FROM eval_queries ORDER BY query_id"
            )
            rows = cur.fetchall()
        if limit:
            rows = rows[:limit]

        cache = QueryCache(cache_path(cfg, scaler_name))
        for query_id, track_id, path in tqdm(rows, desc="extracting queries",
                                             unit="query"):
            if not Path(path).exists():
                continue
            y = decode(path, cfg.audio.sample_rate)

            t0 = time.perf_counter()
            _, vecs = embed_buffer(y, cfg.audio, cfg.embedding, scaler)
            embed_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            hashes, offsets = fingerprint_buffer(y, cfg.audio, cfg.landmark)
            hash_ms = (time.perf_counter() - t0) * 1000.0

            cache.items[query_id] = CachedQuery(
                query_id=query_id,
                track_id=int(track_id),
                vectors=np.ascontiguousarray(vecs, dtype=np.float32),
                hashes=hashes.astype(np.int64),
                offsets=offsets.astype(np.int32),
                embed_extract_ms=embed_ms,
                hash_extract_ms=hash_ms,
            )
        return cache

    # -- persistence ---------------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        meta_ids: list[str] = []
        meta_rows: list[tuple[int, float, float]] = []
        for query_id, item in self.items.items():
            payload[f"v::{query_id}"] = item.vectors
            payload[f"h::{query_id}"] = item.hashes
            payload[f"o::{query_id}"] = item.offsets
            meta_ids.append(query_id)
            meta_rows.append((item.track_id, item.embed_extract_ms,
                              item.hash_extract_ms))
        payload["__ids__"] = np.array(meta_ids, dtype=object)
        payload["__meta__"] = np.array(meta_rows, dtype=np.float64)
        np.savez(self.path, **payload, allow_pickle=True)

    @staticmethod
    def load(path: Path) -> "QueryCache":
        cache = QueryCache(path)
        with np.load(path, allow_pickle=True) as z:
            ids = [str(x) for x in z["__ids__"]]
            meta = z["__meta__"]
            for i, query_id in enumerate(ids):
                cache.items[query_id] = CachedQuery(
                    query_id=query_id,
                    track_id=int(meta[i, 0]),
                    vectors=z[f"v::{query_id}"],
                    hashes=z[f"h::{query_id}"],
                    offsets=z[f"o::{query_id}"],
                    embed_extract_ms=float(meta[i, 1]),
                    hash_extract_ms=float(meta[i, 2]),
                )
        return cache

    def __len__(self) -> int:
        return len(self.items)


def cache_path(cfg: Config, scaler_name: str) -> Path:
    return cfg.paths.results / "cache" / f"queries_{scaler_name}.npz"


def load_or_build(cfg: Config, scaler_name: str,
                  rebuild: bool = False) -> QueryCache:
    path = cache_path(cfg, scaler_name)
    if path.exists() and not rebuild:
        return QueryCache.load(path)
    scaler = fit_scaler.load(cfg, scaler_name)
    cache = QueryCache.build(cfg, scaler, scaler_name)
    cache.save()
    return cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaler", default=fit_scaler.DEFAULT_MODEL_NAME)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    cache = load_or_build(cfg, args.scaler, rebuild=args.rebuild)
    n_seg = [c.vectors.shape[0] for c in cache.items.values()]
    n_hash = [c.hashes.size for c in cache.items.values()]
    e_ms = [c.embed_extract_ms for c in cache.items.values()]
    h_ms = [c.hash_extract_ms for c in cache.items.values()]
    print(f"cached {len(cache)} queries -> {cache.path}")
    print(f"  segments per query : mean {np.mean(n_seg):.2f} "
          f"(min {min(n_seg)}, max {max(n_seg)})")
    print(f"  hashes per query   : mean {np.mean(n_hash):.0f} "
          f"(min {min(n_hash)}, max {max(n_hash)})")
    print(f"  embed extract (ms) : p50 {np.percentile(e_ms, 50):.1f}  "
          f"p95 {np.percentile(e_ms, 95):.1f}")
    print(f"  hash  extract (ms) : p50 {np.percentile(h_ms, 50):.1f}  "
          f"p95 {np.percentile(h_ms, 95):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
