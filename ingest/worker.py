"""Stage 2 of ingestion: parallel feature-extraction workers.

Each worker runs the same loop:

1. claim a batch of pending rows with ``SELECT ... FOR UPDATE SKIP LOCKED``;
2. decode each file and extract both feature paths;
3. ``COPY`` the results in and mark the row done;
4. repeat until the queue is empty.

``SKIP LOCKED`` is the whole trick. Without it, concurrent workers running the same
``SELECT ... FOR UPDATE`` would serialise: the second worker would block on the rows the
first had locked, and N workers would give roughly the throughput of one. With it, a
worker steps over any row another transaction already holds and claims the next free
one, so the queue distributes itself with no coordinator, no partitioning scheme and no
risk of two workers doing the same file.

The claim is committed before any audio work begins. Holding a transaction open across
several seconds of CPU-bound decoding would pin rows, bloat the snapshot horizon and
block vacuum for no benefit.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio import DecodeError, decode
from ..config import Config, load_config
from ..db import connect, copy_embeddings, copy_fingerprints
from ..features.embedding import Standardiser, embed_buffer
from ..features.landmarks import fingerprint_buffer
from . import fit_scaler
from .loader import track_id_from_path


@dataclass
class WorkerStats:
    files: int = 0
    segments: int = 0
    fingerprints: int = 0
    failed: int = 0
    seconds: float = 0.0


def claim(conn, worker_id: str, limit: int) -> list[int]:
    """Claim up to ``limit`` pending rows. Returns the claimed staging ids."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT staging_id
            FROM audio_staging
            WHERE status = 'pending'
            ORDER BY staging_id
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (limit,),
        )
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            cur.execute(
                """
                UPDATE audio_staging
                   SET status = 'processing', worker_id = %s,
                       started_at = now(), attempts = attempts + 1
                 WHERE staging_id = ANY(%s)
                """,
                (worker_id, ids),
            )
    conn.commit()
    return ids


def process_one(conn, cfg: Config, scaler: Standardiser, staging_id: int,
                do_embeddings: bool, do_fingerprints: bool) -> tuple[int, int]:
    """Decode and index one staged file. Returns ``(n_segments, n_fingerprints)``."""
    with conn.cursor() as cur:
        cur.execute("SELECT path, raw_blob FROM audio_staging WHERE staging_id = %s",
                    (staging_id,))
        path, blob = cur.fetchone()

    # Prefer the staged bytes: that is the pipeline's own copy of the input, and using
    # it means a worker needs nothing but a database connection to do its job.
    source = bytes(blob) if blob is not None else path
    y = decode(source, cfg.audio.sample_rate)
    duration = y.size / cfg.audio.sample_rate
    if duration < cfg.audio.min_track_seconds:
        raise DecodeError(f"too short: {duration:.2f}s")

    track_id = track_id_from_path(Path(path))

    starts, vecs = (embed_buffer(y, cfg.audio, cfg.embedding, scaler)
                    if do_embeddings else (np.zeros(0), np.zeros((0, cfg.embedding.dim))))
    hashes, offsets = (fingerprint_buffer(y, cfg.audio, cfg.landmark)
                       if do_fingerprints else (np.zeros(0, np.int64), np.zeros(0, np.int32)))

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tracks (track_id, source_path, duration, n_segments,
                                n_fingerprints)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (track_id) DO UPDATE
               SET source_path = EXCLUDED.source_path,
                   duration = EXCLUDED.duration,
                   n_segments = EXCLUDED.n_segments,
                   n_fingerprints = EXCLUDED.n_fingerprints
            """,
            (track_id, path, float(duration), int(vecs.shape[0]), int(hashes.size)),
        )
        # Re-processing a file must not collide with its previous output.
        if do_embeddings:
            cur.execute("DELETE FROM embeddings WHERE track_id = %s", (track_id,))
            copy_embeddings(
                cur,
                ((track_id, float(s), v) for s, v in zip(starts, vecs)),
            )
        if do_fingerprints:
            cur.execute("DELETE FROM fingerprints WHERE track_id = %s", (track_id,))
            copy_fingerprints(
                cur,
                zip(hashes.tolist(), [track_id] * hashes.size, offsets.tolist()),
            )
        cur.execute(
            "UPDATE audio_staging SET status = 'done', finished_at = now(), "
            "error = NULL WHERE staging_id = %s",
            (staging_id,),
        )
    conn.commit()
    return int(vecs.shape[0]), int(hashes.size)


def mark_failed(conn, staging_id: int, message: str) -> None:
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE audio_staging SET status = 'failed', finished_at = now(), "
            "error = %s WHERE staging_id = %s",
            (message[:500], staging_id),
        )
    conn.commit()


def worker_loop(cfg: Config, worker_index: int, do_embeddings: bool,
                do_fingerprints: bool, scaler_name: str,
                result_queue: "mp.Queue | None" = None) -> WorkerStats:
    worker_id = f"{socket.gethostname()}:{os.getpid()}:w{worker_index}"
    stats = WorkerStats()
    started = time.perf_counter()

    with connect(cfg) as conn:
        scaler = (fit_scaler.load(cfg, scaler_name) if do_embeddings
                  else Standardiser(np.zeros(cfg.embedding.dim),
                                    np.ones(cfg.embedding.dim)))
        while True:
            ids = claim(conn, worker_id, cfg.ingest.claim_batch)
            if not ids:
                break
            for staging_id in ids:
                try:
                    n_seg, n_fp = process_one(conn, cfg, scaler, staging_id,
                                              do_embeddings, do_fingerprints)
                    stats.files += 1
                    stats.segments += n_seg
                    stats.fingerprints += n_fp
                except Exception as exc:  # noqa: BLE001 - a bad file must not kill the worker
                    stats.failed += 1
                    try:
                        mark_failed(conn, staging_id, f"{type(exc).__name__}: {exc}")
                    except Exception:
                        pass

    stats.seconds = time.perf_counter() - started
    if result_queue is not None:
        result_queue.put(stats)
    return stats


def requeue_stale(cfg: Config, older_than_minutes: int = 30) -> int:
    """Return rows abandoned by a dead worker to the pending state.

    A worker that is killed mid-file leaves its claim in 'processing' with nobody
    holding the lock. This is the operational counterpart to SKIP LOCKED: cheap
    claiming in exchange for needing an explicit sweep for orphans.
    """
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE audio_staging
               SET status = 'pending', worker_id = NULL, started_at = NULL
             WHERE status = 'processing'
               AND started_at < now() - (%s || ' minutes')::interval
            """,
            (str(older_than_minutes),),
        )
        n = cur.rowcount
        conn.commit()
    return n


def run_pool(cfg: Config, workers: int, do_embeddings: bool, do_fingerprints: bool,
             scaler_name: str) -> WorkerStats:
    """Spawn ``workers`` processes and report combined throughput."""
    ctx = mp.get_context("spawn")      # the only option on Windows
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker_loop,
                    args=(cfg, i, do_embeddings, do_fingerprints, scaler_name, queue),
                    daemon=False)
        for i in range(workers)
    ]
    wall_start = time.perf_counter()
    for p in procs:
        p.start()

    # The parent reports progress by polling the queue table, so it stays useful even
    # if a worker is wedged.
    with connect(cfg, autocommit=True) as conn:
        last = -1
        while any(p.is_alive() for p in procs):
            time.sleep(3.0)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FILTER (WHERE status='done'), "
                    "       count(*) FILTER (WHERE status='pending'), "
                    "       count(*) FILTER (WHERE status='processing'), "
                    "       count(*) FILTER (WHERE status='failed') "
                    "FROM audio_staging"
                )
                done, pending, processing, failed = cur.fetchone()
            if done != last:
                elapsed = time.perf_counter() - wall_start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = pending / rate if rate > 0 else float("nan")
                print(f"  done={done} pending={pending} active={processing} "
                      f"failed={failed}  {rate:.1f} files/s  eta={eta/60:.1f} min",
                      flush=True)
                last = done

    for p in procs:
        p.join()

    total = WorkerStats()
    while not queue.empty():
        s = queue.get()
        total.files += s.files
        total.segments += s.segments
        total.fingerprints += s.fingerprints
        total.failed += s.failed
    total.seconds = time.perf_counter() - wall_start
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--no-fingerprints", action="store_true")
    parser.add_argument("--scaler", default=fit_scaler.DEFAULT_MODEL_NAME)
    parser.add_argument("--requeue-stale", type=int, default=None, metavar="MINUTES")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.requeue_stale is not None:
        n = requeue_stale(cfg, args.requeue_stale)
        print(f"requeued {n} stale rows")
        return 0

    workers = args.workers or cfg.ingest.workers
    print(f"starting {workers} workers "
          f"(embeddings={not args.no_embeddings}, "
          f"fingerprints={not args.no_fingerprints})")
    stats = run_pool(cfg, workers, not args.no_embeddings,
                     not args.no_fingerprints, args.scaler)
    print()
    print(f"files      : {stats.files}  (failed {stats.failed})")
    print(f"segments   : {stats.segments}")
    print(f"fingerprint: {stats.fingerprints}")
    print(f"wall clock : {stats.seconds:.1f}s")
    if stats.seconds > 0:
        print(f"throughput : {stats.files / stats.seconds:.2f} files/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
