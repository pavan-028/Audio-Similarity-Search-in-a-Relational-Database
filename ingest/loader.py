"""Stage 1 of ingestion: put files on the queue.

The loader does no audio work at all. It walks the corpus, writes one
``audio_staging`` row per file with ``status = 'pending'``, and stops. Decoding and
feature extraction are left entirely to the workers.

Splitting the pipeline at this point is what makes the rest of it scalable and
restartable: the queue is durable, so workers can be added, killed or restarted at any
time without losing or duplicating work, and a crash mid-run costs only the rows that
were in flight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_config
from ..db import connect


def discover(root: Path, limit: int | None = None) -> list[Path]:
    """All .mp3 files under ``root``, in a deterministic order."""
    files = sorted(root.rglob("*.mp3"))
    return files[:limit] if limit else files


def track_id_from_path(path: Path) -> int:
    """FMA encodes the track id in the filename: 000002.mp3 -> 2."""
    return int(path.stem)


def stage_files(cfg: Config, files: list[Path], store_blob: bool,
                batch: int = 500) -> int:
    """Insert staging rows, skipping paths already queued. Returns rows added."""
    added = 0
    with connect(cfg) as conn:
        with conn.cursor() as cur:
            buf: list[tuple] = []
            for path in tqdm(files, desc="staging", unit="file"):
                blob = path.read_bytes() if store_blob else None
                buf.append((str(path), blob, path.stat().st_size))
                if len(buf) >= batch:
                    added += _flush(cur, buf)
                    conn.commit()
                    buf.clear()
            if buf:
                added += _flush(cur, buf)
                conn.commit()
    return added


def _flush(cur, rows: list[tuple]) -> int:
    # ON CONFLICT DO NOTHING makes the loader idempotent: re-running it after a partial
    # load adds only what is missing instead of failing on the UNIQUE(path) constraint.
    cur.executemany(
        """
        INSERT INTO audio_staging (path, raw_blob, byte_size, status)
        VALUES (%s, %s, %s, 'pending')
        ON CONFLICT (path) DO NOTHING
        """,
        rows,
    )
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="corpus root (default: Paths.fma_audio)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stage only the first N files (for smoke tests)")
    parser.add_argument("--no-blob", action="store_true",
                        help="record the path only, do not copy bytes into the table")
    args = parser.parse_args(argv)

    cfg = load_config()
    root = args.root or cfg.paths.fma_audio
    if not root.exists():
        print(f"corpus root not found: {root}", file=sys.stderr)
        return 2

    files = discover(root, args.limit)
    print(f"found {len(files)} files under {root}")
    store_blob = cfg.ingest.store_blob and not args.no_blob
    added = stage_files(cfg, files, store_blob=store_blob)
    print(f"staged {added} rows (store_blob={store_blob})")

    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute("SELECT status, count(*) FROM audio_staging GROUP BY status "
                    "ORDER BY status")
        for status, n in cur.fetchall():
            print(f"  {status:<11} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
