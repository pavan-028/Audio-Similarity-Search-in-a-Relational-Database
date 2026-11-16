"""Stage 3 of ingestion: attach catalogue metadata and fix the experiment splits.

Three things happen here, all of which need the full set of successfully ingested
tracks to be known, and none of which the per-file workers can do on their own.

1. **Metadata.** Title, artist and genre come from the FMA metadata CSV. The workers
   never touch it: making 8 000 independent processes each parse a 248 MB
   multi-index CSV would be absurd, so it is joined once at the end.

2. **ingest_order** -- a fixed random permutation of the corpus. The scale experiments
   select ``ingest_order <= N``, which gives *nested* subsets (the 1 000-track index is
   a subset of the 5 000-track one). Nesting matters: it means a change in accuracy
   across scales is caused by the extra distractors and nothing else.

3. **is_query_source** -- the ~200 held-out tracks the evaluation clips are cut from.
   These are drawn only from ``ingest_order <= min_scale``, so that every query's
   ground-truth track is present in the index *at every scale*. Picking them from the
   whole corpus instead would mean most queries at the 1 000-track scale had no correct
   answer available, and the resulting accuracy numbers would measure the sampling
   accident rather than the index.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from ..config import Config, load_config
from ..db import connect

SEED = 20260211


def attach_metadata(cfg: Config) -> int:
    """Join title/artist/genre from the FMA metadata CSV. Returns rows updated."""
    import pandas as pd

    path = cfg.paths.fma_metadata / "tracks.csv"
    if not path.exists():
        print(f"  metadata CSV not found ({path}); leaving title/artist NULL")
        return 0

    # tracks.csv has a two-level column header and a type row; pandas reads it with
    # header=[0, 1] and the track id as the index.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(path, index_col=0, header=[0, 1], low_memory=False)

    def col(group: str, name: str):
        return df[(group, name)] if (group, name) in df.columns else None

    title = col("track", "title")
    artist = col("artist", "name")
    genre = col("track", "genre_top")

    rows = []
    for track_id in df.index:
        rows.append((
            None if title is None or pd.isna(title.loc[track_id]) else str(title.loc[track_id])[:300],
            None if artist is None or pd.isna(artist.loc[track_id]) else str(artist.loc[track_id])[:300],
            None if genre is None or pd.isna(genre.loc[track_id]) else str(genre.loc[track_id])[:100],
            int(track_id),
        ))

    with connect(cfg) as conn, conn.cursor() as cur:
        # Only tracks we actually ingested are touched; the CSV covers all 106 k FMA
        # tracks, we hold 8 000 of them.
        cur.executemany(
            """
            UPDATE tracks SET title = %s, artist = %s, genre = %s
             WHERE track_id = %s
            """,
            rows,
        )
        conn.commit()
        cur.execute("SELECT count(*) FROM tracks WHERE title IS NOT NULL")
        n = cur.fetchone()[0]
    return int(n)


def assign_splits(cfg: Config, n_query_sources: int = 200,
                  min_scale: int = 1000, seed: int = SEED) -> tuple[int, int]:
    """Assign ingest_order and pick the query-source tracks."""
    rng = np.random.default_rng(seed)

    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute("SELECT track_id FROM tracks ORDER BY track_id")
        track_ids = np.array([r[0] for r in cur.fetchall()], dtype=np.int64)
        if track_ids.size == 0:
            raise RuntimeError("no tracks ingested yet")

        order = rng.permutation(track_ids.size) + 1          # 1..N
        cur.executemany(
            "UPDATE tracks SET ingest_order = %s WHERE track_id = %s",
            [(int(o), int(t)) for o, t in zip(order, track_ids)],
        )

        # Query sources: drawn from the smallest scale so they exist in every index.
        eligible_n = min(min_scale, track_ids.size)
        cur.execute(
            """
            SELECT track_id FROM tracks
             WHERE ingest_order <= %s AND duration >= %s
             ORDER BY ingest_order
            """,
            (eligible_n, 16.0),
        )
        eligible = np.array([r[0] for r in cur.fetchall()], dtype=np.int64)
        take = min(n_query_sources, eligible.size)
        chosen = rng.choice(eligible, size=take, replace=False)

        cur.execute("UPDATE tracks SET is_query_source = FALSE")
        cur.executemany(
            "UPDATE tracks SET is_query_source = TRUE WHERE track_id = %s",
            [(int(t),) for t in chosen],
        )
        conn.commit()

    return int(track_ids.size), int(take)


def report(cfg: Config) -> None:
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*), sum(n_segments), sum(n_fingerprints),
                   min(duration), avg(duration), max(duration)
            FROM tracks
            """
        )
        n, seg, fp, dmin, davg, dmax = cur.fetchone()
        print(f"tracks            : {n}")
        print(f"segments          : {seg}")
        print(f"fingerprint rows  : {fp}")
        print(f"duration (s)      : min {dmin:.1f}  mean {davg:.1f}  max {dmax:.1f}")

        cur.execute("SELECT count(*) FROM tracks WHERE is_query_source")
        print(f"query sources     : {cur.fetchone()[0]}")
        cur.execute("SELECT count(*) FROM tracks WHERE title IS NOT NULL")
        print(f"with metadata     : {cur.fetchone()[0]}")
        cur.execute(
            "SELECT genre, count(*) FROM tracks GROUP BY genre ORDER BY 2 DESC LIMIT 10"
        )
        print("top genres        :")
        for genre, cnt in cur.fetchall():
            print(f"    {str(genre):<22} {cnt}")
        cur.execute("SELECT status, count(*) FROM audio_staging GROUP BY status")
        print("staging           :")
        for status, cnt in cur.fetchall():
            print(f"    {status:<22} {cnt}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-sources", type=int, default=200)
    parser.add_argument("--min-scale", type=int, default=1000)
    args = parser.parse_args(argv)

    cfg = load_config()
    print("attaching metadata ...")
    n_meta = attach_metadata(cfg)
    print(f"  {n_meta} tracks now have a title")
    print("assigning splits ...")
    n_tracks, n_q = assign_splits(cfg, args.query_sources, args.min_scale)
    print(f"  ingest_order assigned to {n_tracks} tracks; {n_q} query sources")
    print()
    report(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
