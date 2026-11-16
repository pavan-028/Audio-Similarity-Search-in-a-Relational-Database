"""Path B retrieval: hash lookup plus time-offset alignment voting.

The matching rule is what makes landmark fingerprinting precise. Looking up a query
hash in ``fingerprints`` returns every place in the corpus where that particular
(f1, f2, dt) triple occurs, and most of those are coincidences. The signal is not *how
many* hashes a track shares with the query but whether many of them agree on a single
constant time offset:

    delta = t_offset_in_track - t_offset_in_query

If the query really is an excerpt of that track, every genuine match has the same
delta -- the position of the excerpt. Coincidental matches scatter their deltas
uniformly. So the score is the height of the tallest bin in the per-track histogram of
delta, and that histogram is also the answer to "where in the track does this come
from".

The histogram is computed inside PostgreSQL. Pulling the raw matching rows to the
client and counting them in Python would move hundreds of thousands of rows across the
wire per query to produce a few dozen numbers; a GROUP BY next to the data moves only
the result. The query hashes are shipped as two arrays and unnested into a join.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import psycopg

from ..config import Config
from ..features.landmarks import fingerprint_buffer


@dataclass
class LandmarkHit:
    track_id: int
    delta: int           # best alignment offset, in STFT frames
    votes: int           # size of the tallest bin


@dataclass
class LandmarkResult:
    tracks: list[LandmarkHit]
    search_ms: float
    extract_ms: float
    n_query_hashes: int
    n_matched_rows: int


# A hash shared by a large fraction of the corpus carries almost no information but
# dominates the join cost. Dropping the worst offenders is standard practice; the
# threshold is expressed per-query so it scales with the database.
DEFAULT_MAX_HASH_FANOUT = 4000


SEARCH_SQL = """
WITH q AS (
    SELECT hash, q_offset
    FROM unnest(%(hashes)s::bigint[], %(offsets)s::int[]) AS t(hash, q_offset)
),
matched AS (
    SELECT f.track_id,
           f.t_offset - q.q_offset AS delta
    FROM q
    JOIN fingerprints f ON f.hash = q.hash
    {scale_join}
)
SELECT track_id, delta, count(*) AS votes
FROM matched
GROUP BY track_id, delta
ORDER BY votes DESC, track_id
LIMIT %(limit)s
"""

COUNT_SQL = """
SELECT count(*)
FROM unnest(%(hashes)s::bigint[]) AS t(hash)
JOIN fingerprints f ON f.hash = t.hash
"""


def search_hashes(cur: psycopg.Cursor, hashes: np.ndarray, offsets: np.ndarray,
                  *, scale_tracks: int | None = None,
                  limit: int = 200) -> list[tuple[int, int, int]]:
    """Run the alignment histogram. Returns ``(track_id, delta, votes)`` rows."""
    scale_join = ""
    params: dict = {
        "hashes": [int(h) for h in hashes],
        "offsets": [int(o) for o in offsets],
        "limit": limit,
    }
    if scale_tracks is not None:
        # Restricting by ingest_order is how the scale experiments shrink the corpus
        # without physically rebuilding the fingerprint table.
        scale_join = "JOIN tracks tr ON tr.track_id = f.track_id " \
                     "AND tr.ingest_order <= %(scale)s"
        params["scale"] = scale_tracks

    cur.execute(SEARCH_SQL.format(scale_join=scale_join), params)
    return [(int(t), int(d), int(v)) for t, d, v in cur.fetchall()]


def collapse_to_tracks(rows: list[tuple[int, int, int]]) -> list[LandmarkHit]:
    """Keep each track's best-aligned bin and rank tracks by it."""
    best: dict[int, LandmarkHit] = {}
    for track_id, delta, votes in rows:
        current = best.get(track_id)
        if current is None or votes > current.votes:
            best[track_id] = LandmarkHit(track_id, delta, votes)
    out = list(best.values())
    out.sort(key=lambda h: (-h.votes, h.track_id))
    return out


def search_clip(conn: psycopg.Connection, cfg: Config, y: np.ndarray, *,
                scale_tracks: int | None = None,
                limit: int = 200,
                count_matched: bool = False) -> LandmarkResult:
    """Full Path B query for one decoded clip."""
    t0 = time.perf_counter()
    hashes, offsets = fingerprint_buffer(y, cfg.audio, cfg.landmark)
    extract_ms = (time.perf_counter() - t0) * 1000.0

    if hashes.size == 0:
        return LandmarkResult([], 0.0, extract_ms, 0, 0)

    with conn.cursor() as cur:
        t0 = time.perf_counter()
        rows = search_hashes(cur, hashes, offsets, scale_tracks=scale_tracks,
                             limit=limit)
        search_ms = (time.perf_counter() - t0) * 1000.0

        n_matched = 0
        if count_matched:
            # Diagnostic only, and deliberately excluded from the timed section.
            cur.execute(COUNT_SQL, {"hashes": [int(h) for h in hashes]})
            n_matched = int(cur.fetchone()[0])

    return LandmarkResult(
        tracks=collapse_to_tracks(rows),
        search_ms=search_ms,
        extract_ms=extract_ms,
        n_query_hashes=int(hashes.size),
        n_matched_rows=n_matched,
    )


def hash_fanout_stats(conn: psycopg.Connection) -> dict:
    """How skewed is the hash distribution across the whole table?

    A perfectly uniform hash function would put the same number of rows in every bucket
    and a B-tree lookup would always return a handful. Real audio is far from uniform:
    some (f1, f2, dt) triples are generated by whole genres.

    The aggregate is computed over the entire table rather than a prefix. Sampling with
    ``LIMIT`` would read rows in physical order, which here correlates with ingestion
    order and therefore with track and genre -- a biased sample of exactly the quantity
    being measured.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH c AS (SELECT hash, count(*) AS n FROM fingerprints GROUP BY hash)
            SELECT count(*)                                         AS distinct_hashes,
                   sum(n)                                           AS total_rows,
                   avg(n)                                           AS mean_rows,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY n)  AS p50,
                   percentile_cont(0.99) WITHIN GROUP (ORDER BY n)  AS p99,
                   max(n)                                           AS max_rows
            FROM c
            """
        )
        row = cur.fetchone()
    return {
        "distinct_hashes": int(row[0]),
        "total_rows": int(row[1] or 0),
        "mean_rows_per_hash": float(row[2] or 0.0),
        "p50": float(row[3] or 0.0),
        "p99": float(row[4] or 0.0),
        "max_rows": int(row[5] or 0),
    }


def query_weighted_fanout(conn: psycopg.Connection,
                          hashes: np.ndarray) -> dict:
    """How many rows an actual query's hashes match.

    This, not the table-wide average, is what sets Path B's latency: a query draws its
    hashes from real music, so it is far more likely to ask for a popular bucket than a
    uniformly random one. The gap between this number and ``mean_rows_per_hash`` is the
    cost of that skew.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH q AS (SELECT DISTINCT h FROM unnest(%s::bigint[]) AS t(h)),
                 c AS (SELECT q.h, count(f.hash) AS n
                       FROM q LEFT JOIN fingerprints f ON f.hash = q.h
                       GROUP BY q.h)
            SELECT count(*)                                         AS query_hashes,
                   sum(n)                                           AS matched_rows,
                   avg(n)                                           AS mean_rows,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY n)  AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY n)  AS p95,
                   max(n)                                           AS max_rows
            FROM c
            """,
            ([int(h) for h in hashes],),
        )
        row = cur.fetchone()
    return {
        "distinct_query_hashes": int(row[0]),
        "matched_rows": int(row[1] or 0),
        "mean_rows_per_hash": float(row[2] or 0.0),
        "p50": float(row[3] or 0.0),
        "p95": float(row[4] or 0.0),
        "max_rows": int(row[5] or 0),
    }
