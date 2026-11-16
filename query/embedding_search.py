"""Path A retrieval: segment-level vector search, then aggregation to tracks.

A query clip is cut into the same 5 s / 2.5 s segments as the reference material, so a
5 s query yields one vector, a 10 s query three and a 15 s query five. Each vector is
searched independently and the segment hits are then combined per track.

Two-stage retrieval like this is the natural consequence of indexing segments: the
index answers "which five-second windows in the corpus resemble this one", and turning
that into "which track is this" is a separate, cheap aggregation step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import psycopg

from ..config import Config
from ..features.embedding import Standardiser, embed_buffer

Aggregation = Literal["sumsim", "votes", "mindist"]


@dataclass
class SegmentHit:
    track_id: int
    segment_start: float
    distance: float


@dataclass
class TrackHit:
    track_id: int
    score: float
    votes: int
    min_distance: float


@dataclass
class SearchResult:
    tracks: list[TrackHit]
    segment_hits: list[list[SegmentHit]]     # one list per query segment
    search_ms: float                         # database time only
    extract_ms: float                        # decode + feature extraction
    n_query_segments: int


# ---------------------------------------------------------------------------
# Index configuration
# ---------------------------------------------------------------------------

def apply_search_params(cur: psycopg.Cursor, index_type: str,
                        ef_search: int | None = None,
                        probes: int | None = None) -> None:
    """Set the per-session knob that controls the accuracy/speed trade-off.

    These are ``SET LOCAL``-style session GUCs, not index properties: the same HNSW
    index can be queried at many points on its accuracy-latency curve just by changing
    ``hnsw.ef_search``, which is exactly what the sweep in the report exploits. The
    IVFFlat equivalent is ``ivfflat.probes``, the number of cell centroids visited.

    For the exact baseline both are irrelevant, but enable_indexscan is turned off so
    that an index left on the table cannot be used accidentally.

    ``set_config()`` is used rather than ``SET``: the ``SET`` statement is parsed before
    parameter binding, so ``SET hnsw.ef_search = %s`` fails with "syntax error at or
    near $1". ``set_config(name, value, is_local)`` is an ordinary function call and
    takes bind parameters normally, which keeps the value out of the SQL text.
    """
    if index_type == "hnsw" and ef_search is not None:
        cur.execute("SELECT set_config('hnsw.ef_search', %s, false)",
                    (str(int(ef_search)),))
    elif index_type == "ivfflat" and probes is not None:
        cur.execute("SELECT set_config('ivfflat.probes', %s, false)",
                    (str(int(probes)),))
    elif index_type == "exact":
        cur.execute("SELECT set_config('enable_indexscan', 'off', false)")
        cur.execute("SELECT set_config('enable_bitmapscan', 'off', false)")


# ---------------------------------------------------------------------------
# Segment-level search
# ---------------------------------------------------------------------------

def search_segments(cur: psycopg.Cursor, vector: np.ndarray, k: int,
                    table: str = "embeddings") -> list[SegmentHit]:
    """Nearest ``k`` indexed segments to one query vector, by cosine distance.

    The ORDER BY must be on the raw ``<=>`` expression for pgvector to recognise it as
    an indexable ordering; wrapping it in arithmetic (``1 - (a <=> b)``, say) silently
    disables the index and turns every query into a sequential scan.
    """
    cur.execute(
        f"""
        SELECT track_id, segment_start, embedding <=> %s AS dist
        FROM {table}
        ORDER BY dist
        LIMIT %s
        """,
        (vector, k),
    )
    return [SegmentHit(int(t), float(s), float(d)) for t, s, d in cur.fetchall()]


def aggregate(segment_hits: Sequence[Sequence[SegmentHit]],
              how: Aggregation = "sumsim") -> list[TrackHit]:
    """Combine per-segment hits into a ranked track list.

    ``sumsim``  sum of (1 - distance) over every hit belonging to the track. This is
                the default because it blends both kinds of evidence: a track gains
                from being hit by several query segments *and* from being close to
                them. A pure vote count throws away margin, and a pure minimum
                distance throws away corroboration.
    ``votes``   number of distinct query segments that hit the track, ties broken by
                the closest hit.
    ``mindist`` closest single hit, i.e. plain nearest-neighbour ranking.
    """
    per_track: dict[int, list[float]] = {}
    votes: dict[int, set[int]] = {}
    for q_index, hits in enumerate(segment_hits):
        for h in hits:
            per_track.setdefault(h.track_id, []).append(h.distance)
            votes.setdefault(h.track_id, set()).add(q_index)

    out: list[TrackHit] = []
    for track_id, dists in per_track.items():
        n_votes = len(votes[track_id])
        min_d = min(dists)
        if how == "sumsim":
            score = float(sum(max(0.0, 1.0 - d) for d in dists))
        elif how == "votes":
            # Votes dominate; the closest hit only separates equal vote counts.
            score = n_votes + (1.0 - min_d) * 1e-3
        else:
            score = -min_d
        out.append(TrackHit(track_id, score, n_votes, min_d))

    out.sort(key=lambda t: (-t.score, t.min_distance))
    return out


# ---------------------------------------------------------------------------
# End-to-end query
# ---------------------------------------------------------------------------

def search_vectors(conn: psycopg.Connection, vecs: np.ndarray, *, k: int = 20,
                   table: str = "embeddings",
                   how: Aggregation = "sumsim") -> SearchResult:
    """Search pre-extracted query vectors.

    Separate from :func:`search_clip` because the benchmarks extract each query's
    features once and then replay the same vectors against every index configuration.
    That keeps feature extraction out of the timed section and guarantees that any
    difference between runs comes from the index and not from re-extraction.

    Session parameters are assumed to have been applied already by the caller.
    """
    if vecs.shape[0] == 0:
        return SearchResult([], [], 0.0, 0.0, 0)
    with conn.cursor() as cur:
        t0 = time.perf_counter()
        segment_hits = [search_segments(cur, v, k, table) for v in vecs]
        search_ms = (time.perf_counter() - t0) * 1000.0
    return SearchResult(
        tracks=aggregate(segment_hits, how),
        segment_hits=segment_hits,
        search_ms=search_ms,
        extract_ms=0.0,
        n_query_segments=int(vecs.shape[0]),
    )


def search_clip(conn: psycopg.Connection, cfg: Config, y: np.ndarray,
                scaler: Standardiser, *, k: int = 20, index_type: str = "hnsw",
                ef_search: int | None = None, probes: int | None = None,
                table: str = "embeddings",
                how: Aggregation = "sumsim",
                apply_params: bool = True) -> SearchResult:
    """Full Path A query for one decoded clip, including feature extraction."""
    t0 = time.perf_counter()
    _, vecs = embed_buffer(y, cfg.audio, cfg.embedding, scaler)
    extract_ms = (time.perf_counter() - t0) * 1000.0

    if vecs.shape[0] == 0:
        return SearchResult([], [], 0.0, extract_ms, 0)

    with conn.cursor() as cur:
        if apply_params:
            apply_search_params(cur, index_type, ef_search, probes)
        t0 = time.perf_counter()
        segment_hits = [search_segments(cur, v, k, table) for v in vecs]
        search_ms = (time.perf_counter() - t0) * 1000.0

    return SearchResult(
        tracks=aggregate(segment_hits, how),
        segment_hits=segment_hits,
        search_ms=search_ms,
        extract_ms=extract_ms,
        n_query_segments=int(vecs.shape[0]),
    )


def recall_at_k(ann: Sequence[Sequence[SegmentHit]],
                exact: Sequence[Sequence[SegmentHit]]) -> float:
    """Fraction of the exact top-k segments that the ANN search also returned.

    Compared on ``(track_id, segment_start)`` identity rather than on distance, and
    averaged over the query's segments. This is recall of the *neighbour set*, which is
    what an ANN index actually approximates -- distinct from end-to-end accuracy, since
    a query can lose several true neighbours and still rank the right track first.
    """
    if not exact:
        return float("nan")
    scores = []
    for ann_hits, exact_hits in zip(ann, exact):
        if not exact_hits:
            continue
        truth = {(h.track_id, round(h.segment_start, 3)) for h in exact_hits}
        got = {(h.track_id, round(h.segment_start, 3)) for h in ann_hits}
        scores.append(len(truth & got) / len(truth))
    return float(np.mean(scores)) if scores else float("nan")
