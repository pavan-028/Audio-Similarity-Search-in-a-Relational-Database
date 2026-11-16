"""The benchmark sweep.

What is measured
----------------
For every configuration: top-1 and top-5 track accuracy, recall@k of the approximate
neighbour set against an exact scan, p50/p95 database latency, index build time and
index size. Everything is written to ``experiments`` / ``query_results`` so the numbers
in the report are reproducible from SQL.

How the sweep is organised
--------------------------
Index *construction* and index *querying* are separated, because they cost wildly
different amounts and because several measurements share one build. An HNSW graph is
built once per ``(scale, m)`` and then queried at every ``ef_search`` in the sweep --
``hnsw.ef_search`` is a session variable, not an index property, so one graph yields a
whole accuracy-latency curve. The same holds for IVFFlat and ``ivfflat.probes``.

Methodological choices worth stating explicitly:

* **The exact scan comes first at each scale**, and its segment-level results are kept
  as the ground truth that ANN recall is measured against. This makes the baseline free
  rather than doubling the work.
* **Query features are pre-extracted** (see query_cache.py), so every configuration
  sees numerically identical query vectors and the timed section contains only database
  work.
* **Each run is warmed up** with a handful of untimed queries. The first query against
  a freshly built index pays for pulling its pages into shared buffers, and including
  that in a p50 would be measuring the buffer cache, not the index.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass

import numpy as np
import psycopg
from tqdm import tqdm

from ..config import Config, load_config
from ..db import connect, relation_bytes, server_settings
from ..indexing import (BuildResult, build_fingerprint_index, build_index,
                        create_scale_table)
from ..ingest import fit_scaler
from ..query import embedding_search as es
from ..query import hash_search as hs
from .query_cache import CachedQuery, QueryCache, load_or_build

DEFAULT_K = 20
WARMUP_QUERIES = 15

# Set by main() when the fingerprint B-tree is (re)built, so that every landmark run
# records the same measured build cost rather than re-timing it per scale.
LANDMARK_BUILD_MS: dict[str, float] = {}


@dataclass
class Grid:
    """The configurations swept. Defaults are what the report's figures use."""

    scales: tuple[int, ...] = (1000, 5000, 8000)
    # Parameter sweeps run only at the largest scale, where differences are clearest;
    # the other scales use the single reference setting so the scale trend is clean.
    ef_search: tuple[int, ...] = (10, 40, 100, 200)
    hnsw_m: tuple[int, ...] = (8, 16, 32)
    hnsw_ef_construction: int = 64
    ivf_lists: tuple[int, ...] = (100, 400)
    ivf_probes: tuple[int, ...] = (1, 10, 40)
    reference_m: int = 16
    reference_ef_search: int = 40
    reference_lists: int = 100
    reference_probes: int = 10
    k: int = DEFAULT_K


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def record_experiment(conn: psycopg.Connection, kind: str, index_type: str,
                      scale: int, params: dict, build: BuildResult | None,
                      notes: str = "") -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiments (kind, index_type, scale_tracks, params,
                                     build_ms, index_bytes, table_bytes, n_vectors,
                                     notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING run_id
            """,
            (kind, index_type, scale, json.dumps(params),
             None if build is None else build.build_ms,
             None if build is None else build.index_bytes,
             None if build is None else build.table_bytes,
             None if build is None else build.n_vectors,
             notes),
        )
        run_id = int(cur.fetchone()[0])
    conn.commit()
    return run_id


def store_results(conn: psycopg.Connection, run_id: int, rows: list[tuple]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO query_results (run_id, query_id, true_track_id, top1_track_id,
                                       hit_rank, latency_ms, extract_ms, ann_recall,
                                       score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, query_id) DO NOTHING
            """,
            [(run_id, *r) for r in rows],
        )
    conn.commit()


def rank_of(track_id: int, ranked: list) -> int | None:
    for i, hit in enumerate(ranked, start=1):
        if hit.track_id == track_id:
            return i
    return None


# ---------------------------------------------------------------------------
# Query selection
# ---------------------------------------------------------------------------

def select_queries(cfg: Config, cache: QueryCache, max_queries: int | None,
                   seed: int = 7) -> list[CachedQuery]:
    """Pick the queries to run, stratified across degradation conditions.

    Subsampling uniformly at random would leave the per-condition cells uneven and make
    the robustness breakdown noisy in an uninformative way. Stratifying keeps every
    (length, SNR, codec) cell equally represented.
    """
    items = list(cache.items.values())
    if max_queries is None or max_queries >= len(items):
        return items

    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute("SELECT query_id, clip_seconds, snr_db, codec FROM eval_queries")
        cond = {q: (c, s, k) for q, c, s, k in cur.fetchall()}

    buckets: dict[tuple, list[CachedQuery]] = {}
    for item in items:
        buckets.setdefault(cond.get(item.query_id, ()), []).append(item)

    rng = np.random.default_rng(seed)
    per_bucket = max(1, max_queries // max(1, len(buckets)))
    chosen: list[CachedQuery] = []
    for bucket in buckets.values():
        take = min(per_bucket, len(bucket))
        idx = rng.choice(len(bucket), size=take, replace=False)
        chosen.extend(bucket[i] for i in sorted(idx))
    return chosen


# ---------------------------------------------------------------------------
# Path A runs
# ---------------------------------------------------------------------------

def run_embedding(conn: psycopg.Connection, cfg: Config, queries: list[CachedQuery],
                  table: str, index_type: str, params: dict, k: int,
                  truth: dict[str, list] | None,
                  collect_truth: bool = False) -> tuple[list[tuple], dict[str, list]]:
    """Execute one Path A configuration over the query set."""
    collected: dict[str, list] = {}
    rows: list[tuple] = []

    with conn.cursor() as cur:
        es.apply_search_params(cur, index_type, params.get("ef_search"),
                               params.get("probes"))

    for item in queries[:WARMUP_QUERIES]:
        es.search_vectors(conn, item.vectors, k=k, table=table)

    for item in tqdm(queries, desc=f"{index_type:<8} {table}", unit="q", leave=False):
        result = es.search_vectors(conn, item.vectors, k=k, table=table)
        rank = rank_of(item.track_id, result.tracks)
        recall = None
        if truth is not None and item.query_id in truth:
            recall = es.recall_at_k(result.segment_hits, truth[item.query_id])
            if not np.isfinite(recall):
                recall = None
        if collect_truth:
            collected[item.query_id] = result.segment_hits
        rows.append((
            item.query_id,
            item.track_id,
            result.tracks[0].track_id if result.tracks else None,
            rank,
            result.search_ms,
            item.embed_extract_ms,
            recall,
            result.tracks[0].score if result.tracks else None,
        ))

    # Session settings are per-connection; reset so the next run starts clean.
    with conn.cursor() as cur:
        cur.execute("RESET enable_indexscan")
        cur.execute("RESET enable_bitmapscan")
    conn.commit()
    return rows, collected


# ---------------------------------------------------------------------------
# Path B runs
# ---------------------------------------------------------------------------

def run_landmark(conn: psycopg.Connection, cfg: Config, queries: list[CachedQuery],
                 scale: int | None) -> list[tuple]:
    rows: list[tuple] = []
    with conn.cursor() as cur:
        for item in queries[:WARMUP_QUERIES]:
            if item.hashes.size:
                hs.search_hashes(cur, item.hashes, item.offsets, scale_tracks=scale)

    for item in tqdm(queries, desc=f"landmark s{scale}", unit="q", leave=False):
        if item.hashes.size == 0:
            continue
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            raw = hs.search_hashes(cur, item.hashes, item.offsets,
                                   scale_tracks=scale)
            search_ms = (time.perf_counter() - t0) * 1000.0
        ranked = hs.collapse_to_tracks(raw)
        rank = rank_of(item.track_id, ranked)
        rows.append((
            item.query_id,
            item.track_id,
            ranked[0].track_id if ranked else None,
            rank,
            search_ms,
            item.hash_extract_ms,
            None,                                  # no approximation to measure
            float(ranked[0].votes) if ranked else None,
        ))
    conn.commit()
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def sweep(cfg: Config, grid: Grid, queries: list[CachedQuery],
          do_embedding: bool = True, do_landmark: bool = True) -> list[int]:
    run_ids: list[int] = []

    with connect(cfg) as conn:
        for scale in grid.scales:
            print(f"\n=== scale {scale} tracks ===")
            table, n_vectors = create_scale_table(conn, scale, cfg.embedding.dim)
            print(f"  {table}: {n_vectors} vectors")

            truth: dict[str, list] = {}

            if do_embedding:
                # --- exact baseline, and the ground truth for recall ---
                build = build_index(conn, table, "exact")
                run_id = record_experiment(
                    conn, "embedding", "exact", scale,
                    {"k": grid.k}, build,
                    notes="sequential scan; also the recall ground truth",
                )
                rows, truth = run_embedding(conn, cfg, queries, table, "exact",
                                            {}, grid.k, None, collect_truth=True)
                store_results(conn, run_id, rows)
                run_ids.append(run_id)
                print(f"  exact                        run_id={run_id}")

                # --- IVFFlat ---
                lists_grid = (grid.ivf_lists if scale == max(grid.scales)
                              else (grid.reference_lists,))
                probes_grid = (grid.ivf_probes if scale == max(grid.scales)
                               else (grid.reference_probes,))
                for lists in lists_grid:
                    build = build_index(conn, table, "ivfflat", lists=lists)
                    print(f"  ivfflat lists={lists:<4} built in "
                          f"{build.build_ms / 1000:6.2f}s "
                          f"({build.index_bytes / 1024 / 1024:.1f} MB)")
                    for probes in probes_grid:
                        params = {"lists": lists, "probes": probes, "k": grid.k}
                        run_id = record_experiment(conn, "embedding", "ivfflat",
                                                   scale, params, build)
                        rows, _ = run_embedding(conn, cfg, queries, table, "ivfflat",
                                                params, grid.k, truth)
                        store_results(conn, run_id, rows)
                        run_ids.append(run_id)
                        print(f"    probes={probes:<4}              run_id={run_id}")

                # --- HNSW ---
                m_grid = (grid.hnsw_m if scale == max(grid.scales)
                          else (grid.reference_m,))
                ef_grid = (grid.ef_search if scale == max(grid.scales)
                           else (grid.reference_ef_search,))
                for m in m_grid:
                    build = build_index(conn, table, "hnsw", m=m,
                                        ef_construction=grid.hnsw_ef_construction)
                    print(f"  hnsw m={m:<4} ef_c={grid.hnsw_ef_construction} built in "
                          f"{build.build_ms / 1000:6.2f}s "
                          f"({build.index_bytes / 1024 / 1024:.1f} MB)")
                    for ef in ef_grid:
                        params = {"m": m,
                                  "ef_construction": grid.hnsw_ef_construction,
                                  "ef_search": ef, "k": grid.k}
                        run_id = record_experiment(conn, "embedding", "hnsw",
                                                   scale, params, build)
                        rows, _ = run_embedding(conn, cfg, queries, table, "hnsw",
                                                params, grid.k, truth)
                        store_results(conn, run_id, rows)
                        run_ids.append(run_id)
                        print(f"    ef_search={ef:<4}           run_id={run_id}")

            if do_landmark:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM fingerprints f JOIN tracks t "
                                "ON t.track_id = f.track_id WHERE t.ingest_order <= %s",
                                (scale,))
                    n_rows = int(cur.fetchone()[0])
                # The B-tree covers the whole table; n_vectors records how many of its
                # rows are in scope at this scale.
                build = BuildResult(
                    "idx_fingerprints_hash", "btree",
                    LANDMARK_BUILD_MS.get("build_ms"),
                    relation_bytes(conn, "idx_fingerprints_hash"),
                    relation_bytes(conn, "fingerprints"), n_rows, {})
                run_id = record_experiment(
                    conn, "landmark", "btree", scale,
                    {"fanout": cfg.landmark.fanout,
                     "max_peaks_per_second": cfg.landmark.max_peaks_per_second,
                     "freq_quant_div": cfg.landmark.freq_quant_div},
                    build,
                    notes="B-tree hash lookup + alignment vote")
                rows = run_landmark(conn, cfg, queries, scale)
                store_results(conn, run_id, rows)
                run_ids.append(run_id)
                print(f"  landmark                     run_id={run_id} "
                      f"({n_rows:,} fingerprint rows in scope)")

    return run_ids


def export(cfg: Config) -> None:
    """Dump the summary views to CSV for the report."""
    import warnings

    import pandas as pd

    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    # pandas warns that it only *tests* SQLAlchemy connectables. A psycopg 3
    # connection is a conforming DBAPI2 object and these are plain SELECTs from
    # views, so the warning is noise here rather than a signal.
    warnings.filterwarnings("ignore", message=".*only supports SQLAlchemy.*")
    with connect(cfg) as conn:
        for view, name in (("run_summary", "run_summary.csv"),
                           ("run_by_condition", "run_by_condition.csv")):
            df = pd.read_sql_query(f"SELECT * FROM {view} ORDER BY run_id", conn)
            df.to_csv(cfg.paths.results / name, index=False)
            print(f"  wrote results/{name}  ({len(df)} rows)")


def environment(cfg: Config) -> dict:
    with connect(cfg) as conn:
        settings = server_settings(conn, [
            "server_version", "shared_buffers", "work_mem", "maintenance_work_mem",
            "max_parallel_maintenance_workers", "max_parallel_workers_per_gather",
            "effective_cache_size", "random_page_cost",
        ])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "postgres": settings,
        "audio": {"sample_rate": cfg.audio.sample_rate,
                  "segment_seconds": cfg.audio.segment_seconds,
                  "hop_seconds": cfg.audio.hop_seconds},
        "embedding_dim": cfg.embedding.dim,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaler", default=fit_scaler.DEFAULT_MODEL_NAME)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="stratified subsample of the query set")
    parser.add_argument("--scales", type=int, nargs="*", default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--no-embedding", action="store_true")
    parser.add_argument("--no-landmark", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-fp-index", action="store_true",
                        help="drop and rebuild the fingerprint B-tree, timing it")
    parser.add_argument("--fresh", action="store_true",
                        help="clear previous experiment rows first")
    args = parser.parse_args(argv)

    cfg = load_config()
    grid = Grid(k=args.k)
    if args.scales:
        grid = Grid(scales=tuple(args.scales), k=args.k)

    if args.fresh:
        with connect(cfg) as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE query_results, experiments")
            conn.commit()
        print("cleared previous experiment rows")

    cache = load_or_build(cfg, args.scaler, rebuild=args.rebuild_cache)
    queries = select_queries(cfg, cache, args.max_queries)
    print(f"query set: {len(queries)} of {len(cache)} cached queries")

    if args.rebuild_fp_index and not args.no_landmark:
        with connect(cfg) as conn:
            print("rebuilding fingerprint B-tree ...")
            br = build_fingerprint_index(conn)
            LANDMARK_BUILD_MS["build_ms"] = br.build_ms
            print(f"  {br.n_vectors:,} rows, {br.build_ms / 1000:.2f}s, "
                  f"{br.index_bytes / 1024 / 1024:.1f} MB")

    started = time.perf_counter()
    run_ids = sweep(cfg, grid, queries,
                    do_embedding=not args.no_embedding,
                    do_landmark=not args.no_landmark)
    print(f"\n{len(run_ids)} runs in {(time.perf_counter() - started) / 60:.1f} min")

    print("\nexporting ...")
    export(cfg)
    env_path = cfg.paths.results / "environment.json"
    env_path.write_text(json.dumps(environment(cfg), indent=2), encoding="utf-8")
    print(f"  wrote results/environment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
