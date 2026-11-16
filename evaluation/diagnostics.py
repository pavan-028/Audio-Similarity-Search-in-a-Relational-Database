"""Structural measurements that explain the benchmark numbers.

These are not accuracy or latency results; they are properties of the stored data that
account for the results. Three questions:

1. **How skewed is the fingerprint hash distribution?** This sets Path B's latency far
   more than the row count does.
2. **How much does the Path A standardisation actually matter?** Measured as an
   ablation, by searching the same corpus with and without it.
3. **Where does the time go end to end?** Extraction versus database search, per path.

Written to ``results/diagnostics.json`` and printed as Markdown for the report.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..config import Config, load_config
from ..db import connect, relation_bytes, table_total_bytes
from ..ingest import fit_scaler
from ..query import hash_search as hs
from .query_cache import load_or_build


def fingerprint_skew(cfg: Config, cache) -> dict:
    with connect(cfg) as conn:
        table_wide = hs.hash_fanout_stats(conn)
        # Pool the hashes of a sample of real queries.
        items = list(cache.items.values())[:40]
        pooled = np.concatenate([i.hashes for i in items]) if items else np.zeros(0, np.int64)
        query_side = hs.query_weighted_fanout(conn, pooled)
    return {"table_wide": table_wide, "query_weighted": query_side}


def storage(cfg: Config) -> dict:
    out = {}
    with connect(cfg) as conn:
        for rel in ("tracks", "embeddings", "fingerprints", "audio_staging",
                    "embeddings_s1000", "embeddings_s5000", "embeddings_s8000"):
            heap = relation_bytes(conn, rel)
            if heap == 0 and table_total_bytes(conn, rel) == 0:
                continue
            out[rel] = {"heap_bytes": heap,
                        "total_bytes": table_total_bytes(conn, rel)}
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.relname, i.relname, am.amname, pg_relation_size(i.oid)
                FROM pg_index x
                JOIN pg_class i ON i.oid = x.indexrelid
                JOIN pg_class t ON t.oid = x.indrelid
                JOIN pg_am   am ON am.oid = i.relam
                WHERE t.relname LIKE 'embeddings%' OR t.relname = 'fingerprints'
                ORDER BY t.relname, i.relname
                """
            )
            out["indexes"] = [
                {"table": t, "index": n, "method": m, "bytes": int(b)}
                for t, n, m, b in cur.fetchall()
            ]
    return out


def standardisation_ablation(cfg: Config, cache, scale: int = 8000,
                             n_queries: int = 400, k: int = 20) -> dict:
    """Search with the fitted transform vs. L2 normalisation of the raw vector.

    The cached query vectors are already standardised, so the comparison is made by
    *inverting* the stored transform to recover each raw vector, then re-normalising it
    the naive way. The reference side of the comparison needs a matching index, which
    is built on a temporary copy of the table holding naively normalised vectors.

    This isolates one decision -- per-dimension rescaling -- while holding the features,
    the corpus, the index type and the queries fixed.
    """
    from ..indexing import build_index
    from ..query import embedding_search as es

    scaler = fit_scaler.load(cfg)
    items = list(cache.items.values())[:n_queries]
    table = "embeddings_ablation_raw"

    results: dict = {}
    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(
                f"""
                CREATE TABLE {table} (
                    embedding_id  BIGINT PRIMARY KEY,
                    track_id      BIGINT NOT NULL,
                    segment_start REAL   NOT NULL,
                    embedding     vector({cfg.embedding.dim}) NOT NULL
                )
                """
            )
        conn.commit()

        # Recover the raw vectors, re-normalise without rescaling, and load them.
        # Read and write are done in separate phases rather than streamed: mixing a
        # server-side cursor's fetches with another cursor's COPY on the same
        # connection is fragile, and at ~90 k x 159 float32 the whole set is well
        # under 100 MB.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.embedding_id, e.track_id, e.segment_start, e.embedding
                FROM embeddings e JOIN tracks t ON t.track_id = e.track_id
                WHERE t.ingest_order <= %s
                """,
                (scale,),
            )
            source = cur.fetchall()

        rows = []
        for emb_id, track_id, start, vec in source:
            # Depending on how the vector type is registered, psycopg hands back either
            # a numpy array or a pgvector Vector wrapper; normalise both.
            arr = vec.to_numpy() if hasattr(vec, "to_numpy") else np.asarray(vec)
            raw = arr.astype(np.float64) * scaler.std + scaler.mean
            n = float(np.linalg.norm(raw)) or 1.0
            rows.append((emb_id, track_id, float(start),
                         (raw / n).astype(np.float32)))
        del source

        with conn.cursor() as ins:
            for i in range(0, len(rows), 20000):
                _copy_ablation(ins, table, rows[i:i + 20000])
            ins.execute(f"ANALYZE {table}")
        conn.commit()
        del rows

        for label, tbl, transform in (
            ("standardised", f"embeddings_s{scale}", None),
            ("raw_l2_only", table, scaler),
        ):
            build_index(conn, tbl, "hnsw", m=16, ef_construction=64)
            with conn.cursor() as cur:
                es.apply_search_params(cur, "hnsw", ef_search=40)
            hits = 0
            top5 = 0
            for item in items:
                vecs = item.vectors
                if transform is not None:
                    raw = vecs.astype(np.float64) * transform.std + transform.mean
                    norms = np.linalg.norm(raw, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    vecs = (raw / norms).astype(np.float32)
                res = es.search_vectors(conn, vecs, k=k, table=tbl)
                rank = next((i for i, t in enumerate(res.tracks, 1)
                             if t.track_id == item.track_id), None)
                if rank == 1:
                    hits += 1
                if rank is not None and rank <= 5:
                    top5 += 1
            results[label] = {"n": len(items),
                              "top1": hits / len(items) if items else float("nan"),
                              "top5": top5 / len(items) if items else float("nan")}

        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
    return results


def _copy_ablation(cur, table: str, rows) -> None:
    with cur.copy(f"COPY {table} (embedding_id, track_id, segment_start, embedding) "
                  f"FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(["int8", "int8", "float4", "vector"])
        for r in rows:
            copy.write_row(r)


def parallel_overhead(cfg: Config, cache, scales=(1000, 5000, 8000),
                      n_queries: int = 60) -> dict:
    """Is the exact scan's latency the scan, or the cost of starting parallel workers?

    The sweep's exact-scan latencies look badly non-linear: 5x the data between the
    1 000- and 5 000-track tables costs 20x the time, and then 1.6x more data costs
    almost nothing. Neither matches a sequential scan.

    The explanation is a planner threshold. ``min_parallel_table_scan_size`` defaults
    to 8 MB; the 1 000-track table's heap is 7.4 MB and stays serial, while the larger
    two cross the threshold and get a Parallel Seq Scan. On Windows there is no
    ``fork()``, so every parallel worker is a fresh process that must be created and
    re-attached to shared memory -- a fixed cost of roughly 20 ms per query here, which
    is far more than the scan it is meant to accelerate.

    This re-measures each table with ``max_parallel_workers_per_gather`` at its normal
    value and at 0, warming every table first and interleaving the queries so ordering
    cannot bias one table against another. If the diagnosis is right, the serial plan
    is *faster*, and the serial per-vector cost is constant across scales.
    """
    import statistics

    from ..query import embedding_search as es

    vectors = [it.vectors[0] for it in list(cache.items.values())[:n_queries]]
    counts: dict[int, int] = {}
    results: dict[int, dict[int, float]] = {}

    for workers in (None, 0):
        with connect(cfg) as conn:
            with conn.cursor() as cur:
                es.apply_search_params(cur, "exact")
                if workers is not None:
                    cur.execute("SELECT set_config("
                                "'max_parallel_workers_per_gather', %s, false)",
                                (str(workers),))
                for s in scales:
                    cur.execute(f"SELECT count(*) FROM embeddings_s{s}")
                    counts[s] = int(cur.fetchone()[0])
            for v in vectors[:10]:                       # warm-up, discarded
                for s in scales:
                    es.search_vectors(conn, v.reshape(1, -1), k=20,
                                      table=f"embeddings_s{s}")
            lat: dict[int, list[float]] = {s: [] for s in scales}
            for v in vectors:
                for s in scales:                         # interleaved
                    r = es.search_vectors(conn, v.reshape(1, -1), k=20,
                                          table=f"embeddings_s{s}")
                    lat[s].append(r.search_ms)
            key = -1 if workers is None else workers
            results[key] = {s: statistics.median(lat[s]) for s in scales}

    out = {}
    for s in scales:
        par, ser = results[-1][s], results[0][s]
        out[str(s)] = {
            "vectors": counts[s],
            "parallel_p50_ms": par,
            "serial_p50_ms": ser,
            "serial_speedup": par / ser if ser else float("nan"),
            "serial_ns_per_vector": ser * 1e6 / counts[s] if counts[s] else float("nan"),
        }
    return out


def extraction_cost(cfg: Config, cache) -> dict:
    e = np.array([c.embed_extract_ms for c in cache.items.values()])
    h = np.array([c.hash_extract_ms for c in cache.items.values()])
    return {
        "embedding_ms": {"p50": float(np.percentile(e, 50)),
                         "p95": float(np.percentile(e, 95)),
                         "mean": float(e.mean())},
        "landmark_ms": {"p50": float(np.percentile(h, 50)),
                        "p95": float(np.percentile(h, 95)),
                        "mean": float(h.mean())},
        "segments_per_query": float(np.mean([c.vectors.shape[0]
                                             for c in cache.items.values()])),
        "hashes_per_query": float(np.mean([c.hashes.size
                                           for c in cache.items.values()])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaler", default=fit_scaler.DEFAULT_MODEL_NAME)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-queries", type=int, default=400)
    args = parser.parse_args(argv)

    cfg = load_config()
    cache = load_or_build(cfg, args.scaler)
    out: dict = {}

    print("fingerprint hash skew ...")
    out["fingerprint_skew"] = fingerprint_skew(cfg, cache)
    tw = out["fingerprint_skew"]["table_wide"]
    qw = out["fingerprint_skew"]["query_weighted"]
    print(f"  table-wide : {tw['distinct_hashes']:,} distinct hashes over "
          f"{tw['total_rows']:,} rows")
    print(f"               mean {tw['mean_rows_per_hash']:.1f} rows/hash, "
          f"p50 {tw['p50']:.0f}, p99 {tw['p99']:.0f}, max {tw['max_rows']:,}")
    print(f"  query-side : mean {qw['mean_rows_per_hash']:.1f} rows/hash, "
          f"p50 {qw['p50']:.0f}, p95 {qw['p95']:.0f}, max {qw['max_rows']:,}")

    print("storage ...")
    out["storage"] = storage(cfg)
    for rel, v in out["storage"].items():
        if rel == "indexes":
            continue
        print(f"  {rel:<22} heap {v['heap_bytes']/1024/1024:9.1f} MB  "
              f"total {v['total_bytes']/1024/1024:9.1f} MB")
    for ix in out["storage"].get("indexes", []):
        print(f"  {ix['index']:<40} {ix['method']:<9} "
              f"{ix['bytes']/1024/1024:8.1f} MB")

    print("parallel-plan overhead ...")
    out["parallel_overhead"] = parallel_overhead(cfg, cache)
    print(f"  {'scale':>6} {'vectors':>9} {'parallel ms':>12} {'serial ms':>10} "
          f"{'serial speedup':>15} {'ns/vector':>11}")
    for s, v in out["parallel_overhead"].items():
        print(f"  {s:>6} {v['vectors']:>9,} {v['parallel_p50_ms']:>12.2f} "
              f"{v['serial_p50_ms']:>10.2f} {v['serial_speedup']:>14.2f}x "
              f"{v['serial_ns_per_vector']:>11.1f}")

    print("extraction cost ...")
    out["extraction"] = extraction_cost(cfg, cache)
    print(f"  Path A  p50 {out['extraction']['embedding_ms']['p50']:.1f} ms  "
          f"p95 {out['extraction']['embedding_ms']['p95']:.1f} ms")
    print(f"  Path B  p50 {out['extraction']['landmark_ms']['p50']:.1f} ms  "
          f"p95 {out['extraction']['landmark_ms']['p95']:.1f} ms")

    if not args.skip_ablation:
        print("standardisation ablation (this rebuilds an index, ~1 min) ...")
        out["standardisation_ablation"] = standardisation_ablation(
            cfg, cache, n_queries=args.ablation_queries)
        for label, v in out["standardisation_ablation"].items():
            print(f"  {label:<14} n={v['n']}  top1={v['top1']:.3f}  "
                  f"top5={v['top5']:.3f}")

    path = cfg.paths.results / "diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
