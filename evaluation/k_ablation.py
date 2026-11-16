"""How much does the retrieval depth ``k`` matter, and are two runs really different?

**Origin of this module.** The first version of the ``run_summary`` view computed
accuracy as ``avg((hit_rank = 1)::int)``. Because ``hit_rank`` is NULL when the true
track was not retrieved at all, and ``avg()`` skips NULL rows, the denominator silently
became "queries where the track was found somewhere" rather than "queries" -- which
inflates every run, and inflates the *worst* runs most, since a configuration that
misses outright has those queries dropped from its own denominator. Under that broken
view, HNSW at ``m=32, ef_search=10`` appeared to beat the exact scan while retrieving
under half the true neighbours, and accuracy appeared non-monotonic in ``ef_search``.

The McNemar test below is what exposed it: it works on the raw ``query_results`` rows
rather than the view, and reported that exact was right-and-HNSW-wrong on 326 queries
against 87 the other way -- flatly contradicting the view. With the view corrected
(``CASE WHEN hit_rank = 1 THEN 1 ELSE 0 END``), accuracy is monotonic in recall, as it
should be, and the exact scan is the ceiling for Path A.

Two things are measured here, both still worth having:

1. **A paired comparison** of configurations over the identical 2 800 queries. McNemar
   looks only at discordant pairs -- queries one method got right and the other got
   wrong -- which is the correct test when the same queries are answered by both.
   Treating the two accuracies as independent proportions would overstate the
   uncertainty.

2. **A sweep over the retrieval depth ``k``** on an exact scan. Each query segment
   contributes its ``k`` nearest indexed segments and ``sumsim`` adds
   ``max(0, 1 - dist)`` over all of them; neighbours ranked 10-20 are largely *other*
   tracks that sit nearby in MFCC space, so a larger ``k`` may add more distractor
   score than signal. This isolates the aggregation's sensitivity to ``k`` from
   anything to do with the index.

The sweep costs only one exact run: the search executes once at the largest ``k`` and
the ranked per-segment hit lists are truncated to each smaller ``k`` and re-aggregated.
Truncating a ranked list is exactly equivalent to having searched at that ``k``, so no
approximation is introduced.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from tqdm import tqdm

from ..config import Config, load_config
from ..db import connect
from ..indexing import build_index, create_scale_table
from ..ingest import fit_scaler
from ..query import embedding_search as es
from .query_cache import load_or_build

K_GRID = (1, 2, 3, 5, 10, 20)


def _rank_of(track_id: int, ranked) -> int | None:
    for i, hit in enumerate(ranked, start=1):
        if hit.track_id == track_id:
            return i
    return None


def k_sweep(cfg: Config, cache, scale: int = 8000,
            how: str = "sumsim") -> dict:
    """One exact run at k=20, scored at every k in K_GRID by truncation."""
    queries = list(cache.items.values())
    table = f"embeddings_s{scale}"

    correct = {k: 0 for k in K_GRID}
    top5 = {k: 0 for k in K_GRID}
    n = 0

    with connect(cfg) as conn:
        create_scale_table(conn, scale, cfg.embedding.dim)
        build_index(conn, table, "exact")          # drops any vector index
        with conn.cursor() as cur:
            es.apply_search_params(cur, "exact")

        for item in tqdm(queries, desc=f"exact k-sweep s{scale}", unit="q"):
            if item.vectors.shape[0] == 0:
                continue
            res = es.search_vectors(conn, item.vectors, k=max(K_GRID), table=table)
            n += 1
            for k in K_GRID:
                truncated = [hits[:k] for hits in res.segment_hits]
                ranked = es.aggregate(truncated, how)
                r = _rank_of(item.track_id, ranked)
                if r == 1:
                    correct[k] += 1
                if r is not None and r <= 5:
                    top5[k] += 1

        with conn.cursor() as cur:
            cur.execute("RESET enable_indexscan")
            cur.execute("RESET enable_bitmapscan")
        conn.commit()

    return {
        "n": n,
        "aggregation": how,
        "top1": {str(k): correct[k] / n for k in K_GRID},
        "top5": {str(k): top5[k] / n for k in K_GRID},
    }


def mcnemar(cfg: Config, run_a: int, run_b: int) -> dict:
    """Paired comparison of two runs over the queries both answered.

    McNemar's test looks only at the discordant pairs -- queries one method got right
    and the other got wrong. Comparing the two accuracies as independent proportions
    would ignore that both methods saw the identical query set, and would therefore
    overstate the uncertainty.
    """
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT sum(CASE WHEN a.hit_rank = 1 AND b.hit_rank IS DISTINCT FROM 1
                            THEN 1 ELSE 0 END) AS a_only,
                   sum(CASE WHEN b.hit_rank = 1 AND a.hit_rank IS DISTINCT FROM 1
                            THEN 1 ELSE 0 END) AS b_only,
                   sum(CASE WHEN a.hit_rank = 1 AND b.hit_rank = 1
                            THEN 1 ELSE 0 END) AS both,
                   count(*) AS n
            FROM query_results a
            JOIN query_results b USING (query_id)
            WHERE a.run_id = %s AND b.run_id = %s
            """,
            (run_a, run_b),
        )
        a_only, b_only, both, n = cur.fetchone()

    a_only, b_only = int(a_only or 0), int(b_only or 0)
    discordant = a_only + b_only
    if discordant == 0:
        return {"run_a": run_a, "run_b": run_b, "n": int(n),
                "a_only": 0, "b_only": 0, "chi2": 0.0, "p_value": 1.0}

    # Continuity-corrected chi-square with 1 degree of freedom.
    chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, 1))
    return {
        "run_a": run_a, "run_b": run_b, "n": int(n), "both_correct": int(both),
        "a_only": a_only, "b_only": b_only,
        "chi2": float(chi2), "p_value": p,
    }


def best_runs(cfg: Config, scale: int) -> dict:
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT run_id, index_type, params, top1, p50_ms
            FROM run_summary
            WHERE kind = 'embedding' AND scale_tracks = %s
            ORDER BY top1 DESC
            """,
            (scale,),
        )
        rows = cur.fetchall()
    out = {"ranked": [{"run_id": r[0], "index_type": r[1],
                       "params": r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                       "top1": float(r[3]), "p50_ms": float(r[4])} for r in rows]}
    exact = next((r for r in out["ranked"] if r["index_type"] == "exact"), None)
    best = out["ranked"][0] if out["ranked"] else None
    out["exact"] = exact
    out["best"] = best
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=8000)
    parser.add_argument("--scaler", default=fit_scaler.DEFAULT_MODEL_NAME)
    parser.add_argument("--skip-k-sweep", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    cache = load_or_build(cfg, args.scaler)
    out: dict = {"scale": args.scale}

    ranking = best_runs(cfg, args.scale)
    out["ranking"] = ranking
    exact = ranking["exact"]
    if exact:
        print(f"exact : run {exact['run_id']}  top1 {exact['top1']:.4f}  "
              f"p50 {exact['p50_ms']:.2f} ms")

        # Compare the exact ceiling against the strongest approximate configuration,
        # and against the fastest one staying within 5 % relative accuracy of it.
        approx = [r for r in ranking["ranked"] if r["index_type"] != "exact"]
        candidates: dict[str, dict] = {}
        if approx:
            candidates["best_approx"] = max(approx, key=lambda r: r["top1"])
            within = [r for r in approx if r["top1"] >= 0.95 * exact["top1"]]
            if within:
                candidates["fastest_within_5pct"] = min(within,
                                                        key=lambda r: r["p50_ms"])

        out["mcnemar"] = {}
        for label, r in candidates.items():
            knobs = {k: v for k, v in r["params"].items() if k != "k"}
            print(f"\n{label} : run {r['run_id']}  {r['index_type']} {knobs}")
            print(f"  top1 {r['top1']:.4f}  p50 {r['p50_ms']:.2f} ms  "
                  f"({exact['p50_ms'] / r['p50_ms']:.1f}x faster than exact, "
                  f"{100 * r['top1'] / exact['top1']:.1f}% of its accuracy)")
            m = mcnemar(cfg, r["run_id"], exact["run_id"])
            out["mcnemar"][label] = m
            print(f"  McNemar vs exact, n={m['n']}: approx-only {m['a_only']}, "
                  f"exact-only {m['b_only']}, chi2={m['chi2']:.2f}, "
                  f"p={m['p_value']:.4g}")

    if not args.skip_k_sweep:
        print(f"\nexact-scan k sweep at scale {args.scale} "
              f"(one pass, scored by truncation) ...")
        out["k_sweep"] = k_sweep(cfg, cache, args.scale)
        ks = out["k_sweep"]
        print(f"  n = {ks['n']}, aggregation = {ks['aggregation']}")
        print(f"  {'k':>4} {'top-1':>8} {'top-5':>8}")
        for k in K_GRID:
            print(f"  {k:>4} {ks['top1'][str(k)]:>8.4f} {ks['top5'][str(k)]:>8.4f}")

    path = cfg.paths.results / "k_ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
