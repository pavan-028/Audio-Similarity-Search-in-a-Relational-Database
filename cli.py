"""Single entry point for the pipeline.

    set PYTHONPATH=src
    python -m audiosim.cli <command> [options]

Commands, in the order a fresh run uses them::

    initdb        apply sql/01_schema.sql
    load          walk the corpus into audio_staging         (ingest.loader)
    fit-scaler    fit and store the Path A standardisation   (ingest.fit_scaler)
    ingest        run the parallel workers                    (ingest.worker)
    finalize      attach metadata, assign splits              (ingest.finalize)
    status        where everything currently stands
    reset         drop feature rows (keeps the staged audio)
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .db import apply_sql_file, connect, relation_bytes, server_settings


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return str(n)


def cmd_initdb(args) -> int:
    cfg = load_config()
    schema = cfg.paths.project_root / "sql" / "01_schema.sql"
    print(f"applying {schema}")
    apply_sql_file(cfg, schema)
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        print("tables/views:", ", ".join(r[0] for r in cur.fetchall()))
    return 0


def cmd_status(args) -> int:
    cfg = load_config()
    with connect(cfg) as conn, conn.cursor() as cur:
        print("server:")
        for k, v in server_settings(conn, [
            "server_version", "shared_buffers", "work_mem",
            "maintenance_work_mem", "max_parallel_maintenance_workers",
        ]).items():
            print(f"  {k:<34} {v}")

        print("\nstaging:")
        cur.execute("SELECT status, count(*) FROM audio_staging GROUP BY status "
                    "ORDER BY status")
        rows = cur.fetchall()
        if not rows:
            print("  (empty)")
        for status, n in rows:
            print(f"  {status:<34} {n}")

        print("\ncontent:")
        for table in ("tracks", "embeddings", "fingerprints", "eval_queries",
                      "experiments", "query_results"):
            cur.execute(f"SELECT count(*) FROM {table}")
            print(f"  {table:<34} {cur.fetchone()[0]:>12,}")

        print("\nsizes:")
        for rel in ("audio_staging", "tracks", "embeddings", "fingerprints"):
            heap = relation_bytes(conn, rel)
            cur.execute(
                "SELECT coalesce(sum(pg_relation_size(indexrelid)), 0) "
                "FROM pg_index WHERE indrelid = to_regclass(%s)", (rel,)
            )
            idx = int(cur.fetchone()[0])
            print(f"  {rel:<22} heap {_human(heap):>10}   indexes {_human(idx):>10}")

        print("\nindexes on embeddings / fingerprints:")
        cur.execute(
            """
            SELECT i.relname, am.amname, pg_relation_size(i.oid)
            FROM pg_index x
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_class t ON t.oid = x.indrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE t.relname IN ('embeddings', 'fingerprints')
            ORDER BY t.relname, i.relname
            """
        )
        for name, am, size in cur.fetchall():
            print(f"  {name:<40} {am:<10} {_human(size)}")

        cur.execute("SELECT name, dim, n_samples FROM feature_models ORDER BY name")
        models = cur.fetchall()
        print("\nfeature models:")
        if not models:
            print("  (none -- run fit-scaler)")
        for name, dim, n in models:
            print(f"  {name:<40} dim={dim} fitted on {n} segments")
    return 0


def cmd_reset(args) -> int:
    cfg = load_config()
    if not args.yes:
        print("refusing without --yes: this deletes all extracted features")
        return 1
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE query_results, experiments, eval_queries, "
                    "embeddings, fingerprints, tracks CASCADE")
        cur.execute("UPDATE audio_staging SET status='pending', worker_id=NULL, "
                    "started_at=NULL, finished_at=NULL, error=NULL, attempts=0")
        conn.commit()
    print("feature tables truncated; staging reset to pending")
    return 0


# Commands implemented by their own module, which owns its argument parsing.
DELEGATED = {
    "load": "audiosim.ingest.loader",
    "fit-scaler": "audiosim.ingest.fit_scaler",
    "ingest": "audiosim.ingest.worker",
    "finalize": "audiosim.ingest.finalize",
    "query-cache": "audiosim.evaluation.query_cache",
    "build-queries": "audiosim.evaluation.build_queries",
    "benchmark": "audiosim.evaluation.run_experiments",
    "diagnostics": "audiosim.evaluation.diagnostics",
    "k-ablation": "audiosim.evaluation.k_ablation",
    "plots": "audiosim.evaluation.plots",
    "search": "audiosim.query.search",
}

LOCAL = {
    "initdb": cmd_initdb,
    "status": cmd_status,
    "reset": cmd_reset,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Delegation is resolved before argparse sees anything. argparse.REMAINDER looks
    # like the natural way to forward arguments to a sub-module, but it stops
    # collecting at the first token that begins with '-', so "fit-scaler --files 500"
    # would be rejected as an unrecognised argument of the top-level parser.
    if argv and argv[0] in DELEGATED:
        import importlib

        module = importlib.import_module(DELEGATED[argv[0]])
        return int(module.main(argv[1:]) or 0)

    parser = argparse.ArgumentParser(prog="audiosim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("initdb", help="apply the schema")
    sub.add_parser("status", help="show pipeline state")
    p_reset = sub.add_parser("reset", help="truncate feature tables")
    p_reset.add_argument("--yes", action="store_true")
    for name, module in DELEGATED.items():
        sub.add_parser(name, add_help=False,
                       help=f"delegates to {module} (use -h after the command)")

    args = parser.parse_args(argv)
    return int(LOCAL[args.command](args) or 0)


if __name__ == "__main__":
    sys.exit(main())
