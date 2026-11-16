"""Building, timing and measuring the vector indexes.

Three retrieval strategies are compared on identical data:

``exact``     no index at all; a sequential scan computing every distance. Always
              correct by definition, so it doubles as the ground truth that ANN recall
              is measured against.
``ivfflat``   the vector space is partitioned into ``lists`` Voronoi cells by k-means at
              build time; a query scans only the ``probes`` nearest cells. Cheap to
              build, but accuracy depends on the query landing in a well-chosen cell,
              and a neighbour sitting just across a cell boundary is invisible unless
              enough probes are used.
``hnsw``      a multi-layer proximity graph. Search enters at a sparse top layer, walks
              greedily towards the query, and descends, keeping a candidate list of
              size ``ef_search``. Build is far more expensive, but search cost grows
              logarithmically rather than linearly with the number of vectors.

The scale experiments need the same vectors in several corpus sizes. Rather than
re-ingesting, a physical copy of the relevant subset is materialised per scale
(``embeddings_s1000`` and so on). Indexes have to be built on real tables to be
measured at all -- ``pg_relation_size`` needs a relation -- and building on a filtered
subset also keeps IVFFlat's k-means honest, since its centroids should describe the
data actually being searched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import psycopg

from .db import relation_bytes


@dataclass
class BuildResult:
    index_name: str | None
    index_type: str
    build_ms: float
    index_bytes: int
    table_bytes: int
    n_vectors: int
    params: dict


def scale_table_name(scale: int) -> str:
    return f"embeddings_s{scale}"


def create_scale_table(conn: psycopg.Connection, scale: int, dim: int,
                       force: bool = False) -> tuple[str, int]:
    """Materialise the subset of ``embeddings`` for the first ``scale`` tracks.

    The table is created with an explicit ``vector(dim)`` column rather than with
    ``CREATE TABLE AS``: a CTAS result can end up with an unconstrained ``vector``
    column, and pgvector refuses to index a column whose dimension it cannot see
    ("column does not have dimensions"). Declaring it up front avoids that entirely.
    """
    table = scale_table_name(scale)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        exists = cur.fetchone()[0] is not None
        if exists and not force:
            cur.execute(f"SELECT count(*) FROM {table}")
            return table, int(cur.fetchone()[0])
        if exists:
            cur.execute(f"DROP TABLE {table}")

        cur.execute(
            f"""
            CREATE TABLE {table} (
                embedding_id  BIGINT PRIMARY KEY,
                track_id      BIGINT NOT NULL,
                segment_start REAL   NOT NULL,
                embedding     vector({dim}) NOT NULL
            )
            """
        )
        cur.execute(
            f"""
            INSERT INTO {table} (embedding_id, track_id, segment_start, embedding)
            SELECT e.embedding_id, e.track_id, e.segment_start, e.embedding
            FROM embeddings e
            JOIN tracks t ON t.track_id = e.track_id
            WHERE t.ingest_order <= %s
            """,
            (scale,),
        )
        n = cur.rowcount
        # ANALYZE matters here: the planner's row estimate for a brand-new table is a
        # guess, and a wrong one changes whether the ANN index is used at all.
        cur.execute(f"ANALYZE {table}")
    conn.commit()
    return table, int(n)


def drop_vector_indexes(conn: psycopg.Connection, table: str) -> None:
    """Remove any ivfflat/hnsw index on ``table`` so the next build starts clean."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.relname
            FROM pg_index x
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_class t ON t.oid = x.indrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE t.relname = %s AND am.amname IN ('ivfflat', 'hnsw')
            """,
            (table,),
        )
        for (name,) in cur.fetchall():
            cur.execute(f"DROP INDEX IF EXISTS {name}")
    conn.commit()


def build_index(conn: psycopg.Connection, table: str, index_type: str,
                *, lists: int = 100, m: int = 16, ef_construction: int = 64,
                parallel_workers: int | None = None) -> BuildResult:
    """Build one index and measure how long it took and how big it is.

    ``exact`` builds nothing; it is still reported so that the baseline appears in the
    same table as the indexes, with a zero build cost and no index bytes.
    """
    drop_vector_indexes(conn, table)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        n_vectors = int(cur.fetchone()[0])

    if index_type == "exact":
        with conn.cursor() as cur:
            table_bytes = relation_bytes(conn, table)
        return BuildResult(None, "exact", 0.0, 0, table_bytes, n_vectors, {})

    if index_type == "ivfflat":
        index_name = f"idx_{table}_ivfflat_l{lists}"
        ddl = (f"CREATE INDEX {index_name} ON {table} "
               f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})")
        params = {"lists": lists}
    elif index_type == "hnsw":
        index_name = f"idx_{table}_hnsw_m{m}_ef{ef_construction}"
        ddl = (f"CREATE INDEX {index_name} ON {table} "
               f"USING hnsw (embedding vector_cosine_ops) "
               f"WITH (m = {m}, ef_construction = {ef_construction})")
        params = {"m": m, "ef_construction": ef_construction}
    else:
        raise ValueError(f"unknown index type: {index_type}")

    with conn.cursor() as cur:
        if parallel_workers is not None:
            cur.execute("SET max_parallel_maintenance_workers = %s",
                        (parallel_workers,))
        conn.commit()
        started = time.perf_counter()
        cur.execute(ddl)
        conn.commit()
        build_ms = (time.perf_counter() - started) * 1000.0

    index_bytes = relation_bytes(conn, index_name)
    table_bytes = relation_bytes(conn, table)
    return BuildResult(index_name, index_type, build_ms, index_bytes, table_bytes,
                       n_vectors, params)


def build_fingerprint_index(conn: psycopg.Connection,
                            drop_first: bool = True) -> BuildResult:
    """(Re)build the B-tree on ``fingerprints.hash`` and time it.

    Included so the report can compare like with like: this is the index the landmark
    path depends on, and its build cost and size belong in the same table as the
    vector indexes.
    """
    name = "idx_fingerprints_hash"
    with conn.cursor() as cur:
        if drop_first:
            cur.execute(f"DROP INDEX IF EXISTS {name}")
            conn.commit()
        cur.execute("SELECT count(*) FROM fingerprints")
        n_rows = int(cur.fetchone()[0])
        started = time.perf_counter()
        cur.execute(f"CREATE INDEX {name} ON fingerprints (hash)")
        conn.commit()
        build_ms = (time.perf_counter() - started) * 1000.0
    return BuildResult(name, "btree", build_ms, relation_bytes(conn, name),
                       relation_bytes(conn, "fingerprints"), n_rows, {})


def explain(conn: psycopg.Connection, sql: str, params, analyze: bool = True) -> str:
    """EXPLAIN (ANALYZE) a query, used to show in the report which access path ran."""
    prefix = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE)" if analyze else "EXPLAIN (VERBOSE)"
    with conn.cursor() as cur:
        cur.execute(f"{prefix} {sql}", params)
        return "\n".join(row[0] for row in cur.fetchall())
