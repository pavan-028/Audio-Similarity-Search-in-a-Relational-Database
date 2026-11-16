"""Database access helpers.

Thin wrappers around psycopg 3. Two things here are worth more than convenience:

* :func:`copy_embeddings` / :func:`copy_fingerprints` use binary ``COPY``, which is
  roughly an order of magnitude faster than row-at-a-time ``INSERT`` for the volumes
  involved (~88 k vectors, ~18 M fingerprint rows).
* :func:`assert_schema_matches` compares the declared ``vector(n)`` width against
  :class:`~audiosim.config.EmbeddingConfig`, so a config edit cannot quietly write
  wrong-width vectors into an existing table.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import Config


@contextlib.contextmanager
def connect(cfg: Config, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Open a connection with the vector type registered."""
    conn = psycopg.connect(cfg.db.conninfo(), autocommit=autocommit)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def apply_sql_file(cfg: Config, path: Path) -> None:
    """Execute a .sql file as a single transaction."""
    sql = Path(path).read_text(encoding="utf-8")
    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def vector_dim(conn: psycopg.Connection, table: str = "embeddings",
               column: str = "embedding") -> int | None:
    """Return the declared width of a pgvector column, or None if absent.

    pgvector encodes the width in ``atttypmod``; for ``vector`` the modifier *is* the
    dimension (unlike, say, ``numeric``), so no decoding is needed.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = %s AND a.attname = %s AND a.attnum > 0
            """,
            (table, column),
        )
        row = cur.fetchone()
    return None if row is None else int(row[0])


def assert_schema_matches(cfg: Config) -> None:
    with connect(cfg) as conn:
        declared = vector_dim(conn)
    if declared is None:
        raise RuntimeError("table 'embeddings' not found -- run the schema first")
    if declared != cfg.embedding.dim:
        raise RuntimeError(
            f"embeddings.embedding is vector({declared}) but EmbeddingConfig.dim is "
            f"{cfg.embedding.dim}. Either edit sql/01_schema.sql and rebuild the "
            f"table, or restore the matching feature configuration."
        )


# ---------------------------------------------------------------------------
# Bulk loading
# ---------------------------------------------------------------------------

def copy_embeddings(cur: psycopg.Cursor,
                    rows: Iterable[tuple[int, float, np.ndarray]]) -> int:
    """Binary COPY into ``embeddings``. Returns the number of rows written."""
    n = 0
    with cur.copy(
        "COPY embeddings (track_id, segment_start, embedding) "
        "FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["int8", "float4", "vector"])
        for track_id, start, vec in rows:
            copy.write_row((track_id, float(start), vec))
            n += 1
    return n


def copy_fingerprints(cur: psycopg.Cursor,
                      rows: Iterable[tuple[int, int, int]]) -> int:
    """Binary COPY into ``fingerprints``. Returns the number of rows written."""
    n = 0
    with cur.copy(
        "COPY fingerprints (hash, track_id, t_offset) FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["int8", "int8", "int4"])
        for h, track_id, t in rows:
            copy.write_row((int(h), int(track_id), int(t)))
            n += 1
    return n


# ---------------------------------------------------------------------------
# Size / timing introspection used by the benchmarks
# ---------------------------------------------------------------------------

def relation_bytes(conn: psycopg.Connection, relation: str) -> int:
    """pg_relation_size of a table or index; 0 if it does not exist."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (relation,))
        if cur.fetchone()[0] is None:
            return 0
        cur.execute("SELECT pg_relation_size(%s)", (relation,))
        return int(cur.fetchone()[0])


def table_total_bytes(conn: psycopg.Connection, relation: str) -> int:
    """pg_total_relation_size: heap + indexes + TOAST."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (relation,))
        if cur.fetchone()[0] is None:
            return 0
        cur.execute("SELECT pg_total_relation_size(%s)", (relation,))
        return int(cur.fetchone()[0])


def server_settings(conn: psycopg.Connection,
                    names: Sequence[str]) -> dict[str, str]:
    """Read current_setting() for a list of GUCs, for the record in the report."""
    out: dict[str, str] = {}
    with conn.cursor() as cur:
        for name in names:
            try:
                cur.execute("SELECT current_setting(%s)", (name,))
                out[name] = cur.fetchone()[0]
            except psycopg.Error:
                conn.rollback()
                out[name] = "<unset>"
    return out
