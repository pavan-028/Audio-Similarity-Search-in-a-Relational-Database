"""A small FastAPI service around the retrieval pipeline.

    set PYTHONPATH=src
    python -m uvicorn audiosim.api.main:app --port 8000

``GET  /``        minimal upload page
``POST /query``   multipart audio upload -> ranked matches from both paths
``GET  /health``  corpus and index state

Its purpose is to show that the whole system is an ordinary database application: the
service holds no index and caches no vectors, it decodes the upload, extracts features
and issues SQL. Scaling it out means adding processes, because all the state that
matters lives in PostgreSQL.

The connection is opened per request rather than pooled. For a demonstration endpoint
that is the right trade -- it keeps the request path obvious, and the measurements in
the report are taken from the benchmark harness, not from here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ..audio import DecodeError
from ..config import load_config
from ..db import connect
from ..query.search import identify

app = FastAPI(title="Audio similarity search",
              description="MFCC embeddings in pgvector, plus landmark fingerprints.",
              version="1.0.0")

INDEX_PAGE = Path(__file__).with_name("static") / "index.html"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/health")
def health() -> JSONResponse:
    cfg = load_config()
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM tracks")
        tracks = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM embeddings")
        embeddings = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM fingerprints")
        fingerprints = cur.fetchone()[0]
        cur.execute(
            """
            SELECT i.relname, am.amname
            FROM pg_index x
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_class t ON t.oid = x.indrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE t.relname = 'embeddings' AND am.amname IN ('hnsw', 'ivfflat')
            """
        )
        vector_indexes = [{"name": n, "method": a} for n, a in cur.fetchall()]
    return JSONResponse({
        "tracks": tracks,
        "embedding_segments": embeddings,
        "fingerprint_rows": fingerprints,
        "vector_indexes": vector_indexes,
        "embedding_dim": cfg.embedding.dim,
        "sample_rate": cfg.audio.sample_rate,
    })


@app.post("/query")
async def query(file: UploadFile = File(...), top: int = 5,
                ef_search: int = 40, which: str = "both") -> JSONResponse:
    if which not in ("a", "b", "both"):
        raise HTTPException(400, "which must be one of: a, b, both")

    payload = await file.read()
    if not payload:
        raise HTTPException(400, "empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")

    cfg = load_config()
    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    # libsndfile and ffmpeg both want a real file; the temp file is removed on exit.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(payload)
        try:
            result = identify(cfg, path, which=which, top=top, ef_search=ef_search)
        except DecodeError as exc:
            raise HTTPException(415, f"could not decode audio: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    result["filename"] = file.filename
    return JSONResponse(result)
