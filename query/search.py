"""Interactive identification: given an audio file, say which track it is.

    python -m audiosim.cli search path\\to\\clip.wav --path both

This is the user-facing form of what the benchmarks exercise, and it is also what the
FastAPI service calls. It prints both paths' answers side by side, which is the
quickest way to see the difference the report quantifies: on a clean or lightly
degraded clip the landmark path returns a single track with an unmistakable vote count,
while the embedding path returns a ranked list of plausible neighbours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..audio import decode
from ..config import Config, load_config
from ..db import connect
from ..ingest import fit_scaler
from . import embedding_search as es
from . import hash_search as hs


def track_labels(conn, track_ids: list[int]) -> dict[int, str]:
    if not track_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT track_id, coalesce(artist, '?'), coalesce(title, '?'), "
            "coalesce(genre, '?') FROM tracks WHERE track_id = ANY(%s)",
            (track_ids,),
        )
        return {int(t): f"{a} - {ti} [{g}]" for t, a, ti, g in cur.fetchall()}


def identify(cfg: Config, path: Path, *, which: str = "both", k: int = 20,
             index_type: str = "hnsw", ef_search: int = 40, top: int = 5,
             scaler_name: str = fit_scaler.DEFAULT_MODEL_NAME) -> dict:
    y = decode(path, cfg.audio.sample_rate)
    out: dict = {"duration": y.size / cfg.audio.sample_rate}

    with connect(cfg) as conn:
        if which in ("a", "both"):
            scaler = fit_scaler.load(cfg, scaler_name)
            res = es.search_clip(conn, cfg, y, scaler, k=k, index_type=index_type,
                                 ef_search=ef_search)
            labels = track_labels(conn, [t.track_id for t in res.tracks[:top]])
            out["embedding"] = {
                "search_ms": res.search_ms,
                "extract_ms": res.extract_ms,
                "n_query_segments": res.n_query_segments,
                "tracks": [
                    {"track_id": t.track_id, "label": labels.get(t.track_id, "?"),
                     "score": round(t.score, 4), "votes": t.votes,
                     "min_distance": round(t.min_distance, 4)}
                    for t in res.tracks[:top]
                ],
            }

        if which in ("b", "both"):
            res_b = hs.search_clip(conn, cfg, y)
            labels = track_labels(conn, [t.track_id for t in res_b.tracks[:top]])
            out["landmark"] = {
                "search_ms": res_b.search_ms,
                "extract_ms": res_b.extract_ms,
                "n_query_hashes": res_b.n_query_hashes,
                "tracks": [
                    {"track_id": t.track_id, "label": labels.get(t.track_id, "?"),
                     "votes": t.votes,
                     "offset_seconds": round(
                         t.delta * cfg.landmark.hop_length / cfg.audio.sample_rate, 2)}
                    for t in res_b.tracks[:top]
                ],
            }
    return out


def _print(result: dict) -> None:
    print(f"clip duration: {result['duration']:.2f}s")
    if "embedding" in result:
        e = result["embedding"]
        print(f"\nPath A - embeddings  "
              f"(extract {e['extract_ms']:.1f} ms, search {e['search_ms']:.1f} ms, "
              f"{e['n_query_segments']} query segments)")
        for i, t in enumerate(e["tracks"], 1):
            print(f"  {i}. [{t['track_id']:>6}] score={t['score']:.3f} "
                  f"votes={t['votes']} dmin={t['min_distance']:.3f}  {t['label']}")
    if "landmark" in result:
        b = result["landmark"]
        print(f"\nPath B - landmarks   "
              f"(extract {b['extract_ms']:.1f} ms, search {b['search_ms']:.1f} ms, "
              f"{b['n_query_hashes']} query hashes)")
        for i, t in enumerate(b["tracks"], 1):
            print(f"  {i}. [{t['track_id']:>6}] votes={t['votes']:<5} "
                  f"offset={t['offset_seconds']:>7.2f}s  {t['label']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--path", dest="which", choices=("a", "b", "both"),
                        default="both")
    parser.add_argument("-k", type=int, default=20)
    parser.add_argument("--index", default="hnsw",
                        choices=("hnsw", "ivfflat", "exact"))
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)

    cfg = load_config()
    if not args.audio.exists():
        print(f"not found: {args.audio}")
        return 2
    _print(identify(cfg, args.audio, which=args.which, k=args.k,
                    index_type=args.index, ef_search=args.ef_search, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
