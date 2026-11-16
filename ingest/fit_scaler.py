"""Fit the Path A standardisation and store it in the database.

This has to happen before the workers run, because every embedding written to the
table must have been produced by the same transform. The fitted mean/std go into
``feature_models`` so that query time can load exactly the transform the index was
built with.

Fitting on a random sample rather than the whole corpus is a deliberate shortcut and a
safe one: we only need per-dimension location and scale, and a few thousand segments
estimate those to far better precision than the downstream cosine ranking can notice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from ..audio import DecodeError, decode
from ..config import Config, load_config
from ..db import connect
from ..features.embedding import Standardiser, raw_segment_features

DEFAULT_MODEL_NAME = "mfcc40_meanstd_delta_v1"


def sample_raw_features(cfg: Config, n_files: int, seed: int = 20260211,
                        max_segments_per_file: int = 4) -> np.ndarray:
    """Collect raw pooled features from a random sample of staged files.

    Capping the segments taken from any single file keeps one long or unusual track
    from dominating the estimate.
    """
    rng = np.random.default_rng(seed)
    with connect(cfg) as conn, conn.cursor() as cur:
        # TABLESAMPLE would be cheaper but is block-based and so correlates with insert
        # order; an explicit random ordering is slower here but unbiased.
        cur.execute(
            "SELECT path FROM audio_staging ORDER BY md5(path || %s) LIMIT %s",
            (str(seed), n_files),
        )
        paths = [r[0] for r in cur.fetchall()]

    if not paths:
        raise RuntimeError("audio_staging is empty -- run the loader first")

    collected: list[np.ndarray] = []
    skipped = 0
    for path in tqdm(paths, desc="fitting scaler", unit="file"):
        try:
            y = decode(path, cfg.audio.sample_rate)
        except DecodeError:
            skipped += 1
            continue
        _, raw = raw_segment_features(y, cfg.audio, cfg.embedding)
        if raw.shape[0] == 0:
            skipped += 1
            continue
        if raw.shape[0] > max_segments_per_file:
            idx = rng.choice(raw.shape[0], max_segments_per_file, replace=False)
            raw = raw[idx]
        collected.append(raw)

    if not collected:
        raise RuntimeError("no features could be extracted from the sample")
    X = np.vstack(collected)
    print(f"fitted on {X.shape[0]} segments from {len(paths) - skipped} files "
          f"({skipped} skipped)")
    return X


def store(cfg: Config, scaler: Standardiser, n_samples: int,
          name: str = DEFAULT_MODEL_NAME) -> int:
    params = {
        "audio": asdict(cfg.audio),
        "embedding": asdict(cfg.embedding),
    }
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feature_models (name, dim, params, mean, std, n_samples)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
               SET dim = EXCLUDED.dim, params = EXCLUDED.params,
                   mean = EXCLUDED.mean, std = EXCLUDED.std,
                   n_samples = EXCLUDED.n_samples, created_at = now()
            RETURNING model_id
            """,
            (name, scaler.dim, json.dumps(params),
             scaler.mean.tolist(), scaler.std.tolist(), n_samples),
        )
        model_id = cur.fetchone()[0]
        conn.commit()
    return int(model_id)


def load(cfg: Config, name: str = DEFAULT_MODEL_NAME) -> Standardiser:
    """Load a stored transform, checking it matches the current configuration."""
    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute("SELECT dim, mean, std FROM feature_models WHERE name = %s",
                    (name,))
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(
            f"feature model '{name}' not found -- run: python -m audiosim.cli fit-scaler"
        )
    dim, mean, std = row
    if dim != cfg.embedding.dim:
        raise RuntimeError(
            f"stored model '{name}' has dim {dim} but the current EmbeddingConfig "
            f"produces {cfg.embedding.dim}"
        )
    return Standardiser(mean=np.asarray(mean, dtype=np.float64),
                        std=np.asarray(std, dtype=np.float64))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=600,
                        help="number of staged files to sample (default 600)")
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args(argv)

    cfg = load_config()
    X = sample_raw_features(cfg, args.files)
    scaler = Standardiser.fit(X)
    model_id = store(cfg, scaler, n_samples=int(X.shape[0]), name=args.name)

    print(f"stored feature_models.model_id = {model_id}  name={args.name} dim={scaler.dim}")
    print(f"  mean range {scaler.mean.min():+.3f} .. {scaler.mean.max():+.3f}")
    print(f"  std  range {scaler.std.min():.4f} .. {scaler.std.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
