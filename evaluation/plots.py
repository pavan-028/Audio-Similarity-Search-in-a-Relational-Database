"""Figures and tables for the report, generated from the measurement tables.

Everything here reads ``run_summary`` / ``run_by_condition``, so no figure can show a
number that is not in the database. Each figure is paired with a Markdown table in
``results/tables.md``: several of the palette's hues sit below 3:1 contrast on a light
surface, and the written-out table is the accessible counterpart to the colour channel
as well as being what actually gets quoted in the text.

Deliberately avoided: any chart with two y-axes. Where two quantities of different
scale belong together (index build time and index size, say) they are drawn as two
panels sharing an x-axis instead, which keeps both readable without inviting the
spurious crossing-point reading a twin-axis plot always suggests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import Config, load_config
from ..db import connect

# Reference categorical palette, light mode. Slots are assigned by entity and never
# cycled; the first three are the all-pairs-safe subset, which is why index type
# (exactly three values) uses them.
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_YELLOW = "#eda100"
C_VIOLET = "#4a3aa7"
C_RED = "#e34948"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# The retrieval depth the index-comparison figures are drawn at. k is a property of
# the query, not of the index, and is swept on its own in evaluation/k_ablation.py.
MAIN_K = 20

INDEX_COLOR = {"exact": C_BLUE, "ivfflat": C_ORANGE, "hnsw": C_AQUA}
INDEX_LABEL = {"exact": "Exact scan", "ivfflat": "IVFFlat", "hnsw": "HNSW"}
PATH_COLOR = {"embedding": C_BLUE, "landmark": C_ORANGE}
PATH_LABEL = {"embedding": "Path A - embeddings", "landmark": "Path B - landmarks"}


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7.0,
        "legend.frameon": False,
        "figure.dpi": 200,
    })
    return plt


def _style(ax, *, xlabel="", ylabel="", title=""):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def _plain_log_x(ax):
    """Label a log x-axis with readable numbers instead of 2 x 10^0 notation."""
    import matplotlib.ticker as mticker

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:.2g}"))
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:.2g}"))
    ax.tick_params(axis="x", which="minor", labelsize=7.5)


def fetch(cfg: Config):
    import warnings

    import pandas as pd

    warnings.filterwarnings("ignore", message=".*only supports SQLAlchemy.*")
    with connect(cfg) as conn:
        summary = pd.read_sql_query(
            "SELECT * FROM run_summary ORDER BY run_id", conn)
        by_cond = pd.read_sql_query(
            "SELECT * FROM run_by_condition ORDER BY run_id", conn)
    for frame in (summary, by_cond):
        if "params" in frame.columns:
            frame["params"] = frame["params"].apply(
                lambda v: v if isinstance(v, dict) else json.loads(v or "{}"))
    # Retrieval depth is swept separately (see evaluation/k_ablation.py), so every
    # figure pins it; mixing k values on one axis would compare different systems.
    summary["k"] = summary["params"].apply(lambda p: int(p.get("k", MAIN_K)))
    return summary, by_cond


def _param(row, key, default=None):
    return row["params"].get(key, default) if isinstance(row["params"], dict) else default


def _pareto(sub, x: str = "p50_ms", y: str = "top1"):
    """Split rows into the Pareto frontier (cheap and accurate) and the rest.

    A configuration is on the frontier when nothing else is both faster and at least
    as accurate. Walking the rows in increasing latency, a row joins the frontier only
    if it improves on the best accuracy seen so far.
    """
    rows = list(sub.sort_values(x).itertuples())
    frontier, dominated, best = [], [], -float("inf")
    for r in rows:
        value = getattr(r, y)
        if value > best:
            frontier.append(r)
            best = value
        else:
            dominated.append(r)
    return frontier, dominated


def representative(by_cond, scale: int, k: int = 20):
    """One run per retrieval path, for the robustness comparisons.

    The sweep produces a dozen HNSW configurations per scale, so filtering only on
    ``index_type`` would put twelve rows in every condition cell and make the per-
    condition tables meaningless.

    Path A is represented by its **exact** run rather than by a chosen HNSW setting.
    The robustness question is about the two feature representations, and using the
    exact scan removes index approximation from the comparison entirely -- otherwise a
    reader could not tell whether Path A degraded because MFCC statistics are fragile
    or because ``ef_search`` was set too low. Path B has only one configuration.
    """
    df = by_cond[by_cond.scale_tracks == scale]
    a = df[(df.kind == "embedding") & (df.index_type == "exact") & (df.k == k)]
    b = df[df.kind == "landmark"]
    import pandas as pd
    return pd.concat([a, b], ignore_index=True)


def _missing(v) -> bool:
    """True for a SQL NULL however pandas chose to represent it.

    A nullable text column can come back as None, NaN or pd.NA depending on the
    pandas version and the column's inferred dtype, so an `is None` test is not
    enough to recognise "clean" (no codec) or "no noise" (no SNR).
    """
    import pandas as pd

    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_accuracy_vs_latency(summary, out: Path, scale: int) -> Path | None:
    """The headline figure: what accuracy costs in latency."""
    plt = _setup_matplotlib()
    df = summary[(summary.kind == "embedding") & (summary.scale_tracks == scale)
                 & (summary.k == MAIN_K)]
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # The exact scan's accuracy as a reference line, so every approximate point can be
    # read against the ceiling it is approximating without consulting the table.
    exact_rows = df[df.index_type == "exact"]
    if not exact_rows.empty:
        ceiling = float(exact_rows.top1.iloc[0])
        ax.axhline(ceiling, color=AXIS, linewidth=1.0, linestyle="--", zorder=1)
        ax.annotate(f"exact scan accuracy = {ceiling:.3f}", (0.02, ceiling),
                    xycoords=("axes fraction", "data"), ha="left", va="bottom",
                    fontsize=7.5, color=MUTED)

    for index_type in ("exact", "ivfflat", "hnsw"):
        sub = df[df.index_type == index_type].sort_values("p50_ms")
        if sub.empty:
            continue
        colour = INDEX_COLOR[index_type]
        if index_type == "exact":
            ax.scatter(sub.p50_ms, sub.top1, s=110, color=colour, zorder=5,
                       marker="D", edgecolors=SURFACE, linewidths=1.5,
                       label=INDEX_LABEL[index_type])
            continue

        # Each index type contributes several build configurations (m = 8/16/32,
        # lists = 100/400). Connecting every point in latency order would draw a line
        # that zig-zags between different builds and suggests a trade-off curve that
        # no single index offers. The connected line is therefore the *Pareto
        # frontier* -- the configurations not beaten on both axes at once -- and the
        # dominated configurations are shown as hollow markers in the same colour.
        frontier, dominated = _pareto(sub)
        if dominated:
            ax.scatter([r.p50_ms for r in dominated], [r.top1 for r in dominated],
                       s=26, facecolors="none", edgecolors=colour, linewidths=1.2,
                       zorder=3)
        ax.plot([r.p50_ms for r in frontier], [r.top1 for r in frontier], "o-",
                color=colour, zorder=4, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=INDEX_LABEL[index_type])

        # Label only the frontier, to keep the middle of the plot legible. The search
        # knob's parameter name and its display name differ for HNSW.
        search_key, search_label = (("ef_search", "ef") if index_type == "hnsw"
                                    else ("probes", "probes"))
        build_key = "m" if index_type == "hnsw" else "lists"
        # Frontier points bunch up towards the high-accuracy end, so successive labels
        # alternate above/below the marker instead of stacking on one another.
        for i, r in enumerate(frontier):
            knob = r.params.get(search_key)
            build = r.params.get(build_key)
            if knob is None:
                continue
            offset = (6, -12) if i % 2 == 0 else (6, 7)
            ax.annotate(f"{build_key}={build}, {search_label}={knob}",
                        (r.p50_ms, r.top1), textcoords="offset points",
                        xytext=offset, fontsize=7, color=MUTED)

    ax.set_xscale("log")
    _plain_log_x(ax)
    _style(ax, xlabel="p50 database latency per query (ms, log scale)",
           ylabel="Top-1 track accuracy",
           title=f"Accuracy against latency at {scale:,} tracks")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out / "fig_accuracy_vs_latency.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_recall_vs_latency(summary, out: Path, scale: int) -> Path | None:
    plt = _setup_matplotlib()
    df = summary[(summary.kind == "embedding") & (summary.scale_tracks == scale)
                 & (summary.k == MAIN_K)
                 & summary.index_type.isin(["ivfflat", "hnsw"])]
    if df.empty or df.mean_recall.isna().all():
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index_type in ("ivfflat", "hnsw"):
        sub = df[df.index_type == index_type].sort_values("p50_ms")
        if sub.empty:
            continue
        ax.plot(sub.p50_ms, sub.mean_recall, "o-", color=INDEX_COLOR[index_type],
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=INDEX_LABEL[index_type])
    ax.axhline(1.0, color=AXIS, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate("exact scan = 1.0", (0.99, 1.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xscale("log")
    _plain_log_x(ax)
    _style(ax, xlabel="p50 database latency per query (ms, log scale)",
           ylabel="Mean recall@20 of the neighbour set",
           title=f"Neighbour-set recall against latency at {scale:,} tracks")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out / "fig_recall_vs_latency.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_ef_search_sweep(summary, out: Path, scale: int, m: int = 16) -> Path | None:
    """Accuracy and recall share a 0-1 axis, so one axis suffices."""
    plt = _setup_matplotlib()
    df = summary[(summary.kind == "embedding") & (summary.index_type == "hnsw")
                 & (summary.scale_tracks == scale)
                 & (summary.k == MAIN_K)].copy()
    if df.empty:
        return None
    df["m"] = df.apply(lambda r: _param(r, "m"), axis=1)
    df["ef"] = df.apply(lambda r: _param(r, "ef_search"), axis=1)
    df = df[df.m == m].sort_values("ef")
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(df.ef, df.top1, "o-", color=C_BLUE, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Top-1 accuracy")
    ax.plot(df.ef, df.mean_recall, "s-", color=C_ORANGE, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Recall@20")
    for _, row in df.iterrows():
        ax.annotate(f"{row.p50_ms:.1f} ms", (row.ef, min(row.top1, row.mean_recall)),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=7.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(df.ef.tolist())
    ax.set_xticklabels([str(int(v)) for v in df.ef])
    _style(ax, xlabel="hnsw.ef_search (log scale)", ylabel="Fraction",
           title=f"HNSW search effort at m={m}, {scale:,} tracks "
                 f"(p50 latency annotated)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out / "fig_ef_search_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_scale(summary, out: Path) -> Path | None:
    """Two panels sharing an x-axis rather than one chart with two y-scales."""
    plt = _setup_matplotlib()
    df = summary[(summary.kind == "embedding") & (summary.k == MAIN_K)].copy()
    if df.empty:
        return None
    df["ef"] = df.apply(lambda r: _param(r, "ef_search"), axis=1)
    df["probes"] = df.apply(lambda r: _param(r, "probes"), axis=1)
    df["m"] = df.apply(lambda r: _param(r, "m"), axis=1)
    df["lists"] = df.apply(lambda r: _param(r, "lists"), axis=1)

    # Reference settings only, so the scale trend is not confounded by the sweeps.
    ref = df[
        (df.index_type == "exact")
        | ((df.index_type == "hnsw") & (df.m == 16) & (df.ef == 40))
        | ((df.index_type == "ivfflat") & (df.lists == 100) & (df.probes == 10))
    ]
    if ref.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for index_type in ("exact", "ivfflat", "hnsw"):
        sub = ref[ref.index_type == index_type].sort_values("scale_tracks")
        if sub.empty:
            continue
        ax1.plot(sub.scale_tracks, sub.p50_ms, "o-", color=INDEX_COLOR[index_type],
                 markeredgecolor=SURFACE, markeredgewidth=1.5,
                 label=INDEX_LABEL[index_type])
        ax2.plot(sub.scale_tracks, sub.top1, "o-", color=INDEX_COLOR[index_type],
                 markeredgecolor=SURFACE, markeredgewidth=1.5,
                 label=INDEX_LABEL[index_type])
    ax1.set_yscale("log")
    import matplotlib.ticker as mticker
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.tick_params(axis="y", which="minor", labelsize=7.5)
    _style(ax1, xlabel="Tracks in the index", ylabel="p50 latency (ms, log scale)",
           title="Latency against corpus size")
    _style(ax2, xlabel="Tracks in the index", ylabel="Top-1 track accuracy",
           title="Accuracy against corpus size")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    path = out / "fig_scale.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_build_cost(summary, out: Path) -> Path | None:
    plt = _setup_matplotlib()
    df = summary[(summary.kind == "embedding") & (summary.k == MAIN_K)
                 & summary.index_type.isin(["ivfflat", "hnsw"])].copy()
    if df.empty:
        return None
    df["m"] = df.apply(lambda r: _param(r, "m"), axis=1)
    df["lists"] = df.apply(lambda r: _param(r, "lists"), axis=1)
    ref = df[((df.index_type == "hnsw") & (df.m == 16))
             | ((df.index_type == "ivfflat") & (df.lists == 100))]
    ref = ref.drop_duplicates(subset=["index_type", "scale_tracks"])
    if ref.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for index_type in ("ivfflat", "hnsw"):
        sub = ref[ref.index_type == index_type].sort_values("scale_tracks")
        if sub.empty:
            continue
        colour = INDEX_COLOR[index_type]
        ax1.plot(sub.scale_tracks, sub.build_ms / 1000.0, "o-", color=colour,
                 markeredgecolor=SURFACE, markeredgewidth=1.5,
                 label=INDEX_LABEL[index_type])
        ax2.plot(sub.scale_tracks, sub.index_bytes / 1024 / 1024, "o-", color=colour,
                 markeredgecolor=SURFACE, markeredgewidth=1.5,
                 label=INDEX_LABEL[index_type])
    _style(ax1, xlabel="Tracks indexed", ylabel="Build time (s)",
           title="Index build time")
    _style(ax2, xlabel="Tracks indexed", ylabel="Index size (MB)",
           title="Index size on disk")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    path = out / "fig_build_cost.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_robustness(by_cond, out: Path, scale: int) -> Path | None:
    """Path A vs Path B under noise, one panel per clip length."""
    plt = _setup_matplotlib()
    df = representative(by_cond, scale)
    df = df[df.codec.isna()].copy()
    if df.empty:
        return None
    lengths = sorted(df.clip_seconds.dropna().unique())
    if not lengths:
        return None

    fig, axes = plt.subplots(1, len(lengths), figsize=(3.6 * len(lengths), 4.0),
                             sharey=True)
    if len(lengths) == 1:
        axes = [axes]

    # "clean" has no SNR; place it on the axis to the right of the noisy levels.
    def snr_key(v):
        return 99.0 if _missing(v) else float(v)

    for ax, length in zip(axes, lengths):
        panel = df[df.clip_seconds == length]
        for kind in ("embedding", "landmark"):
            sub = panel[panel.kind == kind].copy()
            if sub.empty:
                continue
            sub["snr_sort"] = sub.snr_db.apply(snr_key)
            sub = sub.groupby("snr_sort", as_index=False).agg(top1=("top1", "max"))
            sub = sub.sort_values("snr_sort")
            xs = range(len(sub))
            ax.plot(list(xs), sub.top1, "o-", color=PATH_COLOR[kind],
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=PATH_LABEL[kind])
            ax.set_xticks(list(xs))
            ax.set_xticklabels(["0 dB" if v == 0 else
                                ("clean" if v == 99 else f"{int(v)} dB")
                                for v in sub.snr_sort])
        _style(ax, xlabel="Noise level (SNR)", title=f"{int(length)} s query")
    axes[0].set_ylabel("Top-1 track accuracy")
    axes[0].set_ylim(-0.03, 1.03)
    axes[-1].legend(loc="lower right")
    fig.suptitle(f"Robustness to additive noise at {scale:,} tracks",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "fig_robustness_snr.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_codec(by_cond, out: Path, scale: int) -> Path | None:
    plt = _setup_matplotlib()
    df = representative(by_cond, scale).copy()
    if df.empty:
        return None
    df["cond"] = df.apply(
        lambda r: ("clean" if _missing(r.codec) and _missing(r.snr_db)
                   else (None if _missing(r.codec) else r.codec)), axis=1)
    df = df[(df.cond.notna()) & (df.clip_seconds == 10.0)]
    if df.empty:
        return None

    order = ["clean", "mp3@64k", "mp3@32k"]
    present = [c for c in order if c in set(df.cond)]
    if not present:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    width = 0.36
    for i, kind in enumerate(("embedding", "landmark")):
        sub = df[df.kind == kind]
        vals = []
        for cond in present:
            row = sub[sub.cond == cond]
            vals.append(float(row.top1.max()) if not row.empty else np.nan)
        xs = np.arange(len(present)) + (i - 0.5) * (width + 0.02)
        bars = ax.bar(xs, vals, width=width, color=PATH_COLOR[kind],
                      label=PATH_LABEL[kind], zorder=3,
                      edgecolor=SURFACE, linewidth=2.0)
        for rect, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels(present)
    ax.set_ylim(0, 1.08)
    _style(ax, xlabel="Query condition (10 s clips)",
           ylabel="Top-1 track accuracy",
           title=f"Effect of low-bitrate re-encoding at {scale:,} tracks")
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = out / "fig_codec.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_clip_length(by_cond, out: Path, scale: int) -> Path | None:
    plt = _setup_matplotlib()
    df = representative(by_cond, scale)
    df = df[df.codec.isna() & df.snr_db.isna()].copy()
    if df.empty:
        return None

    lengths = sorted(df.clip_seconds.dropna().unique())
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    width = 0.36
    for i, kind in enumerate(("embedding", "landmark")):
        sub = df[df.kind == kind]
        vals = [float(sub[sub.clip_seconds == L].top1.max())
                if not sub[sub.clip_seconds == L].empty else np.nan
                for L in lengths]
        xs = np.arange(len(lengths)) + (i - 0.5) * (width + 0.02)
        bars = ax.bar(xs, vals, width=width, color=PATH_COLOR[kind],
                      label=PATH_LABEL[kind], zorder=3,
                      edgecolor=SURFACE, linewidth=2.0)
        for rect, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(len(lengths)))
    ax.set_xticklabels([f"{int(L)} s" for L in lengths])
    ax.set_ylim(0, 1.08)
    _style(ax, xlabel="Clean query length", ylabel="Top-1 track accuracy",
           title=f"Effect of query length at {scale:,} tracks")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out / "fig_clip_length.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _fmt(v, nd=3, dash="-"):
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return dash if not np.isfinite(f) else f"{f:.{nd}f}"


def export_tables(cfg: Config, summary, by_cond, out: Path) -> Path:
    lines = ["# Measured results",
             "",
             "Generated by `python -m audiosim.cli plots`. Every row comes from the "
             "`run_summary` / `run_by_condition` views in the `audiosim` database.",
             ""]

    scales = sorted(summary.scale_tracks.unique())
    largest = max(scales) if scales else None

    for scale in scales:
        lines += [f"## Index comparison at {scale:,} tracks", ""]
        df = summary[(summary.kind == "embedding")
                     & (summary.scale_tracks == scale)].copy()
        df["k"] = df.apply(lambda r: _param(r, "k", 20), axis=1)
        df = df.sort_values(["k", "index_type", "p50_ms"])
        rows = []
        for _, r in df.iterrows():
            knobs = []
            for key in ("lists", "probes", "m", "ef_construction", "ef_search"):
                v = _param(r, key)
                if v is not None:
                    knobs.append(f"{key}={v}")
            rows.append([
                INDEX_LABEL.get(r.index_type, r.index_type),
                str(int(r.k)),
                ", ".join(knobs) or "-",
                f"{int(r.n_queries)}",
                _fmt(r.top1), _fmt(r.top5), _fmt(r.mean_recall),
                _fmt(r.p50_ms, 2), _fmt(r.p95_ms, 2),
                "-" if _missing(r.build_ms) else f"{float(r.build_ms)/1000:.2f}",
                "-" if _missing(r.index_bytes) or not r.index_bytes
                else f"{int(r.index_bytes)/1024/1024:.1f}",
            ])
        lines.append(_md_table(
            ["Index", "k", "Parameters", "n", "Top-1", "Top-5", "Recall@k",
             "p50 ms", "p95 ms", "Build s", "Index MB"], rows))
        lines.append("")

        lm = summary[(summary.kind == "landmark")
                     & (summary.scale_tracks == scale)]
        if not lm.empty:
            rows = [[
                "Landmark B-tree",
                "-" if _missing(r.n_vectors) else f"{int(r.n_vectors):,} rows in scope",
                f"{int(r.n_queries)}",
                _fmt(r.top1), _fmt(r.top5),
                _fmt(r.p50_ms, 2), _fmt(r.p95_ms, 2),
                "-" if _missing(r.index_bytes) or not r.index_bytes
                else f"{int(r.index_bytes)/1024/1024:.1f}",
            ] for _, r in lm.iterrows()]
            lines += ["### Path B at the same scale", "",
                      _md_table(["Index", "Scope", "n", "Top-1", "Top-5",
                                 "p50 ms", "p95 ms", "Index MB"], rows), ""]

    if largest is not None:
        lines += [f"## Robustness by condition at {largest:,} tracks", "",
                  "Path A is represented by the **exact scan** (k = 20) so that index "
                  "approximation does not confound the comparison of the two feature "
                  "representations; Path B has a single configuration.", ""]
        df = representative(by_cond, largest)
        rows = []
        for _, r in df.sort_values(
                ["kind", "clip_seconds", "snr_db", "codec"]).iterrows():
            rows.append([
                PATH_LABEL.get(r.kind, r.kind),
                f"{r.clip_seconds:.0f}",
                "clean" if _missing(r.snr_db) else f"{float(r.snr_db):.0f} dB",
                "-" if _missing(r.codec) else str(r.codec),
                f"{int(r.n_queries)}",
                _fmt(r.top1), _fmt(r.top5), _fmt(r.p50_ms, 2),
            ])
        lines.append(_md_table(
            ["Path", "Clip s", "SNR", "Codec", "n", "Top-1", "Top-5", "p50 ms"],
            rows))
        lines.append("")

    path = out / "tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=None,
                        help="scale to use for the single-scale figures "
                             "(default: largest available)")
    args = parser.parse_args(argv)

    cfg = load_config()
    cfg.paths.figures.mkdir(parents=True, exist_ok=True)
    summary, by_cond = fetch(cfg)
    if summary.empty:
        print("no experiment rows -- run the benchmark first")
        return 2

    scale = args.scale or int(summary.scale_tracks.max())
    out = cfg.paths.figures
    produced = [
        fig_accuracy_vs_latency(summary, out, scale),
        fig_recall_vs_latency(summary, out, scale),
        fig_ef_search_sweep(summary, out, scale),
        fig_scale(summary, out),
        fig_build_cost(summary, out),
        fig_robustness(by_cond, out, scale),
        fig_codec(by_cond, out, scale),
        fig_clip_length(by_cond, out, scale),
    ]
    for p in produced:
        print(f"  {'wrote' if p else 'skipped'} {p if p else '(no data)'}")

    table_path = export_tables(cfg, summary, by_cond, cfg.paths.results)
    print(f"  wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

