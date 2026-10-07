"""Inline-SVG charts for a cycle: Elo history, review-score heatmap, lineage, similarity.

Pure Python (no plotting dependency) so the same markup works in the Gradio
results panel and in the standalone HTML report. Text uses currentColor and
grid lines are translucent, so charts read in both light and dark themes.
"""

from __future__ import annotations

import html
import math
from typing import Dict, List, Optional

from .models import REVIEW_SCORE_KEYS

PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7", "#84cc16", "#ec4899"]
GRID = "rgba(127,127,127,0.28)"
MUTED = "rgba(127,127,127,0.55)"
OPERATOR_COLORS = {"REFINE": "#10b981", "MUTATE": "#f59e0b", "HYBRIDIZE": "#a855f7", "SIMPLIFY": "#06b6d4"}
SCORE_LABELS = {
    "scientific_soundness": "Sound",
    "novelty": "Novel",
    "relevance": "Relev.",
    "feasibility": "Feas.",
    "testability": "Test.",
    "clarity": "Clarity",
    "potential_impact": "Impact",
}
MAX_LABELLED_LINES = 6
MAX_HEATMAP_ROWS = 12


def _e(value) -> str:
    return html.escape(str(value))


def _svg(width: int, height: int, body: str, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;font-family:inherit;'
        f'font-size:12px;color:inherit" role="img" aria-label="{_e(label)}" xmlns="http://www.w3.org/2000/svg">'
        f"{body}</svg>"
    )


def _figure(title: str, caption: str, svg: str) -> str:
    return (
        '<figure style="margin:18px 0;padding:14px 16px;border:1px solid rgba(127,127,127,0.3);border-radius:12px">'
        f'<figcaption style="margin-bottom:8px"><strong>{_e(title)}</strong>'
        f'<div style="opacity:0.7;font-size:0.9em">{_e(caption)}</div></figcaption>{svg}</figure>'
    )


def _elo_color(elo: float, low: float, high: float) -> str:
    """Grey (lowest Elo in the pool) to green (highest)."""
    t = 0.5 if high <= low else (elo - low) / (high - low)
    r, g, b = (round(148 + (16 - 148) * t), round(163 + (185 - 163) * t), round(184 + (129 - 184) * t))
    return f"rgb({r},{g},{b})"


def _elo_range(hypotheses: List[Dict]) -> tuple:
    elos = [float(h.get("elo_score") or 1200) for h in hypotheses] or [1200.0]
    return min(elos), max(elos)


def _created_cycle(h: Dict) -> int:
    history = h.get("elo_history") or []
    return int(history[0][0]) if history else 1


# --- 1. Elo history ---


def elo_history_svg(hypotheses: List[Dict]) -> Optional[str]:
    series = [(h, [(int(c), float(e)) for c, e in (h.get("elo_history") or [])]) for h in hypotheses]
    series = [(h, pts) for h, pts in series if pts]
    if not series:
        return None
    width, height, left, right, top, bottom = 760, 300, 52, 120, 16, 36
    cycles = sorted({c for _, pts in series for c, _ in pts})
    elos = [e for _, pts in series for _, e in pts]
    lo, hi = min(elos), max(elos)
    pad = max(10.0, (hi - lo) * 0.1)
    lo, hi = lo - pad, hi + pad
    c0, c1 = cycles[0], cycles[-1]

    def x(c):
        return left + (width - left - right) * (0.5 if c1 == c0 else (c - c0) / (c1 - c0))

    def y(e):
        return top + (height - top - bottom) * (1 - (e - lo) / (hi - lo))

    parts = []
    for i in range(5):
        value = lo + (hi - lo) * i / 4
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y(value) + 4:.1f}" text-anchor="end" fill="currentColor">{value:.0f}</text>'
        )
    for c in cycles:
        parts.append(
            f'<text x="{x(c):.1f}" y="{height - 12}" text-anchor="middle" fill="currentColor">cycle {c}</text>'
        )

    ranked = sorted(series, key=lambda s: s[1][-1][1], reverse=True)
    label_slots: List[float] = []
    # Draw lowest-ranked first so the labelled top lines sit on top.
    for rank in range(len(ranked) - 1, -1, -1):
        h, pts = ranked[rank]
        top_line = rank < MAX_LABELLED_LINES
        color = PALETTE[rank % len(PALETTE)] if top_line else MUTED
        coords = " ".join(f"{x(c):.1f},{y(e):.1f}" for c, e in pts)
        stroke = 2.5 if top_line else 1.2
        if len(pts) > 1:
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{stroke}"/>')
        for c, e in pts:
            parts.append(f'<circle cx="{x(c):.1f}" cy="{y(e):.1f}" r="{3.5 if top_line else 2.5}" fill="{color}"/>')
        if top_line:
            ly = y(pts[-1][1]) + 4
            while any(abs(ly - s) < 13 for s in label_slots):
                ly += 13
            label_slots.append(ly)
            parts.append(
                f'<text x="{x(pts[-1][0]) + 8:.1f}" y="{ly:.1f}" fill="{color}" font-weight="600">'
                f"{_e(h.get('id'))} {pts[-1][1]:.0f}</text>"
            )
    return _svg(width, height, "".join(parts), "Elo rating per cycle")


# --- 2. Review-score heatmap ---


def score_heatmap_svg(hypotheses: List[Dict]) -> Optional[str]:
    rows = [h for h in hypotheses if any((h.get("review_scores") or {}).values())]
    if not rows:
        return None
    rows = sorted(rows, key=lambda h: h.get("elo_score") or 0, reverse=True)[:MAX_HEATMAP_ROWS]
    cell_w, cell_h, left, top = 70, 26, 230, 28
    width = left + cell_w * len(REVIEW_SCORE_KEYS) + 10
    height = top + cell_h * len(rows) + 8
    parts = []
    for j, key in enumerate(REVIEW_SCORE_KEYS):
        cx = left + cell_w * j + cell_w / 2
        parts.append(f'<text x="{cx:.0f}" y="18" text-anchor="middle" fill="currentColor">{SCORE_LABELS[key]}</text>')
    for i, h in enumerate(rows):
        ry = top + cell_h * i
        title = str(h.get("title") or "")
        short = title if len(title) <= 28 else title[:27] + "…"
        parts.append(
            f'<text x="{left - 8}" y="{ry + 17}" text-anchor="end" fill="currentColor">'
            f"<title>{_e(title)}</title>{_e(h.get('id'))} · {_e(short)}</text>"
        )
        scores = h.get("review_scores") or {}
        for j, key in enumerate(REVIEW_SCORE_KEYS):
            score = scores.get(key) or 0
            fill = f"rgba(99,102,241,{0.12 + 0.17 * score:.2f})" if score else "rgba(127,127,127,0.08)"
            text_color = "#fff" if score >= 4 else "currentColor"
            rx = left + cell_w * j
            parts.append(
                f'<rect x="{rx + 2}" y="{ry + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" fill="{fill}"/>'
            )
            parts.append(
                f'<text x="{rx + cell_w / 2:.0f}" y="{ry + 17}" text-anchor="middle" fill="{text_color}">'
                f"{score or '–'}</text>"
            )
    return _svg(width, height, "".join(parts), "Review scores per hypothesis")


# --- 3. Lineage ---


def lineage_svg(hypotheses: List[Dict]) -> Optional[str]:
    if not hypotheses:
        return None
    by_id = {h.get("id"): h for h in hypotheses}
    if not any(p in by_id for h in hypotheses for p in h.get("parent_ids") or []):
        return None
    lo, hi = _elo_range(hypotheses)
    columns: Dict[int, List[Dict]] = {}
    for h in hypotheses:
        columns.setdefault(_created_cycle(h), []).append(h)
    for col in columns.values():
        col.sort(key=lambda h: h.get("elo_score") or 0, reverse=True)
    cycles = sorted(columns)
    col_w, row_h, top, left = 190, 48, 34, 70
    width = left + col_w * (len(cycles) - 1) + 120
    height = top + row_h * max(len(c) for c in columns.values()) + 16
    pos = {}
    for ci, cycle in enumerate(cycles):
        for ri, h in enumerate(columns[cycle]):
            pos[h.get("id")] = (left + col_w * ci, top + row_h * ri + 18)

    parts = [
        f'<text x="{left + col_w * ci}" y="16" text-anchor="middle" fill="currentColor" font-weight="600">'
        f"cycle {cycle}</text>"
        for ci, cycle in enumerate(cycles)
    ]
    for h in hypotheses:
        child = pos[h.get("id")]
        color = OPERATOR_COLORS.get(str(h.get("evolution_operator") or "").upper(), MUTED)
        for parent_id in h.get("parent_ids") or []:
            if parent_id not in pos:
                continue
            px, py = pos[parent_id]
            mid = (px + child[0]) / 2
            parts.append(
                f'<path d="M{px + 14},{py} C{mid},{py} {mid},{child[1]} {child[0] - 14},{child[1]}" '
                f'fill="none" stroke="{color}" stroke-width="2" opacity="0.85"/>'
            )
    for h in hypotheses:
        hx, hy = pos[h.get("id")]
        elo = float(h.get("elo_score") or 1200)
        op = str(h.get("evolution_operator") or "")
        tip = f"{h.get('title', '')} (Elo {elo:.0f}{', ' + op if op else ''})"
        parts.append(
            f'<circle cx="{hx}" cy="{hy}" r="13" fill="{_elo_color(elo, lo, hi)}" stroke="currentColor" '
            f'stroke-opacity="0.35"><title>{_e(tip)}</title></circle>'
        )
        parts.append(f'<text x="{hx + 18}" y="{hy + 4}" fill="currentColor">{_e(h.get("id"))} · {elo:.0f}</text>')

    legend_x = 8
    legend = []
    for op, color in OPERATOR_COLORS.items():
        legend.append(f'<rect x="{legend_x}" y="{height - 12}" width="12" height="4" fill="{color}"/>')
        legend.append(f'<text x="{legend_x + 16}" y="{height - 7}" fill="currentColor">{op.lower()}</text>')
        legend_x += 100
    return _svg(max(width, legend_x), height + 12, "".join(parts + legend), "Hypothesis lineage")


# --- 4. Similarity graph ---


def similarity_svg(hypotheses: List[Dict], adjacency: Dict[str, List[Dict]]) -> Optional[str]:
    ids = [h.get("id") for h in hypotheses if h.get("id") in (adjacency or {})]
    if len(ids) < 2:
        return None
    pairs = {}
    for a, links in adjacency.items():
        for link in links:
            b = link.get("other_id")
            if a in ids and b in ids and a != b:
                pairs[tuple(sorted((a, b)))] = float(link.get("similarity") or 0)
    if not pairs:
        return None
    values = sorted(pairs.values())
    cutoff = values[min(len(values) - 1, int(len(values) * 0.75))]
    by_id = {h.get("id"): h for h in hypotheses}
    lo, hi = _elo_range([by_id[i] for i in ids])
    size, radius = 420, 150
    cx = cy = size / 2
    pos = {}
    for k, hid in enumerate(ids):
        angle = 2 * math.pi * k / len(ids) - math.pi / 2
        pos[hid] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    s_lo, s_hi = cutoff, max(values)
    parts = []
    for (a, b), sim in sorted(pairs.items(), key=lambda p: p[1]):
        if sim < cutoff:
            continue
        t = 1.0 if s_hi <= s_lo else (sim - s_lo) / (s_hi - s_lo)
        (ax, ay), (bx, by) = pos[a], pos[b]
        parts.append(
            f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="#6366f1" '
            f'stroke-width="{1 + 4 * t:.1f}" opacity="{0.3 + 0.6 * t:.2f}"><title>{_e(a)}–{_e(b)}: {sim:.2f}</title></line>'
        )
    for hid in ids:
        hx, hy = pos[hid]
        elo = float(by_id[hid].get("elo_score") or 1200)
        parts.append(
            f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="11" fill="{_elo_color(elo, lo, hi)}" stroke="currentColor" '
            f'stroke-opacity="0.35"><title>{_e(by_id[hid].get("title", ""))}</title></circle>'
        )
        lx, ly = cx + (radius + 28) * (hx - cx) / radius, cy + (radius + 28) * (hy - cy) / radius
        parts.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="middle" fill="currentColor">{_e(hid)}</text>')
    return _svg(size, size, "".join(parts), "Hypothesis similarity graph")


# --- Assembly ---


def _pool(cycle_details: Dict) -> List[Dict]:
    steps = cycle_details.get("steps") or {}
    for name in ("ranking_final", "ranking", "reflection"):
        hypotheses = (steps.get(name) or {}).get("hypotheses") or []
        if hypotheses:
            return hypotheses
    return []


def cycle_plots_html(cycle_details: Dict) -> str:
    """All charts that have data for this cycle, as one HTML block ('' if none)."""
    pool = _pool(cycle_details)
    adjacency = ((cycle_details.get("steps") or {}).get("proximity") or {}).get("adjacency_graph") or {}
    figures = []
    chart = elo_history_svg(pool)
    if chart:
        figures.append(
            _figure("Elo rating across cycles", "Rating after each cycle's tournament; top 6 labelled.", chart)
        )
    chart = lineage_svg(pool)
    if chart:
        figures.append(
            _figure(
                "Hypothesis family tree",
                "Columns are the cycle a hypothesis was created; lines link parents to evolved children, "
                "colored by operator. Node color: grey (lowest Elo) to green (highest).",
                chart,
            )
        )
    chart = score_heatmap_svg(pool)
    if chart:
        figures.append(_figure("Review scores (1–5)", "Rows sorted by Elo; hover a row for its full title.", chart))
    chart = similarity_svg(pool, adjacency)
    if chart:
        figures.append(
            _figure(
                "Similarity graph",
                "Edges join the most similar 25% of pairs; thicker means more similar. Clusters suggest "
                "near-duplicate ideas.",
                chart,
            )
        )
    return "".join(figures)
