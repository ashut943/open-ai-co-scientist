"""Regenerate the Parley cost figures used by docs/parley-models.md.

    pip install matplotlib   # docs-only dependency
    python scripts/make_parley_figures.py

Writes PNG and PDF versions of the cycle schematic (pastel agent colors),
one combined measured Pareto plot (cycle-1 $ → cycles-1+2 $ as a range
bar), and the planning-estimate Pareto plot.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _save_fig(fig, path: Path, *, facecolor: str) -> list[Path]:
    """Write PNG (for the markdown preview) and a vector PDF beside it."""
    path = Path(path)
    written = []
    for suffix in (".png", ".pdf"):
        out = path.with_suffix(suffix)
        fig.savefig(out, facecolor=facecolor)
        written.append(out)
    return written

BG = "#0d1017"
PANEL = "#161b24"
GREEN = "#3fb950"
LLM_FILL = "#12261a"
GRAY = "#8b949e"
TEXT = "#e6edf3"
MUTED = "#9aa4b2"
BUDGET_USD = 30.0

# --- Token model (see "Token usage estimate" in docs/parley-models.md) ---
# (input tokens, visible output tokens) per call.
GENERATION = (250, 900)  # ~120-token template + goal + existing IDs -> 4 hypotheses x ~215 tokens
REFLECTION = (1800, 600)  # ~450-token template + goal + hypothesis + 6 papers x ~185 -> JSON review
JUDGE = (1300, 200)  # ~200-token template + goal + 2 x (hypothesis ~215 + formatted review ~300)
EVOLUTION = (2250, 300)  # template + parent(s) with reviews + match feedback + meta-review summary
META_OUT = 700
REASONING_PER_CALL = 1000  # hidden reasoning tokens, billed as output (assumption)


def meta_input(active: int, matches_seen: int) -> int:
    # ~400 template/goal + ~435 per hypothesis snapshot + ~60 per recent match (last 30)
    return 400 + 435 * active + 60 * min(matches_seen, 30)


# label, evolution calls, reflections, judge matches, active hypotheses at meta-review, matches so far
CYCLES = [
    ("Cycle 1", 0, 4, 6, 4, 6),
    ("Cycle 2", 4, 8, 19, 12, 25),
    ("Cycle 3", 4, 8, 31, 20, 56),
]


def cycle_tokens():
    rows = []
    for label, evo, refl, judges, active, seen in CYCLES:
        tokens_in = (
            evo * EVOLUTION[0] + GENERATION[0] + refl * REFLECTION[0] + judges * JUDGE[0] + meta_input(active, seen)
        )
        tokens_out = evo * EVOLUTION[1] + GENERATION[1] + refl * REFLECTION[1] + judges * JUDGE[1] + META_OUT
        calls = evo + 1 + refl + judges + 1
        rows.append((label, calls, tokens_in, tokens_out))
    return rows


# label, TB-Science %, $ in / 1M, $ out / 1M  (from the tables in docs/parley-models.md)
MODELS = [
    ("Luna", 3.3, 0.20, 1.20),
    ("Gemini 3.8 Flash", 12.4, 0.75, 3.75),
    ("Gemini 3.7 Flash", 5.7, 0.75, 3.75),
    ("Terra", 8.6, 2.0, 12.0),
    ("5.6 Sol", 22.4, 4.0, 20.0),
    ("Opus 5.5", 63.3, 4.0, 20.0),
    ("Opus 5", 30.0, 5.0, 25.0),
    ("Opus 4.8", 10.5, 5.0, 25.0),
    ("Astra", 68.1, 10.0, 50.0),
]
LABEL_OFFSETS = {
    "Luna": (8, -14, "left"),
    "Gemini 3.7 Flash": (8, -14, "left"),
    "5.6 Sol": (8, -14, "left"),
    "Terra": (8, -14, "left"),
    "Opus 5.5": (8, -16, "left"),
    "Opus 4.8": (8, -14, "left"),
    "Astra": (-10, 8, "right"),
}


# Pastel agent palette (matches the Gradio "Agent workflow" schematic).
AGENT_COLORS = {
    "generation": ("#d6eaf8", "#5dade2", "#1a5276"),  # fill, header, edge
    "evolution": ("#fadbd8", "#ec7063", "#922b21"),
    "reflection": ("#d5f5e3", "#58d68d", "#1e8449"),
    "ranking": ("#fcf3cf", "#f4d03f", "#9a7d0a"),
    "proximity": ("#e8daef", "#af7ac5", "#6c3483"),
    "meta": ("#d1f2eb", "#48c9b0", "#0e6655"),
    "local": ("#f4f6f7", "#aab7b8", "#566573"),
}


def _agent_box(ax, x, y, w, h, title, sub, palette):
    fill, header, edge = palette
    header_h = 3.6
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.8",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - header_h),
            w,
            header_h,
            boxstyle="round,pad=0,rounding_size=0.8",
            linewidth=0,
            facecolor=header,
        )
    )
    # Square off the bottom of the header so it meets the body cleanly.
    ax.add_patch(plt.Rectangle((x, y + h - header_h - 0.05), w, 1.2, linewidth=0, facecolor=header, zorder=2))
    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        weight="bold",
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + (h - header_h) / 2 + 0.2,
        sub,
        ha="center",
        va="center",
        color="#2c3e50",
        fontsize=8.5,
        linespacing=1.3,
    )


def _arrow(ax, start, end, color="#7f8c8d", style="-|>", ls="-"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=2, linestyle=ls, shrinkA=0, shrinkB=0),
    )


def draw_schematic():
    """Pipeline schematic in the pastel agent-workflow colorscheme (no token footer)."""
    fig = plt.figure(figsize=(10.24, 4.6), dpi=100, facecolor="#f8f9f9")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.set_facecolor("#f8f9f9")

    ax.text(50, 43.2, "One Open AI Co-Scientist cycle", ha="center", color="#1c2833", fontsize=17, weight="bold")
    ax.text(
        50,
        40.0,
        "Defaults: num_hypotheses = 4 · top_k = 2 · 3 matches per hypothesis   ·   "
        "Pastel cards = agents (purple Proximity is local, no API cost)",
        ha="center",
        color="#5d6d7e",
        fontsize=9,
    )

    # 1. Generation step grouping.
    ax.add_patch(
        FancyBboxPatch(
            (2, 17.5),
            36,
            19.5,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.4,
            edgecolor=AGENT_COLORS["generation"][2],
            facecolor="#ebf5fb",
            linestyle="--",
        )
    )
    ax.text(
        20,
        34.8,
        "1. Generation step",
        ha="center",
        va="center",
        color=AGENT_COLORS["generation"][2],
        fontsize=11,
        weight="bold",
    )
    _agent_box(
        ax,
        3.5,
        19.5,
        16,
        13.5,
        "1a. Evolution",
        "cycle ≥ 2 · 4× LLM\nrefine / mutate /\nsimplify / hybridize",
        AGENT_COLORS["evolution"],
    )
    _agent_box(
        ax,
        21,
        19.5,
        16,
        13.5,
        "1b. New ideas",
        "1× LLM\nfresh hypotheses\nfrom the goal",
        AGENT_COLORS["generation"],
    )
    _arrow(ax, (19.5, 26.2), (21, 26.2), color=AGENT_COLORS["generation"][2])

    _agent_box(
        ax,
        41,
        19.5,
        16,
        13.5,
        "2. Reflection",
        "1× LLM per new hyp.\n+ free literature search\n(4 in cycle 1, then 8)",
        AGENT_COLORS["reflection"],
    )
    _agent_box(
        ax,
        60,
        19.5,
        16,
        13.5,
        "3. Tournament",
        "LLM judge per match\n≈ n × 3 / 2 matches\n+ Elo update (local)",
        AGENT_COLORS["ranking"],
    )
    _agent_box(
        ax,
        79,
        19.5,
        18,
        13.5,
        "4. Meta-review",
        "1× LLM over reviews\n+ match outcomes\n→ next evolution plan",
        AGENT_COLORS["meta"],
    )
    _agent_box(
        ax,
        79,
        3.5,
        18,
        11.5,
        "5. Proximity",
        "local embeddings\nsimilarity graph",
        AGENT_COLORS["proximity"],
    )

    _arrow(ax, (38, 26.2), (41, 26.2))
    _arrow(ax, (57, 26.2), (60, 26.2))
    _arrow(ax, (76, 26.2), (79, 26.2))
    _arrow(ax, (88, 19.5), (88, 15), color=AGENT_COLORS["proximity"][2])

    # Feedback: this cycle's meta-review steers the next cycle's evolution.
    loop = AGENT_COLORS["meta"][2]
    ax.plot([84, 84, 11.5], [19.5, 15.5, 15.5], color=loop, lw=2, ls="--")
    _arrow(ax, (11.5, 15.5), (11.5, 17.5), color=loop, ls="--")
    ax.text(
        47,
        14.2,
        "Iterate: meta-review strategy picks the next cycle's evolution operators and parents",
        ha="center",
        va="top",
        color=loop,
        fontsize=9,
    )

    written = _save_fig(fig, DOCS / "parley-cycle-schematic.png", facecolor="#f8f9f9")
    plt.close(fig)
    return written


def _pareto(points):
    front, best = [], -1.0
    for p in sorted(points, key=lambda p: (p["cost"], -p["tb"])):
        if p["tb"] > best:
            front.append(p)
            best = p["tb"]
    return front


# Measured Claude Opus 5.5 token volumes from a real Danionella session
# (Parley total tokens + billed USD; in/out split from list prices $4/$20).
# Flash $ from same tokens at $0.75 / $3.75.
OPUS_MEASURED = {
    "cycle1": {
        "in": 94_000,
        "out": 65_000,
        "total": 159_000,
        "usd": 1.67,
        "flash_usd": 0.31,
        "minutes": 5.5,
        "calls": 12,
    },
    "cycle2": {
        "in": 328_000,
        "out": 195_000,
        "total": 523_000,
        "usd": 5.21,
        "flash_usd": 0.98,
        "minutes": 15.3,
        "calls": 33,
    },
}


def _model_cost(tokens_in, tokens_out, price_in, price_out):
    return tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6


def draw_measured_pareto_combined():
    """One scatter: bar from cycle-1 $ to cycles-1+2 $; light pastel theme like the schematic."""
    c1, c2 = OPUS_MEASURED["cycle1"], OPUS_MEASURED["cycle2"]
    tin1, tout1 = c1["in"], c1["out"]
    tin12, tout12 = c1["in"] + c2["in"], c1["out"] + c2["out"]

    points = []
    for label, tb, price_in, price_out in MODELS:
        points.append(
            {
                "label": label,
                "tb": tb,
                "cost_c1": _model_cost(tin1, tout1, price_in, price_out),
                "cost": _model_cost(tin12, tout12, price_in, price_out),
            }
        )
    front = _pareto(points)
    front_labels = {p["label"] for p in front}

    light_bg = "#f8f9f9"
    ink = "#1c2833"
    muted = "#5d6d7e"
    grid = "#d5d8dc"
    bar = "#85929e"
    front_color = AGENT_COLORS["meta"][2]  # teal, matches meta-review card
    front_soft = AGENT_COLORS["reflection"][1]  # green for cycle-1 dashed front
    dominated_color = "#aab7b8"
    panel = "#ffffff"

    fig, ax = plt.subplots(figsize=(10.24, 6.77), dpi=100, facecolor=light_bg)
    ax.set_facecolor(light_bg)
    for spine in ax.spines.values():
        spine.set_color("#bdc3c7")
    ax.tick_params(colors=muted)
    ax.grid(True, color=grid, lw=0.8)
    ax.set_xscale("log")

    for p in points:
        ax.plot([p["cost_c1"], p["cost"]], [p["tb"], p["tb"]], color=bar, lw=1.8, alpha=0.75, zorder=2)
        ax.plot(p["cost_c1"], p["tb"], marker="|", color=bar, ms=11, mew=1.8, zorder=3)
    ax.plot(
        [p["cost"] for p in front],
        [p["tb"] for p in front],
        color=front_color,
        lw=2.5,
        zorder=4,
        label="Pareto front (cycles 1+2 $)",
    )
    ax.plot(
        [p["cost_c1"] for p in front],
        [p["tb"] for p in front],
        color=front_soft,
        lw=1.6,
        ls="--",
        alpha=0.9,
        zorder=3,
        label="Pareto front (cycle 1 $)",
    )
    dominated = [p for p in points if p["label"] not in front_labels]
    ax.scatter(
        [p["cost"] for p in dominated],
        [p["tb"] for p in dominated],
        s=70,
        color=dominated_color,
        edgecolor=light_bg,
        zorder=5,
        label="Dominated (at cycles 1+2 $)",
    )
    ax.scatter(
        [p["cost"] for p in front],
        [p["tb"] for p in front],
        s=140,
        color=front_color,
        edgecolor="white",
        linewidths=1.2,
        zorder=6,
        label="Pareto (at cycles 1+2 $)",
    )
    ax.plot([], [], color=bar, marker="|", ms=10, lw=1.5, label="Left = cycle 1 $ · bar → cycles 1+2 $")

    for p in points:
        dx, dy, ha = LABEL_OFFSETS.get(p["label"], (8, 6, "left"))
        on_front = p["label"] in front_labels
        ax.annotate(
            p["label"],
            (p["cost"], p["tb"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            color=ink if on_front else muted,
            fontsize=10,
            weight="bold" if on_front else "normal",
        )

    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.05, 22)
    ax.set_ylim(-2, 78)
    ax.set_title(
        "Parley models: cost vs science (measured Opus 5.5 tokens)",
        color=ink,
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel(
        f"Est. USD, log scale · left tick = cycle 1 (~{tin1 / 1000:.0f}k in + ~{tout1 / 1000:.0f}k out) · "
        f"right = cycles 1+2 (~{tin12 / 1000:.0f}k in + ~{tout12 / 1000:.0f}k out)",
        color=ink,
        fontsize=10,
    )
    ax.set_ylabel("Terminal-Bench-Science resolution rate (%)", color=ink, fontsize=11)
    legend = ax.legend(loc="upper left", facecolor=panel, edgecolor="#bdc3c7", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(ink)
    flash = next(p for p in points if p["label"] == "Gemini 3.8 Flash")
    fig.text(
        0.5,
        0.012,
        "Sources: Snorkel Terminal-Bench-Science · vendor list prices · "
        "Opus 5.5 Danionella session (Parley bill + list-price in/out split) · "
        f"${BUDGET_USD:.0f} ≈ {BUDGET_USD / flash['cost_c1']:.0f} Flash cycle-1 runs "
        f"or ~{BUDGET_USD / flash['cost']:.0f} Flash two-cycle sessions",
        ha="center",
        color=muted,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    written = _save_fig(fig, DOCS / "parley-cost-science-pareto-opus-measured.png", facecolor=light_bg)
    plt.close(fig)
    return written, points, front


def draw_pareto(
    tokens_in,
    tokens_out,
    *,
    filename,
    title,
    xlabel,
    footer_extra,
    tokens_vis_out=None,
    xlim=(0.012, 12),
    ticks=None,
):
    """Scatter of $/workload vs TB-Science. If tokens_vis_out is set, draw a grey
    bar from visible-only cost to full (incl. reasoning) cost; otherwise a single
    measured point (billed output already includes thinking)."""
    if ticks is None:
        ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    points = []
    for label, tb, price_in, price_out in MODELS:
        cost_in = tokens_in * price_in / 1e6
        cost = cost_in + tokens_out * price_out / 1e6
        cost_visible = cost_in + (tokens_vis_out if tokens_vis_out is not None else tokens_out) * price_out / 1e6
        points.append({"label": label, "tb": tb, "cost": cost, "cost_visible": cost_visible})
    front = _pareto(points)
    front_labels = {p["label"] for p in front}
    show_range = tokens_vis_out is not None

    fig, ax = plt.subplots(figsize=(10.24, 6.77), dpi=100, facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=MUTED)
    ax.grid(True, color="#30363d", lw=0.8)
    ax.set_xscale("log")

    if show_range:
        for p in points:
            ax.plot([p["cost_visible"], p["cost"]], [p["tb"], p["tb"]], color=GRAY, lw=1.5, alpha=0.6, zorder=2)
            ax.plot(p["cost_visible"], p["tb"], marker="|", color=GRAY, ms=10, mew=1.5, zorder=2)
    ax.plot([p["cost"] for p in front], [p["tb"] for p in front], color=GREEN, lw=2.5, zorder=3, label="Pareto front")
    dominated = [p for p in points if p["label"] not in front_labels]
    ax.scatter(
        [p["cost"] for p in dominated],
        [p["tb"] for p in dominated],
        s=70,
        color="#9aa4b2",
        edgecolor=BG,
        zorder=4,
        label="Dominated",
    )
    ax.scatter(
        [p["cost"] for p in front],
        [p["tb"] for p in front],
        s=130,
        color=GREEN,
        edgecolor=TEXT,
        zorder=5,
        label="Pareto",
    )
    if show_range:
        ax.plot([], [], color=GRAY, marker="|", ms=10, lw=1.5, label="Range down to visible-output-only cost")

    for p in points:
        dx, dy, ha = LABEL_OFFSETS.get(p["label"], (8, 6, "left"))
        on_front = p["label"] in front_labels
        ax.annotate(
            p["label"],
            (p["cost"], p["tb"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            color=TEXT if on_front else MUTED,
            fontsize=10,
            weight="bold" if on_front else "normal",
        )

    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(-2, 78)
    ax.set_title(title, color=TEXT, fontsize=14, pad=12)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=11)
    ax.set_ylabel("Terminal-Bench-Science resolution rate (%)", color=TEXT, fontsize=11)
    legend = ax.legend(loc="upper left", facecolor=PANEL, edgecolor="#30363d", fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT)
    flash = next(p for p in points if p["label"] == "Gemini 3.8 Flash")
    fig.text(
        0.5,
        0.012,
        "Sources: Snorkel Terminal-Bench-Science · vendor list prices · "
        f"{footer_extra} · ${BUDGET_USD:.0f} ≈ {BUDGET_USD / flash['cost']:.0f} Flash runs at this token volume",
        ha="center",
        color=MUTED,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    written = _save_fig(fig, DOCS / filename, facecolor=BG)
    plt.close(fig)
    return written, points, front


def _print_table(points, cost_label):
    print(f"\n| Model | TB-Science | {cost_label} | Runs on $30 |")
    for p in sorted(points, key=lambda p: p["cost"]):
        print(f"| {p['label']} | {p['tb']}% | ${p['cost']:.2f} | ~{BUDGET_USD / p['cost']:.0f} |")
    print("Pareto front:", " -> ".join(p["label"] for p in _pareto(points)))


def main():
    rows = cycle_tokens()
    avg_calls = sum(r[1] for r in rows) / len(rows)
    avg_in = sum(r[2] for r in rows) / len(rows)
    avg_vis_out = sum(r[3] for r in rows) / len(rows)
    avg_reasoning = avg_calls * REASONING_PER_CALL
    avg_out = avg_vis_out + avg_reasoning

    for label, calls, tokens_in, tokens_out in rows:
        print(f"{label}: {calls} calls, {tokens_in:,} in, {tokens_out:,} visible out")
    print(f"Average: {avg_calls:.0f} calls, {avg_in:,.0f} in, {avg_vis_out:,.0f} visible out, {avg_out:,.0f} out")

    written: list[Path] = []
    written.extend(draw_schematic())

    pareto_m, points_m, front_m = draw_measured_pareto_combined()
    written.extend(pareto_m)
    print("\n=== Measured Opus (combined cycle 1 | cycles 1+2) ===")
    print("| Model | TB-Science | $/cycle 1 | $/cycles 1+2 | Cycle-1 runs on $30 |")
    for p in sorted(points_m, key=lambda p: p["cost"]):
        print(
            f"| {p['label']} | {p['tb']}% | ${p['cost_c1']:.2f} | ${p['cost']:.2f} | "
            f"~{BUDGET_USD / p['cost_c1']:.0f} |"
        )
    print("Pareto front:", " -> ".join(p["label"] for p in front_m))

    pareto, points, front = draw_pareto(
        avg_in,
        avg_out,
        filename="parley-cost-science-pareto.png",
        title="Parley models: cost per cycle vs science-agent score (planning estimate)",
        xlabel=(
            f"Est. USD per cycle, log scale (avg of cycles 1–3: ~{avg_in / 1000:.0f}k in + "
            f"~{avg_out / 1000:.0f}k out incl. reasoning)"
        ),
        footer_extra="tokens estimated from app/agents.py prompts",
        tokens_vis_out=avg_vis_out,
        xlim=(0.012, 4),
        ticks=[0.02, 0.05, 0.1, 0.2, 0.5, 1, 2],
    )
    written.extend(pareto)
    print("\n=== Planning estimate (avg cycles 1–3) ===")
    _print_table(points, "$/cycle")
    print("Pareto front:", " -> ".join(p["label"] for p in front))

    # Remove superseded single-workload measured plots if present.
    for stale in (
        "parley-cost-science-pareto-opus-cycle1.png",
        "parley-cost-science-pareto-opus-cycles1-2.png",
        "parley-cost-science-pareto-opus-cycle1.pdf",
        "parley-cost-science-pareto-opus-cycles1-2.pdf",
    ):
        path = DOCS / stale
        if path.exists():
            path.unlink()
            print(f"Removed {path}")

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
