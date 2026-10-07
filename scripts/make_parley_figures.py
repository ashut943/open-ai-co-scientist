"""Regenerate the Parley cost figures used by docs/parley-models.md.

    venv/bin/pip install matplotlib   # docs-only dependency
    venv/bin/python scripts/make_parley_figures.py

Writes docs/parley-cycle-schematic.png and docs/parley-cost-science-pareto.png
and prints the token/cost numbers quoted in the doc. Token counts are planning
estimates derived from the prompt templates in app/agents.py, not measurements.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"

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


def _box(ax, x, y, w, h, title, sub, llm=True):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=2,
            edgecolor=GREEN if llm else GRAY,
            facecolor=LLM_FILL if llm else PANEL,
        )
    )
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center", color=TEXT, fontsize=12, weight="bold")
    ax.text(x + w / 2, y + h - 5.4, sub, ha="center", va="top", color=MUTED, fontsize=9, linespacing=1.35)


def _arrow(ax, start, end, color=GREEN, style="-|>", ls="-"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=2, linestyle=ls, shrinkA=0, shrinkB=0),
    )


def draw_schematic(rows, avg_reasoning):
    fig = plt.figure(figsize=(10.24, 6.46), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 63)
    ax.axis("off")

    ax.text(50, 59.5, "One Open AI Co-Scientist cycle", ha="center", color=TEXT, fontsize=18, weight="bold")
    ax.text(
        50,
        56.0,
        "Defaults: num_hypotheses = 4 · top_k = 2 · 3 matches per hypothesis   ·   "
        "Green = LLM API calls   ·   Gray = local (no API cost)",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )

    # 1. Generation step: evolution (from cycle 2) then fresh generation.
    ax.add_patch(
        FancyBboxPatch(
            (2, 31.5),
            36,
            21,
            boxstyle="round,pad=0,rounding_size=1.5",
            linewidth=1.5,
            edgecolor=GREEN,
            facecolor="none",
            linestyle="--",
        )
    )
    ax.text(20, 50.6, "1. Generation step", ha="center", va="center", color=TEXT, fontsize=12, weight="bold")
    _box(ax, 3.5, 34, 16, 14, "1a. Evolution", "cycle ≥ 2 · 4× LLM\nrefine / mutate /\nsimplify / hybridize")
    _box(ax, 21, 34, 16, 14, "1b. New ideas", "1× LLM\nfresh hypotheses\nfrom the goal")
    _arrow(ax, (19.5, 41), (21, 41))

    _box(ax, 41, 34, 16, 14, "2. Reflection", "1× LLM per new hyp.\n+ free literature search\n(4 in cycle 1, then 8)")
    _box(ax, 60, 34, 16, 14, "3. Tournament", "LLM judge per match\n≈ n × 3 / 2 matches\n+ Elo update (local)")
    _box(ax, 79, 34, 18, 14, "4. Meta-review", "1× LLM over reviews\n+ match outcomes\n→ next evolution plan")
    _box(ax, 79, 15, 18, 12, "5. Proximity", "local embeddings\nsimilarity graph", llm=False)

    _arrow(ax, (38, 41), (41, 41))
    _arrow(ax, (57, 41), (60, 41))
    _arrow(ax, (76, 41), (79, 41))
    _arrow(ax, (91, 34), (91, 27), color=GRAY)

    # Feedback: this cycle's meta-review steers the next cycle's evolution.
    ax.plot([84, 84, 11.5], [34, 29.5, 29.5], color=GREEN, lw=2, ls="--")
    _arrow(ax, (11.5, 29.5), (11.5, 31.5), ls="--")
    ax.text(
        47,
        28.6,
        "next cycle: meta-review strategy picks the evolution operators and parents",
        ha="center",
        va="top",
        color=GREEN,
        fontsize=9,
    )

    ax.add_patch(
        FancyBboxPatch(
            (2, 1.5),
            74,
            22,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1,
            edgecolor="#30363d",
            facecolor=PANEL,
        )
    )
    ax.text(4, 20.8, "LLM calls and tokens per cycle (estimates, not measured)", color=TEXT, fontsize=11, weight="bold")
    notes = {
        "Cycle 1": "no evolution yet",
        "Cycle 2": "4 evolved + 4 new + 4 surviving",
        "Cycle 3": "pool keeps growing; nothing is retired",
    }
    y = 17.4
    for label, calls, tokens_in, tokens_out in rows:
        ax.text(
            4,
            y,
            f"• {label}: {calls} calls · ~{tokens_in / 1000:.0f}k in + ~{tokens_out / 1000:.0f}k visible out  ({notes[label]})",
            color=MUTED,
            fontsize=9.5,
        )
        y -= 3.2
    ax.text(
        4,
        y + 1.4,
        f"• Reasoning models add hidden reasoning tokens billed as output:\n"
        f"  ~{REASONING_PER_CALL:,} per call assumed → ~{avg_reasoning / 1000:.0f}k extra out per cycle on average",
        color=MUTED,
        fontsize=9.5,
        va="top",
        linespacing=1.4,
    )

    out = DOCS / "parley-cycle-schematic.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def _pareto(points):
    front, best = [], -1.0
    for p in sorted(points, key=lambda p: (p["cost"], -p["tb"])):
        if p["tb"] > best:
            front.append(p)
            best = p["tb"]
    return front


def draw_pareto(avg_in, avg_vis_out, avg_out):
    points = []
    for label, tb, price_in, price_out in MODELS:
        cost_in = avg_in * price_in / 1e6
        points.append(
            {
                "label": label,
                "tb": tb,
                "cost": cost_in + avg_out * price_out / 1e6,
                "cost_visible": cost_in + avg_vis_out * price_out / 1e6,
            }
        )
    front = _pareto(points)
    front_labels = {p["label"] for p in front}

    fig, ax = plt.subplots(figsize=(10.24, 6.77), dpi=100, facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=MUTED)
    ax.grid(True, color="#30363d", lw=0.8)
    ax.set_xscale("log")

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

    ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.012, 4)
    ax.set_ylim(-2, 78)
    ax.set_title("Parley models: cost per cycle vs science-agent score", color=TEXT, fontsize=14, pad=12)
    ax.set_xlabel(
        f"Est. USD per cycle, log scale (avg of cycles 1–3: ~{avg_in / 1000:.0f}k in + "
        f"~{avg_out / 1000:.0f}k out incl. reasoning)",
        color=TEXT,
        fontsize=11,
    )
    ax.set_ylabel("Terminal-Bench-Science resolution rate (%)", color=TEXT, fontsize=11)
    legend = ax.legend(loc="upper left", facecolor=PANEL, edgecolor="#30363d", fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT)
    flash = next(p for p in points if p["label"] == "Gemini 3.8 Flash")
    fig.text(
        0.5,
        0.012,
        "Sources: Snorkel Terminal-Bench-Science · vendor list prices · tokens estimated from app/agents.py prompts · "
        f"${BUDGET_USD:.0f} ≈ {BUDGET_USD / flash['cost']:.0f} Flash cycles",
        ha="center",
        color=MUTED,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = DOCS / "parley-cost-science-pareto.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out, points, front


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

    schematic = draw_schematic(rows, avg_reasoning)
    pareto, points, front = draw_pareto(avg_in, avg_vis_out, avg_out)
    print("\n| Model | TB-Science | $/cycle (visible only) | $/cycle (incl. reasoning) | Cycles on $30 |")
    for p in sorted(points, key=lambda p: p["cost"]):
        print(
            f"| {p['label']} | {p['tb']}% | ${p['cost_visible']:.3f} | ${p['cost']:.2f} | "
            f"~{BUDGET_USD / p['cost']:.0f} |"
        )
    print("Pareto front:", " -> ".join(p["label"] for p in front))
    print(f"Wrote {schematic}\nWrote {pareto}")


if __name__ == "__main__":
    main()
