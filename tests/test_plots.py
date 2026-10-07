"""Offline tests for the inline-SVG cycle charts."""

from app.plots import cycle_plots_html, elo_history_svg, lineage_svg, score_heatmap_svg, similarity_svg
from app.run_store import render_report

SCORES = {
    "scientific_soundness": 4,
    "novelty": 5,
    "relevance": 4,
    "feasibility": 2,
    "testability": 3,
    "clarity": 4,
    "potential_impact": 5,
}


def _pool():
    return [
        {"id": "G1", "title": "TUR with feedback <b>", "elo_score": 1240, "elo_history": [[1, 1216], [2, 1240]],
         "review_scores": SCORES, "parent_ids": []},
        {"id": "G2", "title": "Second idea", "elo_score": 1170, "elo_history": [[1, 1184], [2, 1170]],
         "review_scores": {}, "parent_ids": []},
        {"id": "E1", "title": "Refined TUR", "elo_score": 1230, "elo_history": [[2, 1230]],
         "review_scores": SCORES, "parent_ids": ["G1"], "evolution_operator": "REFINE"},
    ]  # fmt: skip


ADJACENCY = {
    "G1": [{"other_id": "G2", "similarity": 0.4}, {"other_id": "E1", "similarity": 0.9}],
    "G2": [{"other_id": "G1", "similarity": 0.4}, {"other_id": "E1", "similarity": 0.5}],
    "E1": [{"other_id": "G1", "similarity": 0.9}, {"other_id": "G2", "similarity": 0.5}],
}


def _cycle(pool):
    return {"iteration": 2, "steps": {"ranking": {"hypotheses": pool}, "proximity": {"adjacency_graph": ADJACENCY}}}


def test_all_four_charts_render_for_an_evolved_pool():
    charts = cycle_plots_html(_cycle(_pool()))

    for title in ("Elo rating across cycles", "Hypothesis family tree", "Review scores", "Similarity graph"):
        assert title in charts
    assert charts.count("<svg") == 4
    assert "<b>" not in charts and "&lt;b&gt;" in charts


def test_elo_chart_plots_each_cycle_and_labels_final_elo():
    svg = elo_history_svg(_pool())
    assert "cycle 1" in svg and "cycle 2" in svg
    assert "G1 1240" in svg
    assert svg.count("<polyline") == 2  # E1 has a single point


def test_heatmap_skips_unreviewed_and_shows_scores():
    svg = score_heatmap_svg(_pool())
    assert "G1 ·" in svg and "E1 ·" in svg
    assert "G2 ·" not in svg
    assert "Novel" in svg


def test_family_tree_needs_a_parent_link():
    first_cycle = [h for h in _pool() if not h["parent_ids"]]
    assert lineage_svg(first_cycle) is None
    svg = lineage_svg(_pool())
    assert "<path" in svg and "#10b981" in svg  # REFINE edge color


def test_similarity_graph_draws_only_the_most_similar_pairs():
    svg = similarity_svg(_pool(), ADJACENCY)
    assert "E1–G1: 0.90" in svg
    assert "G1–G2" not in svg
    assert similarity_svg(_pool(), {}) is None


def test_no_charts_without_data():
    assert cycle_plots_html({"steps": {}}) == ""


def test_report_includes_charts():
    report = render_report({"run_id": "r1", "research_goal": {"description": "Goal"}, "cycle_details": _cycle(_pool())})
    assert "<h2>Charts</h2>" in report and "<svg" in report
