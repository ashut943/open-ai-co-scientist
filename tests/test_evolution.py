"""Offline tests for LLM evolution operators (mocked at the LLM boundary)."""

import json
from unittest.mock import patch

from app.agents import (
    _OPERATOR_INSTRUCTIONS,
    EvolutionAgent,
    call_llm_for_evolution,
    evolve_hypothesis,
)
from app.models import ContextMemory, Hypothesis, ResearchGoal


def _hypo(hid: str, title: str, text: str, elo: float = 1200.0) -> Hypothesis:
    h = Hypothesis(hid, title, text)
    h.novelty_review = "MEDIUM"
    h.feasibility_review = "HIGH"
    h.review_comments = [f"Review of {hid}"]
    h.elo_score = elo
    return h


def _goal():
    return ResearchGoal(description="Increase solar panel efficiency", llm_model="test/model")


def test_call_llm_for_evolution_includes_goal_reviews_and_operator():
    goal = _goal()
    context = ContextMemory()
    parent = _hypo("G1", "Perovskite", "Use perovskite tandem cells.")
    context.add_hypothesis(parent)
    context.tournament_results.append(
        {
            "hypothesis_a": "G1",
            "hypothesis_b": "G2",
            "winner": "G1",
            "loser": "G2",
            "tie": False,
            "judgment": {"reasoning": "G1 is more novel."},
        }
    )
    context.meta_review_feedback.append(
        {
            "recurring_weaknesses": ["Some ideas lack novelty."],
            "recommended_evolution_strategy": [
                {"operator": "REFINE", "parent_ids": ["G1"], "rationale": "Address novelty gaps"}
            ],
            "meta_review_critique": ["Some ideas lack novelty."],
            "research_overview": {"suggested_next_steps": ["Refine top hypotheses."]},
        }
    )
    payload = json.dumps(
        {
            "title": "Improved perovskite stack",
            "text": "Use a graded perovskite tandem with stable contacts.",
            "reasoning": "Addressed feasibility of contacts.",
        }
    )

    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        result = call_llm_for_evolution(goal, "REFINE", [parent], context)

    assert result["title"] == "Improved perovskite stack"
    prompt = mock_call.call_args.args[0]
    assert "Increase solar panel efficiency" in prompt
    assert "Evolution operator: REFINE" in prompt
    assert _OPERATOR_INSTRUCTIONS["REFINE"][:40] in prompt
    assert "Use perovskite tandem cells." in prompt
    assert "Novelty: MEDIUM" in prompt
    assert "G1 is more novel." in prompt
    assert "Some ideas lack novelty." in prompt
    assert "Latest meta-review guidance" in prompt
    assert mock_call.call_args.kwargs["model"] == "test/model"


def test_evolve_hypothesis_creates_child_with_parent_ids_without_mutating_parent():
    goal = _goal()
    context = ContextMemory()
    parent = _hypo("G1", "Parent", "Original parent text.")
    original_text = parent.text
    context.add_hypothesis(parent)
    payload = json.dumps(
        {
            "title": "Child",
            "text": "Evolved child text.",
            "reasoning": "Clearer mechanism.",
            "search_keywords": ["perovskite tandem", "interface passivation"],
        }
    )

    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        child = evolve_hypothesis("MUTATE", [parent], goal, context)

    assert "search_keywords" in mock_call.call_args.args[0]
    assert child is not None
    assert child.hypothesis_id != parent.hypothesis_id
    assert child.parent_ids == ["G1"]
    assert child.evolution_operator == "MUTATE"
    assert child.text == "Evolved child text."
    assert child.search_keywords == '"perovskite tandem" "interface passivation"'
    assert parent.text == original_text  # no in-place mutation
    assert any("[MUTATE]" in c for c in child.review_comments)


def test_hybridize_sets_both_parent_ids():
    goal = _goal()
    context = ContextMemory()
    a = _hypo("G1", "A", "Hypothesis A text.", elo=1300)
    b = _hypo("G2", "B", "Hypothesis B text.", elo=1250)
    context.add_hypothesis(a)
    context.add_hypothesis(b)
    payload = json.dumps({"title": "Hybrid", "text": "Merged complementary mechanism.", "reasoning": "Best of both."})

    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        child = evolve_hypothesis("HYBRIDIZE", [a, b], goal, context)

    assert child is not None
    assert child.parent_ids == ["G1", "G2"]
    assert child.evolution_operator == "HYBRIDIZE"
    prompt = mock_call.call_args.args[0]
    assert "Hypothesis A text." in prompt
    assert "Hypothesis B text." in prompt
    assert "HYBRIDIZE" in prompt


def test_evolve_hypothesis_returns_none_on_llm_error():
    goal = _goal()
    context = ContextMemory()
    parent = _hypo("G1", "Parent", "Text")
    context.add_hypothesis(parent)

    errors: list[str] = []
    with patch("app.agents.call_llm", return_value="Error: API call failed"):
        assert evolve_hypothesis("SIMPLIFY", [parent], goal, context, errors=errors) is None

    assert len(errors) == 1
    assert "SIMPLIFY" in errors[0] and "G1" in errors[0]
    assert "Error: API call failed" in errors[0]


def test_evolution_agent_runs_all_operators_and_preserves_lineage():
    goal = _goal()
    context = ContextMemory()
    top = _hypo("G1", "Top", "Top hypothesis text.", elo=1400)
    second = _hypo("G2", "Second", "Second hypothesis text.", elo=1300)
    context.add_hypothesis(top)
    context.add_hypothesis(second)

    def fake_llm(prompt, temperature=0.7, model=None):
        if "REFINE" in prompt:
            op = "REFINE"
        elif "MUTATE" in prompt:
            op = "MUTATE"
        elif "SIMPLIFY" in prompt:
            op = "SIMPLIFY"
        else:
            op = "HYBRIDIZE"
        return json.dumps(
            {
                "title": f"{op} child",
                "text": f"Body for {op}",
                "reasoning": f"Applied {op}",
            }
        )

    with patch("app.agents.call_llm", side_effect=fake_llm):
        children, errors = EvolutionAgent().evolve_hypotheses(context, goal)

    assert errors == []
    assert len(children) == 4
    ops = {c.evolution_operator for c in children}
    assert ops == {"REFINE", "MUTATE", "SIMPLIFY", "HYBRIDIZE"}
    for child in children:
        assert child.hypothesis_id.startswith("E")
        assert child.parent_ids
        assert top.text == "Top hypothesis text."
        assert second.text == "Second hypothesis text."
    hybrid = next(c for c in children if c.evolution_operator == "HYBRIDIZE")
    assert hybrid.parent_ids == ["G1", "G2"]
    singles = [c for c in children if c.evolution_operator != "HYBRIDIZE"]
    assert all(c.parent_ids == ["G1"] for c in singles)


def test_evolution_agent_skips_hybridize_with_single_hypothesis():
    goal = _goal()
    context = ContextMemory()
    context.add_hypothesis(_hypo("G1", "Only", "Only one.", elo=1200))

    with patch(
        "app.agents.call_llm",
        return_value=json.dumps({"title": "Child", "text": "Body", "reasoning": "ok"}),
    ):
        children, _ = EvolutionAgent().evolve_hypotheses(context, goal)

    assert len(children) == 3
    assert all(c.evolution_operator != "HYBRIDIZE" for c in children)
