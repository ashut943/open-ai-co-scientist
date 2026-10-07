"""Offline tests for LLM meta-review and cycle ordering (mocked LLM boundary)."""

import json
from unittest.mock import patch

from app.agents import (
    EvolutionAgent,
    MetaReviewAgent,
    SupervisorAgent,
    call_llm_for_meta_review,
)
from app.models import ContextMemory, Hypothesis, ResearchGoal


def _hypo(hid: str, title: str, text: str, elo: float = 1200.0) -> Hypothesis:
    h = Hypothesis(hid, title, text)
    h.novelty_review = "MEDIUM"
    h.feasibility_review = "HIGH"
    h.review_scores = {"novelty": 3, "feasibility": 4, "testability": 4}
    h.review_weaknesses = [f"Weakness of {hid}"]
    h.critical_assumptions = [f"Assumption of {hid}"]
    h.elo_score = elo
    return h


def _meta_payload(**overrides):
    payload = {
        "recurring_strengths": ["Clear mechanisms"],
        "recurring_weaknesses": ["Costly materials"],
        "unexplored_mechanisms": ["Photon recycling"],
        "shared_assumptions": ["Stable contacts"],
        "contradictions": ["High novelty vs low feasibility tradeoff"],
        "promising_hypothesis_pairs": [{"ids": ["G1", "G2"], "reason": "Complementary"}],
        "research_gaps": ["No outdoor durability test"],
        "recommended_evolution_strategy": [
            {"operator": "REFINE", "parent_ids": ["G1"], "rationale": "Fix weaknesses"},
            {"operator": "HYBRIDIZE", "parent_ids": ["G1", "G2"], "rationale": "Combine strengths"},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_call_llm_for_meta_review_parses_rich_schema():
    goal = ResearchGoal(description="Increase solar efficiency")
    context = ContextMemory()
    context.add_hypothesis(_hypo("G1", "A", "text a", elo=1400))
    context.add_hypothesis(_hypo("G2", "B", "text b", elo=1300))

    with patch("app.agents.call_llm", return_value=_meta_payload()) as mock_call:
        meta = call_llm_for_meta_review(goal, context)

    assert meta["recurring_weaknesses"] == ["Costly materials"]
    assert meta["recommended_evolution_strategy"][0]["operator"] == "REFINE"
    prompt = mock_call.call_args.args[0]
    assert "Increase solar efficiency" in prompt
    assert "G1" in prompt and "G2" in prompt
    assert "recommended_evolution_strategy" in prompt


def test_meta_review_agent_keeps_legacy_ui_keys():
    goal = ResearchGoal(description="Goal")
    context = ContextMemory()
    context.add_hypothesis(_hypo("G1", "A", "text a", elo=1400))

    with patch("app.agents.call_llm", return_value=_meta_payload()):
        overview = MetaReviewAgent().summarize_and_feedback(context, research_goal=goal)

    assert "meta_review_critique" in overview
    assert "research_overview" in overview
    assert overview["research_overview"]["top_ranked_hypotheses"]
    assert overview["recommended_evolution_strategy"]
    assert context.meta_review_feedback[-1] is overview


def test_meta_review_falls_back_on_llm_error():
    goal = ResearchGoal(description="Goal")
    context = ContextMemory()
    context.add_hypothesis(_hypo("G1", "A", "text a", elo=1400))
    context.add_hypothesis(_hypo("G2", "B", "text b", elo=1300))

    with patch("app.agents.call_llm", return_value="Error: API call failed"):
        meta = call_llm_for_meta_review(goal, context)

    assert meta["recommended_evolution_strategy"]
    assert any(item.get("operator") == "HYBRIDIZE" for item in meta["recommended_evolution_strategy"])


def test_evolution_uses_meta_review_strategy():
    goal = ResearchGoal(description="Goal", llm_model="test/model")
    context = ContextMemory()
    a = _hypo("G1", "A", "text a", elo=1400)
    b = _hypo("G2", "B", "text b", elo=1300)
    context.add_hypothesis(a)
    context.add_hypothesis(b)
    context.meta_review_feedback.append(
        {
            "recommended_evolution_strategy": [
                {"operator": "MUTATE", "parent_ids": ["G2"]},
                {"operator": "HYBRIDIZE", "parent_ids": ["G1", "G2"]},
            ],
            "promising_hypothesis_pairs": [],
        }
    )

    with patch(
        "app.agents.call_llm",
        return_value=json.dumps({"title": "Child", "text": "Body", "reasoning": "ok"}),
    ) as mock_call:
        children = EvolutionAgent().evolve_hypotheses(context, goal)

    assert len(children) == 2
    assert {c.evolution_operator for c in children} == {"MUTATE", "HYBRIDIZE"}
    mutate = next(c for c in children if c.evolution_operator == "MUTATE")
    assert mutate.parent_ids == ["G2"]
    # Prompts should include the meta-review guidance block.
    assert any("Latest meta-review guidance" in call.args[0] for call in mock_call.call_args_list)


def test_run_cycle_orders_meta_review_before_evolution():
    goal = ResearchGoal(description="Goal", num_hypotheses=2, top_k_hypotheses=2)
    context = ContextMemory()
    step_order: list[str] = []

    def fake_llm(prompt, temperature=0.7, model=None):
        if "propose" in prompt.lower() or "novel and feasible hypotheses" in prompt.lower():
            return json.dumps(
                [
                    {"title": "H1", "text": "idea one"},
                    {"title": "H2", "text": "idea two"},
                ]
            )
        if "peer reviewer" in prompt.lower() or "Score the hypothesis" in prompt:
            return json.dumps(
                {
                    "review_scores": {
                        "scientific_soundness": 4,
                        "novelty": 4,
                        "relevance": 4,
                        "feasibility": 3,
                        "testability": 4,
                        "clarity": 4,
                        "potential_impact": 3,
                    },
                    "review_strengths": ["ok"],
                    "review_weaknesses": [],
                    "critical_assumptions": [],
                    "falsification_conditions": ["null result"],
                    "safety_ethical_concerns": [],
                    "recommended_improvements": [],
                    "comment": "ok",
                    "references": [],
                }
            )
        if "tournament judge" in prompt.lower() or "scientific tournament judge" in prompt.lower():
            return json.dumps(
                {
                    "winner": "A",
                    "confidence": 0.7,
                    "reasoning": "A wins",
                    "criterion_scores": {},
                }
            )
        if "Evolution operator:" in prompt:
            step_order.append("evolution_prompt")
            return json.dumps({"title": "Evo", "text": "evolved body", "reasoning": "guided"})
        if "conducting a meta-review" in prompt:
            step_order.append("meta_prompt")
            return _meta_payload()
        return json.dumps({"title": "X", "text": "Y", "reasoning": "z"})

    with patch("app.agents.call_llm", side_effect=fake_llm):
        details = SupervisorAgent().run_cycle(goal, context)

    keys = list(details["steps"].keys())
    assert keys.index("meta_review") < keys.index("evolution")
    assert keys.index("evolution") < keys.index("ranking2")
    assert keys.index("ranking2") < keys.index("proximity")
    assert "meta_prompt" in step_order
    assert "evolution_prompt" in step_order
    assert step_order.index("meta_prompt") < step_order.index("evolution_prompt")
    assert details["meta_review"]["recurring_strengths"]


def test_run_cycle_skips_reflection_for_already_reviewed_hypotheses():
    goal = ResearchGoal(description="Goal", num_hypotheses=1)
    context = ContextMemory()
    context.add_hypothesis(_hypo("G0", "Old", "previously reviewed idea"))
    reviewed_texts: list[str] = []

    def fake_llm(prompt, temperature=0.7, model=None):
        if "Score the hypothesis" in prompt:
            reviewed_texts.append(prompt)
            return json.dumps({"review_scores": {"novelty": 4}, "comment": "ok", "references": []})
        if "tournament judge" in prompt.lower():
            return json.dumps({"winner": "A", "confidence": 0.7, "reasoning": "A", "criterion_scores": {}})
        if "Evolution operator:" in prompt:
            return json.dumps({"title": "Evo", "text": "evolved body", "reasoning": "r"})
        if "conducting a meta-review" in prompt:
            return _meta_payload()
        return json.dumps([{"title": "New", "text": "fresh idea"}])

    with (
        patch("app.agents.call_llm", side_effect=fake_llm),
        patch(
            "app.agents.ProximityAgent.build_proximity_graph",
            return_value={"adjacency_graph": {}, "nodes": [], "edges": []},
        ),
    ):
        SupervisorAgent().run_cycle(goal, context)

    assert reviewed_texts
    assert not any("previously reviewed idea" in p for p in reviewed_texts)
    assert any("fresh idea" in p for p in reviewed_texts)
