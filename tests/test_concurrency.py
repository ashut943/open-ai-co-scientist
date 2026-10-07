"""Reflection reviews and tournament judging run concurrently; results apply in a fixed order."""

import copy
import json
import threading
import time
from unittest.mock import patch

from app import agents
from app.agents import RankingAgent, ReflectionAgent, update_elo
from app.models import ContextMemory, Hypothesis, ResearchGoal


class _InFlight:
    """Fake call_llm that records how many calls overlap."""

    def __init__(self, reply):
        self.reply = reply
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __call__(self, prompt, **kwargs):
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        time.sleep(0.1)
        with self.lock:
            self.current -= 1
        return self.reply(prompt)


def _review_reply(prompt):
    marker = next(m for m in ("body-0", "body-1", "body-2") if m in prompt)
    return json.dumps(
        {
            "review_scores": {"novelty": 4, "feasibility": 3},
            "comment": f"Review of {marker}",
            "references": [],
        }
    )


def _hypotheses(n, prefix="G"):
    return [Hypothesis(f"{prefix}{i}", f"Title {i}", f"body-{i}") for i in range(n)]


def test_reviews_run_concurrently_and_each_lands_on_its_hypothesis(monkeypatch):
    monkeypatch.setitem(agents.config, "llm_max_concurrency", 4)
    hypos = _hypotheses(3)
    fake = _InFlight(_review_reply)

    with patch("app.agents.call_llm", side_effect=fake):
        errors = ReflectionAgent().review_hypotheses(hypos, ContextMemory(), ResearchGoal(description="Goal"))

    assert errors == []
    assert fake.peak > 1
    assert [h.review_comments for h in hypos] == [[f"Review of body-{i}"] for i in range(3)]


def test_concurrency_of_one_runs_calls_one_at_a_time(monkeypatch):
    monkeypatch.setitem(agents.config, "llm_max_concurrency", 1)
    fake = _InFlight(_review_reply)

    with patch("app.agents.call_llm", side_effect=fake):
        ReflectionAgent().review_hypotheses(_hypotheses(3), ContextMemory(), ResearchGoal(description="Goal"))

    assert fake.peak == 1


def test_tournament_judges_concurrently_but_updates_elo_in_pair_order(monkeypatch):
    monkeypatch.setitem(agents.config, "llm_max_concurrency", 4)
    hypos = _hypotheses(4)
    hypos[0].elo_score, hypos[1].elo_score = 1300.0, 1100.0
    reference = {h.hypothesis_id: copy.deepcopy(h) for h in hypos}
    context = ContextMemory()
    for h in hypos:
        context.add_hypothesis(h)
    fake = _InFlight(lambda prompt: json.dumps({"winner": "A", "confidence": 0.9, "reasoning": "A."}))

    with (
        patch("app.agents.random.shuffle"),
        patch("app.agents.random.random", return_value=0.0),
        patch("app.agents.call_llm", side_effect=fake),
    ):
        RankingAgent().run_tournament(hypos, context, ResearchGoal(description="Goal"))

    played = [(r["hypothesis_a"], r["hypothesis_b"]) for r in context.tournament_results]
    assert played == [(a.hypothesis_id, b.hypothesis_id) for i, a in enumerate(hypos) for b in hypos[i + 1 :]]
    assert fake.peak > 1
    for a_id, b_id in played:
        update_elo(reference[a_id], reference[b_id], k_factor=ResearchGoal(description="Goal").elo_k_factor)
    assert [h.elo_score for h in hypos] == [reference[h.hypothesis_id].elo_score for h in hypos]
