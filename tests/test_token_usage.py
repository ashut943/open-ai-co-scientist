"""Token usage is counted per cycle step (also across parallel calls) and shown with a cost estimate."""

import importlib.util
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import app.utils as utils
from app import agents
from app.agents import SupervisorAgent, _parallel_map
from app.models import ContextMemory, ResearchGoal
from app.run_store import render_report, save_run
from app.utils import TokenUsage, call_llm, estimate_cost, track_usage, usage_step


@pytest.fixture(scope="module")
def gradio_app_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("gradio_app_usage", os.path.join(repo_root, "app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completion(content="ok", prompt_tokens=100, completion_tokens=40, reasoning_tokens=10):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    completion.usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return completion


def test_call_llm_records_usage_under_the_current_step(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    meter = TokenUsage()
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion()
        with track_usage(meter):
            with usage_step("reflection"):
                call_llm("p1")
                call_llm("p2")
            call_llm("p3")
        call_llm("outside the meter")

    usage = meter.to_dict()
    assert usage["steps"]["reflection"] == {
        "calls": 2,
        "input_tokens": 200,
        "output_tokens": 80,
        "reasoning_tokens": 20,
    }
    assert usage["steps"]["other"]["calls"] == 1
    assert usage["total"]["input_tokens"] == 300


def test_usage_from_parallel_workers_is_attributed_to_the_callers_step(monkeypatch):
    monkeypatch.setitem(agents.config, "llm_max_concurrency", 4)
    meter = TokenUsage()

    def fake_llm_call(item):
        utils._record_usage(_completion(prompt_tokens=item, completion_tokens=1))
        return item

    with track_usage(meter), usage_step("tournament"):
        assert _parallel_map(fake_llm_call, [1, 2, 3, 4]) == [1, 2, 3, 4]

    assert meter.to_dict()["steps"] == {
        "tournament": {"calls": 4, "input_tokens": 10, "output_tokens": 4, "reasoning_tokens": 40}
    }


def test_missing_usage_field_is_ignored():
    meter = TokenUsage()
    completion = _completion()
    completion.usage = None
    with track_usage(meter):
        utils._record_usage(completion)
    assert meter.to_dict()["total"]["calls"] == 0


def test_anthropic_style_usage_field_names():
    meter = TokenUsage()
    completion = MagicMock()
    completion.usage = SimpleNamespace(input_tokens=5000, output_tokens=800, prompt_tokens=None, completion_tokens=None)
    with track_usage(meter), usage_step("generation"):
        utils._record_usage(completion)
    assert meter.to_dict()["steps"]["generation"]["input_tokens"] == 5000
    assert meter.to_dict()["steps"]["generation"]["output_tokens"] == 800


def test_cached_prompt_tokens_are_added_when_prompt_tokens_is_a_stub():
    """Parley + Opus reported prompt_tokens=4 with the real mass in cached_tokens."""
    meter = TokenUsage()
    completion = MagicMock()
    completion.usage = SimpleNamespace(
        prompt_tokens=4,
        completion_tokens=7309,
        total_tokens=22048,
        prompt_tokens_details=SimpleNamespace(cached_tokens=22033),
        completion_tokens_details=None,
    )
    with track_usage(meter), usage_step("generation"):
        utils._record_usage(completion)
    assert meter.to_dict()["steps"]["generation"]["input_tokens"] == 22037  # 4 + 22033
    assert meter.to_dict()["steps"]["generation"]["output_tokens"] == 7309


def test_total_tokens_recovers_input_when_prompt_fields_are_wrong():
    assert utils._extract_token_counts({"prompt_tokens": 4, "completion_tokens": 100, "total_tokens": 5100}) == (
        5000,
        100,
        0,
    )


def test_run_cycle_stores_cycle_and_session_totals():
    context = ContextMemory()
    supervisor = SupervisorAgent()

    def fake_cycle(goal, ctx):
        with usage_step("generation"):
            utils._record_usage(_completion(prompt_tokens=1000, completion_tokens=200))
        return {"iteration": 1, "steps": {}}

    with patch.object(supervisor, "_run_cycle_steps", side_effect=fake_cycle):
        supervisor.run_cycle(ResearchGoal(description="g"), context)
        details = supervisor.run_cycle(ResearchGoal(description="g"), context)

    usage = details["token_usage"]
    assert usage["total"]["input_tokens"] == 1000
    assert usage["session_total"]["input_tokens"] == 2000
    assert usage["session_total"]["calls"] == 2
    assert len(context.token_usage) == 2


def test_estimate_cost_uses_configured_prices(monkeypatch):
    monkeypatch.setitem(utils.config, "token_prices_per_million", {"m": [0.75, 3.75]})
    totals = {"input_tokens": 1_000_000, "output_tokens": 200_000}
    assert estimate_cost(totals, "m") == pytest.approx(1.5)
    assert estimate_cost(totals, "unpriced-model") is None


def test_status_lines_show_tokens_and_cost(gradio_app_module, monkeypatch):
    monkeypatch.setitem(utils.config, "token_prices_per_million", {"m": [1.0, 2.0]})
    monkeypatch.setattr(gradio_app_module, "get_configured_model", lambda: "m")
    totals = {"calls": 3, "input_tokens": 500_000, "output_tokens": 100_000, "reasoning_tokens": 40_000}
    text = gradio_app_module.token_usage_lines({"total": totals, "session_total": totals})

    assert "Tokens this cycle: 500,000 in / 100,000 out (40,000 reasoning) over 3 calls ≈ $0.70" in text
    assert "Tokens this session:" in text
    assert gradio_app_module.token_usage_lines(None) == ""


def test_report_includes_token_usage_table(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    counts = {"calls": 2, "input_tokens": 12345, "output_tokens": 678, "reasoning_tokens": 9}
    report = render_report(
        save_run(
            research_goal=ResearchGoal(description="g"),
            cycle_details={
                "iteration": 1,
                "steps": {},
                "token_usage": {"steps": {"reflection": counts}, "total": counts, "session_total": counts},
            },
            status="done",
            references_html="",
            results_html="",
            run_id="run-usage",
        )
    )

    assert "<h2>Token Usage</h2>" in report
    assert "<td>reflection</td><td>2</td><td>12,345</td><td>678</td><td>9</td>" in report
    assert "Session so far" in report


def test_goal_search_uses_top_hypotheses_phrases(gradio_app_module):
    details = {
        "steps": {
            "ranking": {
                "hypotheses": [
                    {"elo_score": 1100, "search_keywords": '"low ranked" "other"'},
                    {"elo_score": 1300, "search_keywords": '"escape behavior" "point process"'},
                    {"elo_score": 1250, "search_keywords": '"point process" "looming stimulus"'},
                ]
            }
        }
    }
    query = gradio_app_module.goal_search_query("Long goal. " * 200, details)
    assert query == '"escape behavior" "point process" "looming stimulus" "low ranked"'


def test_goal_search_falls_back_to_short_goal_query(gradio_app_module):
    goal = "Explain collective escape in Danionella cerebrum schools. " + "Background detail. " * 300
    query = gradio_app_module.goal_search_query(goal, {"steps": {}})
    assert 0 < len(query) <= 200
    assert "Danionella" in query
    assert "Background" not in query
