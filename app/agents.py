import json
import math
import random
from typing import Dict, List, Tuple

# Import necessary components from other modules
from .models import REVIEW_SCORE_KEYS, ContextMemory, Hypothesis, ResearchGoal
from .utils import (
    call_llm,
    generate_unique_id,
    generate_visjs_data,
    logger,  # Use the logger configured in utils
    similarity_score,
)

# --- Agent-Specific LLM Calls (Moved from main.py/utils.py for better cohesion) ---


# Updated signature to accept temperature
def call_llm_for_generation(
    prompt: str, num_hypotheses: int = 3, temperature: float = 0.7, model: str | None = None
) -> List[Dict]:
    """Calls LLM for generating hypotheses, handling JSON parsing."""
    logger.info(
        "LLM generation called with prompt: %s, num_hypotheses: %d, temperature: %.2f",
        prompt,
        num_hypotheses,
        temperature,
    )
    full_prompt = (
        prompt
        + "\n\nPlease return the response as a JSON array of objects, where each object has a 'title' and 'text' key."
    )

    # Pass the received temperature down to the actual LLM call
    response = call_llm(full_prompt, temperature=temperature, model=model)
    logger.info("LLM generation response: %s", response)

    if response.startswith("Error:") or response.startswith("Authentication with"):
        logger.error(f"LLM generation call failed: {response}")
        return [{"title": "Error", "text": response}]

    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        hypotheses_data = json.loads(response)

        if not isinstance(hypotheses_data, list) or not all(
            isinstance(h, dict) and "title" in h and "text" in h for h in hypotheses_data
        ):
            error_message = "Invalid JSON format: Expected a list of objects with 'title' and 'text' keys."
            raise ValueError(error_message)
        logger.info("Parsed generated hypotheses: %s", hypotheses_data)
        return hypotheses_data
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Could not parse LLM generation response as JSON: %s", response, exc_info=True)
        return [{"title": "Error", "text": f"Could not parse LLM response: {e}"}]


def _empty_review_scores() -> Dict[str, int]:
    return {key: 0 for key in REVIEW_SCORE_KEYS}


def _score_to_ordinal(score: int | None) -> str:
    """Map a 1-5 review score to the legacy HIGH/MEDIUM/LOW ordinal."""
    if score is None or score <= 0:
        return "Not reviewed"
    if score <= 2:
        return "LOW"
    if score == 3:
        return "MEDIUM"
    return "HIGH"


def _coerce_score(value) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 5:
        return score
    return None


def _coerce_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
            elif item is not None:
                items.append(str(item))
        return items
    return [str(value)]


def call_llm_for_reflection(
    hypothesis_text: str,
    temperature: float = 0.5,
    model: str | None = None,
    research_goal: str | None = None,
) -> Dict:
    """Peer-review a hypothesis with numeric criterion scores and structured critique."""
    logger.info("LLM reflection called with temperature: %.2f", temperature)
    goal_block = research_goal.strip() if research_goal else "Not provided."
    criteria = ", ".join(REVIEW_SCORE_KEYS)
    prompt = (
        f"You are a scientific peer reviewer.\n\n"
        f"Research goal:\n{goal_block}\n\n"
        f"Hypothesis:\n{hypothesis_text}\n\n"
        f"Score the hypothesis from 1-5 on each criterion: {criteria}.\n"
        f"Also identify strengths, weaknesses, critical assumptions, falsification "
        f"conditions (what observation/experiment would refute it), safety/ethical "
        f"concerns, and recommended improvements.\n\n"
        f"For references, provide arXiv IDs (e.g., '2301.12345'), DOIs, or paper "
        f"titles with venues. Do not provide PubMed IDs unless this is specifically "
        f"a biomedical/life-sciences hypothesis.\n\n"
        f"Return ONLY a JSON object with keys:\n"
        f'  "review_scores": object with keys {list(REVIEW_SCORE_KEYS)} and integer values 1-5,\n'
        f'  "review_strengths": list of strings,\n'
        f'  "review_weaknesses": list of strings,\n'
        f'  "critical_assumptions": list of strings,\n'
        f'  "falsification_conditions": list of strings,\n'
        f'  "safety_ethical_concerns": list of strings,\n'
        f'  "recommended_improvements": list of strings,\n'
        f'  "comment": brief overall summary string,\n'
        f'  "references": list of strings.\n'
    )
    response = call_llm(prompt, temperature=temperature, model=model)
    logger.info("LLM reflection response for hypothesis: %s", response)

    if response.startswith("Error:") or response.startswith("Authentication with"):
        logger.error(f"LLM reflection call failed: {response}")
        return {
            "novelty_review": "Not reviewed",
            "feasibility_review": "Not reviewed",
            "review_scores": _empty_review_scores(),
            "review_strengths": [],
            "review_weaknesses": [],
            "critical_assumptions": [],
            "falsification_conditions": [],
            "safety_ethical_concerns": [],
            "recommended_improvements": [],
            "comment": f"LLM review failed: {response}",
            "references": [],
        }

    review_data = {
        "novelty_review": "MEDIUM",
        "feasibility_review": "MEDIUM",
        "review_scores": _empty_review_scores(),
        "review_strengths": [],
        "review_weaknesses": [],
        "critical_assumptions": [],
        "falsification_conditions": [],
        "safety_ethical_concerns": [],
        "recommended_improvements": [],
        "comment": "Could not parse LLM response.",
        "references": [],
    }

    try:
        parsed_data = json.loads(_strip_json_fences(response))
        raw_scores = parsed_data.get("review_scores", {})
        if not isinstance(raw_scores, dict):
            raw_scores = {}
        scores: Dict[str, int] = {}
        for key in REVIEW_SCORE_KEYS:
            coerced = _coerce_score(raw_scores.get(key))
            if coerced is not None:
                scores[key] = coerced
            else:
                # Accept legacy HIGH/MEDIUM/LOW keys if a model still emits them.
                legacy_key = f"{key}_review" if key in {"novelty", "feasibility"} else None
                legacy_val = None
                if legacy_key:
                    legacy_val = parsed_data.get(legacy_key)
                if key == "novelty":
                    legacy_val = legacy_val or parsed_data.get("novelty_review")
                if key == "feasibility":
                    legacy_val = legacy_val or parsed_data.get("feasibility_review")
                if isinstance(legacy_val, str):
                    mapping = {"HIGH": 5, "MEDIUM": 3, "LOW": 1}
                    scores[key] = mapping.get(legacy_val.upper(), 0)
                else:
                    scores[key] = 0
        review_data["review_scores"] = scores
        review_data["novelty_review"] = _score_to_ordinal(scores.get("novelty"))
        review_data["feasibility_review"] = _score_to_ordinal(scores.get("feasibility"))

        for list_key in (
            "review_strengths",
            "review_weaknesses",
            "critical_assumptions",
            "falsification_conditions",
            "safety_ethical_concerns",
            "recommended_improvements",
        ):
            review_data[list_key] = _coerce_str_list(parsed_data.get(list_key))

        comment = parsed_data.get("comment", "No comment provided.")
        review_data["comment"] = comment if isinstance(comment, str) else str(comment)
        references = parsed_data.get("references", [])
        review_data["references"] = references if isinstance(references, list) else []

        # Legacy-only payloads: allow ordinal novelty/feasibility without review_scores.
        if not any(scores.values()):
            novelty = str(parsed_data.get("novelty_review", "MEDIUM")).upper()
            feasibility = str(parsed_data.get("feasibility_review", "MEDIUM")).upper()
            if novelty in {"HIGH", "MEDIUM", "LOW"}:
                review_data["novelty_review"] = novelty
                scores["novelty"] = {"HIGH": 5, "MEDIUM": 3, "LOW": 1}[novelty]
            if feasibility in {"HIGH", "MEDIUM", "LOW"}:
                review_data["feasibility_review"] = feasibility
                scores["feasibility"] = {"HIGH": 5, "MEDIUM": 3, "LOW": 1}[feasibility]
            review_data["review_scores"] = scores

    except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as e:
        logger.warning("Error parsing LLM reflection response: %s", response, exc_info=True)
        review_data["comment"] = f"Could not parse LLM response: {e}"

    logger.info("Parsed reflection data: %s", review_data)
    return review_data


# --- Ranking Helpers (Moved from main.py) ---


def _format_hypothesis_reviews(hypothesis: Hypothesis) -> str:
    """Serialize prior reflection reviews for tournament/evolution prompts."""
    parts = []
    if hypothesis.review_scores:
        scored = ", ".join(f"{k}={v}" for k, v in hypothesis.review_scores.items() if v)
        if scored:
            parts.append(f"Scores: {scored}")
    if hypothesis.novelty_review:
        parts.append(f"Novelty: {hypothesis.novelty_review}")
    if hypothesis.feasibility_review:
        parts.append(f"Feasibility: {hypothesis.feasibility_review}")
    if hypothesis.critical_assumptions:
        parts.append("Critical assumptions: " + "; ".join(hypothesis.critical_assumptions))
    if hypothesis.falsification_conditions:
        parts.append("Falsification conditions: " + "; ".join(hypothesis.falsification_conditions))
    if hypothesis.review_weaknesses:
        parts.append("Weaknesses: " + "; ".join(hypothesis.review_weaknesses))
    if hypothesis.safety_ethical_concerns:
        parts.append("Safety/ethics: " + "; ".join(hypothesis.safety_ethical_concerns))
    if hypothesis.recommended_improvements:
        parts.append("Recommended improvements: " + "; ".join(hypothesis.recommended_improvements))
    if hypothesis.review_comments:
        parts.append(f"Comments: {'; '.join(hypothesis.review_comments)}")
    return "\n".join(parts) if parts else "No prior reviews available."


def _score_based_winner(hypoA: Hypothesis, hypoB: Hypothesis) -> Hypothesis:
    """Fallback judge: prefer summed 1-5 review_scores; else novelty+feasibility ordinals."""
    mapping = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, None: 0, "ERROR": 0, "NOT REVIEWED": 0}

    def score(h: Hypothesis) -> int:
        if h.review_scores and any(h.review_scores.values()):
            return sum(int(v) for v in h.review_scores.values() if isinstance(v, int))
        score_novelty = mapping.get(h.novelty_review, 0) if isinstance(h.novelty_review, str) else 0
        score_feasibility = mapping.get(h.feasibility_review, 0) if isinstance(h.feasibility_review, str) else 0
        return score_novelty + score_feasibility

    scoreA = score(hypoA)
    scoreB = score(hypoB)
    if scoreA > scoreB:
        return hypoA
    if scoreB > scoreA:
        return hypoB
    return random.choice([hypoA, hypoB])


def _strip_json_fences(response: str) -> str:
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()


def judge_pair(
    research_goal: ResearchGoal,
    hypothesis_a: Hypothesis,
    hypothesis_b: Hypothesis,
    reviews_a: str | None = None,
    reviews_b: str | None = None,
    temperature: float | None = None,
) -> Dict:
    """LLM tournament judge: compare two hypotheses against the real research goal.

    Returns a dict with keys: winner ("A"|"B"|"TIE"), confidence, reasoning,
    criterion_scores. On LLM/parse failure, winner is "ERROR" and reasoning
    explains the failure (caller should fall back).
    """
    if reviews_a is None:
        reviews_a = _format_hypothesis_reviews(hypothesis_a)
    if reviews_b is None:
        reviews_b = _format_hypothesis_reviews(hypothesis_b)
    judge_temp = temperature if temperature is not None else research_goal.reflection_temperature

    prompt = (
        f"You are a scientific tournament judge comparing two research hypotheses.\n\n"
        f"Research goal:\n{research_goal.description}\n\n"
        f"Hypothesis A:\nTitle: {hypothesis_a.title}\n{hypothesis_a.text}\n\n"
        f"Prior reviews for A:\n{reviews_a}\n\n"
        f"Hypothesis B:\nTitle: {hypothesis_b.title}\n{hypothesis_b.text}\n\n"
        f"Prior reviews for B:\n{reviews_b}\n\n"
        f"Compare them on:\n"
        f"- scientific soundness\n"
        f"- novelty\n"
        f"- relevance to the research goal\n"
        f"- feasibility\n"
        f"- testability / falsifiability\n"
        f"- clarity\n"
        f"- potential impact\n\n"
        f"Return ONLY a JSON object with these keys:\n"
        f'  "winner": "A" or "B" or "TIE",\n'
        f'  "confidence": number between 0.0 and 1.0,\n'
        f'  "reasoning": brief explanation of the decision,\n'
        f'  "criterion_scores": object with per-criterion {{"A": score, "B": score}} '
        f"where each score is 1-5.\n"
    )

    response = call_llm(prompt, temperature=judge_temp, model=research_goal.llm_model)
    logger.info(
        "Tournament judge response for %s vs %s: %s",
        hypothesis_a.hypothesis_id,
        hypothesis_b.hypothesis_id,
        response,
    )

    if response.startswith("Error:") or response.startswith("Authentication with"):
        logger.error("Tournament judge LLM call failed: %s", response)
        return {
            "winner": "ERROR",
            "confidence": 0.0,
            "reasoning": response,
            "criterion_scores": {},
        }

    try:
        parsed = json.loads(_strip_json_fences(response))
        winner = str(parsed.get("winner", "")).upper().strip()
        if winner not in {"A", "B", "TIE"}:
            raise ValueError(f"Invalid winner value: {parsed.get('winner')!r}")
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        criterion_scores = parsed.get("criterion_scores", {})
        if not isinstance(criterion_scores, dict):
            criterion_scores = {}
        return {
            "winner": winner,
            "confidence": confidence,
            "reasoning": reasoning,
            "criterion_scores": criterion_scores,
        }
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Could not parse tournament judge response: %s", response, exc_info=True)
        return {
            "winner": "ERROR",
            "confidence": 0.0,
            "reasoning": f"Could not parse LLM response: {e}",
            "criterion_scores": {},
        }


def run_pairwise_debate(
    hypoA: Hypothesis,
    hypoB: Hypothesis,
    research_goal: ResearchGoal | None = None,
) -> Tuple[Hypothesis | None, Dict]:
    """Judge a pair and return (winner_or_None_on_tie, judgment_dict).

    Uses an LLM comparison against the real research goal when research_goal is
    provided. Falls back to the legacy novelty+feasibility score sum on LLM
    failure. Position of A/B is randomized to reduce presentation bias.
    """
    if research_goal is None:
        winner = _score_based_winner(hypoA, hypoB)
        judgment = {
            "winner": "A" if winner is hypoA else "B",
            "confidence": 1.0,
            "reasoning": "Legacy score-based comparison (no research_goal provided).",
            "criterion_scores": {},
            "method": "score_fallback",
        }
        logger.info(
            "Debate (score fallback): %s vs %s => Winner: %s",
            hypoA.hypothesis_id,
            hypoB.hypothesis_id,
            winner.hypothesis_id,
        )
        return winner, judgment

    # Randomize presentation order to mitigate position bias, then map back.
    if random.random() < 0.5:
        presented_a, presented_b = hypoA, hypoB
        swap = False
    else:
        presented_a, presented_b = hypoB, hypoA
        swap = True

    judgment = judge_pair(research_goal, presented_a, presented_b)
    judgment["method"] = "llm"
    raw_winner = judgment["winner"]

    if raw_winner == "ERROR":
        winner = _score_based_winner(hypoA, hypoB)
        judgment["method"] = "score_fallback"
        judgment["fallback_reason"] = judgment.get("reasoning", "LLM judge failed")
        judgment["winner"] = "A" if winner is hypoA else "B"
        logger.warning(
            "Debate LLM failed; score fallback: %s vs %s => Winner: %s (%s)",
            hypoA.hypothesis_id,
            hypoB.hypothesis_id,
            winner.hypothesis_id,
            judgment["fallback_reason"][:120],
        )
        return winner, judgment

    if raw_winner == "TIE":
        judgment["winner"] = "TIE"
        logger.info(
            "Debate: %s vs %s => TIE (confidence=%.2f)",
            hypoA.hypothesis_id,
            hypoB.hypothesis_id,
            judgment.get("confidence", 0.0),
        )
        return None, judgment

    # Map presented A/B back to original hypoA/hypoB labels for the record.
    if swap:
        # presented_a was hypoB; LLM "A" means hypoB won.
        actual = hypoB if raw_winner == "A" else hypoA
        judgment["winner"] = "B" if actual is hypoB else "A"
    else:
        actual = hypoA if raw_winner == "A" else hypoB
        judgment["winner"] = "A" if actual is hypoA else "B"

    logger.info(
        "Debate: %s vs %s => Winner: %s (confidence=%.2f, method=llm)",
        hypoA.hypothesis_id,
        hypoB.hypothesis_id,
        actual.hypothesis_id,
        judgment.get("confidence", 0.0),
    )
    return actual, judgment


def update_elo(winner: Hypothesis, loser: Hypothesis, k_factor: int):
    """Updates Elo scores after a comparison, using provided k_factor."""
    # k_factor is now passed as an argument
    ratingA = winner.elo_score
    ratingB = loser.elo_score
    expectedA = 1 / (1 + math.pow(10, (ratingB - ratingA) / 400))
    expectedB = 1 - expectedA  # Or 1 / (1 + math.pow(10, (ratingA - ratingB) / 400))
    winner.elo_score = ratingA + k_factor * (1 - expectedA)
    loser.elo_score = ratingB + k_factor * (0 - expectedB)  # Loser's score update
    logger.info(
        "Updated Elo: Winner %s -> %.2f, Loser %s -> %.2f",
        winner.hypothesis_id,
        winner.elo_score,
        loser.hypothesis_id,
        loser.elo_score,
    )


# --- Evolution Helpers ---

EVOLUTION_OPERATORS = ("REFINE", "MUTATE", "HYBRIDIZE", "SIMPLIFY")

_OPERATOR_INSTRUCTIONS = {
    "REFINE": (
        "Improve the parent hypothesis: fix weaknesses from reviews, strengthen "
        "scientific soundness, clarity, testability/falsifiability, and safety. "
        "Keep the core claim but produce a clearly improved child — do not only rephrase."
    ),
    "MUTATE": (
        "Explore a different mechanism, assumption, or pathway while still addressing "
        "the same research goal. Preserve useful insights from the parent but change "
        "the underlying approach enough that this is a distinct alternative."
    ),
    "HYBRIDIZE": (
        "Synthesize a new hypothesis that combines complementary strengths of the two "
        "parents. Do not concatenate their text; produce a coherent hybrid mechanism "
        "or proposal that is stronger than either alone."
    ),
    "SIMPLIFY": (
        "Remove unnecessary assumptions, jargon, and side claims. Preserve the core "
        "testable idea in a clearer, leaner form that is easier to falsify experimentally."
    ),
}


def _tournament_feedback_for(context: ContextMemory, hypothesis_ids: List[str]) -> str:
    """Summarize recent tournament matches involving the given hypotheses."""
    wanted = set(hypothesis_ids)
    lines = []
    for result in context.tournament_results[-20:]:
        a = result.get("hypothesis_a") or result.get("winner")
        b = result.get("hypothesis_b") or result.get("loser")
        if a not in wanted and b not in wanted and result.get("winner") not in wanted:
            continue
        judgment = result.get("judgment") or {}
        if result.get("tie"):
            lines.append(
                f"Tie between {result.get('hypothesis_a')} and {result.get('hypothesis_b')}: "
                f"{judgment.get('reasoning', 'no reasoning')}"
            )
        else:
            lines.append(
                f"Winner {result.get('winner')} over {result.get('loser')}: {judgment.get('reasoning', 'no reasoning')}"
            )
    return "\n".join(lines) if lines else "No tournament feedback available yet."


def _latest_meta_review_summary(context: ContextMemory) -> str:
    if not context.meta_review_feedback:
        return "No meta-review feedback from prior cycles yet."
    latest = context.meta_review_feedback[-1]
    critiques = latest.get("meta_review_critique") or []
    next_steps = (latest.get("research_overview") or {}).get("suggested_next_steps") or []
    parts = []
    if critiques:
        parts.append("Critiques: " + "; ".join(critiques))
    if next_steps:
        parts.append("Suggested next steps: " + "; ".join(next_steps))
    return "\n".join(parts) if parts else "No meta-review feedback from prior cycles yet."


def _format_parent_block(hypothesis: Hypothesis) -> str:
    return (
        f"ID: {hypothesis.hypothesis_id}\n"
        f"Title: {hypothesis.title}\n"
        f"Text: {hypothesis.text}\n"
        f"Prior reviews:\n{_format_hypothesis_reviews(hypothesis)}"
    )


def call_llm_for_evolution(
    research_goal: ResearchGoal,
    operator: str,
    parents: List[Hypothesis],
    context: ContextMemory,
    temperature: float | None = None,
) -> Dict:
    """Ask the LLM to produce a child hypothesis for the given evolution operator."""
    if operator not in _OPERATOR_INSTRUCTIONS:
        raise ValueError(f"Unknown evolution operator: {operator}")
    if not parents:
        raise ValueError("Evolution requires at least one parent hypothesis")
    if operator == "HYBRIDIZE" and len(parents) < 2:
        raise ValueError("HYBRIDIZE requires two parent hypotheses")

    parent_ids = [p.hypothesis_id for p in parents]
    parent_blocks = "\n\n".join(
        f"Parent {chr(ord('A') + i)}:\n{_format_parent_block(p)}" for i, p in enumerate(parents)
    )
    evo_temp = temperature if temperature is not None else research_goal.generation_temperature

    prompt = (
        f"You are evolving scientific hypotheses for a research program.\n\n"
        f"Research goal:\n{research_goal.description}\n\n"
        f"Constraints: {research_goal.constraints}\n\n"
        f"Evolution operator: {operator}\n"
        f"{_OPERATOR_INSTRUCTIONS[operator]}\n\n"
        f"{parent_blocks}\n\n"
        f"Tournament feedback involving parent(s):\n"
        f"{_tournament_feedback_for(context, parent_ids)}\n\n"
        f"Meta-review feedback (from prior cycles if any):\n"
        f"{_latest_meta_review_summary(context)}\n\n"
        f"Return ONLY a JSON object with keys:\n"
        f'  "title": short title for the NEW child hypothesis,\n'
        f'  "text": full statement of the NEW child hypothesis,\n'
        f'  "reasoning": brief note of what changed and why.\n'
        f"Do not modify the parents in place; invent a distinct child.\n"
    )

    response = call_llm(prompt, temperature=evo_temp, model=research_goal.llm_model)
    logger.info("Evolution (%s) LLM response for parents %s: %s", operator, parent_ids, response)

    if response.startswith("Error:") or response.startswith("Authentication with"):
        logger.error("Evolution LLM call failed (%s): %s", operator, response)
        return {"title": "Error", "text": response, "reasoning": ""}

    try:
        parsed = json.loads(_strip_json_fences(response))
        title = parsed.get("title")
        text = parsed.get("text")
        if not isinstance(title, str) or not title.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError("Evolution response missing non-empty title/text")
        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        return {"title": title.strip(), "text": text.strip(), "reasoning": reasoning.strip()}
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Could not parse evolution response: %s", response, exc_info=True)
        return {"title": "Error", "text": f"Could not parse LLM response: {e}", "reasoning": ""}


def evolve_hypothesis(
    operator: str,
    parents: List[Hypothesis],
    research_goal: ResearchGoal,
    context: ContextMemory,
) -> Hypothesis | None:
    """Create a new child hypothesis via an evolution operator. Never mutates parents."""
    idea = call_llm_for_evolution(research_goal, operator, parents, context)
    if idea["title"] == "Error":
        logger.error("Skipping %s evolution: %s", operator, idea["text"])
        return None

    new_id = generate_unique_id("E")
    while new_id in context.hypotheses:
        new_id = generate_unique_id("E")

    child = Hypothesis(new_id, idea["title"], idea["text"])
    child.parent_ids = [p.hypothesis_id for p in parents]
    child.evolution_operator = operator
    if idea["reasoning"]:
        child.review_comments.append(f"[{operator}] {idea['reasoning']}")
    logger.info(
        "Evolved %s -> %s via %s (parents=%s)",
        [p.hypothesis_id for p in parents],
        child.hypothesis_id,
        operator,
        child.parent_ids,
    )
    return child


###############################################################################
# Agent Implementations
###############################################################################


class GenerationAgent:
    def generate_new_hypotheses(
        self, research_goal: ResearchGoal, context: ContextMemory
    ) -> Tuple[List[Hypothesis], List[str]]:
        """Generates new hypotheses using LLM, based on research_goal settings.

        Returns (hypotheses, errors). Error markers are kept out of the
        hypothesis list (they must not be ranked) but their messages are
        returned so the caller can surface the real failure cause instead of
        letting the run silently end with no rankings (see issue llnl#36).
        """
        # Use settings from research_goal object
        num_to_generate = research_goal.num_hypotheses
        gen_temp = research_goal.generation_temperature

        prompt = (
            f"Research Goal: {research_goal.description}\n"
            f"Constraints: {research_goal.constraints}\n"
            f"Existing Hypothesis IDs: {list(context.hypotheses.keys())}\n"  # Provide context
            f"Please propose {num_to_generate} novel and feasible hypotheses with rationale, avoiding duplication with existing IDs.\n"
        )
        # Pass the specific temperature and num_hypotheses
        raw_output = call_llm_for_generation(
            prompt,
            num_hypotheses=num_to_generate,
            temperature=gen_temp,
            model=research_goal.llm_model,
        )
        new_hypos = []
        errors = []
        for idea in raw_output:
            # An error response carries the real failure cause; collect it
            # (do not add it to the hypothesis list) so the caller can report it.
            if idea["title"] == "Error":
                logger.error("Hypothesis generation failed: %s", idea["text"])
                errors.append(idea["text"])
                continue

            hypo_id = generate_unique_id("G")
            # Ensure ID is unique within the current context
            while hypo_id in context.hypotheses:
                hypo_id = generate_unique_id("G")
            h = Hypothesis(hypo_id, idea["title"], idea["text"])
            logger.info("Generated hypothesis: %s", h.to_dict())
            new_hypos.append(h)
        return new_hypos, errors


class ReflectionAgent:
    def review_hypotheses(
        self, hypotheses: List[Hypothesis], context: ContextMemory, research_goal: ResearchGoal
    ) -> None:
        """Peer-review hypotheses with structured scores and scientific critique."""
        reflect_temp = research_goal.reflection_temperature

        for h in hypotheses:
            result = call_llm_for_reflection(
                h.text,
                temperature=reflect_temp,
                model=research_goal.llm_model,
                research_goal=research_goal.description,
            )
            h.novelty_review = result["novelty_review"]
            h.feasibility_review = result["feasibility_review"]
            h.review_scores = result.get("review_scores") or {}
            h.review_strengths = result.get("review_strengths") or []
            h.review_weaknesses = result.get("review_weaknesses") or []
            h.critical_assumptions = result.get("critical_assumptions") or []
            h.falsification_conditions = result.get("falsification_conditions") or []
            h.safety_ethical_concerns = result.get("safety_ethical_concerns") or []
            h.recommended_improvements = result.get("recommended_improvements") or []
            if result["comment"] not in {"Could not parse LLM response.", ""}:
                h.review_comments.append(result["comment"])
            if result["references"]:
                h.references.extend(result["references"])
            logger.info(
                "Reviewed hypothesis: %s, Novelty: %s, Feasibility: %s, Scores: %s",
                h.hypothesis_id,
                h.novelty_review,
                h.feasibility_review,
                h.review_scores,
            )


class RankingAgent:
    def run_tournament(self, hypotheses: List[Hypothesis], context: ContextMemory, research_goal: ResearchGoal) -> None:
        """Runs a pairwise tournament to rank hypotheses, using research_goal settings."""
        # Use k_factor from research_goal
        k_factor = research_goal.elo_k_factor

        if len(hypotheses) < 2:
            logger.info("Not enough hypotheses to run a tournament.")
            return

        active_hypotheses = [h for h in hypotheses if h.is_active]
        if len(active_hypotheses) < 2:
            logger.info("Not enough *active* hypotheses to run a tournament.")
            return

        random.shuffle(active_hypotheses)  # Shuffle only active ones

        # Simple round-robin: each active hypothesis debates every other active one once
        pairs = []
        for i in range(len(active_hypotheses)):
            for j in range(i + 1, len(active_hypotheses)):
                pairs.append((active_hypotheses[i], active_hypotheses[j]))

        logger.info(f"Running tournament with {len(pairs)} pairs.")
        for hA, hB in pairs:
            winner, judgment = run_pairwise_debate(hA, hB, research_goal=research_goal)
            result_record = {
                "iteration": context.iteration_number,
                "hypothesis_a": hA.hypothesis_id,
                "hypothesis_b": hB.hypothesis_id,
                "judgment": {
                    "winner": judgment.get("winner"),
                    "confidence": judgment.get("confidence"),
                    "reasoning": judgment.get("reasoning"),
                    "criterion_scores": judgment.get("criterion_scores", {}),
                    "method": judgment.get("method"),
                },
            }
            if winner is None:
                # True tie: leave Elo unchanged; still record the match.
                result_record["winner"] = None
                result_record["loser"] = None
                result_record["tie"] = True
                result_record["winner_score_after"] = hA.elo_score
                result_record["loser_score_after"] = hB.elo_score
            else:
                loser = hB if winner is hA else hA
                update_elo(winner, loser, k_factor=k_factor)
                result_record["winner"] = winner.hypothesis_id
                result_record["loser"] = loser.hypothesis_id
                result_record["tie"] = False
                result_record["winner_score_after"] = winner.elo_score
                result_record["loser_score_after"] = loser.elo_score
            context.tournament_results.append(result_record)


class EvolutionAgent:
    def evolve_hypotheses(self, context: ContextMemory, research_goal: ResearchGoal) -> List[Hypothesis]:
        """Evolve top hypotheses into new children via LLM operators.

        Operators (each creates a distinct child; parents are never mutated):
          REFINE / MUTATE / SIMPLIFY on the top-ranked hypothesis
          HYBRIDIZE on the top two when available

        Uses research goal, reflection reviews, tournament feedback, and any
        prior-cycle meta-review stored on the context.
        """
        top_k = research_goal.top_k_hypotheses
        active = context.get_active_hypotheses()
        if not active:
            logger.info("No active hypotheses to evolve.")
            return []

        top_candidates = sorted(active, key=lambda h: h.elo_score, reverse=True)[: max(1, top_k)]
        primary = top_candidates[0]
        new_hypotheses: List[Hypothesis] = []

        for operator in ("REFINE", "MUTATE", "SIMPLIFY"):
            child = evolve_hypothesis(operator, [primary], research_goal, context)
            if child is not None:
                new_hypotheses.append(child)

        if len(top_candidates) >= 2:
            hybrid = evolve_hypothesis("HYBRIDIZE", [top_candidates[0], top_candidates[1]], research_goal, context)
            if hybrid is not None:
                new_hypotheses.append(hybrid)
        else:
            logger.info("Skipping HYBRIDIZE: need at least two active hypotheses.")

        logger.info(
            "Evolution produced %d child hypotheses from top candidates %s",
            len(new_hypotheses),
            [h.hypothesis_id for h in top_candidates],
        )
        return new_hypotheses


class ProximityAgent:
    def build_proximity_graph(self, context: ContextMemory) -> Dict:
        """Builds proximity graph data based on hypothesis similarity."""
        active_hypotheses = context.get_active_hypotheses()
        adjacency = {}
        if not active_hypotheses:
            logger.info("No active hypotheses to build proximity graph.")
            return {"adjacency_graph": {}, "nodes": [], "edges": []}

        for i in range(len(active_hypotheses)):
            hypo_i = active_hypotheses[i]
            adjacency[hypo_i.hypothesis_id] = []
            for j in range(len(active_hypotheses)):
                if i == j:
                    continue
                hypo_j = active_hypotheses[j]
                if hypo_i.text and hypo_j.text:
                    sim = similarity_score(hypo_i.text, hypo_j.text)
                    adjacency[hypo_i.hypothesis_id].append({"other_id": hypo_j.hypothesis_id, "similarity": sim})
                else:
                    logger.warning(
                        f"Skipping similarity for {hypo_i.hypothesis_id} or {hypo_j.hypothesis_id} due to empty text."
                    )

        visjs_data = generate_visjs_data(adjacency)  # Use utility function
        logger.info("Built proximity graph adjacency with %d nodes.", len(active_hypotheses))
        return {"adjacency_graph": adjacency, "nodes": visjs_data["nodes"], "edges": visjs_data["edges"]}


class MetaReviewAgent:
    def summarize_and_feedback(self, context: ContextMemory, adjacency: Dict) -> Dict:
        """Summarizes research state and provides feedback."""
        active_hypotheses = context.get_active_hypotheses()
        if not active_hypotheses:
            return {
                "meta_review_critique": ["No active hypotheses."],
                "research_overview": {"top_ranked_hypotheses": [], "suggested_next_steps": []},
            }

        comment_summary = set()
        for h in active_hypotheses:
            if h.novelty_review == "LOW" or (h.review_scores.get("novelty") or 0) <= 2:
                comment_summary.add("Some ideas lack novelty.")
            if h.feasibility_review == "LOW" or (h.review_scores.get("feasibility") or 0) <= 2:
                comment_summary.add("Some ideas may have low feasibility.")
            if (h.review_scores.get("testability") or 0) <= 2:
                comment_summary.add("Some ideas are weakly testable or hard to falsify.")
            if h.safety_ethical_concerns:
                comment_summary.add("Safety or ethical concerns were raised for one or more hypotheses.")
            if h.critical_assumptions:
                comment_summary.add("Critical assumptions should be examined before investing further.")

        best_hypotheses = sorted(active_hypotheses, key=lambda h: h.elo_score, reverse=True)[:3]
        logger.info("Top hypotheses for meta-review: %s", [h.hypothesis_id for h in best_hypotheses])

        next_steps = [
            "Refine top hypotheses using recommended improvements and weaknesses from reflection.",
            "Probe critical assumptions and design tests aimed at falsification conditions.",
            "Consider exploring areas with fewer, less connected hypotheses (if any).",
            "Seek external expert feedback on top candidates.",
        ]
        if not comment_summary:
            comment_summary.add("Overall hypothesis quality seems reasonable based on automated review.")

        overview = {
            "meta_review_critique": list(comment_summary),
            "research_overview": {
                "top_ranked_hypotheses": [h.to_dict() for h in best_hypotheses],  # Use to_dict for serialization
                "suggested_next_steps": next_steps,
            },
        }
        context.meta_review_feedback.append(overview)  # Store feedback in context
        logger.info("Meta-review complete: %s", overview)
        return overview


class SupervisorAgent:
    """Orchestrates the Open AI Co-Scientist workflow."""

    def __init__(self):
        self.generation_agent = GenerationAgent()
        self.reflection_agent = ReflectionAgent()
        self.ranking_agent = RankingAgent()
        self.evolution_agent = EvolutionAgent()
        self.proximity_agent = ProximityAgent()
        self.meta_review_agent = MetaReviewAgent()

    def run_cycle(self, research_goal: ResearchGoal, context: ContextMemory) -> Dict:
        """Runs a single cycle of hypothesis generation and refinement."""
        logger.info("--- Starting Cycle %d ---", context.iteration_number + 1)
        cycle_details = {"iteration": context.iteration_number + 1, "steps": {}, "meta_review": {}}

        # 1. Generation
        logger.info("Step 1: Generation")
        new_hypotheses, generation_errors = self.generation_agent.generate_new_hypotheses(research_goal, context)
        for nh in new_hypotheses:
            context.add_hypothesis(nh)  # Add to central context
        cycle_details["steps"]["generation"] = {"hypotheses": [h.to_dict() for h in new_hypotheses]}

        # Propagate LLM errors to top-level errors field for frontend display, so a
        # generation failure surfaces its real cause instead of an empty ranking.
        if generation_errors:
            cycle_details["errors"] = generation_errors

        # Get all active hypotheses for subsequent steps
        active_hypos = context.get_active_hypotheses()

        # 2. Reflection
        logger.info("Step 2: Reflection")
        self.reflection_agent.review_hypotheses(active_hypos, context, research_goal)  # Pass research_goal
        cycle_details["steps"]["reflection"] = {"hypotheses": [h.to_dict() for h in active_hypos]}

        # 3. Ranking (Tournament 1)
        logger.info("Step 3: Ranking 1")
        self.ranking_agent.run_tournament(active_hypos, context, research_goal)  # Pass research_goal
        cycle_details["steps"]["ranking1"] = {"hypotheses": [h.to_dict() for h in active_hypos]}

        # 4. Evolution
        logger.info("Step 4: Evolution")
        evolved_hypotheses = self.evolution_agent.evolve_hypotheses(context, research_goal)  # Pass research_goal
        if evolved_hypotheses:
            for eh in evolved_hypotheses:
                context.add_hypothesis(eh)
            logger.info("Step 4a: Reviewing Evolved Hypotheses")
            self.reflection_agent.review_hypotheses(evolved_hypotheses, context, research_goal)  # Pass research_goal
            active_hypos = context.get_active_hypotheses()  # Update active list
            cycle_details["steps"]["evolution"] = {"hypotheses": [h.to_dict() for h in evolved_hypotheses]}
            # Add explicit step for reviewing evolved hypotheses AFTER evolution
            cycle_details["steps"]["reflection_evolved"] = {"hypotheses": [h.to_dict() for h in evolved_hypotheses]}
        else:
            cycle_details["steps"]["evolution"] = {"hypotheses": []}

        # 5. Ranking (Tournament 2 - includes evolved)
        logger.info("Step 5: Ranking 2")
        self.ranking_agent.run_tournament(active_hypos, context, research_goal)  # Pass research_goal
        cycle_details["steps"]["ranking2"] = {"hypotheses": [h.to_dict() for h in active_hypos]}

        # Ensure context.active_hypotheses reflects the final ranked hypotheses for meta-review
        # Use all hypotheses from the final ranking step (not just active_hypos, which may be filtered)
        final_ranked_hypos = [h for h in active_hypos]
        context.active_hypotheses = {h.hypothesis_id: h for h in final_ranked_hypos}

        # 6. Proximity Analysis
        logger.info("Step 6: Proximity Analysis")
        proximity_result = self.proximity_agent.build_proximity_graph(context)  # Pass context
        cycle_details["steps"]["proximity"] = {
            "adjacency_graph": proximity_result["adjacency_graph"],
            "nodes": proximity_result["nodes"],
            "edges": proximity_result["edges"],
        }

        # 7. Meta-review
        logger.info("Step 7: Meta-Review")
        overview = self.meta_review_agent.summarize_and_feedback(context, proximity_result["adjacency_graph"])
        cycle_details["meta_review"] = overview
        # Add meta-review to steps for consistency
        cycle_details["steps"]["meta_review"] = overview

        # Increment iteration number at the end of the cycle
        context.iteration_number += 1
        logger.info("--- Cycle %d Complete ---", context.iteration_number)
        return cycle_details
