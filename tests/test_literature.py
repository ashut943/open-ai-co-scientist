"""Offline tests for multi-source literature search and user references.

HTTP is mocked with `responses`; the autouse fixture in conftest.py disables
literature lookups, so tests that exercise them re-enable via `enabled`.
"""

from unittest.mock import patch

import pytest
import responses

from app.agents import GenerationAgent, SupervisorAgent
from app.models import ContextMemory, ResearchGoal
from app.tools import literature as lit
from app.utils import classify_llm_error

CROSSREF_WORK = {
    "message": {
        "DOI": "10.1038/nature12373",
        "title": ["Nanometre-scale thermometry in a living cell"],
        "container-title": ["Nature"],
        "issued": {"date-parts": [[2013, 7, 31]]},
        "author": [{"given": "G.", "family": "Kucsko"}, {"given": "P.", "family": "Maurer"}],
        "abstract": "<jats:p>We report thermometry.</jats:p>",
        "URL": "https://doi.org/10.1038/nature12373",
    }
}

PUBMED_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
<Journal><Title>J Neuro</Title><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Amyloid clearance in sleep</ArticleTitle>
<Abstract><AbstractText>Background.</AbstractText><AbstractText>Results.</AbstractText></Abstract>
<AuthorList><Author><LastName>Lee</LastName><Initials>K</Initials></Author></AuthorList>
<ELocationID EIdType="doi">10.5555/amyloid</ELocationID>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.delenv("CO_SCIENTIST_DISABLE_LITERATURE", raising=False)
    for var in ("OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "NCBI_API_KEY", "LITERATURE_CONTACT_EMAIL"):
        monkeypatch.delenv(var, raising=False)


# --- Sources ---


@responses.activate
def test_openalex_parses_results_and_rebuilds_abstract(enabled, monkeypatch):
    monkeypatch.setenv("LITERATURE_CONTACT_EMAIL", "me@example.org")
    responses.add(
        responses.GET,
        lit.OPENALEX_URL,
        json={
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1000/abc",
                    "title": "Perovskite tandem cells",
                    "publication_year": 2023,
                    "primary_location": {"source": {"display_name": "Nature Energy"}},
                    "authorships": [{"author": {"display_name": "A. Smith"}}],
                    "abstract_inverted_index": {"Tandem": [0], "work": [2], "cells": [1]},
                }
            ]
        },
    )

    papers = lit.search_openalex("perovskite tandem", 3)

    assert papers[0]["title"] == "Perovskite tandem cells"
    assert papers[0]["abstract"] == "Tandem cells work"
    assert papers[0]["doi"] == "10.1000/abc"
    assert papers[0]["year"] == 2023
    assert papers[0]["venue"] == "Nature Energy"
    assert papers[0]["url"] == "https://doi.org/10.1000/abc"
    assert "mailto=me%40example.org" in responses.calls[0].request.url


@responses.activate
def test_semantic_scholar_parses_results_and_sends_key_header(enabled, monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-test-key")
    responses.add(
        responses.GET,
        lit.SEMANTIC_SCHOLAR_URL,
        json={
            "data": [
                {
                    "paperId": "p1",
                    "title": "Graph transformers",
                    "abstract": "We study graphs.",
                    "year": 2021,
                    "venue": "NeurIPS",
                    "authors": [{"name": "X. Li"}],
                    "externalIds": {"DOI": "10.1000/graph"},
                    "url": "https://www.semanticscholar.org/paper/p1",
                }
            ]
        },
    )

    papers = lit.search_semantic_scholar("graph transformers", 3)

    assert papers[0]["doi"] == "10.1000/graph"
    assert papers[0]["authors"] == ["X. Li"]
    assert responses.calls[0].request.headers["x-api-key"] == "s2-test-key"


@responses.activate
def test_pubmed_search_fetches_and_parses_abstracts(enabled):
    responses.add(responses.GET, lit.PUBMED_SEARCH_URL, json={"esearchresult": {"idlist": ["123"]}})
    responses.add(responses.GET, lit.PUBMED_FETCH_URL, body=PUBMED_XML)

    papers = lit.search_pubmed("amyloid sleep", 3)

    assert papers[0]["title"] == "Amyloid clearance in sleep"
    assert papers[0]["abstract"] == "Background. Results."
    assert papers[0]["doi"] == "10.5555/amyloid"
    assert papers[0]["authors"] == ["Lee K"]
    assert papers[0]["year"] == 2019
    assert papers[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/123/"


@responses.activate
def test_crossref_resolves_doi_and_returns_none_when_missing(enabled):
    responses.add(responses.GET, lit.CROSSREF_URL + "10.1038/nature12373", json=CROSSREF_WORK)
    responses.add(responses.GET, lit.CROSSREF_URL + "10.9999/missing", status=404)

    paper = lit.fetch_crossref("10.1038/nature12373")

    assert paper["title"] == "Nanometre-scale thermometry in a living cell"
    assert paper["abstract"] == "We report thermometry."
    assert paper["year"] == 2013
    assert lit.fetch_crossref("10.9999/missing") is None


# --- LiteratureSearch ---


@responses.activate
def test_search_merges_sources_and_reports_failures_without_leaking_keys(enabled, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "oa-secret-key")
    responses.add(responses.GET, lit.OPENALEX_URL, status=429)
    monkeypatch.setitem(
        lit.SOURCES,
        "fake",
        lambda query, n, timeout: [
            lit._paper("fake", "1", "Paper A", doi="10.1000/a"),
            lit._paper("fake", "2", "Paper A again", doi="10.1000/A"),
            lit._paper("fake", "3", "Paper B"),
        ],
    )

    papers, errors = lit.LiteratureSearch(sources=["openalex", "fake"]).search("tandem cells")

    assert [p["title"] for p in papers] == ["Paper A", "Paper B"]
    assert len(errors) == 1
    assert "rate limited" in errors[0]
    assert "oa-secret-key" not in errors[0]
    assert classify_llm_error(errors[0]) == "Literature search unavailable"


def test_search_drops_preprint_duplicate_of_journal_paper(enabled, monkeypatch):
    title = "Quantum thermodynamic uncertainty relation under feedback control"
    monkeypatch.setitem(
        lit.SOURCES, "journal", lambda q, n, t: [lit._paper("openalex", "W1", title, doi="10.1103/PhysRevE.1")]
    )
    monkeypatch.setitem(
        lit.SOURCES,
        "preprint",
        lambda q, n, t: [lit._paper("arxiv", "2301.1", title + ".", doi="10.48550/arXiv.2301.1")],
    )

    papers, errors = lit.LiteratureSearch(sources=["journal", "preprint"]).search("tur feedback")

    assert errors == []
    assert [p["source"] for p in papers] == ["openalex"]


def test_arxiv_query_requires_all_significant_words():
    assert lit._arxiv_query("Thermodynamic uncertainty relation under feedback-control") == (
        "all:Thermodynamic AND all:uncertainty AND all:relation AND all:feedback AND all:control"
    )
    assert lit._arxiv_query("of the") == ""


def test_search_is_skipped_when_disabled():
    with patch.object(lit, "search_openalex", side_effect=AssertionError("network used")):
        assert lit.LiteratureSearch(sources=["openalex"]).search("anything") == ([], [])


def test_hypothesis_query_uses_title_and_pads_short_titles_with_text():
    assert lit.hypothesis_query("Perovskite tandem solar cell efficiency", "ignored") == (
        "Perovskite tandem solar cell efficiency"
    )
    assert lit.hypothesis_query("Tandems", "Stack perovskite on silicon.") == "Tandems Stack perovskite on silicon"


# --- User references ---


@responses.activate
def test_resolve_references_mixes_papers_and_notes(enabled, monkeypatch):
    responses.add(responses.GET, lit.CROSSREF_URL + "10.1038/nature12373", json=CROSSREF_WORK)
    monkeypatch.setattr(
        lit, "fetch_arxiv", lambda arxiv_id, timeout=10: lit._paper("arxiv", arxiv_id, "Attention", "Abstract.", 2017)
    )

    refs, errors = lit.resolve_references(
        ["https://doi.org/10.1038/nature12373", "arXiv:1706.03762", "Our pilot showed a 3% gain"]
    )

    assert errors == []
    assert [r["label"] for r in refs] == ["U1", "U2", "U3"]
    assert refs[0]["kind"] == "paper" and refs[0]["source"] == "crossref"
    assert refs[1]["id"] == "1706.03762"
    assert refs[2] == {
        "label": "U3",
        "kind": "note",
        "text": "Our pilot showed a 3% gain",
        "raw": "Our pilot showed a 3% gain",
        "doi": None,
    }


@responses.activate
def test_unresolvable_reference_is_reported_and_kept_as_note(enabled):
    responses.add(responses.GET, lit.CROSSREF_URL + "10.9999/missing", status=404)

    refs, errors = lit.resolve_references(["10.9999/missing"])

    assert refs[0]["kind"] == "note"
    assert errors and errors[0].startswith("Could not resolve reference U1")
    assert classify_llm_error(errors[0]) == "Reference could not be resolved"


def test_resolve_references_offline_keeps_lines_as_notes():
    refs, errors = lit.resolve_references(["10.1038/nature12373"])
    assert errors == []
    assert refs[0]["kind"] == "note"
    assert refs[0]["doi"] == "10.1038/nature12373"


def test_parse_reference_lines_drops_blanks_and_caps_count():
    lines = lit.parse_reference_lines(" a \n\n b \n" + "\n".join(f"x{i}" for i in range(30)))
    assert lines[:2] == ["a", "b"]
    assert len(lines) == lit.MAX_USER_REFERENCES


# --- Citation grounding ---


def test_ground_references_keeps_only_verifiable_citations():
    retrieved = [
        lit._paper("openalex", "W1", "Paper A", year=2020, authors=["A. Smith"], doi="10.1000/a"),
        lit._paper("openalex", "W2", "Paper B", year=2021),
    ]
    user_refs = [{"label": "U1", "kind": "note", "text": "Pilot data", "raw": "Pilot data"}]

    kept, dropped = lit.ground_references(
        ["P2", "[U1]", "doi:10.1000/A", "10.2000/real", "Made-up et al. 2021", "P9"],
        retrieved,
        user_refs,
        verify_doi=lambda doi: doi == "10.2000/real",
    )

    # The user's note guides prompts but is not a reference, and is not counted as dropped.
    assert kept == [lit.citation(retrieved[1]), lit.citation(retrieved[0]), "doi:10.2000/real"]
    assert dropped == 2


# --- Agents use the user's references ---


def test_user_references_reach_generation_prompt():
    goal = ResearchGoal("Goal", num_hypotheses=1)
    goal.resolved_references = [{"label": "U1", "kind": "note", "text": "Pilot showed a 3% gain", "raw": "x"}]

    with patch("app.agents.call_llm", return_value='[{"title": "H", "text": "t"}]') as mock_call:
        GenerationAgent().generate_new_hypotheses(goal, ContextMemory())

    assert "[U1] Note from the user: Pilot showed a 3% gain" in mock_call.call_args.args[0]


def test_generation_asks_for_and_keeps_search_keywords():
    goal = ResearchGoal("Goal", num_hypotheses=1)
    reply = '[{"title": "IKUR", "text": "t", "search_keywords": "uncertainty relation, feedback,  Markov jump"}]'

    with patch("app.agents.call_llm", return_value=reply) as mock_call:
        hypotheses, errors = GenerationAgent().generate_new_hypotheses(goal, ContextMemory())

    assert "search_keywords" in mock_call.call_args.args[0]
    assert errors == []
    assert hypotheses[0].search_keywords == '"uncertainty relation" "feedback" "Markov jump"'


class _FakeArxivTool:
    def __init__(self, results_for):
        self.results_for, self.queries = results_for, []
        self.client = self

    def results(self, search):
        self.queries.append(search.query)
        return self.results_for(search.query)

    def _format_paper(self, item):
        return item


def test_arxiv_retries_with_fewer_terms_when_strict_query_is_empty(monkeypatch):
    paper = {"arxiv_id": "2401.1", "title": "Feedback TUR", "published": "2024-01-01"}
    tool = _FakeArxivTool(lambda q: [] if q.count("AND") >= 3 else [paper])
    monkeypatch.setattr(lit, "_get_arxiv_tool", lambda: tool)

    papers = lit.search_arxiv("hybrid informational kinetic uncertainty relation feedback motors", 3)

    assert [p["title"] for p in papers] == ["Feedback TUR"]
    assert tool.queries == [
        "all:hybrid AND all:informational AND all:kinetic AND all:uncertainty AND all:relation AND all:feedback",
        "all:hybrid AND all:informational AND all:kinetic",
    ]


def test_arxiv_searches_key_phrases_and_drops_the_last_ones_first(monkeypatch):
    paper = {"arxiv_id": "2401.2", "title": "Activity bounds", "published": "2024-01-01"}
    tool = _FakeArxivTool(lambda q: [] if "Markov" in q else [paper])
    monkeypatch.setattr(lit, "_get_arxiv_tool", lambda: tool)

    papers = lit.search_arxiv('"dynamical activity" "transfer entropy" "Markov jump process"', 3)

    assert [p["title"] for p in papers] == ["Activity bounds"]
    assert tool.queries == [
        'all:"dynamical activity" AND all:"transfer entropy" AND all:"Markov jump process"',
        'all:"dynamical activity" AND all:"transfer entropy"',
    ]


def test_arxiv_never_loosens_below_two_phrases(monkeypatch):
    tool = _FakeArxivTool(lambda q: [])
    monkeypatch.setattr(lit, "_get_arxiv_tool", lambda: tool)

    assert lit.search_arxiv('"dynamical activity" "transfer entropy"', 3) == []
    assert tool.queries == ['all:"dynamical activity" AND all:"transfer entropy"']


@responses.activate
def test_relevance_ranked_sources_get_the_phrases_without_quotes(enabled):
    responses.add(responses.GET, lit.OPENALEX_URL, json={"results": []})

    lit.search_openalex('"dynamical activity" "transfer entropy"', 3)

    assert "search=dynamical+activity+transfer+entropy" in responses.calls[0].request.url


def test_arxiv_query_drops_numbers_and_single_letters():
    assert (
        lit._arxiv_query("Level-2.5 large deviation bound") == "all:Level AND all:large AND all:deviation AND all:bound"
    )


@responses.activate
def test_missing_crossref_abstract_is_filled_from_openalex(enabled):
    work = {"message": {**CROSSREF_WORK["message"], "abstract": None}}
    responses.add(responses.GET, lit.CROSSREF_URL + "10.1038/nature12373", json=work)
    responses.add(
        responses.GET,
        lit.OPENALEX_URL + "/doi:10.1038/nature12373",
        json={"abstract_inverted_index": {"Thermometry": [0], "works": [1]}},
    )

    refs, errors = lit.resolve_references(["10.1038/nature12373"])

    assert errors == []
    assert refs[0]["abstract"] == "Thermometry works"


@responses.activate
def test_openalex_abstract_lookup_failure_does_not_fail_the_reference(enabled):
    work = {"message": {**CROSSREF_WORK["message"], "abstract": None}}
    responses.add(responses.GET, lit.CROSSREF_URL + "10.1038/nature12373", json=work)
    responses.add(responses.GET, lit.OPENALEX_URL + "/doi:10.1038/nature12373", status=500)

    refs, errors = lit.resolve_references(["10.1038/nature12373"])

    assert errors == []
    assert refs[0]["kind"] == "paper" and refs[0]["abstract"] == ""


def test_run_cycle_resolves_user_references_once():
    goal = ResearchGoal("Goal", num_hypotheses=1, user_references=["Pilot showed a 3% gain"])

    with (
        patch("app.agents.call_llm", return_value="Error: Rate limit exceeded: slow down"),
        patch(
            "app.agents.ProximityAgent.build_proximity_graph",
            return_value={"adjacency_graph": {}, "nodes": [], "edges": []},
        ),
        patch("app.agents.resolve_references", wraps=lit.resolve_references) as resolver,
    ):
        supervisor, context = SupervisorAgent(), ContextMemory()
        details = supervisor.run_cycle(goal, context)
        supervisor.run_cycle(goal, context)

    assert resolver.call_count == 1
    assert details["user_references"][0]["label"] == "U1"
    assert goal.resolved_references[0]["kind"] == "note"


def test_arxiv_wait_budget_covers_searches_queued_by_concurrent_reviews(monkeypatch):
    monkeypatch.setitem(lit.config, "llm_max_concurrency", 4)

    assert lit._wait_budget("openalex", 10) == 15
    assert lit._wait_budget("arxiv", 10) == 60
