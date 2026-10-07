"""Multi-source literature search and user-reference resolution.

Sources: OpenAlex, Semantic Scholar, PubMed (E-utilities), arXiv, and Crossref
(DOI lookup/verification). Every function returns normalized paper dicts:

    {"source", "id", "title", "abstract", "year", "venue", "authors", "doi", "url"}

Optional keys/contact come only from the environment: OPENALEX_API_KEY,
SEMANTIC_SCHOLAR_API_KEY, NCBI_API_KEY, LITERATURE_CONTACT_EMAIL. Set
CO_SCIENTIST_DISABLE_LITERATURE=1 to turn all network lookups off.
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, Dict, List, Optional, Tuple

import requests

from ..config import config
from ..utils import redact_secrets

logger = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CROSSREF_URL = "https://api.crossref.org/works/"

DISABLE_ENV = "CO_SCIENTIST_DISABLE_LITERATURE"
MAX_USER_REFERENCES = 20

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([A-Za-z\-.]*/?\d{4}\.?\d{3,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(r"^(?:arxiv:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
_PMID_RE = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/(\d+))|(?:^PMID:?\s*(\d+)$)", re.IGNORECASE)


def literature_settings() -> Dict:
    settings = {
        "enabled": True,
        "sources": ["openalex", "arxiv"],
        "results_per_source": 3,
        "max_papers_in_prompt": 6,
        "timeout_seconds": 10,
        "verify_cited_dois": True,
    }
    settings.update(config.get("literature_search") or {})
    return settings


def literature_enabled() -> bool:
    if os.getenv(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return False
    return bool(literature_settings().get("enabled", True))


# --- HTTP helpers ---


def _contact_email() -> Optional[str]:
    return os.getenv("LITERATURE_CONTACT_EMAIL") or None


def _get(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: float = 10):
    agent = "open-ai-co-scientist"
    if _contact_email():
        agent += f" (mailto:{_contact_email()})"
    response = requests.get(url, params=params, headers={"User-Agent": agent, **(headers or {})}, timeout=timeout)
    response.raise_for_status()
    return response


def _clean(text) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip()


def _normalize_doi(doi) -> Optional[str]:
    if not doi:
        return None
    doi = str(doi).strip()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.IGNORECASE)
    return doi.rstrip(".,;)") or None


def _paper(source, pid, title, abstract="", year=None, venue="", authors=None, doi=None, url="") -> Dict:
    doi = _normalize_doi(doi)
    return {
        "source": source,
        "id": str(pid or ""),
        "title": _clean(title),
        "abstract": _clean(abstract),
        "year": int(year) if str(year or "").isdigit() else None,
        "venue": _clean(venue),
        "authors": [a for a in (authors or []) if a][:3],
        "doi": doi,
        "url": url or (f"https://doi.org/{doi}" if doi else ""),
    }


# --- Sources ---


def _openalex_abstract(inverted) -> str:
    if not isinstance(inverted, dict):
        return ""
    positioned = [(i, word) for word, indexes in inverted.items() for i in indexes]
    return " ".join(word for _, word in sorted(positioned))


def _openalex_params(extra: Dict) -> Dict:
    params = dict(extra)
    if os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = os.getenv("OPENALEX_API_KEY")
    if _contact_email():
        params["mailto"] = _contact_email()
    return params


def openalex_abstract_for_doi(doi: str, timeout: float = 10) -> str:
    """Abstract for a DOI from OpenAlex ('' if it has none). Crossref often lacks abstracts."""
    try:
        work = _get(
            f"{OPENALEX_URL}/doi:{requests.utils.quote(doi, safe='/')}",
            params=_openalex_params({"select": "abstract_inverted_index"}),
            timeout=timeout,
        ).json()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return ""
        raise
    return _clean(_openalex_abstract(work.get("abstract_inverted_index")))


def search_openalex(query: str, max_results: int = 3, timeout: float = 10) -> List[Dict]:
    params = _openalex_params(
        {
            "search": _plain_query(query),
            "per_page": max_results,
            "select": "id,doi,title,publication_year,primary_location,authorships,abstract_inverted_index",
        }
    )
    data = _get(OPENALEX_URL, params=params, timeout=timeout).json()
    papers = []
    for work in data.get("results") or []:
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        authors = [(a.get("author") or {}).get("display_name") for a in work.get("authorships") or []]
        papers.append(
            _paper(
                "openalex",
                work.get("id"),
                work.get("title"),
                _openalex_abstract(work.get("abstract_inverted_index")),
                work.get("publication_year"),
                source.get("display_name") or "",
                authors,
                work.get("doi"),
                location.get("landing_page_url") or "",
            )
        )
    return papers


def search_semantic_scholar(query: str, max_results: int = 3, timeout: float = 10) -> List[Dict]:
    headers = {}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    params = {
        "query": _plain_query(query),
        "limit": max_results,
        "fields": "title,abstract,year,venue,authors,externalIds,url",
    }
    data = _get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=timeout).json()
    papers = []
    for item in data.get("data") or []:
        external = item.get("externalIds") or {}
        papers.append(
            _paper(
                "semantic_scholar",
                item.get("paperId"),
                item.get("title"),
                item.get("abstract"),
                item.get("year"),
                item.get("venue"),
                [a.get("name") for a in item.get("authors") or []],
                external.get("DOI"),
                item.get("url") or "",
            )
        )
    return papers


def _pubmed_params(extra: Dict) -> Dict:
    params = {"db": "pubmed", "tool": "open-ai-co-scientist", **extra}
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.getenv("NCBI_API_KEY")
    if _contact_email():
        params["email"] = _contact_email()
    return params


def fetch_pubmed(pmids: List[str], timeout: float = 10) -> List[Dict]:
    if not pmids:
        return []
    xml_text = _get(
        PUBMED_FETCH_URL, params=_pubmed_params({"id": ",".join(pmids), "retmode": "xml"}), timeout=timeout
    ).text
    papers = []
    for article in ET.fromstring(xml_text).iter("PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue
        pmid = citation.findtext("PMID", "")
        art = citation.find("Article")
        if art is None:
            continue
        abstract = " ".join("".join(node.itertext()) for node in art.iter("AbstractText"))
        authors = []
        for author in art.iter("Author"):
            name = " ".join(filter(None, [author.findtext("LastName"), author.findtext("Initials")]))
            if name:
                authors.append(name)
        doi = None
        for loc in art.iter("ELocationID"):
            if loc.get("EIdType") == "doi":
                doi = loc.text
        title_node = art.find("ArticleTitle")
        title = "".join(title_node.itertext()) if title_node is not None else ""
        papers.append(
            _paper(
                "pubmed",
                pmid,
                title,
                abstract,
                art.findtext("Journal/JournalIssue/PubDate/Year"),
                art.findtext("Journal/Title", ""),
                authors,
                doi,
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
        )
    return papers


def search_pubmed(query: str, max_results: int = 3, timeout: float = 10) -> List[Dict]:
    data = _get(
        PUBMED_SEARCH_URL,
        params=_pubmed_params({"term": query, "retmax": max_results, "retmode": "json"}),
        timeout=timeout,
    ).json()
    return fetch_pubmed((data.get("esearchresult") or {}).get("idlist") or [], timeout=timeout)


_arxiv_tool = None


def _get_arxiv_tool():
    # One shared client so arXiv's one-request-per-3-seconds delay applies across searches.
    global _arxiv_tool
    if _arxiv_tool is None:
        from .arxiv_search import ArxivSearchTool

        _arxiv_tool = ArxivSearchTool()
    return _arxiv_tool


def _from_arxiv_dict(item: Dict) -> Dict:
    venue = "arXiv" + (f" ({item['journal_ref']})" if item.get("journal_ref") else "")
    return _paper(
        "arxiv",
        item.get("arxiv_id"),
        item.get("title"),
        item.get("abstract"),
        (item.get("published") or "")[:4],
        venue,
        item.get("authors"),
        item.get("doi"),
        item.get("arxiv_url") or "",
    )


_STOPWORDS = set("a an and as at by for from in into is of on or the to under using via with without".split())
_ARXIV_TERM_STEPS = (6, 3)  # all of the first 6 significant words, then retry with 3 if nothing matches


def _arxiv_terms(query: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", query or "")
    return [w for w in words if len(w) > 1 and w.lower() not in _STOPWORDS]


def _arxiv_query(query: str, max_terms: int = _ARXIV_TERM_STEPS[0]) -> str:
    """AND the significant words; arXiv treats bare words as OR and returns off-topic hits."""
    return " AND ".join(f"all:{w}" for w in _arxiv_terms(query)[:max_terms])


def _quoted_phrases(query: str) -> List[str]:
    return [p.strip() for p in re.findall(r'"([^"]+)"', query or "") if p.strip()]


def _plain_query(query: str) -> str:
    """Query text without phrase quotes, for relevance-ranked sources (OpenAlex, Semantic Scholar)."""
    return " ".join((query or "").replace('"', " ").split())


def _arxiv_queries(query: str) -> List[str]:
    """Strict-to-loose arXiv queries. Quoted key phrases must each appear as a phrase; when
    nothing matches, the least important (last) phrases are dropped, keeping at least two."""
    phrases = _quoted_phrases(query)
    if phrases:
        queries = []
        for count in range(len(phrases), min(len(phrases), 2) - 1, -1):
            queries.append(" AND ".join(f'all:"{p}"' for p in phrases[:count]))
        return queries
    queries = []
    for max_terms in _ARXIV_TERM_STEPS:
        candidate = _arxiv_query(query, max_terms)
        if candidate and candidate not in queries:
            queries.append(candidate)
    return queries


def search_arxiv(query: str, max_results: int = 3, timeout: float = 10) -> List[Dict]:
    import arxiv

    tool = _get_arxiv_tool()
    for arxiv_query in _arxiv_queries(query):
        search = arxiv.Search(query=arxiv_query, max_results=max_results)
        papers = [_from_arxiv_dict(tool._format_paper(result)) for result in tool.client.results(search)]
        if papers:
            return papers
    return []


def fetch_arxiv(arxiv_id: str, timeout: float = 10) -> Optional[Dict]:
    item = _get_arxiv_tool().get_paper_details(arxiv_id)
    return _from_arxiv_dict(item) if item else None


def fetch_crossref(doi: str, timeout: float = 10) -> Optional[Dict]:
    """Return the paper for a DOI, or None if Crossref has no record (404)."""
    params = {"mailto": _contact_email()} if _contact_email() else None
    try:
        message = _get(CROSSREF_URL + requests.utils.quote(doi, safe="/"), params=params, timeout=timeout).json()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise
    work = message.get("message") or {}
    issued = ((work.get("issued") or {}).get("date-parts") or [[None]])[0]
    authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in work.get("author") or []]
    return _paper(
        "crossref",
        work.get("DOI") or doi,
        (work.get("title") or [""])[0],
        work.get("abstract"),
        issued[0] if issued else None,
        (work.get("container-title") or [""])[0],
        authors,
        work.get("DOI") or doi,
        work.get("URL") or "",
    )


SOURCES: Dict[str, Callable[..., List[Dict]]] = {
    "openalex": search_openalex,
    "semantic_scholar": search_semantic_scholar,
    "pubmed": search_pubmed,
    "arxiv": search_arxiv,
}


def _failure_message(source: str, error: Exception) -> str:
    text = redact_secrets(str(error) or type(error).__name__)
    hint = ""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status == 429:
        hint = " (rate limited; an API key for this source raises the limit)"
    elif status in (401, 403):
        hint = " (access denied; check this source's API key)"
    return f"Literature search ({source}) failed{hint}: {text}"


# --- Searching ---


def _dedupe_keys(paper: Dict) -> set:
    # Title as well as DOI: a preprint and its journal version carry different DOIs.
    keys = {"title:" + re.sub(r"[^a-z0-9]", "", paper.get("title", "").lower())[:80]}
    if paper.get("doi"):
        keys.add("doi:" + paper["doi"].lower())
    return keys


def _interleave(results: List[List[Dict]], limit: int) -> List[Dict]:
    merged, seen = [], set()
    for rank in range(max((len(r) for r in results), default=0)):
        for papers in results:
            if rank < len(papers):
                paper = papers[rank]
                keys = _dedupe_keys(paper)
                if paper.get("title") and not keys & seen:
                    seen |= keys
                    merged.append(paper)
    return merged[:limit]


def hypothesis_query(title: str, text: str) -> str:
    """Short keyword query for a hypothesis; search APIs do poorly on long prose."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", title or "")
    if len(words) < 4:
        words += re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", text or "")[:15]
    return " ".join(words)[:200]


class LiteratureSearch:
    """Searches the configured sources in parallel; caches results per query."""

    def __init__(self, sources: Optional[List[str]] = None):
        self._sources = sources
        self._cache: Dict[str, Tuple[List[Dict], List[str]]] = {}

    def search(self, query: str) -> Tuple[List[Dict], List[str]]:
        """Return (papers, error messages). Never raises."""
        query = (query or "").strip()
        if not query or not literature_enabled():
            return [], []
        if query in self._cache:
            return self._cache[query]

        settings = literature_settings()
        sources = [s for s in (self._sources or settings["sources"]) if s in SOURCES]
        per_source = int(settings["results_per_source"])
        timeout = float(settings["timeout_seconds"])
        results: List[List[Dict]] = []
        errors: List[str] = []
        pool = ThreadPoolExecutor(max_workers=max(1, len(sources)))
        futures = {s: pool.submit(SOURCES[s], query, per_source, timeout) for s in sources}
        for source, future in futures.items():
            try:
                results.append(future.result(timeout=timeout + 5))
            except FutureTimeout:
                errors.append(f"Literature search ({source}) failed: timed out after {timeout + 5:.0f}s")
            except Exception as e:  # noqa: BLE001 - any source failure must be reported, not raised
                errors.append(_failure_message(source, e))
        pool.shutdown(wait=False, cancel_futures=True)

        for message in errors:
            logger.warning(message)
        outcome = (_interleave(results, int(settings["max_papers_in_prompt"])), errors)
        if not errors:
            self._cache[query] = outcome
        return outcome


# --- User-provided references ---


def parse_reference_lines(text: str) -> List[str]:
    lines = [line.strip() for line in (text or "").splitlines()]
    return [line for line in lines if line][:MAX_USER_REFERENCES]


def _identify(line: str) -> Tuple[Optional[str], Optional[str]]:
    match = _ARXIV_URL_RE.search(line) or _ARXIV_ID_RE.match(line)
    if match:
        return "arxiv", match.group(1)
    match = _PMID_RE.search(line)
    if match:
        return "pubmed", match.group(1) or match.group(2)
    match = _DOI_RE.search(line)
    if match:
        return "doi", _normalize_doi(match.group(0))
    return None, None


def resolve_references(lines: List[str]) -> Tuple[List[Dict], List[str]]:
    """Resolve DOIs / arXiv IDs / PubMed IDs to paper metadata; keep other lines as notes.

    Returns (references, errors). Each reference has a label U1..Un, a kind
    ("paper" or "note"), and the original line in "raw".
    """
    timeout = float(literature_settings()["timeout_seconds"])
    resolved, errors = [], []
    for index, line in enumerate(lines, start=1):
        label = f"U{index}"
        kind, identifier = _identify(line)
        paper = None
        if kind and literature_enabled():
            try:
                if kind == "doi":
                    paper = fetch_crossref(identifier, timeout=timeout)
                elif kind == "arxiv":
                    paper = fetch_arxiv(identifier, timeout=timeout)
                else:
                    found = fetch_pubmed([identifier], timeout=timeout)
                    paper = found[0] if found else None
                if paper is None:
                    errors.append(f"Could not resolve reference {label} ({line}): no record found")
            except Exception as e:  # noqa: BLE001
                errors.append(f"Could not resolve reference {label} ({line}): {redact_secrets(str(e))}")
        if paper and not paper.get("abstract") and paper.get("doi"):
            try:
                paper["abstract"] = openalex_abstract_for_doi(paper["doi"], timeout=timeout)
            except Exception as e:  # noqa: BLE001 - a missing abstract is not worth an error
                logger.info("No OpenAlex abstract for %s: %s", paper["doi"], redact_secrets(str(e)))
        if paper:
            resolved.append({**paper, "label": label, "kind": "paper", "raw": line})
        else:
            resolved.append(
                {
                    "label": label,
                    "kind": "note",
                    "text": line[:1500],
                    "raw": line,
                    "doi": identifier if kind == "doi" else None,
                }
            )
    return resolved, errors


# --- Prompt formatting and citation grounding ---


def citation(paper: Dict) -> str:
    if paper.get("kind") == "note":
        return f"Note: {paper.get('text', '')}"
    authors = paper.get("authors") or []
    lead = (authors[0] + (" et al." if len(authors) > 1 else "")) if authors else ""
    parts = [p for p in (lead, f"({paper['year']})" if paper.get("year") else "") if p]
    head = " ".join(parts)
    title = paper.get("title", "")
    venue = f" {paper['venue']}." if paper.get("venue") else ""
    link = f" doi:{paper['doi']}" if paper.get("doi") else (f" {paper['url']}" if paper.get("url") else "")
    return f"{head + '. ' if head else ''}{title}.{venue}{link}".strip()


def format_papers_block(papers: List[Dict], label_prefix: str = "P", abstract_chars: int = 600) -> str:
    lines = []
    for index, paper in enumerate(papers, start=1):
        label = paper.get("label") or f"{label_prefix}{index}"
        if paper.get("kind") == "note":
            lines.append(f"[{label}] Note from the user: {paper.get('text', '')}")
            continue
        abstract = paper.get("abstract") or "(no abstract available)"
        if len(abstract) > abstract_chars:
            abstract = abstract[:abstract_chars].rsplit(" ", 1)[0] + " ..."
        lines.append(f"[{label}] {citation(paper)}\n    Abstract: {abstract}")
    return "\n".join(lines)


_LABEL_RE = re.compile(r"^\[?([PU])(\d+)\]?(?:\W|$)", re.IGNORECASE)


def ground_references(
    cited: List[str],
    retrieved: List[Dict],
    user_references: List[Dict],
    verify_doi: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[str], int]:
    """Map model citations onto real papers; drop anything unverifiable.

    Accepts [P#] labels (retrieved papers), [U#] labels (user references), and
    DOIs that match a known paper or that `verify_doi` confirms exists.
    Citations of user notes are skipped (notes guide the prompt; they are not
    references) without counting as dropped. Returns (citations, number dropped).
    """
    known_dois = {}
    for paper in list(retrieved) + list(user_references):
        if paper.get("doi"):
            known_dois[paper["doi"].lower()] = paper
    kept, dropped = [], 0
    for raw in cited:
        text = str(raw).strip()
        match = _LABEL_RE.match(text)
        paper = None
        if match:
            pool = retrieved if match.group(1).upper() == "P" else user_references
            index = int(match.group(2)) - 1
            if 0 <= index < len(pool):
                paper = pool[index]
        else:
            doi_match = _DOI_RE.search(text)
            if doi_match:
                doi = _normalize_doi(doi_match.group(0))
                paper = known_dois.get(doi.lower())
                if paper is None and verify_doi is not None and verify_doi(doi):
                    paper = {"doi": doi, "title": text if text != doi else "", "kind": "paper"}
        if paper is None:
            dropped += 1
            continue
        if paper.get("kind") == "note":
            continue
        rendered = citation(paper) if paper.get("title") else f"doi:{paper['doi']}"
        if rendered not in kept:
            kept.append(rendered)
    return kept, dropped


def crossref_doi_exists(doi: str) -> bool:
    if not literature_enabled():
        return False
    try:
        return fetch_crossref(doi, timeout=float(literature_settings()["timeout_seconds"])) is not None
    except Exception as e:  # noqa: BLE001
        logger.warning("Crossref DOI check failed for %s: %s", doi, redact_secrets(str(e)))
        return False
