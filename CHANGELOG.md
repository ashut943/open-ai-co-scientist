# Changelog

All notable changes to the AI Co-Scientist project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Token counting: each cycle records input, output, and reasoning tokens per
  step (generation, evolution, reflection, tournament, meta-review). The status
  box shows totals for the cycle and the session, plus a cost estimate for
  models listed under `token_prices_per_million` in `config.yaml`; the report
  has a Token Usage table.
- Charts in the results panel and the saved HTML report: Elo rating across
  cycles, a hypothesis family tree (parents → evolved children, colored by
  operator), a review-score heatmap, and a similarity graph. They are inline SVG,
  so no plotting dependency is needed. Hypotheses now record `elo_history`.
- Reflection reviews are now grounded in a literature search per hypothesis
  (default sources: OpenAlex + arXiv; Semantic Scholar and PubMed are opt-in
  via `literature_search.sources`). Retrieved papers are shown to the reviewer
  as `[P#]`, novelty must be judged against them, and each hypothesis records
  its `closest_prior_work` and the papers it was checked against. Cited
  references are kept only when they map to a retrieved or user-provided paper
  or to a DOI that Crossref confirms exists; others are dropped and counted.
  Search failures (timeouts, rate limits) appear in the error box.
- A "References and notes" box under the research goal. Each line may be a DOI,
  an arXiv ID/URL, a PubMed ID/URL, or a free-text note. Identifiers are resolved
  once per goal (Crossref / arXiv / PubMed) and given to generation,
  reflection and evolution prompts as `[U#]`. The References tab lists your
  references, the literature each review used, and related papers for the goal.
- Optional environment variables: `LITERATURE_CONTACT_EMAIL`,
  `OPENALEX_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `NCBI_API_KEY`;
  `CO_SCIENTIST_DISABLE_LITERATURE=1` turns all lookups off.
- Each run report is also saved as a PDF next to the HTML
  (`results/reports/<run-id>.pdf`), printed with headless Chromium via
  Playwright; skipped with a log line if Playwright/Chromium is missing, or
  when `CO_SCIENTIST_DISABLE_PDF=1`. The history table links both.
  LaTeX in hypotheses and reviews (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`)
  is typeset with KaTeX (loaded from a CDN; raw TeX stays if offline), and
  the PDF is printed after the math has rendered. The app's results panel
  typesets LaTeX the same way each time it updates.
- Literature search retrieves 5 papers per source (was 3), up to 8 per
  review prompt.

### Fixed
- LLM calls send an output cap (`llm_max_output_tokens`, default 32000) as
  `max_completion_tokens`. Without it the gateway's default (~4k for Claude
  models) cut off Opus generations mid-JSON. Output stopped at the limit is now
  reported as "cut off at the output token limit" instead of a JSON parse error.
- Token counting reads Anthropic-style `input_tokens`/`output_tokens` and adds
  `prompt_tokens_details.cached_tokens` when the gateway under-reports
  `prompt_tokens` (Parley + Opus showed 4 input tokens per call).
- The "Related papers for the research goal" search no longer sends the whole
  goal text (OpenAlex answered 400 Bad Request). It searches the key phrases of
  the top-ranked hypotheses, or the goal's first sentence when there are none.
- `call_llm` no longer sends `temperature`, which reasoning models such as
  `gpt-5-mini` reject unless it is 1. The UI temperature sliders were removed.
- The UI was restyled (header card, side panel, examples next to the goal), and
  result cards now use translucent colors that stay readable in dark mode.
- arXiv searches require all significant query words (retrying with fewer
  when nothing matches), and a preprint is merged with its journal version even
  when their DOIs differ.
- Literature searches use `search_keywords` that the generation/evolution
  LLM returns with each hypothesis: 2-4 key phrases (field-standard terms, no
  coined names), falling back to the title; title-word queries pulled in
  off-topic papers. arXiv matches each phrase exactly, dropping the
  least important phrases (down to two) when nothing matches.
- LLM JSON with LaTeX written using single backslashes (`"$\sigma$"`, an
  invalid JSON escape) is repaired and parsed instead of failing the step;
  raw newlines or tabs inside JSON strings are accepted too.
- The reviewer lists which retrieved papers are actually relevant
  (`relevant_papers`); only those are recorded as the literature the
  hypothesis was checked against ("N relevant of M retrieved").
- User references without a Crossref abstract get one from OpenAlex by DOI.
- User notes are no longer listed as a hypothesis's cited references (they
  still go into every prompt). Logs no longer print an unused temperature.
- Clicking "Set Research Goal" again with the same goal and references no
  longer wipes the session, so later cycles (and evolution) actually run.

### Changed
- Reflection reviews (with their literature searches) and tournament judge
  calls now run concurrently, up to `llm_max_concurrency` (default 4) at a
  time. Elo is still updated in pair order after judging, so rankings follow
  the same rules; arXiv searches still queue on one rate-limited client.
- Failures after generation are no longer silent. If a reflection review,
  tournament judgment, meta-review, or evolution operator fails, the cause is
  added to `cycle_details["errors"]` and shown in the results error box and
  status line, while the cycle continues with its fallback. A failed review
  no longer invents MEDIUM ratings or overwrites an earlier successful review,
  and the meta-review section is marked when it is a rule-based fallback.
- Tournaments no longer judge every pair: each hypothesis plays about
  `tournament_matches_per_hypothesis` (default 3) LLM-judged matches, so judge
  calls grow linearly instead of as n*(n-1)/2 (set 0 for a full round-robin).
  Hypotheses that already have a successful reflection review are not
  re-reviewed in later cycles.
- Meta-review is now an LLM analysis (recurring strengths/weaknesses,
  unexplored mechanisms, shared assumptions, contradictions, promising pairs,
  research gaps, recommended evolution strategy) whose strategy controls which
  operators/parents the next cycle's evolution uses. Each cycle is Generation
  (evolved children of the previous meta-review + fresh ideas) → Reflection →
  Tournament → Meta-review → Proximity, with a single tournament per cycle.
  The first cycle has no evolved children.
- Reflection now returns a peer-review schema with 1-5 scores (soundness,
  novelty, relevance, feasibility, testability, clarity, impact) plus
  strengths/weaknesses, critical assumptions, falsification conditions,
  safety/ethical concerns, and recommended improvements. Legacy
  HIGH/MEDIUM/LOW novelty and feasibility fields are derived from those scores.
- Evolution no longer string-concatenates top hypotheses. The Evolution agent
  creates **new child** hypotheses (via `parent_ids`, never in-place mutation)
  using LLM operators: REFINE, MUTATE, SIMPLIFY on the top-ranked idea, and
  HYBRIDIZE on the top two when available. Prompts include the research goal,
  reflection reviews, tournament feedback, and prior-cycle meta-review.
- Default LLM provider is now OpenAI-compatible / Parley (`llm_provider:
  openai`, `openai_base_url: https://parley.api.mit.edu/v1`). OpenRouter
  remains available via `LLM_PROVIDER=openrouter`. Offline tests target the
  OpenAI/Parley path.
- Tournament match winners are now chosen by an LLM judge that compares
  hypotheses against the real research goal (soundness, novelty, relevance,
  feasibility, testability, clarity, impact) instead of summing novelty +
  feasibility ordinals. Existing Elo update math is unchanged; ties leave
  ratings alone; LLM/parse failures fall back to the legacy score comparison.

- Fixed the Hugging Face Space build failure by pinning `pydantic` to the
  range required by `gradio[oauth,mcp]==6.19.0`, pinning the Space runtime to
  Python 3.12, and adding a deploy preflight that runs Hugging Face's Gradio
  extras resolver before pushing to the Space.
- Fixed the Hugging Face Space metadata to match the pinned Gradio dependency
  and added post-push Space status polling so build/runtime errors fail the
  deployment workflow instead of being missed (loop#14).

- Automatic model fallback (llnl#26): when the configured/selected model is
  unavailable, delisted, rate-limited, or erroring, `call_llm` falls back to a
  **working free model fetched live from OpenRouter** (`fetch_free_models`, the
  source of truth) instead of failing every run. Falls back on
  model-unavailable / rate-limit / provider errors — not on auth (a different
  model won't help). Rate-limited free models are skipped immediately (SDK
  Retry-After backoff disabled) rather than blocking the run; attempts are
  bounded. A static list is a last resort only if the live fetch fails.
- Updated the default `config.yaml` model (the previous
  `google/gemini-2.0-flash-exp:free` was delisted).

### Fixed
- Generation failures no longer surface as a silent empty ranking (llnl#36).
  The real cause — missing/invalid API key, model unavailable/delisted, rate
  limiting, or unparsable model output — now appears in the run status line and
  an actionable error box in the results panel, instead of a false
  "completed successfully". Fixes a three-layer silent-failure chain where the
  error text was discarded during generation, making downstream propagation
  dead code.

### Added
- `classify_llm_error()` maps raw LLM/API errors to user-actionable categories;
  `call_llm` now distinguishes model-unavailable/delisted errors.
- Offline-by-default test suite: `integration`/`network` pytest markers, mocked
  OpenRouter boundary tests, and credential-leak regression tests.
- CI workflow (lint, offline tests, boot smoke) pinned to Python 3.12 with
  CPU-only torch.
- `pyproject.toml` (pytest + ruff config), pinned `requirements.txt`,
  `requirements-dev.txt`, `.env.example`.
- Makefile targets: `test`, `test-all`, `lint`, `fmt`, and per-issue worktree
  helpers `wt`/`wt-clean`.
- `AGENTS.md` (AI-agent instructions), `docs/loop/GOALS.md` (loop steering),
  `scripts/setup_labels.sh` (loop label state machine).

### Fixed
- `call_llm` crashed with openai>=1 when `OPENROUTER_API_KEY` was unset instead
  of returning the documented error message.
- API keys are now redacted from log lines and user-facing error text
  (`redact_secrets`).
- `similarity_score` treats whitespace-only input as empty (returns 0.0).

### Removed
- Dead FastAPI-era `tests/test_api.py` (targeted the removed `app/api.py`) and
  the stray `tests/test_graph.html` artifact.

## [1.1.0] - 2025-05-31

### Added - References Section and Literature Integration

#### 🔬 **ArXiv Integration**
- **Comprehensive arXiv API integration** for scientific literature discovery
- **Automatic paper search** based on research goal keywords (up to 5 most relevant papers)
- **Full paper metadata display** including titles, authors, abstracts, publication dates, and categories
- **Direct linking** to arXiv papers, PDF downloads, and DOI references
- **ArXiv testing interface** at `/arxiv/test` for standalone literature search functionality

#### 📚 **Smart Reference Detection**
- **Intelligent reference type detection** from LLM-generated hypothesis reviews
- **arXiv ID linking**: Automatic detection and linking of arXiv identifiers (e.g., `2301.12345`, `arxiv:1706.03762`)
- **DOI linking**: Direct links to journal articles via DOI identifiers (e.g., `10.1145/3394486.3403087`)
- **PubMed integration**: Links to biomedical literature with domain-appropriate usage warnings
- **Generic reference display**: Formatted display for paper titles, conference citations, and other references

#### 🎯 **Domain-Appropriate Literature**
- **Computer science focus**: Prioritizes arXiv papers and CS conference literature
- **Biomedical support**: Maintains PubMed integration for life sciences research
- **Cross-domain warnings**: Alerts users when PubMed references appear in non-biomedical contexts
- **Updated LLM prompts**: Modified reflection prompts to avoid inappropriate PMIDs for CS topics

#### 🎨 **Professional User Interface**
- **New References section** positioned between Results and Errors in main interface
- **Card-based paper display** with professional academic formatting
- **Category tags** showing arXiv subject classifications
- **Responsive design** elements for optimal viewing experience
- **Error state handling** with user-friendly messages and fallbacks

#### 🔧 **API Endpoints**
- `POST /arxiv/search` - Search arXiv papers with filtering options
- `GET /arxiv/paper/{id}` - Retrieve specific paper details
- `GET /arxiv/trends/{query}` - Analyze research trends over time
- `GET /arxiv/categories` - List available arXiv subject categories
- `GET /arxiv/test` - Comprehensive testing interface for arXiv functionality
- `POST /log_frontend_error` - Frontend error logging for debugging

#### 📊 **Enhanced Logging and Debugging**
- **Frontend-to-backend logging** system for comprehensive error tracking
- **Detailed reference processing logs** showing each step of literature discovery
- **ArXiv search status logging** with response codes and paper counts
- **Error handling with stack traces** for debugging JavaScript issues
- **Structured log data** with timestamps and contextual information

#### 🛠 **Technical Improvements**
- **New data models**: `ArxivPaper`, `ArxivSearchRequest`, `ArxivSearchResponse`, `ArxivTrendsResponse`
- **ArXiv search tool**: Comprehensive `ArxivSearchTool` class with filtering and analysis capabilities
- **Updated dependencies**: Added `arxiv`, `feedparser`, `python-dateutil` for arXiv integration
- **Async JavaScript functions** for non-blocking literature search
- **Regex pattern fixes** for reliable reference type detection
- **Graceful error handling** with user-friendly fallback messages

### Changed
- **Enhanced hypothesis reviews** now include domain-appropriate reference types
- **Improved LLM prompts** to generate relevant CS literature references instead of inappropriate PMIDs
- **Updated main interface** to include automatic literature discovery after each cycle
- **Modified reference display** from generic "PMIDs" to "Additional References" with smart type detection

### Fixed
- **JavaScript regex errors** in reference type detection patterns
- **Domain inappropriateness** of PubMed references for computer science research
- **Missing error handling** in frontend reference processing
- **Console errors** that prevented references section from loading properly

### Technical Details
- **Files Added**: `app/tools/arxiv_search.py`, `CHANGELOG.md`
- **Files Modified**: `app/api.py`, `app/models.py`, `app/agents.py`, `requirements.txt`, `README.md`, `claude_planning.md`
- **New Dependencies**: arxiv==2.1.0, feedparser==6.0.10, python-dateutil==2.8.2
- **API Endpoints Added**: 6 new endpoints for arXiv integration and frontend logging
- **JavaScript Functions Added**: `logToBackend()`, enhanced `updateReferences()` and `displayReferences()`

### Impact
- **Dramatically improved research quality** through automatic literature discovery
- **Enhanced user experience** with professional reference display and direct paper access
- **Better domain appropriateness** with CS-focused literature for computer science research
- **Improved debugging capabilities** with comprehensive frontend-to-backend logging
- **Scientific rigor** through integration with arXiv, the primary preprint server for CS and physics

This release transforms the AI Co-Scientist from a hypothesis-only system into a literature-integrated research platform, providing users with immediate access to relevant scientific papers and properly formatted academic references.

## [1.0.0] - 2025-02-28

### Added
- Initial release of AI Co-Scientist hypothesis evolution system
- Multi-agent architecture with Generation, Reflection, Ranking, Evolution, Proximity, and MetaReview agents
- FastAPI web interface with advanced settings
- LLM integration via OpenRouter API
- Elo-based hypothesis ranking system
- Hypothesis similarity analysis and visualization
- YAML configuration management
- Basic HTML frontend with vis.js graph visualization
