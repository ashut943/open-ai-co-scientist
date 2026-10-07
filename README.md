---
title: Open AI Co-Scientist
emoji: 📊
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
short_description: Open-source implementation of Google's AI Co-Scientist
---

# Open AI Co-Scientist - Hypothesis Evolution System

Open AI Co-Scientist is an AI-powered system for generating, reviewing, ranking, and evolving research hypotheses using a multi-agent architecture and Large Language Models (LLMs). The user interface is built with Gradio for rapid prototyping and interactive research. The system helps researchers explore research spaces and identify promising hypotheses through iterative refinement.

A live demonstration can be accessed at: https://huggingface.co/spaces/liaoch/open-ai-co-scientist
* Please note that this demo exclusively utilizes free models from OpenRouter.

## 🚀 Features

- **Multi-Agent System:** Iteratively generates, reviews, ranks, and evolves research hypotheses using specialized agents (Generation, Reflection, Ranking, Evolution, Proximity, Meta-Review).
- **LLM Integration:** Uses an OpenAI-compatible API (Parley by default; OpenRouter optional) with model selection in the UI.
- **Interactive Gradio UI:** Easy-to-use interface for research goal input, advanced settings, and results visualization.
- **References & Literature:** Reviews are grounded in OpenAlex / arXiv (optionally Semantic Scholar, PubMed) searches, cited references are verified, and you can supply your own references and notes.
- **Cost Control:** Automatically filters to cost-effective models in production deployment.
- **Logging:** Each run is logged to a timestamped file in the `results/` directory.

## AI Transparency Statement

In accordance with LLNL policy on Generative Artificial Intelligence, this project contains AI-assisted code and documentation. Various AI models (including OpenAI and Claude) were used to draft components and fix errors. The development process involved switching between models when encountering limitations with a particular model. All AI-generated content has been reviewed and verified by human developers to ensure accuracy, security, and alignment with project requirements.

## 💡 Example Research Goals

- Develop new methods for increasing the efficiency of solar panels.
- Create novel approaches to treat Alzheimer's disease.
- Design sustainable materials for construction.
- Improve machine learning model interpretability.
- Develop new quantum computing algorithms.

## Quick Start

1. **Set up a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Choose an LLM provider and set its API key:**
    - OpenAI / Parley (default):
      ```bash
      export LLM_PROVIDER=openai
      export OPENAI_API_KEY=your_api_key
      ```
      Default `config.yaml` points `openai_base_url` at Parley
      (`https://parley.api.mit.edu/v1`). For stock OpenAI, set it to
      `https://api.openai.com/v1`. Usage is billed by that provider.
    - Or use OpenRouter (e.g. free models / HF Spaces demo):
      ```bash
      export LLM_PROVIDER=openrouter
      export OPENROUTER_API_KEY=your_api_key
      ```

4. **Run the Gradio app:**
    ```bash
    python app.py
    ```
    Or, using the Makefile:
    ```bash
    make run
    ```

5. **Access the web interface:**
    - Open your browser and go to [http://localhost:7860](http://localhost:7860)

## 🎯 How to Use

1. **Enter a research goal** in the provided textbox.
   Optionally add **references and notes**, one per line: a DOI, an arXiv ID/URL, a PubMed ID/URL, or free text (e.g. "our pilot showed a 3% gain").
2. **(Optional) Adjust advanced settings** such as LLM model, number of hypotheses, temperatures, etc.
3. **Click "Run Cycle"** to generate, review, and evolve hypotheses.
4. **View results, meta-review, and related literature** in the web interface.
5. **Iterate** by running additional cycles to refine hypotheses.

## ⚙️ Configuration

- Default settings can be adjusted in `config.yaml`.
- Many settings can be overridden in the Gradio UI under "Advanced Settings".

## 🧠 How It Works

The system uses a multi-agent approach:

1. **Generation step:** the **Evolution Agent** applies LLM operators (refine / mutate / hybridize / simplify) to the previous cycle's top hypotheses, guided by its meta-review, creating child hypotheses linked by `parent_ids`; the **Generation Agent** then adds fresh hypotheses. The first cycle has fresh hypotheses only.
2. **Reflection Agent:** Structured peer review (1-5 criterion scores, assumptions, falsification conditions, safety concerns, improvements) of hypotheses not yet reviewed.
3. **Ranking Agent:** One tournament per cycle: LLM pairwise judging against the research goal, then Elo updates to rank hypotheses.
4. **Meta-Review Agent:** LLM analysis of recurring themes across reviews and matches; its strategy steers the next cycle's evolution.
5. **Proximity Agent:** Analyzes similarity between hypotheses.

## 📚 Literature Integration

- **Grounded reviews:** before reviewing a hypothesis, the Reflection Agent searches the sources in `literature_search.sources` (default `openalex` and `arxiv`; `semantic_scholar` and `pubmed` are also available) in parallel. The top papers (`max_papers_in_prompt`, default 6) are shown to the reviewer as `[P1]`, `[P2]`, ... Novelty is judged against them, and the closest prior work is recorded.
- **Your references:** lines from the "References and notes" box are resolved once per goal (Crossref for DOIs, arXiv, PubMed) and passed to the generation, reflection and evolution prompts as `[U1]`, `[U2]`, ... Lines that are not identifiers are kept as notes.
- **No invented citations:** a cited reference is kept only if it maps to a `[P#]` or `[U#]` paper, or to a DOI that Crossref confirms exists. Anything else is dropped, and the drop is noted in the review.
- **References tab:** shows your references, the papers each review was checked against, and related papers for the goal.
- **Optional environment variables** (never put keys in `config.yaml`):
  - `LITERATURE_CONTACT_EMAIL` sends a polite contact email to OpenAlex, Crossref and PubMed.
  - `OPENALEX_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY` and `NCBI_API_KEY` raise each source's rate limit.
  - `CO_SCIENTIST_DISABLE_LITERATURE=1` turns all lookups off.
- **Failures are visible:** a source that fails (timeout, rate limit) is reported in the error box, and the review continues with whatever was retrieved.

## ⚙️ Technical Details

- **Models:** Uses OpenRouter API with cost-effective models in production.
- **Environment Detection:** Automatically detects Hugging Face Spaces deployment.
- **Cost Control:** Filters to budget-friendly models (Gemini Flash, GPT-3.5-turbo, Claude Haiku, etc.).
- **Iterative Process:** Each cycle builds on previous results for continuous improvement.

## 🔧 Deployment (Hugging Face Spaces)

The system automatically configures itself based on the deployment environment:

- **Production (HF Spaces):** Limited to cost-effective models for budget control.
- **Development:** Full access to all available models.

For the maintained release path from approved loop-repo PRs into the public
upstream repo and then to Hugging Face, see
[`docs/upstream-release-process.md`](docs/upstream-release-process.md).
The process is backed by manual/tag-gated GitHub Actions workflows and offline
tests.

### Hugging Face Spaces Setup

1. **Create a new Space** at [Hugging Face Spaces](https://huggingface.co/spaces).
2. **Upload files:** README.md, app.py, requirements.txt, and the app/ directory.
3. **Set environment variables:** Add your `OPENROUTER_API_KEY` as a secret in Space settings.
4. **Deploy:** The Space will automatically build and deploy the app.

## 📖 Research Paper

Based on the AI Co-Scientist research: https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf

## 🤝 Contributing

This is an open-source project. Feel free to contribute improvements, bug fixes, or new features. 

See CONTRIBUTING.md for details. 

## ⚠️ Note

This system requires an OpenRouter API key to function. The public demo uses a limited budget, so please use it responsibly. For extensive research, consider running your own instance with your API key.


## Acknowledgements

- Based on the idea of Google's AI Co-Scientist system.
- Uses [Gradio](https://gradio.app/) for the user interface.
- LLM access via Parley / OpenAI-compatible APIs (OpenRouter still supported).

---

## Release

LLNL-CODE-2010270

SPDX-License-Identifier: MIT
