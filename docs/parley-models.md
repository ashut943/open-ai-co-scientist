# Parley models — Terminal-Bench science ranking

Intel below is **re-ranked from public agent benchmarks**, not a subjective vibe score.

Primary source: **[Terminal-Bench-Science 0.1](https://snorkel.ai/leaderboard/terminal-bench-science/)** (scientific research workflows).  
Secondary: **[Terminal-Bench 4.0](https://snorkel.ai/leaderboard/terminal-bench-4-0/)** when Science score is missing.  
Costs remain vendor list prices (USD / 1M tokens); Parley/MIT billing may differ.

Scores are **resolution rate %** at the reported high-effort agent setting. Different harnesses/agents change absolute %; relative order is what matters here.

## What a “cycle” is

In this repo, a **cycle** is one click of **Run Cycle** in the Gradio UI: one full Open AI Co-Scientist pipeline pass for your research goal.

![One Open AI Co-Scientist cycle schematic](parley-cycle-schematic.png?v=2026-10-07c)

Each cycle runs these steps in order:

1. **Generation step.** From cycle 2 on, the Evolution agent first makes up to 4 LLM calls (refine / mutate / simplify on the top hypothesis, hybridize on the top two), guided by the previous cycle's meta-review. Then 1 LLM call generates fresh hypotheses.
2. **Reflection.** 1 LLM call per hypothesis that has no review yet. Each call first runs a literature search (OpenAlex + arXiv by default). Those APIs are free, but the retrieved abstracts make the prompt larger.
3. **Tournament.** 1 LLM judge call per match. Each hypothesis plays about `tournament_matches_per_hypothesis` (default 3) matches, so a pool of `n` hypotheses needs roughly `n × 3 / 2` judge calls. Elo updates are local.
4. **Meta-review.** 1 LLM call over all reviews and match outcomes. Its strategy steers the next cycle's evolution.
5. **Proximity.** Local embeddings only; no API cost.

Nothing is retired, so the pool and the number of matches grow every cycle: 4 hypotheses in cycle 1, 12 in cycle 2, 20 in cycle 3.  
“Cycles on $30” in the cost table means how many Run Cycle passes fit in $30 under the token estimate below.

## Token usage estimate

These are **planning estimates built from the prompt templates in `app/agents.py`, not measurements**. Defaults assumed: `num_hypotheses: 4`, `top_k_hypotheses: 2`, `tournament_matches_per_hypothesis: 3`, a ~50-token research goal, and typical model output lengths.

**How tokens were counted.** One token is about 4 characters or ¾ of an English word ([OpenAI: What are tokens](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)). Template sizes come from character counts of each prompt in `app/agents.py`. Output sizes assume a hypothesis is ~150 words (~215 tokens with its title) and a JSON review is ~500 tokens. Every call sends one user message with no system prompt, as `call_llm` in `app/utils.py` shows.

| Call | Input tokens | Visible output tokens | What drives the size |
| --- | ---: | ---: | --- |
| Generation | ~250 | ~900 | ~120-token template + goal + existing IDs → 4 hypotheses |
| Reflection | ~1,800 | ~600 | ~450-token template + goal + 1 hypothesis + 6 retrieved papers × ~185 tokens (citation + 600-char abstract) → JSON review (7 scores + 7 lists) |
| Tournament judge | ~1,300 | ~200 | template + goal + 2 × (hypothesis ~215 + formatted review ~300) → winner, reasoning, criterion scores |
| Evolution | ~2,250 | ~300 | template + parent(s) with reviews + recent match reasoning + rendered meta-review |
| Meta-review | 400 + 435 per hypothesis + 60 per recent match | ~700 | every hypothesis snapshot + up to 30 recent match reasons → 8 lists |

| Cycle | LLM calls | Input | Visible output | Pool size |
| --- | ---: | ---: | ---: | --- |
| 1 | 12 (1 gen, 4 reviews, 6 matches, 1 meta) | ~18k | ~5k | 4 new |
| 2 | 33 (4 evo, 1 gen, 8 reviews, 19 matches, 1 meta) | ~55k | ~11k | 4 evolved + 4 new + 4 surviving |
| 3 | 45 (4 evo, 1 gen, 8 reviews, 31 matches, 1 meta) | ~75k | ~14k | 20 |
| **Average of 1–3** | **30** | **~49k** | **~10k** | |

Tournament judging is the biggest single cost: 45% of input tokens in cycle 2 and 54% in cycle 3. Literature-grounded reflection is next, at about 26% of input in cycle 2. User references add about 100 tokens each to the generation, reflection and evolution prompts; the estimate assumes none. Per cycle it grows roughly linearly with the pool size, because of the 3-matches cap.

**Hidden reasoning tokens are the biggest uncertainty.** All models in the plot are reasoning models. Their internal reasoning is billed as output even though the app never sees it ([OpenAI reasoning guide](https://platform.openai.com/docs/guides/reasoning), [Anthropic extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking), [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking)). `call_llm` sets no `max_tokens` or reasoning-effort limit, so the provider default applies. The plot assumes **~1,000 reasoning tokens per call**. That is an assumption for short, structured tasks at default effort, not a published figure. At 30 calls per cycle it adds ~30k output tokens.

**Planning figure: ~49k input + ~40k output per cycle** (10k visible + 30k reasoning), averaged over a 3-cycle session. Before literature grounding it was ~41k + ~39k. The original figure of 80k + 32k assumed roughly 6 calls per cycle and much longer prompts. To replace these estimates with real numbers, log `completion.usage` (`prompt_tokens`, `completion_tokens`, and `completion_tokens_details.reasoning_tokens`) from `call_llm` for one session.

## Cost vs science scatter (co-scientist use case)

X = estimated USD per cycle (log scale), using ~49k input + ~40k output tokens. Each grey bar extends left to the cost with zero reasoning tokens (~49k in + ~10k out), which is the lower bound.  
Y = Terminal-Bench-Science resolution %.  
**★** = Pareto front (maximize science, minimize $/cycle). Only **measured** TB-Science points define the front. The front is the same at both ends of the reasoning-token range, because each model's input and output prices differ by about the same ratio.

| Model | TB-Science | $ / cycle (no reasoning) | $ / cycle (incl. reasoning) | Cycles on $30 | Pareto |
| --- | ---: | ---: | ---: | ---: | --- |
| `gpt-5.6-luna` | 3.3% | $0.022 | $0.058 | ~517 | ★ cheapest |
| `gemini-3.8-flash` | 12.4% | $0.075 | $0.19 | ~160 | ★ best for ~$30 |
| `gemini-3.7-flash` | 5.7% | $0.075 | $0.19 | ~160 | dominated by 3.8 Flash |
| `gpt-5.6-terra` | 8.6% | $0.22 | $0.58 | ~52 | dominated by Flash |
| `gpt-5.6-sol` | 22.4% | $0.40 | $1.00 | ~30 | dominated by Opus 5.5 |
| `claude-opus-5-5` | 63.3% | $0.40 | $1.00 | ~30 | ★ best quality before Astra |
| `claude-opus-5` | 30.0% | $0.50 | $1.25 | ~24 | dominated |
| `claude-opus-4-8` | 10.5% | $0.50 | $1.25 | ~24 | dominated |
| `gpt-6-astra` | 68.1% | $0.99 | $2.50 | ~12 | ★ highest science |

“Cycles on $30” uses the cost including reasoning. With no reasoning tokens, it is about 2.5× more: for example ~400 Flash cycles or ~75 Opus 5.5 cycles. Per-cycle cost also rises over a session. For Flash it is about $0.08 in cycle 1 and $0.28 in cycle 3, including reasoning.

### Scatter plot

![Cost vs Terminal-Bench-Science with Pareto front](parley-cost-science-pareto.png?v=2026-10-07c)

Pareto polyline (left → right): **Luna → Gemini 3.8 Flash → Claude Opus 5.5 → GPT-6 Astra**.

**Pick on ~$30:** stay at `gemini-3.8-flash` on the front (~160 cycles). Jump to `claude-opus-5-5` only if you need much higher measured science (~30 cycles). Skip `gpt-6-astra` until budget grows.

Both figures are generated by `scripts/make_parley_figures.py`; it also prints the numbers above. To regenerate them after changing an assumption:

```bash
venv/bin/pip install matplotlib
venv/bin/python scripts/make_parley_figures.py
```

## Ranked chat models (your Parley IDs)

| Rank | Model ID | Intel | TB-Science | TB 4.0 | In / Out per 1M | Notes |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `gpt-6-astra` | **5** | **68.1%** | ~58% | $10 / $50 | Best on TB-Science; burns $30 fast |
| 2 | `claude-opus-5-5` | **5** | **63.3%** | ~66%* | $4 / $20 | Close #2 on science agents |
| 3 | `claude-opus-5` | **4** | **30.0%** | ~54% | $5 / $25 | Strong science; expensive |
| 4 | `gpt-5.6-sol` | **4** | **22.4%** | ~37% | $4 / $20 | Best OpenAI mid-frontier on science |
| 5 | `claude-opus-4-8` | **3** | **10.5%** | ~24% | $5 / $25 | Mid pack on TB-Science |
| 6 | `gemini-3.8-flash` | **3** | **12.4%** | ~19% | $0.75* / $3.75* | Best cheap science pick on your list |
| 7 | `gpt-5.6-terra` | **3** | **8.6%** | ~22% | $2 / $12 | Balanced, modest science score |
| 8 | `gemini-3.7-flash` | **2** | **5.7%** | ~11% | $0.75* / $3.75* | Cheaper Flash; weaker science |
| 9 | `gpt-5.6-luna` | **2** | **3.3%** | ~17% | $0.20 / $1.20 | Cheap; weak on TB-Science |
| — | `claude-sonnet-5-5` | **4†** | — | ~70%* | $2 / $10 | No TB-Science row yet; very strong TB 4.0 |
| — | `gpt-6-sol` | **4†** | — | — | $2 / $10 | No TB-Science row; expect below Astra |
| — | `gpt-6-luna` | **2†** | — | — | $0.10 / $0.50 | No TB-Science row; expect ~Luna tier |
| — | `claude-sonnet-5` | **2†** | — | ~12% | $2 / $10 | Weak TB 4.0 vs Opus/Astra |
| — | `claude-sonnet-4-6` | **2†** | — | — | $3 / $15 | Older Sonnet; no TB-Science score |
| — | `claude-opus-4-7` | **3†** | — | — | $5 / $25 | Assume near Opus 4.8 |
| — | `gpt-5.5` | **3†** | — | — | $5 / $30 | Strong GPQA; no TB-Science score |
| — | `gpt-5.4` / `gpt-5.3-codex` / `gpt-5.2` / `gpt-5.1` / `gpt-5` | **2–3†** | — | — | see below | Older GPT-5 line; not on TB-Science board |
| — | `gpt-5.4-mini` / `gpt-5-mini` | **2†** | — | — | cheap | Fine for app smoke tests, not science agents |
| — | `gpt-5.4-nano` / `gpt-5-nano` / `claude-haiku-4-5` / `gemini-3.5-flash-lite` | **1†** | — | — | cheapest | Throughput only |
| — | `gemini-3.1-pro` / `gemini-3.5-flash` / `gemini-3.0-flash` / `gemini-2.5-pro` | **2–3†** | — | — | mid | Not on TB-Science board (3.8 Flash is) |
| — | `llama-4-maverick-17b` | **1†** | — | — | ~$0.24 / $0.97 | No TB-Science score |

\* Gemini Flash intro pricing thru 2026-12-31; vendor TB 4.0 / Anthropic self-reports can differ from Snorkel official board.  
† Estimated from nearby family + TB 4.0 / GPQA; **not** a measured TB-Science score.

## If you only have ~$30

| Priority | Model | Why |
| --- | --- | --- |
| **Best science / $** | `gemini-3.8-flash` | Measured 12.4% TB-Science, cheap intro rates (~$0.19 / cycle, ~160 cycles) |
| **Best unmeasured bet** | `claude-sonnet-5-5` | No TB-Science score yet, but top-tier TB 4.0 at $2/$10 (~$0.50 / cycle, ~60 cycles) |
| **Repo default** | `gpt-5-mini` | `openai_model` in `config.yaml`; fine for smoke tests (~$0.09 / cycle, ~324 cycles) |
| **Avoid on $30** | `gpt-6-astra`, `claude-opus-5*`, `gpt-5.5` | High $/token; few cycles |

Per-cycle costs in this table use the same ~49k input + ~40k output estimate as the scatter plot.

## Older GPT pricing (not on TB-Science)

| Model | In / Out |
| --- | --- |
| `gpt-5.4` | $2.50 / $15 |
| `gpt-5.4-mini` | $0.75 / $4.50 |
| `gpt-5.4-nano` | $0.20 / $1.25 |
| `gpt-5.3-codex` | $1.75 / $14 |
| `gpt-5.2` | $1.75 / $14 |
| `gpt-5.1` / `gpt-5` | $1.25 / $10 |
| `gpt-5-mini` | $0.25 / $2 |
| `gpt-5-nano` | $0.05 / $0.40 |

## Not for chat cycles

| Model ID | Kind |
| --- | --- |
| `text-embedding-3-small` | embeddings |
| `amazon-titan-embed-text-v2-0` | embeddings |
| `gemini-embedding-2` | embeddings |
| `gpt-image-1` / `gpt-image-2` | image |

## Caveats

1. TB-Science measures **terminal scientific workflows with an agent harness** (Codex / Claude Code / etc.). This Gradio app is multi-step hypothesis chat, not the same task — but it’s the closest public science-agent ranking.
2. Absolute % depends on agent + effort; compare models within the same board.
3. Sources: Snorkel Terminal-Bench-Science / Terminal-Bench 4.0 leaderboards (as of ~Oct 2026).
