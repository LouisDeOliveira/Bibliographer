# Day 2: Multi-modal contextual scraping & verification

## Learning objectives

- Connect the agent to real academic APIs (Semantic Scholar and arXiv)
- Score paper relevance via a structured LLM evaluation
- Traverse citation networks forward (citing papers) and backward (references)

---

## Functional tool binding

So far the tools returned fake data, now we will call real HTTP APIs. The binding pattern is:

```
- Agent asks for tool call
- Python function executes HTTP request
- Raw API response is cleaned/truncated
- Cleaned result is returned to the agent as observation
```

The cleaning step is critical. Raw API responses can be several KB: if you inject them into the context verbatim, it uses a lot of context. The rule: **only surface what the agent needs to act on.**

For academic papers that means: `title`, `year`, `authors[0..2]`, `abstract[:400]`, `citation_count`, `paper_id`. Everything else is discarded before the LLM sees it.

---

## Token optimisation

Every token in your context window costs money and latency. More importantly, LLMs degrade with very long contexts: they start "forgetting" early content or producing less coherent outputs.

Practical limits for a research agent:
- Abstract: truncate to 400–500 characters.
- Author list: keep first 3.
- Reference lists: keep titles only, not full metadata.

The `_truncate_paper()` helper in `01_api_tools.py` formalises these limits. Treat them as configurable constants, not hardcoded magic numbers.

---

## Semantic drift

When you crawl citation networks, you will inevitably pull in papers that are adjacent to your topic but not truly relevant. This is **semantic drift** — the agent slowly drifts away from the target topic as it follows references.

The evaluation tool (exercise 2) is the guard against drift. Every paper that enters the pipeline goes through a relevance check before it touches the `curated_library`. Papers that score below the threshold go into `blacklist`, not the library.

The threshold is a design choice. A low threshold produces a broad, inclusive review. A high threshold produces a tight, focused one.

---

## Citation graph traversal

The citation graph has two directions: "Which papers cite Paper A?", to find SOTA and follow-on work, and "What does Paper A cite?", to find foundational work.

Semantic Scholar exposes both directions via the `/paper/{id}/citations` and `/paper/{id}/references` endpoints.

---

## The Semantic Scholar API

Base URL: `https://api.semanticscholar.org/graph/v1`

Key endpoints used today:
- `GET /paper/search?query=...&fields=...` — keyword search
- `GET /paper/{paper_id}?fields=...` — fetch one paper
- `GET /paper/{paper_id}/references?fields=...` — backward traversal
- `GET /paper/{paper_id}/citations?fields=...` — forward traversal

No API key needed for basic use (rate-limited to ~100 req/5 min). Set `SEMANTIC_SCHOLAR_API_KEY` in `.env` for higher limits.

### arXiv API (fallback)

Base URL: `http://export.arxiv.org/api/query`

Useful when Semantic Scholar is down or a paper isn't in its database. Returns Atom XML; the `01_api_tools.py` script parses it with the `xml.etree.ElementTree` module (no third-party XML library needed).

---

## Files

### `01_api_tools.py`

**What it does**: Wraps the Semantic Scholar and arXiv APIs into clean Python functions that return normalised paper dictionaries. The token budget for each paper is enforced here.

**What to focus on**:
- The `_make_request()` helper with retry logic and rate-limit handling
- The `_truncate_paper()` function and its configurable constants
- The arXiv fallback in `search_papers()`

**Exercise**: Change `ABSTRACT_MAX_CHARS` from 400 to 800. Run a search query and observe how the number of papers you can hold in context before overflow shrinks.

---

### `02_evaluator.py`

**What it does**: Takes a raw paper payload and a target topic, and asks the LLM to score relevance on a structured markdown scorecard. Returns a float score and a verdict string.

**What to focus on**:
- The scorecard rubric in the system prompt (this is a prompt-engineering exercise)
- How the threshold comparison routes papers to `curated_library` vs `blacklist`
- The `parse_score()` function that extracts a float from LLM text output

**Exercise**: The current rubric has 3 criteria. Add a fourth: "Does this paper introduce a novel method, or is it a survey/review?" Surveys are less valuable as citations for a related works section.

---

### `03_citation_graph.py`

**What it does**: Given a paper ID already in `curated_library`, fetches its references (backward) and citations (forward), truncates the list to the top-N most relevant by citation count, and adds them to `discovery_queue`.

**What to focus on**:
- How the graph traversal adds to `discovery_queue` rather than `curated_library` directly
- The `already_seen()` check that prevents re-queuing papers we already have
- The depth parameter that limits how many hops you follow

**Exercise**: Implement a depth-limited BFS across the citation graph. Start from all papers in `curated_library` and expand twice. Print the resulting queue size.

---

## Running the exercises

```bash
cd day2
python 01_api_tools.py         # live API call to Semantic Scholar
python 02_evaluator.py         # evaluate a hardcoded paper stub
python 03_citation_graph.py    # graph traversal from a known paper ID
```

`01_api_tools.py` makes real HTTP requests. It will fail if you're offline. All three exercises work standalone.
