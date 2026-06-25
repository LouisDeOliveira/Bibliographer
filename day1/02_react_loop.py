"""
Day 1, Exercise 2: manual ReAct loop

A minimal, self-contained implementation of the ReAct (Reason, Act, Observe) pattern. The agent has three toy tools and runs until it calls `finish` or hits MAX_STEPS.

The critical mechanic to study: the messages list grows on every iteration. Each tool result is injected back as a `user` turn. The model "sees" its own previous actions via this history.

Key lessons:
  - How conversation history encodes agent memory within a single session
  - How tool outputs are surfaced back to the model
  - The stop condition: the model drives termination by calling `finish`

Run:
    python day1/02_react_loop.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json

MAX_STEPS = 8

# ─── Toy tool implementations ─────────────────────────────────────────────────
#
# These return fake data. Day 2 replaces them with real API calls.

def search_papers(query: str, max_results: int = 3) -> dict:
    """Simulates a paper search. Returns fake paper stubs."""
    fake_papers = [
        {
            "paper_id": "paper_001",
            "title": "Hypergraph Neural Networks for Semi-Supervised Classification",
            "year": 2019,
            "abstract_snippet": "We propose HGNN, a general framework for data fitting with hypergraph structure...",
        },
        {
            "paper_id": "paper_002",
            "title": "Self-Supervised Contrastive Learning on Graphs",
            "year": 2021,
            "abstract_snippet": "GraphCL applies contrastive learning to graph-structured data via augmentation...",
        },
        {
            "paper_id": "paper_003",
            "title": "Sparse Hypergraph Representation Learning",
            "year": 2023,
            "abstract_snippet": "We address scalability in hypergraph learning via sparse attention mechanisms...",
        },
    ]
    return {"query": query, "results": fake_papers[:max_results]}


def evaluate_paper(paper_id: str) -> dict:
    """Simulates paper relevance evaluation. Always returns a score."""
    scores = {
        "paper_001": {"score": 0.72, "verdict": "relevant", "reason": "HGNN directly relates to hypergraph learning"},
        "paper_002": {"score": 0.65, "verdict": "relevant", "reason": "Contrastive learning on graphs is closely related"},
        "paper_003": {"score": 0.88, "verdict": "highly_relevant", "reason": "Sparse hypergraphs are the exact target domain"},
    }
    return scores.get(
        paper_id,
        {"score": 0.1, "verdict": "irrelevant", "reason": "Paper not found in mock database"},
    )


def finish(summary: str = "") -> dict:
    """Signals that the agent is done with this reasoning session."""
    return {"status": "finished", "summary": summary}


# ─── Tool registry ────────────────────────────────────────────────────────────

TOOLS = {
    "search_papers": search_papers,
    "evaluate_paper": evaluate_paper,
    "finish": finish,
}

TOOL_DESCRIPTIONS = [
    {"name": "search_papers",  "args": "query: str, max_results: int = 3",
     "description": "Search for academic papers matching a keyword query."},
    {"name": "evaluate_paper", "args": "paper_id: str",
     "description": "Score a paper's relevance to the target topic (0.0–1.0)."},
    {"name": "finish",         "args": "summary: str = ''",
     "description": "End the session. Call this when you have enough information."},
]


# ─── Prompt construction ──────────────────────────────────────────────────────

def build_system_prompt(topic: str, tools: list[dict]) -> str:
    tool_block = "\n".join(
        f"  - {t['name']}({t['args']}): {t['description']}" for t in tools
    )
    return f"""\
You are a research assistant helping build a literature review on:
  "{topic}"

Available tools:
{tool_block}

On every turn, respond with ONLY a valid JSON object:
{{
  "reasoning": "<your thinking about what to do next>",
  "action": {{
    "tool": "<tool_name>",
    "args": {{ "<arg>": "<value>" }}
  }}
}}

No prose. No markdown fences. Only the JSON object.
Call `finish` when you have searched for at least 2 queries and evaluated at least 2 papers.
"""


# ─── ReAct loop ───────────────────────────────────────────────────────────────

def run_react_loop(topic: str) -> None:
    messages = [
        {"role": "system", "content": build_system_prompt(topic, TOOL_DESCRIPTIONS)},
        {"role": "user",   "content": f'Begin the literature search. Current state: {{"curated_library": {{}}, "discovery_queue": []}}'},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n{'─' * 50}")
        print(f"STEP {step}")

        # ── THINK ──
        raw = chat(messages, max_tokens=512, temperature=0.2)
        print(f"[LLM raw]: {raw}")

        # ── PARSE ──
        try:
            parsed = _extract_json(raw)
        except ValueError as exc:
            print(f"[ERROR] Could not parse LLM output: {exc}")
            # Inject the parse error and continue — the model will self-correct
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "error": "Your response was not valid JSON.",
                    "instruction": "Respond with ONLY the JSON object described in the system prompt.",
                }),
            })
            continue

        messages.append({"role": "assistant", "content": json.dumps(parsed)})
        print(f"[Reasoning]: {parsed.get('reasoning', '')}")

        action = parsed.get("action", {})
        tool_name = action.get("tool", "")
        tool_args = action.get("args", {})

        print(f"[Action]: {tool_name}({tool_args})")

        # ── ACT ──
        if tool_name == "finish":
            observation = finish(**{k: v for k, v in tool_args.items()})
            print(f"[Observation]: {observation}")
            print(f"\n{'=' * 50}")
            print(f"Agent finished after {step} steps.")
            print(f"Summary: {tool_args.get('summary', '(none)')}")
            return

        if tool_name not in TOOLS:
            # This is covered fully in exercise 3, but we handle it here too
            observation = {
                "error": f"Tool '{tool_name}' does not exist.",
                "available_tools": list(TOOLS.keys()),
            }
        else:
            try:
                observation = TOOLS[tool_name](**tool_args)
            except TypeError as exc:
                observation = {"error": f"Wrong arguments for {tool_name}: {exc}"}

        print(f"[Observation]: {json.dumps(observation)[:200]}...")

        # ── OBSERVE (inject result into history) ──
        messages.append({
            "role": "user",
            "content": json.dumps({"observation": observation}),
        })

    print(f"\n[LOOP] Reached MAX_STEPS ({MAX_STEPS}). Terminating.")


if __name__ == "__main__":
    topic = "Contrastive learning over sparse hypergraphs"
    run_react_loop(topic)
