"""
Day 1, Exercise 3: hallucination handler

Demonstrates what happens when the model invents a tool that doesn't exist, and how to recover without crashing the agent.

The system prompt deliberately describes a plausible academic workflow that might tempt a model to reach for tools like `search_google_scholar_by_author` or `get_impact_factor`. These don't exist in our registry. When the model calls them, we intercept, log, and retry.

Key lessons:
  - The difference between a crashed agent and a recovering agent
  - How error messages should be worded to guide correction
  - The retry budget (don't loop forever on a stubborn model)

Run:
    python day1/03_hallucination_handler.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json

MAX_STEPS = 10
MAX_CONSECUTIVE_ERRORS = 3  # bail out if the model keeps hallucinating


# ─── Small, intentionally limited tool registry ───────────────────────────────

def search_papers(query: str, max_results: int = 5) -> dict:
    return {"results": [{"id": "p001", "title": f"Paper about {query}"}] * min(max_results, 3)}


def evaluate_paper(paper_id: str) -> dict:
    return {"paper_id": paper_id, "score": 0.75, "verdict": "relevant"}


def finish(summary: str = "") -> dict:
    return {"status": "done", "summary": summary}


TOOLS = {
    "search_papers": search_papers,
    "evaluate_paper": evaluate_paper,
    "finish": finish,
}


# ─── System prompt (intentionally generic — invites hallucination) ─────────────

SYSTEM_PROMPT = """\
You are an expert academic research assistant for PhD students.
You help find and evaluate papers for literature reviews.

You have access to these tools ONLY:
  - search_papers(query: str, max_results: int): Search for academic papers.
  - evaluate_paper(paper_id: str): Assess a paper's relevance score.
  - finish(summary: str): End the session with a summary.

Respond with ONLY a JSON object:
{
  "reasoning": "<your thinking>",
  "action": {
    "tool": "<one of the three tools above>",
    "args": { ... }
  }
}

Important: you must ONLY use the three tools listed. Do not use any other tools.
"""


# ─── Hallucination recovery ───────────────────────────────────────────────────

def build_hallucination_error(bad_tool: str) -> dict:
    """
    Craft an error observation that corrects the model without being unhelpfully
    terse. The goal is to give the model enough signal to self-correct on the
    next turn.
    """
    return {
        "error": f"Tool '{bad_tool}' does not exist in this agent.",
        "available_tools": list(TOOLS.keys()),
        "instruction": (
            f"'{bad_tool}' was not in the list of available tools you were given. "
            "Please re-read the system prompt and choose from the available tools only."
        ),
    }


def run_with_hallucination_recovery(topic: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Find papers for this topic: {topic}"},
    ]

    consecutive_errors = 0
    hallucination_log: list[str] = []

    for step in range(1, MAX_STEPS + 1):
        print(f"\n{'─' * 50}")
        print(f"STEP {step}  (consecutive errors: {consecutive_errors})")

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"[ABORT] Model produced {MAX_CONSECUTIVE_ERRORS} consecutive errors. "
                  "This indicates a systemic prompt or model issue.")
            break

        raw = chat(messages, max_tokens=400, temperature=0.3)
        print(f"[LLM raw]: {raw}")

        # ── Parse ──
        try:
            parsed = _extract_json(raw)
        except ValueError:
            consecutive_errors += 1
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "error": "Response was not valid JSON.",
                    "instruction": "You must respond with ONLY a JSON object. No other text.",
                }),
            })
            continue

        messages.append({"role": "assistant", "content": json.dumps(parsed)})
        print(f"[Reasoning]: {parsed.get('reasoning', '')}")

        action = parsed.get("action", {})
        tool_name = action.get("tool", "")
        tool_args = action.get("args", {})

        # ── Hallucination check ──
        if tool_name not in TOOLS:
            consecutive_errors += 1
            hallucination_log.append(tool_name)
            error_obs = build_hallucination_error(tool_name)
            print(f"[HALLUCINATION DETECTED] Model called: '{tool_name}'")
            print(f"[Injecting error]: {error_obs}")
            messages.append({"role": "user", "content": json.dumps(error_obs)})
            continue

        # ── Valid tool — reset error counter and execute ──
        consecutive_errors = 0

        if tool_name == "finish":
            print(f"\n{'=' * 50}")
            print("Agent finished successfully.")
            print(f"Summary: {tool_args.get('summary', '(none)')}")
            break

        try:
            observation = TOOLS[tool_name](**tool_args)
        except TypeError as exc:
            observation = {"error": f"Bad arguments for '{tool_name}': {exc}"}
            consecutive_errors += 1

        print(f"[Tool result]: {json.dumps(observation)}")
        messages.append({"role": "user", "content": json.dumps({"observation": observation})})

    # ── Post-run report ──
    print(f"\n{'=' * 50}")
    print("HALLUCINATION REPORT")
    print(f"  Total steps: {step}")
    print(f"  Hallucinated tools: {len(hallucination_log)}")
    if hallucination_log:
        from collections import Counter
        counts = Counter(hallucination_log)
        for name, count in counts.most_common():
            print(f"    '{name}' called {count} time(s)")
    else:
        print("  None detected — model stayed within the tool registry.")


if __name__ == "__main__":
    topic = "Contrastive learning over sparse hypergraphs"
    run_with_hallucination_recovery(topic)
