"""
main.py — CLI Entry Point for the Semantic Rating Agent.

Usage:
    python main.py

The agent will:
  1. Ask for a CSV path (defaults to data/Mobile Reviews Sentiment.csv)
  2. Run full analysis or reload from memory.json
  3. Start interactive loop supporting both:
     A) Dataset Questions (e.g., "Which variables are strongly associated with ratings?")
     B) Unseen Review Validation (e.g., "Loving the clean UI and fast updates. Best purchase of the year!")
"""

from __future__ import annotations

import os
import sys

# ── UTF-8 output on Windows ──────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agent import SemanticRatingAgent

DEFAULT_CSV    = os.path.join("data", "Mobile Reviews Sentiment.csv")
DEFAULT_MEMORY = "memory.json"

BANNER = """==================================================
SEMANTIC RATING AGENT
=================================================="""


def main() -> None:
    print(BANNER)
    agent = SemanticRatingAgent()

    # ── CSV path ──────────────────────────────────────────────────────────────
    print(f"\n  Default CSV path: {DEFAULT_CSV}")
    raw = input("  Enter CSV path (press Enter for default): ").strip()
    csv_path = raw if raw else DEFAULT_CSV

    if not os.path.exists(csv_path):
        print(f"\n  ERROR: File not found: {csv_path}")
        sys.exit(1)

    # ── Offer to reuse existing memory ────────────────────────────────────────
    if os.path.exists(DEFAULT_MEMORY):
        print(f"\n  Found existing memory file: {DEFAULT_MEMORY}")
        reuse = input("  Reuse existing analysis? [Y/n]: ").strip().lower()
        if reuse != "n":
            agent.load_memory(DEFAULT_MEMORY)
        else:
            agent.run_full_analysis(csv_path, DEFAULT_MEMORY)
    else:
        agent.run_full_analysis(csv_path, DEFAULT_MEMORY)

    # ── Interactive Loop ──────────────────────────────────────────────────────
    print("\nDataset understanding loaded.\n")
    print("Agent ready.\n")
    print("Ask any natural-language question or paste an unseen review to analyze.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        low = user_input.lower()
        if low in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        response = agent.answer_question(user_input)
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
