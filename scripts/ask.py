"""Ask FinQuery one question from the command line.

Usage:
    python scripts/ask.py "What were Costco's total assets at the end of FY2021?"
    python scripts/ask.py "What is the FY2017 DPO?" --company Amazon --year 2017
    python scripts/ask.py "What is Amcor's FY2023 quick ratio?" --agent
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.config import get_settings
from finquery.generation.pipeline import build_agent, build_answerer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--company")
    parser.add_argument("--year", type=int, help="fiscal year (doc_period)")
    parser.add_argument("--doc", help="document name, e.g. AMAZON_2017_10K")
    parser.add_argument("--agent", action="store_true", help="use the tool-calling agent (step 5)")
    args = parser.parse_args()

    settings = get_settings()
    answerer, client = build_agent(settings) if args.agent else build_answerer(settings)
    try:
        result = answerer.answer(args.question, {"company": args.company, "doc_period": args.year,
                                                 "doc_name": args.doc})
    finally:
        client.close()

    for t in result.get("trace", []):
        print(f"  step {t['step']}: {t['tool']}({t['arguments']})")
    print("\n" + result["answer"])
    print("\nSources given to the model:")
    for p in result["passages"]:
        cited = "cited" if (p["doc_name"], p["page"]) in result["citations"] else "     "
        print(f"  [{cited}] {p['doc_name']} p.{p['page']}")
    if result["invalid_citations"]:
        print(f"  WARNING: citations not in the sources: {result['invalid_citations']}")
    print(f"\n{result['seconds']} s, {result['usage'].get('prompt_tokens', '?')} prompt tokens")


if __name__ == "__main__":
    main()
