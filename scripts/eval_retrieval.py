"""Evaluate retrieval on the dev set: dense vs BM25 vs hybrid vs hybrid + rerank,
over the whole corpus and inside the question's document.

Usage:
    python scripts/eval_retrieval.py                 # dev set (use this while tuning)
    python scripts/eval_retrieval.py --split test    # only once, at the very end (step 8)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.config import ROOT, get_settings, resolve
from finquery.data.financebench import read_jsonl
from finquery.evaluation.retrieval_eval import evaluate, format_summary
from finquery.retrieval.retriever import MODES, build_retriever


def first_hit_table(questions: list[dict], setting: str) -> str:
    """Rank of the first chunk from an evidence page, per question and mode ('-' = not in top k)."""
    lines = [f"{'question':<24}{'type':<19}" + "".join(f"{m:>15}" for m in MODES)]
    for q in questions:
        ranks = [q["runs"].get(f"{setting}/{m}", {}).get("first_hit") for m in MODES]
        lines.append(f"{q['financebench_id']:<24}{q['question_type']:<19}"
                     + "".join(f"{r if r else '-':>15}" for r in ranks))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    settings = get_settings()
    questions = read_jsonl(resolve(settings["data"]["eval_dir"]) / f"{args.split}.jsonl")
    retriever, client = build_retriever(settings)

    start = time.perf_counter()
    report = evaluate(retriever, questions)
    client.close()

    print(f"\n{args.split}: {len(questions)} questions, {time.perf_counter() - start:.0f} s "
          f"(chunks of {settings['chunking']['max_words']} words, "
          f"{settings['retrieval']['candidates']} candidates)\n")
    print(format_summary(report["summary"]))
    for setting in ("corpus", "document"):
        print(f"\nFirst evidence hit per question ({setting}):")
        print(first_hit_table(report["questions"], setting))

    out = ROOT / "results" / f"retrieval_{args.split}.json"
    out.parent.mkdir(exist_ok=True)
    report["config"] = {k: settings[k] for k in ("chunking", "embedding", "reranker", "retrieval")}
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nDetails saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
