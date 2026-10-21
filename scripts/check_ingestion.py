"""Check that the evidence of each question survived parsing and chunking.

For every evidence passage of the chosen split:
- page coverage:  share of its words found in the parsed text of its page
                  (low = parsing problem: missing text, wrong page number)
- chunk coverage: same, in the best single chunk of that page
                  (much lower than page coverage = the chunks cut the evidence)

Usage:
    python scripts/check_ingestion.py              # dev set
    python scripts/check_ingestion.py --split test
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.config import get_settings, resolve
from finquery.data.financebench import read_jsonl
from finquery.evaluation.metrics import token_coverage
from finquery.ingestion.chunker import text_for_embedding

LOW = 0.8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    data = get_settings()["data"]
    processed = resolve(data["processed_dir"])
    questions = read_jsonl(resolve(data["eval_dir"]) / f"{args.split}.jsonl")

    chunks_by_page = defaultdict(list)
    for chunk in read_jsonl(processed / "chunks.jsonl"):
        chunks_by_page[chunk["doc_name"], chunk["page"]].append(text_for_embedding(chunk))
    page_text = {}
    for path in (processed / "pages").glob("*.jsonl"):
        for p in read_jsonl(path):
            page_text[p["doc_name"], p["page"]] = p["text"]

    rows = []
    for q in questions:
        for ev in q["evidence"]:
            key = (ev["doc_name"], ev["page"])
            page_cov = token_coverage(ev["text"], page_text.get(key, ""))
            chunk_cov = max((token_coverage(ev["text"], c) for c in chunks_by_page[key]), default=0.0)
            rows.append((q["financebench_id"], *key, page_cov, chunk_cov))

    print(f"{'question':<24}{'document':<42}{'page':>5}{'page cov':>10}{'chunk cov':>11}")
    for qid, doc, page, page_cov, chunk_cov in rows:
        flag = "  <-- check" if min(page_cov, chunk_cov) < LOW else ""
        print(f"{qid:<24}{doc:<42}{page:>5}{page_cov:>10.0%}{chunk_cov:>11.0%}{flag}")
    n = len(rows)
    print(f"\n{args.split}: {n} evidence passages | "
          f"mean page coverage {sum(r[3] for r in rows) / n:.0%} | "
          f"mean chunk coverage {sum(r[4] for r in rows) / n:.0%} | "
          f"below {LOW:.0%}: {sum(min(r[3], r[4]) < LOW for r in rows)}")


if __name__ == "__main__":
    main()
