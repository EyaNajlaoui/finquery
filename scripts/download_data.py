"""Download the FinanceBench subset and build data/eval/dev.jsonl and test.jsonl.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.config import get_settings, resolve
from finquery.data.financebench import (
    merge_documents,
    normalize,
    read_jsonl,
    select,
    stratified_split,
    write_jsonl,
)

TIMEOUT = 120


def download(url: str, target: Path) -> None:
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    target.write_bytes(response.content)


def load_questions(cfg: dict) -> list[dict]:
    """Try the merged Hugging Face file first, then the two GitHub files."""
    with tempfile.TemporaryDirectory() as tmp:
        merged = Path(tmp) / "merged.jsonl"
        try:
            download(cfg["questions_url"], merged)
            print("Questions: downloaded financebench_merged.jsonl (Hugging Face)")
            return read_jsonl(merged)
        except Exception as error:  # network error, 404, blocked host...
            print(f"Hugging Face unavailable ({error.__class__.__name__}), using GitHub instead")
        questions, documents = Path(tmp) / "q.jsonl", Path(tmp) / "d.jsonl"
        download(cfg["fallback_questions_url"], questions)
        download(cfg["fallback_documents_url"], documents)
        return merge_documents(read_jsonl(questions), read_jsonl(documents))


def download_pdfs(cfg: dict, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in cfg["documents"]:
        target = raw_dir / f"{name}.pdf"
        if target.exists() and target.read_bytes()[:4] == b"%PDF":
            print(f"  already here: {target.name}")
            continue
        download(cfg["pdf_url"].format(doc_name=name), target)
        if target.read_bytes()[:4] != b"%PDF":
            target.unlink()
            raise RuntimeError(f"{name}: the downloaded file is not a PDF")
        print(f"  downloaded:   {target.name} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    settings = get_settings()
    cfg = settings["data"]

    questions = [normalize(q) for q in select(load_questions(cfg), cfg["documents"])]
    missing = set(cfg["documents"]) - {q["doc_name"] for q in questions}
    if missing:
        raise RuntimeError(f"No questions found for: {sorted(missing)}")

    dev, test = stratified_split(questions, cfg["dev_size"], cfg["seed"])
    eval_dir = resolve(cfg["eval_dir"])
    write_jsonl(dev, eval_dir / "dev.jsonl")
    write_jsonl(test, eval_dir / "test.jsonl")
    print(f"\n{len(questions)} questions -> dev {len(dev)}, test {len(test)}")
    for name, rows in (("dev", dev), ("test", test)):
        print(f"  {name:<4} {dict(Counter(q['question_type'] for q in rows))}")

    print(f"\nPDFs -> {cfg['raw_dir']}/")
    download_pdfs(cfg, resolve(cfg["raw_dir"]))
    print("\nDone.")


if __name__ == "__main__":
    main()
