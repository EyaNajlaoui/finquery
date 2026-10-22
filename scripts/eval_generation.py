"""Generate answers for the dev questions, grade them, and explain the errors.

Answers and grades are cached in results/answers_<split>_<context>_<setting>.jsonl. Re-running only
calls the LLM for what is missing or out of date: an answer is regenerated when the prompt or the
settings change, a grade when the answer or the judge prompt changes (useful with Groq rate limits).

Usage:
    python scripts/eval_generation.py                       # dev, context from settings.yaml
    python scripts/eval_generation.py --context chunks      # compare context modes
    python scripts/eval_generation.py --rejudge             # keep answers, grade again
    python scripts/eval_generation.py --regenerate          # generate everything again
    python scripts/eval_generation.py --limit 5             # quick try on 5 questions
    python scripts/eval_generation.py --system agent        # step 5: tool-calling agent
    python scripts/eval_generation.py --limit 10 --model openai/gpt-oss-20b   # smaller model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.agent.agent import AGENT_SYSTEM, TOOLS
from finquery.config import ROOT, get_settings, resolve
from finquery.data.financebench import read_jsonl, write_jsonl
from finquery.evaluation.answer_eval import code_checks, judge, summarize
from finquery.generation.answerer import check_citations
from finquery.generation.llm import DailyLimitError, GroqLLM
from finquery.generation.pipeline import build_agent, build_answerer
from finquery.generation.prompts import ANSWER_SYSTEM, ANSWER_USER, JUDGE_SYSTEM, JUDGE_USER, signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--system", default="rag", choices=["rag", "agent"],
                        help="rag: one search then one answer (step 4); agent: tool calling (step 5)")
    parser.add_argument("--context", choices=["chunks", "window", "page"])
    parser.add_argument("--setting", default="corpus", choices=["corpus", "document"],
                        help="search all filings, or only the question's filing")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--model", help="answer with another Groq model than settings.yaml "
                                        "(e.g. openai/gpt-oss-20b when the main model's daily quota is used)")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--rejudge", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    context_mode = args.context or settings["generation"]["context_mode"]
    questions = read_jsonl(resolve(settings["data"]["eval_dir"]) / f"{args.split}.jsonl")[:args.limit]
    model = args.model or settings["llm"]["model"]
    name = f"{args.split}_{context_mode}_{args.setting}" if args.system == "rag" else \
        f"{args.split}_agent_{args.setting}"
    if args.model:  # results of another model never mix with the main ones
        name += "_" + args.model.split("/")[-1]
    out = ROOT / "results" / f"answers_{name}.jsonl"
    cache = {} if args.regenerate or not out.exists() else {r["financebench_id"]: r for r in read_jsonl(out)}
    pause = settings["llm"].get("pause_seconds", 0)

    # A cached answer is reused only if prompts and settings are unchanged (same for grades)
    gen, llm_cfg = settings["generation"], settings["llm"]
    answer_sig = signature(ANSWER_SYSTEM, ANSWER_USER, model, context_mode, gen.get("neighbours"),
                           gen.get("retrieval_mode"), settings["retrieval"], settings["chunking"])
    if args.system == "agent":
        answer_sig = signature(answer_sig, AGENT_SYSTEM, json.dumps(TOOLS), settings["agent"])
    judge_sig = signature(JUDGE_SYSTEM, JUDGE_USER, llm_cfg["judge_model"])

    answerer = client = None
    judge_llm = GroqLLM.from_settings(settings, role="judge_model", fallback=False)
    answer_llm = GroqLLM(model, llm_cfg.get("temperature", 0.0), llm_cfg.get("max_retries", 10))
    # no fallback: every answer of a run comes from the same model
    rows = []
    try:
        for i, q in enumerate(questions, start=1):
            row = cache.get(q["financebench_id"])
            if row is None or row.get("answer_signature") != answer_sig:
                if answerer is None:  # load the models only if something must be generated
                    answerer, client = build_agent(settings, llm=answer_llm) if args.system == "agent" \
                        else build_answerer(settings, context_mode, llm=answer_llm)
                filters = {"doc_name": q["doc_name"]} if args.setting == "document" else None
                try:
                    result = answerer.answer(q["question"], filters)
                except DailyLimitError:
                    raise
                except Exception as error:  # one failing question must not stop the whole run
                    print(f"{i:>2}/{len(questions)} !  {q['financebench_id']} skipped: {str(error)[:150]}")
                    continue
                row = {"financebench_id": q["financebench_id"], "question_type": q["question_type"],
                       "question": q["question"], "reference": q["answer"], **result,
                       "answer_signature": answer_sig}
                time.sleep(pause)
            row.update(check_citations(row["answer"], row["passages"]))  # parser fixes apply to old answers
            row["checks"] = code_checks(q, row)
            grade_sig = signature(judge_sig, row["answer"])
            if args.rejudge or row.get("judge_signature") != grade_sig:
                row["judge"] = judge(judge_llm, q["question"], q["answer"], row["answer"])
                row["judge_signature"] = grade_sig
                time.sleep(pause)
            rows.append(row)
            cache[q["financebench_id"]] = row
            write_jsonl(cache.values(), out)  # saved after each question: a crash loses nothing
            mark = {"correct": "OK ", "partial": "~  ", "incorrect": "X  "}[row["judge"]["grade"]]
            found = row["checks"]["evidence_in_context"]
            print(f"{i:>2}/{len(questions)} {mark} {q['financebench_id']} "
                  f"(evidence in context: {found}) {row['judge']['reason'][:90]}")
    except DailyLimitError as error:
        print(f"\nStopped: {error}\nEverything done so far is saved in {out.name}: "
              "run the same command later to continue where it stopped.")
        return
    finally:
        if client is not None:
            client.close()

    summary = {"system": args.system, "model": model, "context": context_mode, "setting": args.setting,
               **summarize(rows)}
    if args.system == "agent":
        summary["avg_steps"] = round(sum(r.get("steps", 0) for r in rows) / len(rows), 2)
        summary["avg_searches"] = round(sum(sum(t["tool"] == "search_filings" for t in r.get("trace", []))
                                            for r in rows) / len(rows), 2)
        summary["avg_calculations"] = round(sum(sum(t["tool"] == "calculate" for t in r.get("trace", []))
                                                for r in rows) / len(rows), 2)
    print(f"\n{args.split} | system={args.system} | model={model} | context={context_mode} | "
          f"setting={args.setting} | {len(rows)} questions")
    print(json.dumps(summary, indent=2))
    (ROOT / "results" / f"generation_{name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
