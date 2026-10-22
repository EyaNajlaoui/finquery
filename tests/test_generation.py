import json

from finquery.evaluation.answer_eval import code_checks, judge, numeric_match, summarize
from finquery.generation.answerer import Answerer, is_refusal, parse_citations
from finquery.generation.context import ContextBuilder, format_passages, merge_overlapping
from finquery.generation.llm import LLMResponse


def chunk(page, index, text, doc="AMAZON_2017_10K"):
    return {"chunk_id": f"{doc}:p{page}:c{index}", "doc_name": doc, "company": "Amazon",
            "doc_type": "10k", "doc_period": 2017, "page": page, "chunk_index": index,
            "context": "Amazon", "text": text}


CHUNKS = [chunk(38, 0, "STATEMENTS OF OPERATIONS\n2016 2017"),
          chunk(38, 1, "2016 2017\nNet sales 135,987 177,866"),
          chunk(38, 2, "Net sales 135,987 177,866\nCost of sales 88,265 111,934"),
          chunk(40, 0, "Accounts payable 25,309 34,616")]
AMZN = "AMAZON_2017_10K"


def test_merge_overlapping_removes_repeated_lines():
    assert merge_overlapping(["a\nb\nc", "b\nc\nd", "d\ne"]) == "a\nb\nc\nd\ne"


def test_window_rebuilds_the_page_around_the_hit():
    builder = ContextBuilder(CHUNKS, mode="window", neighbours=1, max_passages=4)
    results = [(CHUNKS[1], 1.0), (CHUNKS[3], 0.9), (CHUNKS[2], 0.8)]
    passages = builder.build(results)
    assert [(p["doc_name"], p["page"]) for p in passages] == [(AMZN, 38), (AMZN, 40)]
    assert passages[0]["text"] == "STATEMENTS OF OPERATIONS\n2016 2017\nNet sales 135,987 177,866\n" \
                                  "Cost of sales 88,265 111,934"


def test_chunks_mode_and_max_passages():
    builder = ContextBuilder(CHUNKS, mode="chunks", max_passages=2)
    assert len(builder.build([(c, 1.0) for c in CHUNKS])) == 2
    builder = ContextBuilder(CHUNKS, mode="window", max_passages=1)
    assert len(builder.build([(CHUNKS[3], 1.0), (CHUNKS[0], 0.5)])) == 1


def test_format_passages_starts_with_citation_tag():
    text = format_passages(ContextBuilder(CHUNKS, mode="chunks").build([(CHUNKS[3], 1.0)]))
    assert text.startswith("[AMAZON_2017_10K p.40] Amazon | 10-K | FY2017")


def test_parse_citations_and_refusal():
    text = f"Sales 177,866 [{AMZN} p.38] and AP 34,616 [{AMZN} p.40] [{AMZN} p.38]."
    assert parse_citations(text) == [("AMAZON_2017_10K", 38), ("AMAZON_2017_10K", 40)]
    assert parse_citations("[JOHNSON_JOHNSON_2023_8K_dated-2023-08-30 p.4]")[0][1] == 4
    assert is_refusal("The provided documents do not contain enough information to answer.")


class FakeRetriever:
    def search(self, question, k, mode, filters):
        return [(CHUNKS[1], 1.0), (CHUNKS[3], 0.5)]


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def chat(self, system, user, json_mode=False):
        self.last_user = user
        return LLMResponse(self.text, "fake", 0.1, {"prompt_tokens": 10, "completion_tokens": 5})


def test_answerer_flags_invalid_citations():
    llm = FakeLLM("Net sales were $177,866 million [AMAZON_2017_10K p.38] [NIKE_2019_10K p.3].")
    answerer = Answerer(FakeRetriever(), ContextBuilder(CHUNKS, mode="window"), llm)
    result = answerer.answer("What were Amazon's FY2017 net sales?")
    assert "[AMAZON_2017_10K p.38]" in llm.last_user
    assert result["invalid_citations"] == [("NIKE_2019_10K", 3)]
    assert not result["refusal"]


def test_judge_parses_json_and_survives_garbage():
    good = FakeLLM(json.dumps({"grade": "correct", "reason": "same number"}))
    assert judge(good, "q", "ref", "cand") == {"grade": "correct", "reason": "same number"}
    assert judge(FakeLLM("not json"), "q", "ref", "cand")["grade"] == "incorrect"


def test_numeric_match():
    assert numeric_match("$11588.00", "Net sales were $11,588 million.") is True
    assert numeric_match("5.4%", "The EBITDA margin is 5.43%.") is True
    assert numeric_match("5.4%", "The EBITDA margin is 7.1%.") is False
    assert numeric_match("The quick ratio improved from 0.67 to 0.69", "anything") is None


def test_code_checks_and_summary():
    question = {"answer": "$34616.00", "evidence": [{"doc_name": "AMAZON_2017_10K", "page": 40}]}
    result = {"answer": "AP was $34,616 million [AMAZON_2017_10K p.40]",
              "passages": [{"doc_name": "AMAZON_2017_10K", "page": 40}],
              "citations": [["AMAZON_2017_10K", 40]], "invalid_citations": []}
    checks = code_checks(question, result)
    assert checks == {"evidence_in_context": True, "has_citation": True, "cites_evidence": True,
                      "invalid_citations": 0, "numeric_match": True}
    row = {"question_type": "metrics-generated", "judge": {"grade": "partial"}, "checks": checks,
           "refusal": False, "seconds": 2.0, "usage": {"prompt_tokens": 1000}}
    summary = summarize([row, {**row, "judge": {"grade": "correct"}}])
    assert summary["score"] == 0.75 and summary["correct"] == 0.5
    assert summary["diagnosis"]["evidence found, answer not correct (generation problem)"] == 1


def test_citations_with_fullwidth_brackets():
    from finquery.generation.answerer import check_citations
    passages = [{"doc_name": AMZN, "page": 38}]
    report = check_citations(f"Sales were $177,866 million【{AMZN} p.38】and 【{AMZN} p. 40】.", passages)
    assert report == {"citations": [(AMZN, 38), (AMZN, 40)], "invalid_citations": [(AMZN, 40)]}


def test_signature_changes_with_prompt():
    from finquery.generation.prompts import signature
    assert signature("prompt v1", 4) == signature("prompt v1", 4) != signature("prompt v2", 4)


def test_judge_accepts_json_inside_text():
    text = 'Here is my grade: {"grade": "partial", "reason": "one item missing"} Done.'
    assert judge(FakeLLM(text), "q", "ref", "cand") == {"grade": "partial", "reason": "one item missing"}
