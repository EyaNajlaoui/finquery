import json

from finquery.agent.agent import Agent, catalog_text
from finquery.generation.context import ContextBuilder
from finquery.generation.llm import ChatTurn, ToolCall

DOC = {"doc_name": "AMCOR_2023_10K", "company": "Amcor", "doc_type": "10k", "doc_period": 2023}
CHUNKS = [
    {**DOC, "chunk_id": "a:p52:c0", "page": 52, "chunk_index": 0, "context": "Amcor",
     "text": "Total current assets 5,853 ... Total current liabilities 5,308"},
    {**DOC, "chunk_id": "a:p50:c0", "page": 50, "chunk_index": 0, "context": "Amcor",
     "text": "Net sales 14,694"},
]


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k, mode, filters):
        self.calls.append((query, filters))
        return [(CHUNKS[0], 1.0)] if "liabilities" in query else [(CHUNKS[1], 1.0)]


class ScriptedLLM:
    """Plays a fixed list of turns and records the messages it receives."""
    model = "scripted"

    def __init__(self, turns):
        self.turns, self.received = list(turns), []

    def complete(self, messages, tools):
        self.received.append(list(messages))
        return self.turns.pop(0)


def call(name, **args):
    return ToolCall(id=f"id-{name}", name=name, arguments=json.dumps(args))


def make_agent(turns, max_steps=6):
    llm = ScriptedLLM(turns)
    agent = Agent(FakeRetriever(), ContextBuilder(CHUNKS, mode="chunks"), llm, [DOC], max_steps=max_steps)
    return agent, llm


def test_agent_searches_calculates_and_cites():
    agent, llm = make_agent([
        ChatTurn("", [call("search_filings", query="current assets current liabilities",
                           doc_name="AMCOR_2023_10K")], 0.1, {"prompt_tokens": 100}),
        ChatTurn("", [call("calculate", expression="5853 / 5308")], 0.1, {"prompt_tokens": 200}),
        ChatTurn("The current ratio is 1.1 [AMCOR_2023_10K p.52].", [], 0.1, {"prompt_tokens": 300}),
    ])
    result = agent.answer("What is Amcor's FY2023 current ratio?")
    assert [t["tool"] for t in result["trace"]] == ["search_filings", "calculate"]
    assert agent.retriever.calls == [("current assets current liabilities", {"doc_name": "AMCOR_2023_10K"})]
    tool_messages = [m for m in llm.received[-1] if m["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("[AMCOR_2023_10K p.52]")
    assert tool_messages[1]["content"] == "1.102675"
    assert result["citations"] == [("AMCOR_2023_10K", 52)] and not result["invalid_citations"]
    assert result["usage"]["prompt_tokens"] == 600 and result["steps"] == 3


def test_tool_errors_go_back_to_the_model():
    agent, _ = make_agent([])
    seen = {}
    assert agent.run_tool("calculate", '{"expression": "__import__(\'os\')"}', seen).startswith("Error")
    assert agent.run_tool("search_filings", '{"query": "x", "doc_name": "NOPE"}', seen).startswith("Error")
    assert agent.run_tool("delete_everything", "{}", seen).startswith("Error")
    assert agent.run_tool("calculate", "not json", seen).startswith("Error")


def test_step_limit_forces_an_answer():
    loop = [ChatTurn("", [call("search_filings", query="net sales")], 0.1) for _ in range(2)]
    agent, llm = make_agent(loop + [ChatTurn("Net sales were 14,694 [AMCOR_2023_10K p.50].", [], 0.1)],
                            max_steps=2)
    result = agent.answer("Net sales?")
    assert "Step limit reached" in llm.received[-1][-1]["content"]
    assert result["citations"] == [("AMCOR_2023_10K", 50)]


def test_invented_citation_is_flagged():
    agent, _ = make_agent([ChatTurn("Revenue was 1 [NIKE_2019_10K p.3].", [], 0.1)])
    assert agent.answer("q")["invalid_citations"] == [("NIKE_2019_10K", 3)]


def test_catalog_in_prompt():
    assert catalog_text([DOC]) == "- AMCOR_2023_10K: Amcor, 10-K, FY2023"


def test_agent_stays_inside_the_user_selection():
    other = {"doc_name": "UP_ACME_1", "company": "", "doc_type": "other", "doc_period": None, "title": "Acme"}
    llm = ScriptedLLM([
        ChatTurn("", [call("search_filings", query="revenue")], 0.1),
        ChatTurn("", [call("search_filings", query="revenue", doc_name="AMCOR_2023_10K")], 0.1),
        ChatTurn("Done.", [], 0.1)])
    agent = Agent(FakeRetriever(), ContextBuilder(CHUNKS, mode="chunks"), llm, [DOC, other])
    result = agent.answer("Revenue?", {"doc_name": ["UP_ACME_1"]})
    assert agent.retriever.calls[0] == ("revenue", {"doc_name": ["UP_ACME_1"]})  # default = selection
    assert result["trace"][1]["output_preview"].startswith("Error")  # outside the selection
    assert "UP_ACME_1: Acme, other, fiscal year unknown" in llm.received[0][0]["content"]
    assert "AMCOR_2023_10K" not in llm.received[0][0]["content"]


def test_search_output_is_capped():
    long_chunk = {**CHUNKS[0], "text": "x " * 5000}

    class LongRetriever(FakeRetriever):
        def search(self, query, k, mode, filters):
            return [(long_chunk, 1.0)]

    agent = Agent(LongRetriever(), ContextBuilder([long_chunk], mode="chunks"), ScriptedLLM([]), [DOC],
                  max_chars_per_search=1000, max_total_chars=1500)
    seen = {}
    first = agent.run_tool("search_filings", '{"query": "a"}', seen)
    assert len(first) < 1200 and first.rstrip().endswith("[...]")
    agent.run_tool("search_filings", '{"query": "b"}', seen)
    assert agent.run_tool("search_filings", '{"query": "c"}', seen).startswith("Search budget used up")
    assert len(seen[("AMCOR_2023_10K", 52)]["text"]) == 10000  # full page kept for citations
