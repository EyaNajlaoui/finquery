"""Daily-limit fallback of the Groq client, without network calls."""
import pytest

groq = pytest.importorskip("groq")  # not installed in the light CI environment

import httpx  # noqa: E402  (a dependency of groq)

from finquery.generation.llm import DailyLimitError, GroqLLM, is_daily_limit  # noqa: E402


def rate_limit(message):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.RateLimitError(message, response=httpx.Response(429, request=request), body=None)


DAILY = "Rate limit reached for model on tokens per day (TPD): Limit 200000"


def test_is_daily_limit():
    assert is_daily_limit(Exception(DAILY))
    assert not is_daily_limit(Exception("Rate limit reached on tokens per minute (TPM)"))


def test_fallback_to_next_model_then_stop():
    llm = GroqLLM("big", api_key="gsk_test", fallback_models=["small"])
    used = []

    def request(model):
        used.append(model)
        if model == "big":
            raise rate_limit(DAILY)
        return "ok"

    assert llm._call(request) == "ok" and used == ["big", "small"] and llm.model == "small"

    def always_limited(model):
        raise rate_limit(DAILY)

    with pytest.raises(DailyLimitError):
        llm._call(always_limited)


def test_per_minute_limit_is_not_a_fallback():
    llm = GroqLLM("big", api_key="gsk_test", fallback_models=["small"])

    def request(model):
        raise rate_limit("Rate limit reached on tokens per minute (TPM)")

    with pytest.raises(groq.RateLimitError):
        llm._call(request)
    assert llm.model == "big"


def test_json_mode_failure_retries_in_normal_mode():
    llm = GroqLLM("small", api_key="gsk_test")
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append("response_format" in kwargs)
            if "response_format" in kwargs:
                request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
                response = httpx.Response(400, request=request)
                raise groq.BadRequestError("json_validate_failed", response=response, body=None)

            class Message:
                content = '{"grade": "correct", "reason": "ok"}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]
                usage = None
                model = "small"
            return Response()

    llm.client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    assert llm.chat("system", "user", json_mode=True).text.startswith('{"grade"')
    assert calls == [True, False]
