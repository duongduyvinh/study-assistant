from __future__ import annotations

import dataclasses

import httpx
import pytest

from t3.config import Settings
from t3.embeddings import GeminiEmbedder
from t3.http import LLMError
from t3.llm import call_llm, strip_think
from t3.types import ChatMessage

MSG = [ChatMessage("user", "xin chào")]


def settings(**kw) -> Settings:
    return dataclasses.replace(Settings(), llm_api_key="k", llm_backoff=0.0, llm_max_retries=2, **kw)


def ok_json(text="trả lời", pt=11, ct=7):
    return {"model": "m", "choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


def test_success_returns_usage_and_strips_think():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json=ok_json("<think>nghĩ</think>Đáp án"))

    text, usage = call_llm(MSG, stage="answer", settings=settings(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert text == "Đáp án" and (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 7, 18)
    assert usage.stage == "answer"
    assert usage.calls == 1 and usage.max_prompt == 11  # 11 là pt trong ok_json


def test_transient_errors_are_retried_and_logged(caplog):
    codes = iter([503, 429, 200])

    def handler(req):
        code = next(codes)
        return httpx.Response(code, json=ok_json()) if code == 200 else httpx.Response(code, text="busy")

    with caplog.at_level("WARNING", logger="t3.http"):
        text, _ = call_llm(MSG, stage="agent", settings=settings(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert text == "trả lời"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("HTTP 503" in m and "1/3" in m for m in msgs) and any("HTTP 429" in m and "2/3" in m for m in msgs)


def test_permanent_error_is_not_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(LLMError, match="HTTP 401"):
        call_llm(MSG, stage="answer", settings=settings(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(calls) == 1


def test_gives_up_after_retries_and_on_timeout():
    def always_500(req):
        return httpx.Response(500, text="x")

    with pytest.raises(LLMError, match="sau 3 lần thử"):
        call_llm(MSG, stage="a", settings=settings(), client=httpx.Client(transport=httpx.MockTransport(always_500)))

    def timeout(req):
        raise httpx.ReadTimeout("chậm", request=req)

    with pytest.raises(LLMError, match="lỗi mạng"):
        call_llm(MSG, stage="a", settings=settings(), client=httpx.Client(transport=httpx.MockTransport(timeout)))


def test_missing_key_and_bad_payload():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        call_llm(MSG, stage="a", settings=dataclasses.replace(Settings(), llm_api_key=""))
    with pytest.raises(LLMError, match="sai định dạng"):
        call_llm(MSG, stage="a", settings=settings(),
                 client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"oops": 1}))))


def test_extra_body_is_merged():
    seen = {}

    def handler(req):
        import json
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=ok_json())

    s = settings(llm_extra_body='{"enable_thinking": false}')
    call_llm(MSG, stage="a", settings=s, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert seen["enable_thinking"] is False and seen["model"] == s.llm_model


def test_strip_think_only_removes_think_blocks():
    assert strip_think("a <think>x\ny</think> b") == "a  b"


def test_gemini_embedder_request_shape_and_normalisation():
    import json
    seen = {}

    def handler(req):
        body = json.loads(req.content)
        seen["body"], seen["key"], seen["url"] = body, req.headers["x-goog-api-key"], str(req.url)
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0]} for _ in body["requests"]]})

    s = dataclasses.replace(Settings(), gemini_api_key="g", embedding_dim=2, llm_backoff=0.0)
    emb = GeminiEmbedder(s, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert emb.embed_query("hi") == [0.6, 0.8]
    assert seen["body"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY" and seen["key"] == "g"
    assert seen["url"].endswith("gemini-embedding-001:batchEmbedContents")
    emb.embed_documents(["a", "b"])
    assert seen["body"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiEmbedder(dataclasses.replace(Settings(), gemini_api_key=""))
