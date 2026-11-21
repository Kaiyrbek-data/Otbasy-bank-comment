import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import app as app_module
from app import Question, search


class FakeRetriever:
    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    def search(self, question, top_k, dedupe):
        self.calls.append(
            {"question": question, "top_k": top_k, "dedupe": dedupe}
        )
        return self.matches


def make_request(retriever):
    state = SimpleNamespace(settings=app_module.SETTINGS, retriever=retriever)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_empty_question_is_rejected_by_schema():
    with pytest.raises(ValidationError):
        Question(question="")


def test_question_over_4000_characters_is_rejected_by_schema():
    with pytest.raises(ValidationError):
        Question(question="x" * 10000)


def test_whitespace_question_is_rejected_by_route():
    with pytest.raises(HTTPException) as error:
        search(Question(question="   "), make_request(FakeRetriever([])))

    assert error.value.status_code == 422


def test_known_question_uses_configured_defaults(monkeypatch):
    monkeypatch.setattr(app_module, "write_query_log", lambda *args: None)
    retriever = FakeRetriever(
        [{"id": 2803, "answer": "Публикуются ежемесячно.", "cosine": 0.71}]
    )

    result = search(
        Question(question="Как часто публикуют списки?"),
        make_request(retriever),
    )

    assert result["found"] is True
    assert result["confidence"] == "high"
    assert result["matches"][0]["id"] == 2803
    assert retriever.calls[0]["dedupe"] is True


def test_out_of_scope_mock_returns_explicit_refusal(monkeypatch):
    monkeypatch.setattr(app_module, "write_query_log", lambda *args: None)
    retriever = FakeRetriever(
        [{"id": 9, "answer": "Несвязанный ответ.", "cosine": 0.2}]
    )

    result = search(
        Question(question="Как приготовить бешбармак?"),
        make_request(retriever),
    )

    assert result == {"matches": [], "found": False, "confidence": "low"}


def test_request_can_override_thresholds(monkeypatch):
    monkeypatch.setattr(app_module, "write_query_log", lambda *args: None)
    retriever = FakeRetriever([{"id": 1, "answer": "Ответ", "cosine": 0.55}])

    result = search(
        Question(question="Вопрос", low_threshold=0.4, high_threshold=0.7),
        make_request(retriever),
    )

    assert result["found"] is True
    assert result["confidence"] == "medium"


def test_invalid_threshold_order_is_rejected(monkeypatch):
    monkeypatch.setattr(app_module, "write_query_log", lambda *args: None)

    with pytest.raises(HTTPException) as error:
        search(
            Question(question="Вопрос", low_threshold=0.7, high_threshold=0.6),
            make_request(FakeRetriever([])),
        )

    assert error.value.status_code == 422


def test_jsonl_log_does_not_store_question_text(tmp_path):
    settings = {
        **app_module.SETTINGS,
        "logging": {"jsonl_path": str(tmp_path / "queries.jsonl")},
    }

    app_module.write_query_log(
        settings,
        "Секретный текст вопроса",
        {"id": 17, "cosine": 0.62},
        "high",
    )

    record = json.loads((tmp_path / "queries.jsonl").read_text(encoding="utf-8"))
    assert record["question_length"] == len("Секретный текст вопроса")
    assert record["top1_id"] == 17
    assert record["cosine"] == 0.62
    assert record["confidence"] == "high"
    assert "question" not in record
    assert "Секретный текст вопроса" not in json.dumps(record, ensure_ascii=False)
