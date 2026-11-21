from threading import Lock

import numpy as np
from torch import nn

from retriever import FAQRetriever


class FakeEncoder:
    def encode(self, questions, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def make_retriever():
    retriever = object.__new__(FAQRetriever)
    retriever.encoder = FakeEncoder()
    retriever.projection = nn.Identity()
    retriever.index = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.70, 0.30],
            [0.60, 0.40],
        ],
        dtype=np.float32,
    )
    retriever.docs = [
        {"id": 10, "question": "Вариант A1", "answer": "Ответ A"},
        {"id": 11, "question": "Вариант A2", "answer": "Ответ A"},
        {"id": 12, "question": "Вариант B", "answer": "Ответ B"},
        {"id": 13, "question": "Вариант C", "answer": "Ответ C"},
    ]
    retriever._lock = Lock()
    return retriever


def test_dedupe_returns_top_k_distinct_answers_and_keeps_best_question():
    matches = make_retriever().search("Вопрос", top_k=2)

    assert [match["id"] for match in matches] == [10, 12]
    assert [match["question"] for match in matches] == ["Вариант A1", "Вариант B"]
    assert [match["rank"] for match in matches] == [1, 2]


def test_dedupe_can_be_disabled_for_comparison():
    matches = make_retriever().search("Вопрос", top_k=2, dedupe=False)

    assert [match["id"] for match in matches] == [10, 11]
    assert matches[0]["answer"] == matches[1]["answer"]
