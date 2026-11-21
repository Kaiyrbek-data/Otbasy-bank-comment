import pytest

from retriever import FAQRetriever


@pytest.mark.slow
def test_known_question_finds_monthly_list_publication_answer():
    retriever = FAQRetriever()

    matches = retriever.search(
        "С какой периодичностью банк публикует списки состоящих на жилищном учёте?",
        top_k=3,
    )

    assert matches
    assert any("ежемесячно" in match["answer"].casefold() for match in matches)
