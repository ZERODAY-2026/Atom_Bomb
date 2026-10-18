import analyze
import pytest


def test_empty_document_is_rejected():
    with pytest.raises(ValueError):
        analyze.analyze_document("")


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        analyze.ask_about_document(
            "The agreement requires 30 days' notice.",
            ""
        )
