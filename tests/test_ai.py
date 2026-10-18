import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
# pyrefly: ignore[missing-import]
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


def test_verify_quote_accepts_real_quote():
    document = "Page 2: The agreement requires 30 days' notice."
    quote = "The agreement requires 30 days' notice."

    assert analyze.verify_quote(quote, document) is True


def test_verify_quote_rejects_invented_quote():
    document = "Page 2: The agreement requires 30 days' notice."
    quote = "The agreement requires 60 days' notice."

    assert analyze.verify_quote(quote, document) is False


def test_verify_quote_handles_extra_whitespace_and_case():
    document = "The agreement requires\n30 DAYS' notice."
    quote = "the agreement requires 30 days' notice."

    assert analyze.verify_quote(quote, document) is True


def test_verify_quote_rejects_empty_inputs():
    assert analyze.verify_quote("", "Some document text.") is False
    assert analyze.verify_quote("Some quotation.", "") is False


def test_answer_evidence_accepts_a_real_quotation():
    document = "The agreement requires 30 days' notice."

    answer = """
    1. ANSWER
    Thirty days' notice is required.

    2. SUPPORTING QUOTATION
    "The agreement requires 30 days' notice."

    3. PAGE NUMBER
    Not provided.
    """

    result = analyze.verify_answer_evidence(answer, document)

    assert result["status"] == "VERIFIED"


def test_answer_evidence_flags_an_invented_quotation():
    document = "The agreement requires 30 days' notice."

    answer = """
    1. ANSWER
    Sixty days' notice is required.

    2. SUPPORTING QUOTATION
    "The agreement requires 60 days' notice."

    3. PAGE NUMBER
    Not provided.
    """

    result = analyze.verify_answer_evidence(answer, document)

    assert result["status"] == "UNVERIFIED"


def test_answer_evidence_flags_missing_quotation_section():
    answer = "The agreement may require notice."
    document = "The agreement requires 30 days' notice."

    result = analyze.verify_answer_evidence(answer, document)

    assert result["status"] == "UNVERIFIED"


def test_verified_workflow_checks_gemma_answer(monkeypatch):
    document = "The agreement requires 30 days' notice."

    fake_answer = """
    1. ANSWER
    The notice period is 30 days.

    2. SUPPORTING QUOTATION
    "The agreement requires 30 days' notice."

    3. PAGE NUMBER
    Not provided.
    """

    def fake_ask_about_document(document_text, question):
        return fake_answer

    monkeypatch.setattr(
        analyze,
        "ask_about_document",
        fake_ask_about_document,
    )

    result = analyze.ask_about_document_with_verification(
        document,
        "How much notice is required?",
    )

    assert result["answer"] == fake_answer
    assert result["evidence_check"]["status"] == "VERIFIED"


def test_passages_to_document_text_preserves_evidence():
    passages = [
        {
            "id": "P1",
            "page": 1,
            "text": "Page 1: A tenant must provide 30 days written notice.",
        },
        {
            "id": "P2",
            "page": 2,
            "text": "Page 2: The security deposit must be returned within 15 days.",
        },
    ]

    result = analyze.passages_to_document_text(passages)

    assert "Passage ID: P1" in result
    assert "Page: 1" in result
    assert "30 days written notice" in result

    assert "Passage ID: P2" in result
    assert "Page: 2" in result
    assert "returned within 15 days" in result


def test_passage_evidence_returns_correct_source_page():
    passages = [
        {
            "id": "P1",
            "page": 1,
            "text": "A tenant must provide 30 days written notice.",
        },
        {
            "id": "P2",
            "page": 2,
            "text": "The security deposit must be returned within 15 days.",
        },
    ]

    answer = """
    2. SUPPORTING QUOTATION
    "The security deposit must be returned within 15 days."
    """

    result = analyze.verify_passage_evidence(answer, passages)

    assert result["status"] == "VERIFIED"
    assert result["passage_id"] == "P2"
    assert result["page"] == 2


def test_ask_gemma_uses_local_model(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {
            "message": {
                "content": "A termination clause explains how an agreement ends."
            }
        }

    monkeypatch.setattr(analyze.ollama, "chat", fake_chat)

    result = analyze.ask_gemma("Explain termination clauses.")

    assert result.startswith("A termination clause")
    assert captured["model"] == "gemma4:e2b"
    assert any(
        message["role"] == "system"
        for message in captured["messages"]
    )


def test_verify_quote_handles_hyphenation_and_smart_quotes():
    doc = "Landlord may terminate the lease with-\nout notice. The Tenant’s deposit is held."
    assert analyze.verify_quote("terminate the lease without notice", doc) is True
    assert analyze.verify_quote("The Tenant's deposit is held.", doc) is True


def test_no_passages_returns_document_is_silent_without_calling_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model must not be called when no evidence exists")
    monkeypatch.setattr(analyze, "ask_gemma", boom)
    res = analyze.ask_about_passages([], "Is there a pet clause?")
    assert res["evidence_check"]["status"] == "NO_EVIDENCE"
    assert "could not find supporting information" in res["answer"]