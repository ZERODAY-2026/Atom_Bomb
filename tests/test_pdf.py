import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf import (
    extract_document,
    extract_pdf,
    extract_text_document,
    find_relevant_passages,
    normalize_text,
    segment_clauses,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a temporary PDF for testing."""
    pdf_path = tmp_path / "sample.pdf"

    document = pymupdf.open()
    page1 = document.new_page()
    page1.insert_text(
        (72, 72),
        "A tenant must provide 30 days written notice."
    )

    page2 = document.new_page()
    page2.insert_text(
        (72, 72),
        "The security deposit must be returned within 15 days."
    )

    document.save(pdf_path)
    document.close()

    return pdf_path


def test_extract_pdf_valid():
    pages = extract_pdf(FIXTURES / "sample.pdf")
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "30 days written notice" in pages[0]["text"]
    assert pages[1]["page"] == 2
    assert "security deposit" in pages[1]["text"]


def test_extract_scanned_pdf_ocr():
    scanned_path = FIXTURES / "scanned_sample.pdf"
    if scanned_path.exists():
        pages = extract_pdf(scanned_path)
        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert "30 days" in pages[0]["text"].lower() or "notice" in pages[0]["text"].lower()


def test_extract_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_pdf(PROJECT_ROOT / "non_existent_file.pdf")


def test_extract_pdf_empty_file(tmp_path):
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError):
        extract_pdf(empty_file)


def test_extract_text_document(tmp_path):
    text_file = tmp_path / "agreement.txt"
    text_file.write_text("Clause 1: Parties\nClause 2: Termination with 30 days notice.", encoding="utf-8")
    pages = extract_text_document(text_file)
    assert len(pages) >= 1
    assert pages[0]["page"] == 1
    assert "Termination with 30 days notice" in pages[0]["text"]
    assert len(extract_document(text_file)) == len(pages)


def test_extract_json_document(tmp_path):
    import json
    json_file = tmp_path / "contract.json"
    json_file.write_text(json.dumps({"contract": "Clause 1: Notice period is 45 days."}), encoding="utf-8")
    pages = extract_document(json_file)
    assert len(pages) >= 1
    assert "Notice period is 45 days" in pages[0]["text"]


def test_extract_cuad_json_document(tmp_path):
    import json
    cuad_file = tmp_path / "cuad.json"
    data = {
        "data": [{
            "title": "Commercial Agreement",
            "paragraphs": [{"context": "Parties agree to arbitrate all disputes in London."}]
        }]
    }
    cuad_file.write_text(json.dumps(data), encoding="utf-8")
    pages = extract_document(cuad_file)
    assert len(pages) >= 1
    assert "arbitrate all disputes in London" in pages[0]["text"]


def test_find_relevant_passages():
    pages = extract_pdf(FIXTURES / "sample.pdf")
    question = "How many days notice must a tenant provide?"
    passages = find_relevant_passages(pages, question, max_passages=5)

    assert len(passages) >= 1
    assert passages[0]["id"] == "P1"
    assert passages[0]["page"] == 1
    assert "30 days written notice" in passages[0]["text"]


def test_find_relevant_passages_title():
    pages = [
        {"page": 1, "text": "RESIDENTIAL LEASE AGREEMENT\nBetween Landlord and Tenant."},
        {"page": 2, "text": "Rent is due on the first of each month."},
    ]
    passages = find_relevant_passages(pages, "What is the document name?")
    assert len(passages) >= 1
    assert passages[0]["page"] == 1
    assert "RESIDENTIAL LEASE AGREEMENT" in passages[0]["text"]


def test_find_relevant_passages_empty_inputs():
    assert find_relevant_passages([], "Any question") == []
    assert find_relevant_passages([{"page": 1, "text": "Test"}], "") == []
    assert find_relevant_passages([{"page": 1, "text": "Test"}], "Question", max_passages=0) == []


def test_preserve_page_numbers():
    pages = [
        {"page": 42, "text": "The confidentiality term lasts for five years."},
        {"page": 77, "text": "Governing law shall be the laws of California."},
    ]
    passages = find_relevant_passages(pages, "What is the governing law?")
    assert len(passages) >= 1
    assert passages[0]["page"] == 77
    assert "California" in passages[0]["text"]


def test_normalize_rejoins_hyphenated_line_breaks_and_smart_quotes():
    out = normalize_text("Landlord may terminate with-\nout notice. Tenant’s “deposit”")
    assert "without notice" in out
    assert "Tenant's \"deposit\"" in out


def test_segment_clauses_keeps_abbreviations_and_clause_ids():
    pages = [
        {"page": 1, "text": "3.1 The Tenant shall pay Rs. 5,000 per month.\n3.2 Landlord may terminate the lease\nif rent is unpaid."},
        {"page": 2, "text": "unless cured within 7 days.\n4. Deposits are refundable after vacating."},
    ]
    clauses = segment_clauses(pages)
    assert [c["clause_id"] for c in clauses] == ["3.1", "3.2", "4"]
    assert "Rs. 5,000 per month" in clauses[0]["text"]
    assert clauses[1]["page"] == 1 and clauses[1]["page_end"] == 2
    assert "cured within 7 days" in clauses[1]["text"]


def test_retrieval_uses_stemming_and_synonyms_and_returns_nothing_off_topic():
    pages = [
        {"page": 1, "text": "3.1 The Tenant shall pay Rs. 5,000 per month.\n3.2 Landlord may terminate the lease without notice if rent is unpaid."},
        {"page": 2, "text": "4. Deposits are refundable after vacating."},
    ]
    hit = find_relevant_passages(pages, "Can I get my deposit back?")
    assert hit and hit[0]["clause_id"] == "4" and hit[0]["page"] == 2
    hit = find_relevant_passages(pages, "Can the landlord evict me?")
    assert hit and hit[0]["clause_id"] == "3.2"
    assert find_relevant_passages(pages, "What is the weather today?") == []


def test_extract_pdf_returns_pages(sample_pdf):
    pages = extract_pdf(sample_pdf)
    assert len(pages) == 2


def test_extract_pdf_preserves_page_text(sample_pdf):
    pages = extract_pdf(sample_pdf)
    combined_text = str(pages)
    assert "30 days written notice" in combined_text
    assert "security deposit" in combined_text


def test_find_relevant_passages_from_fixture(sample_pdf):
    pages = extract_pdf(sample_pdf)
    passages = find_relevant_passages(
        pages,
        "How many days notice must a tenant provide?"
    )
    assert passages
    assert any(
        "30 days written notice" in passage["text"]
        for passage in passages
    )


def test_passages_include_page_numbers(sample_pdf):
    pages = extract_pdf(sample_pdf)
    passages = find_relevant_passages(
        pages,
        "How many days notice must a tenant provide?"
    )
    assert passages
    assert all("page" in passage for passage in passages)


if __name__ == "__main__":
    pages = extract_pdf(FIXTURES / "sample.pdf")
    question = "How many days notice must a tenant provide?"
    passages = find_relevant_passages(pages, question)

    print("\nRelevant evidence:")
    for passage in passages:
        print(f"\nID: {passage['id']}")
        print(f"Page: {passage['page']}")
        print(f"Text: {passage['text']}")
