
import sys
from pathlib import Path

import pytest
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf import extract_pdf, find_relevant_passages


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


def test_extract_pdf_returns_pages(sample_pdf):
    pages = extract_pdf(sample_pdf)

    assert len(pages) == 2


def test_extract_pdf_preserves_page_text(sample_pdf):
    pages = extract_pdf(sample_pdf)

    combined_text = str(pages)

    assert "30 days written notice" in combined_text
    assert "security deposit" in combined_text


def test_find_relevant_passages(sample_pdf):
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
