
import re
from pathlib import Path

import pymupdf


def extract_pdf(pdf_path):
    """
    Extract text page by page.
    Use OCR when a page has no selectable text.
    Preserve page numbers for evidence citations.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    try:
        with pymupdf.open(pdf_path) as document:

            if document.needs_pass:
                raise ValueError(
                    "This PDF is password-protected. "
                    "Password-protected PDFs are not supported."
                )

            if len(document) == 0:
                raise ValueError("The PDF contains no pages.")

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                # Use OCR when no selectable text is available.
                if not text:
                    try:
                        text_page = page.get_textpage_ocr(
                            language="eng",
                            dpi=300,
                            full=True
                        )
                        text = page.get_text(
                            "text",
                            textpage=text_page
                        ).strip()

                    except Exception as error:
                        raise ValueError(
                            f"OCR failed on page {page_number}. "
                            "Check that Tesseract OCR and its "
                            "English language data are installed. "
                            f"Details: {error}"
                        ) from error

                pages.append({
                    "page": page_number,
                    "text": text
                })

    except (FileNotFoundError, ValueError):
        raise
    except Exception as error:
        raise ValueError(
            f"Could not read this PDF: {error}"
        ) from error

    if all(not page["text"] for page in pages):
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be blank, blurry, or require additional OCR support."
        )

    return pages


def find_relevant_passages(pages, question, max_passages=5):
    """
    Find sentences that share keywords with the question.
    Return the best matches with stable IDs and page numbers.
    """

    if max_passages <= 0:
        return []

    stop_words = {
        "what", "when", "where", "which", "who", "whom",
        "why", "how", "is", "are", "was", "were", "the",
        "a", "an", "of", "to", "in", "on", "for", "and",
        "or", "does", "do", "did", "must", "can", "could",
        "would", "should", "with", "from", "by", "be",
        "it", "this", "that", "these", "those"
    }

    question_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    ) - stop_words

    if not question_words:
        return []

    results = []

    for page in pages:
        text = page.get("text", "")
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_words = set(
                re.findall(r"\b[a-zA-Z0-9]+\b", sentence.lower())
            )

            matches = question_words & sentence_words

            if matches:
                results.append({
                    "page": page["page"],
                    "text": sentence,
                    "score": len(matches)
                })

    # Sort by relevance before assigning IDs.
    results.sort(
        key=lambda passage: passage["score"],
        reverse=True
    )

    results = results[:max_passages]

    for index, passage in enumerate(results, start=1):
        passage["id"] = f"P{index}"

    return results
