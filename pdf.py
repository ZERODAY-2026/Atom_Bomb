"""
pdf.py - High-performance document extraction and passage retrieval.
Preserves page tracking and extracts relevant passages strictly on-device.
"""

import re
from pathlib import Path
import pymupdf

STOP_WORDS = frozenset({
    "what", "when", "where", "which", "who", "whom",
    "why", "how", "is", "are", "was", "were", "the",
    "a", "an", "of", "to", "in", "on", "for", "and",
    "or", "does", "do", "did", "must", "can", "could",
    "would", "should", "with", "from", "by", "be",
    "at", "it", "this", "that", "these", "those"
})

WORD_PATTERN = re.compile(r"\b[a-zA-Z0-9]+\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def extract_pdf(pdf_path):
    """
    Extract text from a PDF page by page.
    Keep page numbers so evidence can be cited correctly.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    try:
        with pymupdf.open(pdf_path) as document:
            if document.is_encrypted:
                raise ValueError("This PDF is password-protected.")

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

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

    if not pages:
        raise ValueError("The PDF contains no pages.")

    if all(not page["text"].strip() for page in pages):
        raise ValueError(
            "No extractable text found. "
            "This may be a scanned PDF requiring OCR."
        )

    return pages


def extract_pages_from_bytes(file_bytes, filename="document.pdf"):
    """
    Extract pages from in-memory PDF or TXT bytes directly.
    Avoids disk I/O latency.
    """
    fname = filename.lower()
    pages = []

    if fname.endswith(".pdf"):
        with pymupdf.open(stream=file_bytes, filetype="pdf") as document:
            if document.is_encrypted:
                raise ValueError("This PDF is password-protected.")

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "page": page_number,
                        "text": text
                    })

    elif fname.endswith(".txt"):
        try:
            text = file_bytes.decode("utf-8-sig").strip()
        except Exception:
            text = file_bytes.decode("latin-1", errors="replace").strip()

        if text:
            pages.append({"page": 1, "text": text})
    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not pages or all(not p["text"].strip() for p in pages):
        raise ValueError(
            "No readable text found. The PDF may be scanned "
            "or contain no extractable text."
        )

    return pages


def find_relevant_passages(pages, question, max_passages=5):
    """
    Find passages containing keywords related to the question.
    Preserve page numbers for evidence citations.
    Optimized with pre-compiled regexes and frozen stop-words set.
    """
    question_words = set(
        WORD_PATTERN.findall(question.lower())
    ) - STOP_WORDS

    if not question_words:
        # Fallback to general terms if all question words were stop words
        question_words = set(WORD_PATTERN.findall(question.lower()))

    results = []

    for page in pages:
        text = page.get("text", "")
        sentences = SENTENCE_SPLIT_PATTERN.split(text)

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            sentence_words = set(
                WORD_PATTERN.findall(sentence_clean.lower())
            )

            matches = question_words & sentence_words

            if matches:
                results.append({
                    "id": f"P{len(results) + 1}",
                    "page": page.get("page", 1),
                    "text": sentence_clean,
                    "score": len(matches)
                })

    results.sort(key=lambda passage: passage["score"], reverse=True)

    return results[:max_passages]
