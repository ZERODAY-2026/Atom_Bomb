
import pymupdf
from pathlib import Path


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

    return pages



def find_relevant_passages(pages, question, max_passages=5):
    """
    Find passages containing keywords related to the question.
    Preserve page numbers for evidence citations.
    """

    import re

    stop_words = {
        "what", "when", "where", "which", "who", "whom",
        "why", "how", "is", "are", "was", "were", "the",
        "a", "an", "of", "to", "in", "on", "for", "and",
        "or", "does", "do", "did", "must", "can", "could",
        "would", "should", "with", "from", "by", "be"
    }

    question_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    ) - stop_words

    results = []

    for page in pages:
        text = page["text"]
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence_words = set(
                re.findall(r"\b[a-zA-Z0-9]+\b", sentence.lower())
            )

            matches = question_words & sentence_words

            if matches:
                results.append({
                    "id": f"P{len(results) + 1}",
                    "page": page["page"],
                    "text": sentence.strip(),
                    "score": len(matches)
                })

    results.sort(key=lambda passage: passage["score"], reverse=True)

    return results[:max_passages]

