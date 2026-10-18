import math
import re
import unicodedata
from pathlib import Path

# pyrefly: ignore[missing-import]
import pymupdf


_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
})


def normalize_text(text):
    """
    Normalize extracted text so the model and the quote verifier see the
    same characters. Keeps line breaks (needed for clause detection).
    - NFKC folds ligatures (fi) and full-width forms
    - curly quotes/dashes become plain ASCII
    - words split by a line-break hyphen are rejoined ("with-\nout")
    - runs of spaces/tabs collapse
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"(?<=[a-z])-[ \t]*\n[ \t]*(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_for_match(text):
    """Aggressive form used only for comparing quotes with source text."""
    return " ".join(normalize_text(text).split()).casefold()


def extract_pdf(pdf_path):
    """
    Extract text page by page from a PDF.
    Use OCR when a page has no selectable text.
    Preserve page numbers for evidence citations.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Check for empty files
    if pdf_path.stat().st_size == 0:
        raise ValueError("The PDF file is empty (0 bytes).")

    pages = []

    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass or document.is_encrypted:
                raise ValueError(
                    "This PDF is password-protected. "
                    "Password-protected PDFs are not supported."
                )

            if len(document) == 0:
                raise ValueError("The PDF contains no pages.")

            for page_idx in range(len(document)):
                page_number = page_idx + 1
                page = document[page_idx]
                text = normalize_text(page.get_text("text"))

                # Use OCR when no selectable text is available.
                if not text:
                    try:
                        text_page = page.get_textpage_ocr(
                            language="eng",
                            dpi=300,
                            full=True,
                        )
                        text = normalize_text(page.get_text(
                            "text",
                            textpage=text_page,
                        ))

                    except Exception as error:
                        raise ValueError(
                            f"OCR failed on page {page_number}. "
                            "Check that Tesseract OCR and its "
                            "English language data are installed. "
                            f"Details: {error}"
                        ) from error

                pages.append({
                    "page": page_number,
                    "text": text,
                })

    except (FileNotFoundError, ValueError):
        raise
    except Exception as error:
        raise ValueError(
            f"Could not read this PDF: {error}"
        ) from error

    if all(not page["text"].strip() for page in pages):
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be blank, blurry, or require additional OCR support."
        )

    return pages


def extract_text_document(file_path):
    """
    Extract plain text documents (.txt, .md).
    Maps lines into pages (~50 lines per virtual page) while preserving line content.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError("The document is empty (0 bytes).")

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1")

    content = normalize_text(content)
    lines = content.splitlines()
    if not lines or not content.strip():
        raise ValueError("The document contains no readable text.")

    # Group into pages of 50 lines to keep citations structured
    page_size = 50
    pages = []
    for index in range(0, len(lines), page_size):
        page_lines = lines[index : index + page_size]
        page_text = "\n".join(page_lines).strip()
        if page_text:
            pages.append({
                "page": (index // page_size) + 1,
                "text": page_text,
            })

    if not pages:
        raise ValueError("No readable text was found in this document.")

    return pages


def extract_document(file_path):
    """
    General entry point for document extraction.
    Supports PDF and plain text documents (.txt, .md).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(path)
    elif suffix in [".txt", ".md", ".markdown", ".text"]:
        return extract_text_document(path)
    else:
        # Default try PDF first, fallback to text
        try:
            return extract_pdf(path)
        except Exception:
            return extract_text_document(path)


def _assign_ids(passages):
    """Assign stable sequential IDs (P1, P2, ...) to passages."""
    for index, passage in enumerate(passages, start=1):
        passage["id"] = f"P{index}"
    return passages


def _find_document_title(pages):
    """
    Extract contract or document title from the first page.
    Looks for prominent title lines such as 'AGREEMENT', 'LEASE', etc.
    """
    if not pages:
        return None

    first_page_text = pages[0].get("text", "")
    if not first_page_text.strip():
        return None

    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]

    # Look for common contract title keywords in early lines
    contract_keywords = [
        "agreement", "contract", "lease", "nda", "accord",
        "covenant", "deed", "terms", "policy", "memorandum",
        "license", "undertaking", "addendum", "amendment"
    ]

    for line in lines[:8]:
        cleaned = re.sub(r"[^\w\s-]", "", line).strip()
        if 4 <= len(cleaned) <= 120:
            if any(kw in cleaned.lower() for kw in contract_keywords):
                return {
                    "page": pages[0]["page"],
                    "page_end": pages[0]["page"],
                    "clause_id": None,
                    "text": line,
                    "score": 100,
                }

    if lines:
        return {
            "page": pages[0]["page"],
            "page_end": pages[0]["page"],
            "clause_id": None,
            "text": lines[0],
            "score": 50,
        }

    return None


_CLAUSE_START = re.compile(
    r"^\s*(?:"
    r"(?P<num>\d+(?:\.\d+)*)(?:[.)]|\s)"            # 3.1  /  3.  /  3)
    r"|(?P<letter>\([a-zA-Z0-9]{1,4}\))"              # (a)  (iv)
    r"|(?P<word>(?:article|clause|section|schedule|annexure)\s+[\w.]+)"
    r")",
    re.IGNORECASE,
)

_SYNONYMS = [
    {"deposit", "refund", "refundable", "return", "returned", "advance"},
    {"evict", "eviction", "terminate", "termination", "vacate", "vacating", "expel"},
    {"rent", "payment", "pay", "paid", "monthly", "installment"},
    {"notice", "inform", "notify", "intimate"},
    {"penalty", "fine", "charge", "damages", "late"},
    {"salary", "wage", "compensation", "remuneration", "ctc", "pay"},
    {"resign", "resignation", "quit", "leave", "exit", "notice"},
]


def stem(word):
    """Very small suffix stripper: 'deposits' -> 'deposit'."""
    word = word.lower()
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix in ("ies", "ied"):
                return word[: -len(suffix)] + "y"
            return word[: -len(suffix)]
    return word


_SYNONYM_STEMS = [{stem(w) for w in group} for group in _SYNONYMS]


def segment_clauses(pages):
    """
    Split page text into clause-level passages.
    A new clause starts at a numbered/lettered/'Section' heading line.
    Unnumbered continuation lines (including the first lines of a new page)
    are appended to the previous clause. Each clause keeps the page where
    it STARTS (page) and where it ends (page_end).
    """
    clauses = []
    for page in pages or []:
        text = page.get("text", "") if isinstance(page, dict) else ""
        if not text.strip():
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _CLAUSE_START.match(line)
            # Unnumbered text: a new capitalised line after a finished
            # sentence starts a new passage; otherwise it is a wrapped line.
            starts_new_sentence = (
                bool(clauses)
                and re.search(r"[.;:!?][\"')\]]*$", clauses[-1]["text"])
                and line[:1].isupper()
            )
            if match or not clauses or starts_new_sentence:
                clause_id = None
                if match and not (
                    starts_new_sentence and not match.group("num")
                    and not match.group("letter") and not match.group("word")
                ):
                    clause_id = (
                        match.group("num") or match.group("letter")
                        or match.group("word")
                    )
                clauses.append({
                    "page": page["page"],
                    "page_end": page["page"],
                    "clause_id": clause_id,
                    "text": line,
                })
            else:
                clauses[-1]["text"] += " " + line
                clauses[-1]["page_end"] = page["page"]
    return clauses


def _tokens(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def _expand_query(terms):
    expanded = set(terms)
    for group in _SYNONYM_STEMS:
        if expanded & group:
            expanded |= group
    return expanded


def find_relevant_passages(pages, question, max_passages=5):
    """
    Retrieve relevant evidence using clause-aware keyword scoring
    and short overlapping sentence windows. Preserve PDF page numbers.
    """
    if max_passages <= 0:
        return []

    if not pages or not isinstance(pages, list):
        return []

    question_lower = question.lower()

    name_question = bool(
        re.search(
            r"\b(document name|contract name|name of the contract|"
            r"name of this contract|title of the contract|"
            r"title of this document)\b",
            question_lower,
        )
    )

    if name_question:
        title = _find_document_title(pages)
        if title:
            return _assign_ids([title])

    stop_words = {
        "what", "when", "where", "which", "who", "whom",
        "why", "how", "is", "are", "was", "were", "the",
        "a", "an", "of", "to", "in", "on", "for", "and",
        "or", "does", "do", "did", "must", "can", "could",
        "would", "should", "with", "from", "by", "be",
        "it", "this", "that", "these", "those",
        "highlight", "parts", "related", "reviewed", "lawyer",
        "details", "any", "contract"
    }

    def words(text):
        return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))

    # Extract the clause label and the explanation from CUAD-style questions.
    label_match = re.search(r'"([^"]+)"', question)
    clause_label = label_match.group(1).lower() if label_match else ""

    details_match = re.search(
        r"Details:\s*(.*)",
        question,
        flags=re.IGNORECASE | re.DOTALL,
    )
    details = details_match.group(1).strip() if details_match else ""

    question_words = words(question_lower) - stop_words
    label_words = words(clause_label) - stop_words
    detail_words = words(details) - stop_words

    # Add related terms for common legal clause categories.
    months = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    }

    related_terms = {
        "agreement date": {"made", "signed", "dated", "date", "day", "month", "year", "entered", "witnesseth"} | months,
        "effective date": {"effective", "commence", "commencement", "begin", "start", "starting", "as of"} | months,
        "expiration date": {"expiration", "expire", "term", "end", "ending", "terminate", "conclusion"} | months,
        "renewal term": {"renewal", "renewable", "renew", "extend", "extension", "automatic", "period"},
        "parties": {"between", "company", "distributor", "party", "parties", "client", "vendor", "contractor", "employer", "employee", "landlord", "tenant"},
        "termination": {"terminate", "termination", "notice", "cancel", "cancellation", "default", "breach", "cure"},
        "payment": {"pay", "payment", "fee", "invoice", "amount", "compensation", "due", "remit", "dollar", "price"},
        "penalty": {"penalty", "damages", "fine", "liability", "liquidated", "interest", "indemnity"},
        "governing law": {"governed", "laws", "jurisdiction", "state", "court", "venue", "arbitration"},
        "notice": {"notice", "written", "days", "mail", "notify", "writing", "address"},
        "indemnification": {"indemnify", "indemnification", "hold harmless", "defend", "liability", "loss", "claims"},
        "confidentiality": {"confidential", "proprietary", "disclosure", "secret", "nondisclosure", "trade secret"},
        "non-compete": {"compete", "competition", "restrictive", "solicit", "solicitation", "non-compete"},
        "audit": {"audit", "inspection", "records", "books", "examine", "accounting"},
    }

    expanded_label_words = set(label_words)
    for label, terms in related_terms.items():
        if label in clause_label:
            expanded_label_words.update(terms)

    # Clause-level BM25 retrieval over stemmed terms.
    clauses = segment_clauses(pages)
    if not clauses:
        return []

    query_terms = {stem(w) for w in question_words | expanded_label_words | detail_words}
    query_terms = _expand_query(query_terms)
    query_terms = {t for t in query_terms if len(t) > 1}
    if not query_terms:
        return []

    doc_tokens = [_tokens(c["text"]) for c in clauses]
    n_docs = len(clauses)
    avg_len = sum(len(t) for t in doc_tokens) / n_docs or 1.0
    k1, b = 1.5, 0.75

    doc_freq = {}
    for tokens in doc_tokens:
        for term in set(tokens) & query_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    results = []
    for clause, tokens in zip(clauses, doc_tokens):
        score = 0.0
        for term in query_terms:
            freq = tokens.count(term)
            if not freq:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * freq * (k1 + 1) / (
                freq + k1 * (1 - b + b * len(tokens) / avg_len)
            )
        if score > 0:
            results.append({
                "page": clause["page"],
                "page_end": clause["page_end"],
                "clause_id": clause["clause_id"],
                "text": re.sub(r"\s+", " ", clause["text"]).strip(),
                "score": round(score, 3),
            })

    results.sort(key=lambda passage: passage["score"], reverse=True)
    return _assign_ids(results[:max_passages])