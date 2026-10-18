"""
ClauseGuard safety checks.

These checks provide basic safeguards, not a guarantee that
an answer is legally correct or safe.
"""

import re


# Questions that may require professional legal judgment.
HIGH_STAKES_PATTERNS = [
    r"\bshould i sign\b",
    r"\bshould i sue\b",
    r"\bshould i file a case\b",
    r"\bcan i legally\b",
    r"\bis this clause legal\b",
    r"\bis this contract valid\b",
    r"\bwill i win\b",
    r"\bwhat legal action should i take\b",
    r"\bcan i terminate (?:this|the) contract\b",
]

# Common signs of instructions embedded in untrusted document text.
INJECTION_PATTERNS = [
    r"\bignore (?:all )?(?:previous|prior|above) instructions\b",
    r"\breveal (?:the )?(?:system prompt|hidden instructions)\b",
    r"\bact as (?:a )?(?:different|new) assistant\b",
    r"\bdisregard (?:all )?(?:previous|prior|above) instructions\b",
    r"\bprint (?:your )?(?:system prompt|hidden instructions)\b",
]

# Pre-compiled regular expressions for fast matching without repetitive compilation overhead
HIGH_STAKES_COMPILED = [
    (pattern, re.compile(pattern, flags=re.IGNORECASE))
    for pattern in HIGH_STAKES_PATTERNS
]

INJECTION_COMPILED = [
    (pattern, re.compile(pattern, flags=re.IGNORECASE))
    for pattern in INJECTION_PATTERNS
]


def check_question_safety(question):
    """
    Identify questions that may need a legal-professional warning.

    This does not reject the question. It flags the need for caution.
    """
    if not isinstance(question, str) or not question.strip():
        return {
            "requires_caution": True,
            "reason": "The question is empty or invalid.",
        }

    normalized_question = question.casefold()

    for _, compiled_re in HIGH_STAKES_COMPILED:
        if compiled_re.search(normalized_question):
            return {
                "requires_caution": True,
                "reason": (
                    "This question may require professional legal "
                    "judgment. ClauseGuard can explain document text "
                    "but cannot make the legal decision for you."
                ),
            }

    return {
        "requires_caution": False,
        "reason": "",
    }


def detect_prompt_injection(document_text):
    """
    Detect some common instruction-like phrases in document text.

    A match is only a warning signal. Ordinary legal text can contain
    these phrases, and malicious instructions can use other wording.
    """
    if not isinstance(document_text, str) or not document_text.strip():
        return {
            "detected": False,
            "matches": [],
        }

    matches = []
    for pattern, compiled_re in INJECTION_COMPILED:
        if compiled_re.search(document_text):
            matches.append(pattern)

    return {
        "detected": bool(matches),
        "matches": matches,
    }


def build_safety_notice(question, document_text, cached_injection_check=None):
    """
    Create safety notices for the application to display.
    Accepts an optional cached_injection_check to avoid re-scanning full documents on every query.
    """
    question_check = check_question_safety(question)

    if cached_injection_check is not None:
        injection_check = cached_injection_check
    else:
        injection_check = detect_prompt_injection(document_text)

    notices = []

    if question_check["requires_caution"]:
        notices.append(question_check["reason"])

    if injection_check["detected"]:
        notices.append(
            "This document contains instruction-like text that may "
            "be untrusted. It will be treated as document content, "
            "not as instructions for the assistant."
        )

    return {
        "requires_caution": question_check["requires_caution"],
        "prompt_injection_detected": injection_check["detected"],
        "notices": notices,
    }
