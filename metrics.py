# metrics.py
import re
import math
from typing import Dict

import textstat

_SIMPLE_DEF_PAT = re.compile(r"\b(is|means|refers to|defined as)\b", re.IGNORECASE)
_WORD_PAT = re.compile(r"[A-Za-z]+")
_SENT_SPLIT = re.compile(r"[.!?]+")

_EXAMPLE_PAT = re.compile(r"\b(for example|e\.g\.|e\. g\.|for instance|consider)\b", re.IGNORECASE)
_BULLET_PAT = re.compile(r"(?m)^\s*([-*]|\d+\.)\s+")
_MATH_PAT = re.compile(r"=|->|⇒|∑|Σ|∂|≈|O\(|\bTheta\(|\bOmega\(|\bBigO\b")

def _safe_float(x, default=float("nan")):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

def jargon_rate(text: str) -> float:
    """
    Simple proxy: long words + acronyms / #words.
    Long words >= 9 chars. Acronyms are ALLCAPS length>=2.
    """
    words = _WORD_PAT.findall(text)
    if not words:
        return 0.0
    long_words = sum(1 for w in words if len(w) >= 9)
    acronyms = sum(1 for w in words if w.isupper() and len(w) >= 2)
    return (long_words + acronyms) / max(1, len(words))

def definition_count(text: str) -> int:
    return len(_SIMPLE_DEF_PAT.findall(text))

def example_count(text: str) -> int:
    return len(_EXAMPLE_PAT.findall(text))

def bullet_count(text: str) -> int:
    return len(_BULLET_PAT.findall(text))

def math_symbol_rate(text: str) -> float:
    """
    Proxy for technicality: count math-ish tokens / #words.
    """
    words = re.findall(r"\S+", text)
    if not words:
        return 0.0
    n_math = len(_MATH_PAT.findall(text))
    return float(n_math) / max(1, len(words))

def sentence_count(text: str) -> int:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return len(sents)

def explanation_metrics(text: str) -> Dict[str, float]:
    tokens = re.findall(r"\S+", text)
    n_words = len(tokens)

    # Readability (guard against textstat blowing up on empty strings)
    if n_words < 3:
        fk_grade = float("nan")
        fk_ease = float("nan")
    else:
        fk_grade = _safe_float(textstat.flesch_kincaid_grade(text))
        fk_ease = _safe_float(textstat.flesch_reading_ease(text))

    return {
        "n_words": float(n_words),
        "n_sentences": float(sentence_count(text)),
        "fk_grade": float(fk_grade),
        "fk_ease": float(fk_ease),
        "jargon_rate": float(jargon_rate(text)),
        "definition_count": float(definition_count(text)),
        "example_count": float(example_count(text)),
        "bullet_count": float(bullet_count(text)),
        "math_symbol_rate": float(math_symbol_rate(text)),
    }
