"""Roman-Urdu and script heuristics for session handlers and LLM routing."""

from __future__ import annotations

import re

_URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")

# Roman-Urdu markers, matched as whole words (see looks_like_roman_urdu).
_ROMAN_URDU_MARKERS = (
    "mera", "meri", "mere", "mujhe", "mjy", "mujay", "sar", "saar",
    "dard", "ho raha", "horaha", "hai", "ha", "he", "hain", "mein", "main", "ma",
    "ka", "ki", "ko", "se", "sy", "nahi", "nahin", "kya", "ky", "bukhar", "tabiyat",
    "chahiye", "karein", "theek", "acha", "achha", "kahan", "kaisa", "kaise",
    "kyun", "kyu", "jab", "magar", "lekin", "ya", "bhi", "rahay", "rahe",
    "gayi", "gaya", "diya", "diye", "chalo", "suno", "dekho",
    "batao", "bataiye", "karo", "karen", "aal", "haal", "chal",
    "salam", "assalam", "assalamualaikum", "assalamualikum",
    "wala", "wali", "ne", "par", "aur", "abhi", "phir", "idhar",
    "udhar", "zaroorat", "apni", "apna", "aap", "tum",
)
_ROMAN_URDU_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in _ROMAN_URDU_MARKERS) + r")\b"
)

# Strong markers — even ONE is enough to classify as Roman Urdu
_STRONG_ROMAN_URDU_RE = re.compile(
    r"\b(?:assalam|assalamualaikum|assalamualikum|assalamu|salam|salaam|"
    r"nahi|nahin|kya|kyun|kyu|chahiye|karein|batao|bataiye|karo|"
    r"theek|achha|acha|tabiyat|bukhar|zaroorat|chalo|suno|dekho)\b",
    re.IGNORECASE,
)

# Common English single-word responses that should trigger English mode
# even when they're the only word — prevents the LLM from continuing in
# Urdu after the user switches to English with a short affirmation.
_ENGLISH_SINGLE_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "no", "nope", "okay", "ok", "sure",
    "please", "thanks", "thank", "hello", "hi", "hey", "bye",
    "correct", "right", "exactly", "true", "false", "maybe",
    "done", "ready", "stop", "wait", "go", "continue",
})


def contains_urdu_script(text: str) -> bool:
    return bool(_URDU_SCRIPT_RE.search(text))


def looks_like_roman_urdu(text: str) -> bool:
    """True if Latin-script text looks like Roman Urdu (session handler markers)."""
    low = text.lower()
    if _STRONG_ROMAN_URDU_RE.search(low):
        return True
    return len(set(_ROMAN_URDU_RE.findall(low))) >= 2


def looks_english(text: str) -> bool:
    """True if the text is a real English message (LLM language routing).

    Not Urdu, not Roman Urdu, and not just a number/one-word answer that
    shouldn't flip the language — except common English single-word replies.
    """
    if _URDU_SCRIPT_RE.search(text):
        return False
    low = text.lower()
    if _STRONG_ROMAN_URDU_RE.search(low):
        return False
    alpha_words = re.findall(r"[A-Za-z]+", text)
    if len(alpha_words) < 2:
        if len(alpha_words) == 1 and alpha_words[0].lower() in _ENGLISH_SINGLE_WORDS:
            return True
        return False
    return len(set(_ROMAN_URDU_RE.findall(low))) < 2
