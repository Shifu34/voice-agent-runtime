"""Whisper verbose_json confidence scoring and hallucination filtering."""

from __future__ import annotations

import re

# Urdu (Arabic) script OR Devanagari — Whisper sometimes renders spoken Urdu
# in Devanagari. Either way it is Urdu, not English.
_URDU_SCRIPT_RE = re.compile(r"[؀-ۿऀ-ॿ]")

# Whisper marks pure silence/noise with a high no_speech_prob; penalise it so
# an empty-but-"confident" pass doesn't win.
_NO_SPEECH_PENALTY = 1.0

# If average no_speech_prob exceeds this, the segment is almost certainly
# silence/noise — Whisper hallucinates text on such audio. Discard it.
_NO_SPEECH_THRESHOLD = 0.4

# Minimum raw avg_logprob for a transcript to be considered real speech.
_MIN_LOGPROB = -0.8

_LATIN_RE = re.compile(r"[A-Za-z]")

# Roman-Urdu markers — STT router set (slightly different from session markers).
_ROMAN_URDU_MARKERS = (
    "mera", "meri", "mere", "mujhe", "mjy", "mujay", "sar", "saar",
    "dard", "ho", "raha", "horaha", "hai", "ha", "he", "hain",
    "mein", "main", "ma", "ka", "ki", "ko", "se", "sy", "nahi",
    "nahin", "kya", "ky", "bukhar", "tabiyat", "chahiye", "karein",
    "theek", "acha", "achha", "kahan", "kaisa", "kaise", "kyun",
    "kyu", "jab", "magar", "lekin", "ya", "bhi", "rahay", "rahe",
    "gayi", "gaya", "diya", "diye", "chalo", "suno", "dekho",
    "batao", "bataiye", "karo", "karen", "aal", "haal", "chal",
    "salam", "assalam", "assalamualaikum", "assalamualikum",
    "wala", "wali", "ne", "par", "aur", "abhi", "phir", "idhar",
    "udhar", "zaroorat", "mujhe", "apni", "apna", "aap", "tum",
)
_ROMAN_URDU_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in _ROMAN_URDU_MARKERS) + r")\b"
)

_ENGLISH_WORDS_RE = re.compile(
    r"\b(?:yes|no|not|the|a|an|is|are|was|were|am|i|you|he|she|we|they|"
    r"it|this|that|and|or|but|if|so|do|does|did|have|has|had|will|would|"
    r"can|could|should|want|need|like|feel|pain|head|doctor|appointment|"
    r"hospital|medicine|help|please|thank|hello|hi|my|your|his|her|"
    r"what|when|where|why|how|who|me|us|them|get|make|go|come|see|"
    r"book|schedule|today|tomorrow|now|here|there|fever|cold|cough|"
    r"stomach|chest|back|neck|arm|leg|eye|ear|nose|throat|skin|"
    r"blood|pressure|sugar|test|report|check|up|down|out|good|bad|"
    r"better|worse|okay|ok|sure|right|left|morning|evening|night|"
    r"day|week|month|year|time|date|name|age|phone|number|address|"
    r"start|starts|started|let|lets|with|first|second|third|one|two|"
    r"three|four|five|give|brief|briefing|encounter|patient|pending|"
    r"details|next|pick|choose|select|think|should|would|could|said)\b",
    re.IGNORECASE,
)

# Common doctor selection commands — short but valid English even when
# Whisper confidence is borderline (e.g. "let's start with the first one").
_CLINICAL_SELECTION_RE = re.compile(
    r"(?i)\b(?:let'?s|lets)\s+start\s+with\b"
    r"|\bstart\s+with\b"
    r"|\b(?:first|second|third)\s+(?:one|patient|appointment)\b"
    r"|\b(?:yes|yeah|yep),?\s*(?:start|please)\b",
)

_HALLUCINATION_PATTERNS = (
    re.compile(r"thank you for watching", re.IGNORECASE),
    re.compile(r"thanks for watching", re.IGNORECASE),
    re.compile(r"please subscribe", re.IGNORECASE),
    re.compile(r"subscribe for more", re.IGNORECASE),
    re.compile(r"terima kasih kerana menonton", re.IGNORECASE),
    re.compile(r"продолжение следует", re.IGNORECASE),
    re.compile(r"Спасибо за просмотр", re.IGNORECASE),
    re.compile(r"برای تماشا", re.IGNORECASE),
    re.compile(r"شكرا للمشاهدة", re.IGNORECASE),
    re.compile(r"اگر شما", re.IGNORECASE),
    re.compile(r"بسم الله", re.IGNORECASE),
)

_HALLUCINATION_WORDS = frozenset({
    "nein", "ja", "oui", "non", "si", "danke", "merci", "gracias",
    "hallo", "bonjour", "hola", "ciao", "ja", "nee", "nee.",
    "nein.", "oui.", "non.", "si.", "danke.", "merci.",
})

_GREETING_RE = re.compile(
    r"\b(?:assalam|assalamualaikum|assalamualikum|assalamu|salam|salaam)\b"
    r"|السلام|سلام علیکم|سلام عليكم",
    re.IGNORECASE,
)

_STRONG_ROMAN_URDU_RE = re.compile(
    r"\b(?:assalam|assalamualaikum|assalamualikum|assalamu|salam|salaam|"
    r"nahi|nahin|kya|kyun|kyu|chahiye|karein|batao|bataiye|karo|"
    r"theek|achha|acha|tabiyat|bukhar|zaroorat|chalo|suno|dekho)\b",
    re.IGNORECASE,
)

# If the two passes score within this margin, keep the current language
# instead of switching — prevents rapid back-and-forth on ambiguous audio.
STICKY_MARGIN = 0.15


def is_hallucination(text: str) -> bool:
    """True if the text matches a known Whisper hallucination pattern."""
    low = text.lower().strip()
    if low in _HALLUCINATION_WORDS:
        return True
    return any(p.search(text) for p in _HALLUCINATION_PATTERNS)


def looks_like_roman_urdu_stt(text: str) -> bool:
    """True if Latin-script text looks like Roman Urdu (STT router markers)."""
    if _URDU_SCRIPT_RE.search(text):
        return False
    low = text.lower()
    if _STRONG_ROMAN_URDU_RE.search(low):
        return True
    return len(set(_ROMAN_URDU_RE.findall(low))) >= 2


def looks_like_english_stt(text: str) -> bool:
    """Heuristic: does this Latin-script text contain recognisable English?"""
    if not _LATIN_RE.search(text):
        return False
    if _URDU_SCRIPT_RE.search(text):
        return False
    if looks_like_roman_urdu_stt(text):
        return False
    if _CLINICAL_SELECTION_RE.search(text):
        return True
    if len(text) < 40:
        return bool(_ENGLISH_WORDS_RE.search(text))
    return english_word_count(text) >= 2


def english_word_count(text: str) -> int:
    """Count recognisable English words in Latin-script text."""
    return len(_ENGLISH_WORDS_RE.findall(text))


def is_full_english_sentence(text: str) -> bool:
    """True if the text is a clear, multi-word English utterance."""
    if not text or _URDU_SCRIPT_RE.search(text):
        return False
    if looks_like_roman_urdu_stt(text):
        return False
    return english_word_count(text) >= 3


def score_transcript(result: object) -> tuple[float, str]:
    """Return (confidence_score, text) for one Groq verbose_json result.

    Confidence = mean segment avg_logprob (closer to 0 is better), minus a
    penalty for average no_speech_prob. Empty text scores -inf so it can
    never win.
    """
    if not isinstance(result, dict):
        return float("-inf"), ""
    text = (result.get("text") or "").strip()
    if not text:
        return float("-inf"), ""
    if is_hallucination(text):
        return float("-inf"), ""
    segments = result.get("segments") or []
    if segments:
        avg_logprob = sum(s.get("avg_logprob", -5.0) for s in segments) / len(segments)
        no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
    else:
        avg_logprob, no_speech = -5.0, 0.0
    if no_speech >= _NO_SPEECH_THRESHOLD:
        return float("-inf"), ""
    if avg_logprob < _MIN_LOGPROB:
        return float("-inf"), ""
    return avg_logprob - _NO_SPEECH_PENALTY * no_speech, text


def pick_language(
    en_res: object,
    ur_res: object,
    *,
    current_language: str,
) -> tuple[str, str]:
    """Choose the better of the English/Urdu STT passes. Returns (lang, text)."""
    en_score, en_text = score_transcript(en_res)
    ur_score, ur_text = score_transcript(ur_res)

    if ur_text and _GREETING_RE.search(ur_text):
        return "ur", ur_text
    if en_text and _GREETING_RE.search(en_text):
        return "ur", en_text

    if en_score == float("-inf") and ur_score == float("-inf"):
        return current_language, ""

    cur = current_language
    if abs(en_score - ur_score) < STICKY_MARGIN:
        if cur == "ur" and ur_text:
            if en_text and is_full_english_sentence(en_text):
                lang, text = "en", en_text
            else:
                lang, text = "ur", ur_text
        elif cur == "en" and en_text and looks_like_english_stt(en_text):
            lang, text = "en", en_text
        elif cur == "en" and ur_text:
            lang, text = "ur", ur_text
        elif ur_score > en_score:
            lang, text = "ur", ur_text
        elif en_text and looks_like_english_stt(en_text):
            lang, text = "en", en_text
        else:
            lang, text = "ur", ur_text if ur_text else ""
    elif ur_score > en_score:
        lang, text = "ur", ur_text
    else:
        if not looks_like_english_stt(en_text):
            if ur_text and ur_score > float("-inf"):
                lang, text = "ur", ur_text
            else:
                return current_language, ""
        else:
            lang, text = "en", en_text

    if text and _URDU_SCRIPT_RE.search(text):
        lang = "ur"
    if text and lang == "en" and looks_like_roman_urdu_stt(text):
        lang = "ur"
    return lang, text
