"""Conversation language routing for bilingual English/Urdu voice agents.

Shared helpers used by patient, doctor, and future agents after STT:

- Detect phonetic English written in Urdu script (Whisper Urdu-pass artifact)
- Decide whether the LLM should reply in Urdu
- Sync ``LanguageState`` from user text / transcripts
- Soften first-turn greeting mishears toward the default language (Urdu)
- Switch to Urdu when the transcript is Urdu script / Roman Urdu (router trust)
- Switch to English immediately on clear Latin-script English or phonetic English
- Keep an established language only when the transcript itself is ambiguous

STT selection itself stays in ``LanguageRouterSTT`` / ``scoring.pick_language``.
"""

from __future__ import annotations

import re

from voice_agent_runtime.language_state import LanguageState
from voice_agent_runtime.roman_urdu import (
    contains_urdu_script,
    looks_english,
    looks_like_roman_urdu,
)

# Whisper often turns Urdu greetings / "can you hear me?" into short English.
_FIRST_TURN_EN_GREETING_RE = re.compile(
    r"(?i)(?:\b(?:hello|hi|hey)\b.*\b(?:nora|hear|hearing|listen)\b"
    r"|can you hear me|do you hear me|are you there|are you listening)",
)

_DOMAIN_EN_MARKERS = (
    "appointment",
    "encounter",
    "patient",
    "schedule",
    "pending",
    "briefing",
    "vitals",
    "prescription",
    "medicine",
    "medication",
    "book",
    "lab",
    "report",
)

# Short English hospital commands — STT often transcribes these in Latin letters
# during an Urdu session ("Briefing", "encounter", "yes"). They must NOT flip
# reply language or scripted briefing/encounter speech stays English.
_WORKFLOW_EN_WORDS: frozenset[str] = frozenset({
    "briefing", "encounter", "vitals", "vital", "yes", "yeah", "yep", "yup",
    "ok", "okay", "start", "and", "no", "nope", "next", "continue", "proceed",
    "go", "ready", "schedule", "pending", "patient", "appointment", "record",
    "save", "done", "confirm", "confirmed", "correct", "right", "sure",
    "please", "thanks", "thank", "hi", "hello", "let", "lets", "with", "said",
})

_PARTIAL_CLINICAL_EN_RE = re.compile(
    r"(?i)^(?:let'?s|lets)\s+start\s+with\.?$"
    r"|^start\s+with\.?$"
    r"|^i\s+said\.?$"
    r"|^(?:yes|yeah|yep),?\s*please\.?$"
)


def _latin_words(text: str) -> list[str]:
    low = re.sub(r"let's", "lets", (text or "").lower())
    return re.findall(r"[a-z]+", low)

# Whisper Urdu pass often transliterates spoken English into Urdu script.
_PHONETIC_EN_URDU_TOKENS: tuple[str, ...] = (
    "ہیلو", "ہلو", "ہیلوو", "نورا", "کیمیر", "لیٹ", "لیٹس", "فل", "اپ", "دائن",
    "کاؤنٹر", "کانٹر", "کانٹرز", "انکاؤنٹر", "انکاؤنٹرز", "فور", "فر",
    "فارسٹ", "فاسٹ", "سیکنڈ", "تھرڈ", "اپائنٹمنٹ", "اپائنمنٹ", "اپائنٹ",
    "بریفنگ", "گریفنگ", "پینڈنگ", "شیڈول", "سٹارٹ", "گیٹ", "میٹ", "ویس", "ود",
    "ففٹین", "سیکسٹین", "سیونٹین", "نوجیا", "نازیا", "ناسیا", "یس", "نو",
    "پیشنٹ", "پیشنت", "نیم", "نیمز", "نیمس", "ٹیل", "مے",
    "شفک",
)

_DOMAIN_PHONETIC_MARKERS: tuple[str, ...] = (
    "انکاؤنٹر", "انکاؤنٹرز", "کاؤنٹر", "کانٹر", "کانٹرز",
    "اپائنٹمنٹ", "اپائنمنٹ", "بریفنگ", "گریفنگ", "پینڈنگ", "شیڈول",
    "پیشنٹ", "پیشنت", "نیم", "سٹارٹ",
)

# "Let's start with Shafqat" → لیٹ سٹارٹ ویس شفک …
_START_WITH_PHONETIC_RE = re.compile(
    r"(?:لیٹ|لیٹس).{0,18}(?:سٹارٹ|start)"
    r"|(?:سٹارٹ|start).{0,18}(?:ویس|ود|with)"
    r"|سٹارٹ\s+ویس"
    r"|(?:فارسٹ|فاسٹ|first).{0,18}(?:ون|one|patient|پیشنٹ)?",
    re.IGNORECASE,
)

_STRONG_URDU_GRAMMAR: tuple[str, ...] = (
    "میں", "ہے", "ہیں", "سے", "کو", "کی", "کا", "کے", "چاہ", "بتاؤ", "کرنا",
    "چاہتی", "چاہتے", "ہو", "ہوں", "مجھے", "آپ", "کیا", "رہی", "رہے", "گا", "گی",
    "باقی", "دیکھ", "سن", "بتائ", "کرتے", "کریں", "والی", "والا", "بات",
)


def _urdu_token_in_text(token: str, text: str) -> bool:
    """Match phonetic tokens without short-token substring false positives (نو⊂نوجیا)."""
    if len(token) <= 2:
        return bool(
            re.search(
                rf"(?:^|[\s،۔؟]){re.escape(token)}(?:$|[\s،۔؟])",
                text,
            )
        )
    return token in text


def looks_like_phonetic_english_in_urdu_script(text: str) -> bool:
    """True when English speech was rendered as Urdu-script phonetics by STT."""
    if not contains_urdu_script(text):
        return False
    if looks_like_roman_urdu(text):
        return False

    # Real Urdu request framing around English loanwords
    # (e.g. "ہاں جی پیشنٹ نیم بتا دیں") — not phonetic English.
    if ("ہاں" in text or "جی" in text or "ہنجی" in text) and (
        "بتا" in text or "دیں" in text or "کریں" in text or "سنا" in text or "دے" in text
    ):
        return False

    # Real Urdu patient/action framing — not phonetic English
    # e.g. "شفقت محمود سے سٹارٹ کرتے ہیں", "شفقت محمود سے شروع کر لیں"
    if ("سے" in text or "کو" in text) and any(
        v in text
        for v in ("کرتے", "کریں", "کرنا", "کرتی", "شروع", "لیں", "کر لیں")
    ):
        return False
    if "شروع" in text and "کر" in text:
        return False

    # Pakistani Urdu: Urdu grammar + English hospital words in Urdu script
    # (پینڈنگ، بریفنگ، appointment) — not English spoken as phonetics.
    _real_urdu_markers = (
        "میں", "مجھے", "کیا", "میرے", "کریں", "دوں", "دو", "دیں", "بتاؤ",
        "ہیں", "ہے", "کوئی", "نا", "کہ", "یہی", "بولا", "پہ", "سے",
        "ایسا", "پہلی", "مجھ", "ہو", "جی", "کرتے", "کرنا",
    )
    _real_hits = 0
    for _m in _real_urdu_markers:
        if len(_m) <= 3:
            if _urdu_token_in_text(_m, text):
                _real_hits += 1
        elif _m in text:
            _real_hits += 1
    if _real_hits >= 3:
        return False

    # "Let's start with Shafqat" / "start with first one" (English phonetics)
    if _START_WITH_PHONETIC_RE.search(text):
        return True

    phonetic_hits = sum(1 for token in _PHONETIC_EN_URDU_TOKENS if _urdu_token_in_text(token, text))
    grammar_hits = _grammar_hits(text)
    domain_hit = any(_urdu_token_in_text(marker, text) for marker in _DOMAIN_PHONETIC_MARKERS)

    # "ہیلو نورا … پینڈنگ کانٹرز" — English with light Urdu grammar particles
    if domain_hit and phonetic_hits >= 1 and grammar_hits <= 3:
        return True
    if phonetic_hits >= 2 and grammar_hits <= 3:
        return True
    if phonetic_hits >= 1 and domain_hit and grammar_hits <= 3:
        return True
    # Hello Nora / can you hear me style openers in Urdu letters
    if ("ہیلو" in text or "ہلو" in text or "نورا" in text) and domain_hit:
        return True
    return False


def _grammar_hits(text: str) -> int:
    """Count Urdu grammar particles — avoid substring false positives (ہو in چوہوڑے)."""
    count = 0
    for word in _STRONG_URDU_GRAMMAR:
        if len(word) <= 3:
            if _urdu_token_in_text(word, text):
                count += 1
        elif word in text:
            count += 1
    return count


def strong_urdu_signal(text: str) -> bool:
    """True for clear Urdu (not short code-switch / phonetic English)."""
    if not contains_urdu_script(text) and not looks_like_roman_urdu(text):
        return False
    if looks_like_phonetic_english_in_urdu_script(text):
        return False
    phonetic_hits = sum(1 for token in _PHONETIC_EN_URDU_TOKENS if _urdu_token_in_text(token, text))
    if phonetic_hits >= 1 and len(text.strip()) < 45:
        return False
    if len(text.strip()) < 35:
        return False
    return _grammar_hits(text) >= 3


def user_turn_count(language_state: LanguageState | dict[str, str] | None) -> int:
    if not language_state:
        return 0
    try:
        return int(language_state.get("user_turn_count") or 0)
    except (TypeError, ValueError):
        return 0


def _tracked_is_english(language_state: LanguageState | dict[str, str] | None) -> bool:
    if not language_state:
        return False
    tracked = language_state.get("language")
    return bool(tracked and str(tracked).lower().startswith("en"))


def _tracked_is_urdu(language_state: LanguageState | dict[str, str] | None) -> bool:
    if not language_state:
        return False
    tracked = language_state.get("language")
    return bool(tracked and str(tracked).lower().startswith("ur"))


def is_workflow_english_command(user_text: str) -> bool:
    """True for short Latin hospital commands that should not flip an Urdu session."""
    if not user_text.strip() or contains_urdu_script(user_text):
        return False
    if looks_like_roman_urdu(user_text):
        return False
    words = _latin_words(user_text)
    if not words or len(words) > 4:
        return False
    return all(w in _WORKFLOW_EN_WORDS or w in _DOMAIN_EN_MARKERS for w in words)


def is_partial_clinical_english(user_text: str) -> bool:
    """Clipped English STT during patient/date selection — not a language switch."""
    raw = (user_text or "").strip()
    if not raw or contains_urdu_script(raw) or looks_like_roman_urdu(raw):
        return False
    if not looks_english(raw):
        return False
    low = raw.lower().rstrip(".!?")
    if _PARTIAL_CLINICAL_EN_RE.match(low):
        return True
    words = _latin_words(raw)
    return len(words) <= 3 and "start" in words


def _keep_urdu_despite_latin(
    language_state: LanguageState | dict[str, str] | None,
    user_text: str,
) -> bool:
    if not _tracked_is_urdu(language_state):
        return False
    return is_workflow_english_command(user_text) or is_partial_clinical_english(user_text)


def speech_language(
    language_state: LanguageState | dict[str, str] | None,
) -> str:
    """Normalized reply language for scripted speech (briefing, TTS, on_enter)."""
    if not language_state:
        return "ur"
    lang = language_state.get("language", "ur")
    return "ur" if str(lang).lower().startswith("ur") else "en"


def likely_english_mishear_of_urdu(user_text: str) -> bool:
    """True when Latin-script text is probably Urdu mis-transcribed as English."""
    low = user_text.lower().strip()
    if not low or contains_urdu_script(user_text) or looks_like_roman_urdu(user_text):
        return False
    if any(marker in low for marker in _DOMAIN_EN_MARKERS):
        return False
    if _FIRST_TURN_EN_GREETING_RE.search(low):
        return True
    words = re.findall(r"[a-z]+", low)
    return len(words) <= 6


def clear_latin_english(user_text: str) -> bool:
    """True for unambiguous Latin-script English (not a greeting mishear alone)."""
    if not user_text.strip() or contains_urdu_script(user_text):
        return False
    if not looks_english(user_text):
        return False
    return not likely_english_mishear_of_urdu(user_text)


def reply_in_urdu(
    language_state: LanguageState | dict[str, str] | None,
    user_text: str,
) -> bool:
    """Whether the LLM should reply in Urdu for this turn.

    Trusts STT/router transcript shape the same way patient agents do:
    Urdu script / Roman Urdu → Urdu, unless it is phonetic English.
    Clear Latin English switches immediately.
    """
    if looks_like_phonetic_english_in_urdu_script(user_text):
        return False

    # Clear Latin English switches — except short workflow commands mid Urdu session.
    if clear_latin_english(user_text):
        if _keep_urdu_despite_latin(language_state, user_text):
            return True
        return False

    turns = user_turn_count(language_state)

    if contains_urdu_script(user_text) or looks_like_roman_urdu(user_text):
        return True

    if turns <= 1:
        if user_text.strip() and looks_english(user_text):
            return likely_english_mishear_of_urdu(user_text)
        return True

    if user_text.strip() and looks_english(user_text):
        if _keep_urdu_despite_latin(language_state, user_text):
            return True
        return False

    if _tracked_is_english(language_state):
        return False

    tracked = language_state.get("language") if language_state else None
    if tracked:
        return str(tracked).lower().startswith("ur")
    return True


def sync_language_state(
    language_state: LanguageState | dict[str, str] | None,
    *,
    user_text: str,
    reply_urdu: bool,
) -> None:
    """Write ``language_state['language']`` from the latest user text."""
    if language_state is None:
        return
    language_state["language"] = "ur" if reply_urdu else "en"


def apply_transcript_language(
    language_state: LanguageState | dict[str, str] | None,
    transcript: str,
    stt_lang: str | None,
) -> None:
    """Update conversation language from a final STT transcript.

    Prefers transcript evidence over sticky session language so bilingual
    switching matches LanguageRouterSTT (same as patient agents).
    """
    if language_state is None:
        return

    turns = user_turn_count(language_state)

    if looks_like_phonetic_english_in_urdu_script(transcript):
        language_state["language"] = "en"
        return

    # Clear Latin English switches — except short workflow commands mid Urdu session.
    if clear_latin_english(transcript):
        if _keep_urdu_despite_latin(language_state, transcript):
            language_state["language"] = "ur"
            return
        language_state["language"] = "en"
        return

    if contains_urdu_script(transcript) or looks_like_roman_urdu(transcript):
        language_state["language"] = "ur"
        return

    if transcript.strip() and looks_english(transcript):
        if turns <= 1 and likely_english_mishear_of_urdu(transcript):
            language_state["language"] = "ur"
            return
        if _keep_urdu_despite_latin(language_state, transcript):
            language_state["language"] = "ur"
            return
        language_state["language"] = "en"
        return

    if isinstance(stt_lang, str) and stt_lang:
        if _keep_urdu_despite_latin(language_state, transcript):
            language_state["language"] = "ur"
            return
        if turns <= 1 and stt_lang.lower().startswith("en"):
            language_state["language"] = "ur"
        else:
            language_state["language"] = stt_lang
