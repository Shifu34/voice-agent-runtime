"""Basic tests for voice-agent-runtime (no network required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_agent_runtime import (
    LanguageState,
    apply_transcript_language,
    contains_urdu_script,
    looks_english,
    looks_like_phonetic_english_in_urdu_script,
    looks_like_roman_urdu,
    reply_in_urdu,
    score_transcript,
    speech_language,
)
from voice_agent_runtime.scoring import is_hallucination, pick_language


def test_language_state_dict_compat():
    state = LanguageState("ur")
    assert state["language"] == "ur"
    state["language"] = "en"
    assert state.normalized() == "en"
    assert isinstance(state, dict)
    print("[PASS] test_language_state_dict_compat")


def test_roman_urdu_detection():
    assert looks_like_roman_urdu("mera sar dard ho raha hai")
    assert not looks_like_roman_urdu("I want to book an appointment")
    assert contains_urdu_script("میرا سر درد ہے")
    print("[PASS] test_roman_urdu_detection")


def test_looks_english():
    assert looks_english("yes")
    assert looks_english("book an appointment please")
    assert not looks_english("mera sar dard hai")
    assert not looks_english("میرا سر درد")
    print("[PASS] test_looks_english")


def test_score_transcript_rejects_hallucination():
    score, text = score_transcript({"text": "thank you for watching", "segments": []})
    assert score == float("-inf")
    assert text == ""
    assert is_hallucination("thank you for watching")
    print("[PASS] test_score_transcript_rejects_hallucination")


def test_score_transcript_accepts_real_speech():
    score, text = score_transcript({
        "text": "book an appointment",
        "segments": [{"avg_logprob": -0.3, "no_speech_prob": 0.05}],
    })
    assert score > float("-inf")
    assert text == "book an appointment"
    print("[PASS] test_score_transcript_accepts_real_speech")


def test_pick_language_prefers_higher_confidence():
    en = {"text": "book appointment", "segments": [{"avg_logprob": -0.2, "no_speech_prob": 0.05}]}
    ur = {"text": "بک", "segments": [{"avg_logprob": -1.5, "no_speech_prob": 0.3}]}
    lang, text = pick_language(en, ur, current_language="ur")
    assert lang == "en"
    assert "book" in text
    print("[PASS] test_pick_language_prefers_higher_confidence")


def test_phonetic_english_in_urdu_script():
    phonetic = "لیٹس فل اپ دائن کاؤنٹر فور شفقت محمود ففٹین جولائی فارسٹ اپائنٹمنٹ"
    real_urdu = "ڈاکٹر صاحبہ دو مریضوں پر encounters باقی ہیں"
    assert looks_like_phonetic_english_in_urdu_script(phonetic)
    assert not looks_like_phonetic_english_in_urdu_script(real_urdu)
    print("[PASS] test_phonetic_english_in_urdu_script")


def test_reply_in_urdu_first_turn():
    state = LanguageState("ur")
    state["user_turn_count"] = "1"
    assert reply_in_urdu(state, "Hello Nora, can you hear me?") is True
    assert reply_in_urdu(state, "get my pending encounters") is False
    assert reply_in_urdu(
        state,
        "لیٹس فل اپ دائن کاؤنٹر فور شفقت محمود ففٹین جولائی فارسٹ اپائنٹمنٹ",
    ) is False
    assert reply_in_urdu(state, "کیا میری آواز آ رہی ہے") is True
    print("[PASS] test_reply_in_urdu_first_turn")


def test_english_session_sticky():
    state = LanguageState("en")
    state["user_turn_count"] = "2"
    # Urdu-script turns follow the router (same as patient) — switch to Urdu
    assert reply_in_urdu(state, "نوجیا والی کی بات کر") is True
    assert reply_in_urdu(state, "گریفنگ کرتے ہیں") is False  # phonetic English
    assert reply_in_urdu(
        state,
        "ڈاکٹر صاحبہ مجھے pending encounters کی تفصیل بتائیں اور مریض کا نام بھی بولیں",
    ) is True
    print("[PASS] test_english_session_sticky")


def test_clear_english_switches_from_urdu_session():
    state = LanguageState("ur")
    state["user_turn_count"] = "2"
    assert reply_in_urdu(state, "Yes, start with the first one.") is False
    assert reply_in_urdu(state, "I think we can start with Shafqat Mehmood.") is False
    print("[PASS] test_clear_english_switches_from_urdu_session")


def test_phonetic_hello_pending_encounters():
    text = "ہیلو نورا ہے ایک پینڈنگ کانٹرز ہے؟"
    assert looks_like_phonetic_english_in_urdu_script(text) is True
    state = LanguageState("ur")
    state["user_turn_count"] = "1"
    assert reply_in_urdu(state, text) is False
    print("[PASS] test_phonetic_hello_pending_encounters")


def test_phonetic_lets_start_with_patient():
    text = "لیٹ سٹارٹ ویس شفک چوہوڑے"
    assert looks_like_phonetic_english_in_urdu_script(text) is True
    state = LanguageState("en")
    state["user_turn_count"] = "3"
    apply_transcript_language(state, text, "ur")
    assert state["language"] == "en"
    assert reply_in_urdu(state, text) is False
    print("[PASS] test_phonetic_lets_start_with_patient")


def test_urdu_hold_in_pick_language():
    # English slightly better but Urdu script present — stay Urdu when current=ur
    en = {
        "text": "Do I have any pending encounters?",
        "segments": [{"avg_logprob": -0.25, "no_speech_prob": 0.05}],
    }
    ur = {
        "text": "کیا میرے کوئی pending encounters ہیں",
        "segments": [{"avg_logprob": -0.40, "no_speech_prob": 0.08}],
    }
    lang, text = pick_language(en, ur, current_language="ur")
    assert lang == "ur"
    assert "pending" in text or "کیا" in text
    print("[PASS] test_urdu_hold_in_pick_language")


def test_phonetic_urdu_prefers_english_pass():
    en = {
        "text": "Hello Nora, any pending encounters?",
        "segments": [{"avg_logprob": -0.35, "no_speech_prob": 0.08}],
    }
    ur = {
        "text": "ہیلو نورا ہے ایک پینڈنگ کانٹرز ہے؟",
        "segments": [{"avg_logprob": -0.20, "no_speech_prob": 0.05}],
    }
    lang, text = pick_language(en, ur, current_language="ur")
    assert lang == "en"
    assert "pending" in text.lower() or "Hello" in text or "Nora" in text
    print("[PASS] test_phonetic_urdu_prefers_english_pass")


def test_intentional_urdu_patient_name_request():
    text = "ہاں جی پیشنٹ نیم بتا دیں"
    assert looks_like_phonetic_english_in_urdu_script(text) is False
    state = LanguageState("en")
    state["user_turn_count"] = "2"
    # Router-style: Urdu script switches reply language
    assert reply_in_urdu(state, text) is True
    assert reply_in_urdu(state, "ہنجی دے دیں") is True
    assert reply_in_urdu(state, "شفقت محمود سے سٹارٹ کرتے ہیں") is True
    print("[PASS] test_intentional_urdu_patient_name_request")


def test_clear_english_beats_garbled_urdu_hold():
    en = {
        "text": "Yes, start with the first one please.",
        "segments": [{"avg_logprob": -0.30, "no_speech_prob": 0.06}],
    }
    ur = {
        "text": "لیکن پہلے بھی بھیگیں",
        "segments": [{"avg_logprob": -0.25, "no_speech_prob": 0.05}],
    }
    lang, text = pick_language(en, ur, current_language="ur")
    assert lang == "en"
    assert "first" in text.lower() or "start" in text.lower()
    print("[PASS] test_clear_english_beats_garbled_urdu_hold")


def test_partial_lets_start_with_keeps_urdu_session():
    state = LanguageState("ur")
    state["user_turn_count"] = "3"
    apply_transcript_language(state, "Let's start with", "en")
    assert state["language"] == "ur"
    assert reply_in_urdu(state, "Let's start with") is True
    print("[PASS] test_partial_lets_start_with_keeps_urdu_session")


def test_apply_transcript_follows_urdu_router():
    from voice_agent_runtime import apply_transcript_language

    state = LanguageState("en")
    state["user_turn_count"] = "2"
    apply_transcript_language(state, "ہنجی دے دیں", "ur")
    assert state["language"] == "ur"
    apply_transcript_language(state, "Yes, start with the first patient.", "en")
    assert state["language"] == "en"
    print("[PASS] test_apply_transcript_follows_urdu_router")


def test_workflow_english_keeps_urdu_session():
    from voice_agent_runtime import apply_transcript_language

    state = LanguageState("ur")
    state["user_turn_count"] = "4"
    for cmd in ("Briefing", "encounter", "and", "yes"):
        apply_transcript_language(state, cmd, "en")
        assert state["language"] == "ur", f"Expected ur after {cmd!r}, got {state['language']}"
        assert reply_in_urdu(state, cmd) is True, f"Expected Urdu reply for {cmd!r}"
    apply_transcript_language(
        state,
        "I want to continue in English from here please.",
        "en",
    )
    assert state["language"] == "en"
    print("[PASS] test_workflow_english_keeps_urdu_session")


def test_speech_language_normalized():
    from voice_agent_runtime import speech_language

    assert speech_language(LanguageState("ur")) == "ur"
    assert speech_language(LanguageState("en")) == "en"
    assert speech_language(None) == "ur"
    print("[PASS] test_speech_language_normalized")


def test_soft_fallback_keeps_short_barge_in():
    low_conf = {
        "text": "first one",
        "segments": [{"avg_logprob": -1.05, "no_speech_prob": 0.48}],
    }
    silent = {
        "text": "",
        "segments": [{"avg_logprob": -0.2, "no_speech_prob": 0.05}],
    }
    lang, text = pick_language(low_conf, silent, current_language="en")
    assert lang == "en"
    assert "first" in text
    print("[PASS] test_soft_fallback_keeps_short_barge_in")


def main():
    print("=" * 60)
    print("Testing voice-agent-runtime")
    print("=" * 60)
    tests = [
        test_language_state_dict_compat,
        test_roman_urdu_detection,
        test_looks_english,
        test_score_transcript_rejects_hallucination,
        test_score_transcript_accepts_real_speech,
        test_pick_language_prefers_higher_confidence,
        test_phonetic_english_in_urdu_script,
        test_reply_in_urdu_first_turn,
        test_english_session_sticky,
        test_clear_english_switches_from_urdu_session,
        test_phonetic_hello_pending_encounters,
        test_phonetic_lets_start_with_patient,
        test_intentional_urdu_patient_name_request,
        test_clear_english_beats_garbled_urdu_hold,
        test_partial_lets_start_with_keeps_urdu_session,
        test_apply_transcript_follows_urdu_router,
        test_workflow_english_keeps_urdu_session,
        test_speech_language_normalized,
    ]
    for t in tests:
        t()
    print("=" * 60)
    print(f"Results: {len(tests)} passed, 0 failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
