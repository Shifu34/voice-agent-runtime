"""Basic tests for voice-agent-runtime (no network required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_agent_runtime import (
    LanguageState,
    contains_urdu_script,
    looks_english,
    looks_like_roman_urdu,
    score_transcript,
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
    ]
    for t in tests:
        t()
    print("=" * 60)
    print(f"Results: {len(tests)} passed, 0 failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
