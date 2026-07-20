"""Typed conversation language state shared across STT, TTS, and LLM layers."""

from __future__ import annotations

from typing import Literal

LanguageCode = Literal["en", "ur"]


class LanguageState(dict[str, str]):
    """Mutable conversation language tracker.

    Subclasses ``dict`` so existing components that read/write
    ``language_state["language"]`` keep working unchanged.
    """

    def __init__(self, language: LanguageCode = "ur") -> None:
        super().__init__()
        self["language"] = language

    @property
    def language(self) -> str:
        return self.get("language", "ur")

    @language.setter
    def language(self, value: str) -> None:
        self["language"] = value

    def normalized(self) -> LanguageCode:
        """Return ``"ur"`` or ``"en"`` for the current language."""
        lang = str(self.get("language", "ur")).lower()
        return "ur" if lang.startswith("ur") else "en"
