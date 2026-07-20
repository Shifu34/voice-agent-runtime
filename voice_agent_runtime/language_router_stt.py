"""Language-router STT for bilingual (English + Urdu) voice input.

Whisper's per-utterance *auto-detect* is unreliable on short, quiet, or
code-switched speech — it mis-labels the language and mis-transcribes the
words. This router fixes that by NOT letting Whisper guess:

  1. For each utterance it transcribes the SAME audio twice in parallel —
     once pinned to English, once pinned to Urdu (pinned transcription is
     markedly more accurate than auto-detect).
  2. It asks Groq for ``verbose_json`` so each result carries per-segment
     ``avg_logprob`` (Whisper's own confidence). The language the audio
     actually is in gets the higher confidence; the wrong-language pass is
     forced to romanise/guess and scores lower.
  3. It picks the higher-confidence transcript. Ties fall back to the
     current conversation language (``language_state``) to avoid flapping.

Running both passes concurrently keeps latency close to a single call.
Only English and Urdu are considered — exactly the two languages the
assistant supports.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave

import aiohttp
from livekit.agents import APIConnectOptions, stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from voice_agent_runtime.language_state import LanguageState
from voice_agent_runtime.scoring import pick_language

logger = logging.getLogger("language-router-stt")

_GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def _to_wav_bytes(buffer) -> bytes:
    """Combine LiveKit audio frame(s) into a 16-bit PCM WAV byte string."""
    frames = buffer if isinstance(buffer, list) else [buffer]
    if not frames:
        return b""
    sample_rate = frames[0].sample_rate
    num_channels = frames[0].num_channels
    pcm = b"".join(bytes(f.data) for f in frames)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return bio.getvalue()


class LanguageRouterSTT(stt.STT):
    """Non-streaming STT that routes each utterance to English or Urdu.

    Drop-in replacement for ``openai.STT`` in a VAD-driven AgentSession.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-large-v3",
        language_state: LanguageState | dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._api_key = api_key
        self._model = model
        self._language_state = language_state
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _transcribe(self, wav_bytes: bytes, language: str) -> object:
        """One Groq transcription pinned to ``language``, verbose_json."""
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", self._model)
        form.add_field("language", language)
        form.add_field("response_format", "verbose_json")
        form.add_field("temperature", "0")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with self._get_session().post(_GROQ_URL, data=form, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _current_language(self) -> str:
        if isinstance(self._language_state, LanguageState):
            return self._language_state.normalized()
        if isinstance(self._language_state, dict):
            lang = self._language_state.get("language", "ur")
            return "ur" if str(lang).lower().startswith("ur") else "en"
        return "ur"

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        t0 = time.perf_counter()

        wav_bytes = _to_wav_bytes(buffer)
        t_wav = time.perf_counter()

        en_res, ur_res = await asyncio.gather(
            self._transcribe(wav_bytes, "en"),
            self._transcribe(wav_bytes, "ur"),
            return_exceptions=True,
        )
        t_stt = time.perf_counter()

        for res, lang in ((en_res, "en"), (ur_res, "ur")):
            if isinstance(res, Exception):
                logger.warning("Groq transcription (%s) failed: %s", lang, res)

        chosen_lang, text = pick_language(
            en_res,
            ur_res,
            current_language=self._current_language(),
        )
        t_pick = time.perf_counter()

        logger.info(
            "ROUTER latency — wav=%.0fms stt=%.0fms pick=%.0fms total=%.0fms lang=%s chars=%d",
            (t_wav - t0) * 1000,
            (t_stt - t_wav) * 1000,
            (t_pick - t_stt) * 1000,
            (t_pick - t0) * 1000,
            chosen_lang,
            len(text),
        )

        if not text:
            logger.debug("ROUTER — both passes hallucination/silence, discarding")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(language=self._current_language(), text=""),
                ],
            )

        if self._language_state is not None:
            self._language_state["language"] = chosen_lang
            logger.debug("ROUTER — chose %s: %r", chosen_lang, text)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language=chosen_lang, text=text),
            ],
        )

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        await super().aclose()
