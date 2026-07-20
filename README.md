# voice-agent-runtime

Reusable voice agent runtime for Healix LiveKit agents:

- **LanguageRouterSTT** — dual-pinned English/Urdu Whisper transcription with confidence scoring
- **LanguageState** — typed mutable conversation language tracker
- **roman_urdu** — script and Roman-Urdu heuristics for session handlers and LLM routing
- **scoring** — Whisper verbose_json confidence scoring and hallucination filtering
- **handoff** — agent transfer helper with conversation context preservation

Install from Git:

```bash
pip install "voice-agent-runtime @ git+https://github.com/Shifu34/voice-agent-runtime.git@v0.1.0"
```

Import:

```python
from voice_agent_runtime import LanguageRouterSTT, LanguageState, transfer_with_context
```
