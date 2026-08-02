"""Voice message transcription via faster-whisper (optional, local).

Telegram voice messages are OGG/Opus. faster-whisper reads them via PyAV
and runs fully offline on CPU — nothing leaves your server, which matters
for health-related speech.

Env:
    VOICE_STT        auto|off (default auto: on if faster-whisper installed)
    WHISPER_MODEL    default "small" (tiny/base/small/medium — trade speed
                     for accuracy; "small" is a good CPU default)
    WHISPER_MODELS_DIR  model cache dir (default data/models)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("health-bot")

_model = None


def stt_available() -> bool:
    if os.getenv("VOICE_STT", "auto").strip().lower() in ("0", "no", "false", "off"):
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        from .config import DATA_DIR
        name = os.getenv("WHISPER_MODEL", "small")
        download_root = os.getenv("WHISPER_MODELS_DIR", str(DATA_DIR / "models"))
        log.info("Loading whisper model '%s' (dir=%s)…", name, download_root)
        _model = WhisperModel(name, device="cpu", compute_type="int8",
                              download_root=download_root)
    return _model


def transcribe(audio_path: str | Path, language: str = "ru") -> str:
    """Transcribe an audio file to text. Raises RuntimeError on failure."""
    if not stt_available():
        raise RuntimeError(
            "Распознавание голоса не установлено. Выполни: "
            ".venv/bin/pip install faster-whisper")
    audio_path = Path(audio_path)
    model = _get_model()
    segments, info = model.transcribe(str(audio_path), language=language,
                                      vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    log.info("Transcribed %s: %.1fs audio, %d chars, lang=%s",
             audio_path.name, info.duration, len(text), info.language)
    if not text:
        raise RuntimeError("Не удалось распознать речь — попробуй говорить ближе к микрофону.")
    return text
