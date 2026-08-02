"""LLM provider abstraction: DeepSeek API or Claude CLI.

Provider resolution (first match):
  1. LLM_PROVIDER=deepseek|claude
  2. DEEPSEEK_API_KEY set  → deepseek
  3. claude CLI on PATH    → claude
  4. otherwise deepseek (will error on call with a clear message)

Tiers: "fast" (extraction, classification, OCR cleanup) and
"smart" (Q&A, health profile, digests).

Env:
  DEEPSEEK_API_KEY      API key for api.deepseek.com
  DEEPSEEK_BASE_URL     default https://api.deepseek.com/v1
  DEEPSEEK_MODEL_FAST   default deepseek-v4-flash
  DEEPSEEK_MODEL_SMART  default deepseek-v4-pro
  DEEPSEEK_THINKING     enabled|disabled per tier, comma-separated as
                        fast=disabled,smart=enabled (default exactly that).
                        V4 models think by default and reasoning tokens eat
                        the completion budget — deterministic tasks (extraction,
                        cleanup) run non-thinking; Q&A keeps thinking.
  CLAUDE_MODEL_FAST     default claude-sonnet-4-6 (fallback claude-haiku-4-5-20251001)
  CLAUDE_MODEL_SMART    default claude-opus-4-6 (fallback claude-sonnet-4-6)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

import requests

log = logging.getLogger("health-bot")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
DEEPSEEK_MODELS = {
    "fast": os.getenv("DEEPSEEK_MODEL_FAST", "deepseek-v4-flash"),
    "smart": os.getenv("DEEPSEEK_MODEL_SMART", "deepseek-v4-pro"),
}
_CLAUDE_MODELS = {
    "fast": [os.getenv("CLAUDE_MODEL_FAST", "claude-sonnet-4-6"), "claude-haiku-4-5-20251001"],
    "smart": [os.getenv("CLAUDE_MODEL_SMART", "claude-opus-4-6"), "claude-sonnet-4-6"],
}


def provider() -> str:
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced in ("deepseek", "claude"):
        return forced
    if DEEPSEEK_API_KEY:
        return "deepseek"
    if shutil.which("claude"):
        return "claude"
    return "deepseek"


def available() -> bool:
    return provider() == "claude" or bool(DEEPSEEK_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────
def _thinking_config(tier: str) -> dict:
    """Per-tier thinking toggle. V4 thinks by default; reasoning tokens share
    the completion budget, so deterministic fast-tier tasks run non-thinking."""
    raw = os.getenv("DEEPSEEK_THINKING", "")
    setting = "disabled" if tier == "fast" else "enabled"
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip().lower() == tier:
                setting = v.strip().lower()
    return {"type": "enabled" if setting == "enabled" else "disabled"}


def _deepseek_chat(prompt: str, tier: str, timeout: int,
                   temperature: float, max_retries: int = 2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY не задан — добавь его в .env")
    model = DEEPSEEK_MODELS.get(tier, DEEPSEEK_MODELS["fast"])
    thinking = _thinking_config(tier)
    # Thinking consumes the same token budget — give smart tier room for both
    max_tokens = 8192 if thinking["type"] == "disabled" else 32768
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            log.info("DeepSeek call model=%s tier=%s thinking=%s (%d chars)",
                     model, tier, thinking["type"], len(prompt))
            resp = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "thinking": thinking,
                },
                timeout=timeout,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            answer = (choice["message"].get("content") or "").strip()
            if not answer:
                reason = choice.get("finish_reason", "?")
                raise RuntimeError(
                    f"DeepSeek вернул пустой ответ (finish_reason={reason})")
            log.info("DeepSeek answer (%d chars) model=%s", len(answer), model)
            return answer
        except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
            last_exc = exc
            log.warning("DeepSeek attempt %d failed: %s", attempt + 1, exc)
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DeepSeek недоступен после {max_retries + 1} попыток: {last_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Claude CLI
# ─────────────────────────────────────────────────────────────────────────────
def _claude_chat(prompt: str, tier: str, timeout: int) -> str:
    for model in _CLAUDE_MODELS.get(tier, _CLAUDE_MODELS["fast"]):
        try:
            log.info("claude CLI call model=%s tier=%s (%d chars)", model, tier, len(prompt))
            result = subprocess.run(
                ["claude", "-p", "--model", model, "--", prompt],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                log.warning("claude CLI rc=%d model=%s: %s",
                            result.returncode, model, result.stderr.strip()[:200])
                continue
            answer = result.stdout.strip()
            if answer:
                return answer
        except subprocess.TimeoutExpired:
            log.warning("claude CLI timeout model=%s", model)
        except Exception as exc:
            log.warning("claude CLI error model=%s: %s", model, exc)
    raise RuntimeError("claude CLI недоступен для всех моделей")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def chat(prompt: str, tier: str = "fast", timeout: int = 180,
         temperature: float = 0.1) -> str:
    """Single-turn LLM call. Raises RuntimeError when unavailable."""
    if provider() == "claude":
        return _claude_chat(prompt, tier, timeout)
    return _deepseek_chat(prompt, tier, timeout, temperature)
