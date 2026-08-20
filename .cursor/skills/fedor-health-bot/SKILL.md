---
name: fedor-health-bot
description: Develops the Fedor Telegram health-analytics bot (healthbot package, OCR, diary, ClickHouse, history import, Worker). Use when the user mentions Fedor, Фёдор, healthbot, tg_listener, лабораторные, OCR, дневник, ClickHouse, diagnostician, or asks to change the health bot on this machine.
---

# Fedor health bot

Read [AGENTS.md](../../../AGENTS.md) before edits. Worker notes: [worker.md](../../../docs/worker.md).

## Layout

- `tg_listener.py` → `healthbot.bot.main` (Telegram long polling)
- `diagnostician.py` → L1 digest / L2 profile
- `import_history.py` → Yandex Disk + IMAP bundle
- `healthbot/` — bot, llm, db, ocr, diary, voice, yadisk, ingest, mailscan
- `knowledge/` — protocols, optimal ranges, antagonists
- `tests/test_offline.py`, `tests/test_ingest.py` — no CH/Telegram/LLM

## Do

- Russian replies, phone-readable.
- Surgical diffs. Push `origin` (GitHub) and `gitlab`.
- Keep `owner_id` isolation. Synthetic data only in tests.

## Do not

- Restart `cursor-agent-worker-fedor.service` unless explicitly asked.
- Change GitHub `origin` to GitLab.
- Commit `.env`, `data/`, `logs/`, real labs, tokens.
- Pull upstream petrovich-health or rewrite knowledge YAML unasked.
