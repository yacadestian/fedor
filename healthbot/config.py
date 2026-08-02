"""Shared paths, environment loading and small constants."""
from __future__ import annotations

import os
from datetime import timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
SCHEMA_PATH = BASE_DIR / "schema.sql"
USERS_YAML_PATH = BASE_DIR / "users.yaml"

load_dotenv(BASE_DIR / ".env")

MSK_TZ = timezone(timedelta(hours=3))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
