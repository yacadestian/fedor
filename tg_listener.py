#!/usr/bin/env python3
"""Entry point: run the Telegram bot (long polling).

Usage: .venv/bin/python tg_listener.py
"""
from healthbot.bot import main

if __name__ == "__main__":
    main()
