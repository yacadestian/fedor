#!/usr/bin/env python3
"""Entry point for the daily diagnostician (L1 digest / L2 profile).

Cron:
    55 20 * * * .venv/bin/python diagnostician.py --digest
    0  4  * * * .venv/bin/python diagnostician.py --profile
"""
import sys

from healthbot.diagnostician import main

if __name__ == "__main__":
    sys.exit(main())
