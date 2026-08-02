"""healthbot — personal health assistant Telegram bot.

Layout:
    config.py       paths, env, constants
    bot.py          Telegram long-polling listener (entry point)
    llm.py          LLM provider abstraction (DeepSeek API / Claude CLI)
    db.py           ClickHouse interface
    extractor.py    LLM biomarker extraction / document classification
    pdf_parser.py   PDF text extraction
    ocr.py          robust OCR for low-quality photos (vision or tesseract)
    voice.py        voice message transcription (faster-whisper, optional)
    diary.py        health diary & hypotheses
    yadisk.py       Yandex Disk sync (WebDAV)
    nutrition.py    meal analysis
    charts.py       trend charts
    spc.py          SPC control charts
    report_pdf.py   PDF report for the doctor
    reminders.py    medication reminders
    diagnostician.py daily digest (L1) + health profile (L2), cron
"""

__version__ = "0.2.0"
