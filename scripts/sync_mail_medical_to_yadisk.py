#!/usr/bin/env python3
"""Scan mailbox (all time), OCR via vision API, upload medical docs + sidecar txt
to Yandex Disk next to each original.

    .venv/bin/python scripts/sync_mail_medical_to_yadisk.py

Remote layout (same folder, same basename):
    {YANDEX_DISK_DIR}/mail_medical/{year}/{YYYY-MM-DD}_{title}_{hash8}.pdf
    {YANDEX_DISK_DIR}/mail_medical/{year}/{YYYY-MM-DD}_{title}_{hash8}.txt

OCR is API-only (OCR_BACKEND=vision_api). No local OCR.
Resume-safe: local bundle + remote existence check + state JSON.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from healthbot import ingest, mailscan, yadisk  # noqa: E402
from healthbot.config import LOGS_DIR  # noqa: E402

log = logging.getLogger("health-bot")

STATE_NAME = "yadisk_sync_state.json"
REMOTE_ROOT = "mail_medical"


def _setup() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "sync_mail_yadisk.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _safe_name(text: str, fallback: str = "doc") -> str:
    text = (text or "").strip() or fallback
    text = re.sub(r"[^\wа-яА-ЯёЁ0-9.-]+", "_", text, flags=re.I)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or fallback)[:80]


def _year_and_date(rec: ingest.IngestRecord, att: mailscan.MailAttachment) -> tuple[str, str]:
    if rec.doc_date and re.match(r"\d{4}-\d{2}-\d{2}", rec.doc_date):
        return rec.doc_date[:4], rec.doc_date
    if att.msg_date:
        try:
            dt = parsedate_to_datetime(att.msg_date)
            return f"{dt.year:04d}", dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return "без_даты", "0000-00-00"


def _sidecar_text(rec: ingest.IngestRecord, att: mailscan.MailAttachment, body: str) -> str:
    lines = [
        rec.title or rec.filename,
        "",
        f"Дата документа: {rec.doc_date or 'неизвестно'}",
        f"Тип: {rec.kind or ''}",
        f"Письмо: {att.subject}",
        f"От: {att.sender}",
        f"Папка: {att.folder}",
        f"Дата письма: {att.msg_date}",
        f"Исходный файл: {att.filename}",
        f"SHA256: {rec.file_hash}",
    ]
    if rec.summary:
        lines += ["", "Кратко:", rec.summary]
    lines += ["", "--- распознанный текст ---", "", body.strip()]
    return "\n".join(lines).rstrip() + "\n"


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"uploaded": {}}
    return {"uploaded": {}}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _upload_pair(orig: Path, txt: Path, remote_subdir: str) -> tuple[str, str]:
    orig_remote = yadisk.upload(orig, remote_subdir)
    txt_remote = yadisk.upload(txt, remote_subdir)
    return orig_remote or "", txt_remote or ""


def main() -> int:
    _setup()
    if not yadisk.enabled():
        log.error("Yandex Disk не настроен (.env)")
        return 2
    backend = os.getenv("OCR_BACKEND", "")
    if backend == "tesseract":
        log.error("Локальный OCR запрещён")
        return 2
    os.environ["OCR_BACKEND"] = "vision_api"
    os.environ.setdefault("OCR_TESSERACT", "0")

    work_dir = Path("data/history_bundle")
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / STATE_NAME
    state = _load_state(state_path)
    uploaded: dict = state.setdefault("uploaded", {})

    records: list[ingest.IngestRecord] = []
    seen: set[str] = set()
    if (work_dir / "manifest.json").exists():
        prev = ingest.load_manifest(work_dir)
        for r in prev:
            err = r.error or ""
            ok = (not err) or err == "duplicate" or err.startswith("skip ")
            if ok:
                records.append(r)
                if r.file_hash and r.error != "duplicate":
                    seen.add(r.file_hash)
        log.info("Resume ingest: %d records, %d hashes", len(records), len(seen))

    mail_cache = work_dir / ".mail_cache"
    stats = {"scanned": 0, "medical": 0, "uploaded": 0, "skipped": 0, "failed": 0}

    def handle(att: mailscan.MailAttachment) -> None:
        digest = hashlib.sha256(att.data).hexdigest()
        stats["scanned"] += 1
        rec = ingest.process_file(att.filename, att.data, "mail",
                                  f"{att.folder} | {att.subject} | {att.msg_date} | {digest[:12]}",
                                  work_dir, seen)
        records.append(rec)
        ingest.save_manifest(records, work_dir)
        if not rec.medical:
            stats["skipped"] += 1
            log.info("  skip non-medical %s | %s", att.filename, (rec.error or rec.title or "")[:70])
            return
        stats["medical"] += 1
        if digest in uploaded and uploaded[digest].get("orig") and uploaded[digest].get("txt"):
            log.info("  already on Disk %s", uploaded[digest]["orig"])
            stats["uploaded"] += 1
            return

        year, date_s = _year_and_date(rec, att)
        stem = _safe_name(rec.title or Path(att.filename).stem)
        base = f"{date_s}_{stem}_{digest[:8]}"
        suffix = Path(att.filename).suffix.lower() or ".pdf"
        if suffix not in {".pdf", ".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".pdf"
        remote_subdir = f"{REMOTE_ROOT}/{year}"

        body = ingest._load_text_cache(work_dir, rec.file_hash)
        with tempfile.TemporaryDirectory() as tmp:
            orig = Path(tmp) / f"{base}{suffix}"
            txt = Path(tmp) / f"{base}.txt"
            orig.write_bytes(att.data)
            txt.write_text(_sidecar_text(rec, att, body), encoding="utf-8")
            # Skip re-PUT if both already exist
            dest_orig = f"{os.getenv('YANDEX_DISK_DIR', '/healthbot').rstrip('/')}/{remote_subdir}/{orig.name}"
            dest_txt = f"{os.getenv('YANDEX_DISK_DIR', '/healthbot').rstrip('/')}/{remote_subdir}/{txt.name}"
            if yadisk.exists(dest_orig) and yadisk.exists(dest_txt):
                uploaded[digest] = {"orig": dest_orig, "txt": dest_txt,
                                    "title": rec.title, "date": rec.doc_date}
                _save_state(state_path, state)
                stats["uploaded"] += 1
                log.info("  exists on Disk %s", dest_orig)
                return
            try:
                o_path, t_path = _upload_pair(orig, txt, remote_subdir)
                uploaded[digest] = {"orig": o_path, "txt": t_path,
                                    "title": rec.title, "date": rec.doc_date}
                _save_state(state_path, state)
                stats["uploaded"] += 1
                log.info("  Disk ← %s + .txt", o_path)
            except Exception as exc:
                stats["failed"] += 1
                log.warning("  Disk upload failed: %s", exc)

    log.info("=== Mail → Yandex Disk (API OCR only, all time) ===")
    # Phase 1: cache PDFs quickly while IMAP is alive (no OCR yet)
    log.info("Phase 1: cache all PDF attachments")
    try:
        n_new = 0
        for _att in mailscan.iter_scan_mail(years=0, cache_dir=mail_cache):
            n_new += 1
        log.info("Phase 1 done, newly cached this run: %d", n_new)
    except Exception as exc:
        log.error("Mail scan failed (cache continues from disk): %s", exc)

    # Phase 2: OCR via vision API + upload medical originals + .txt sidecars
    log.info("Phase 2: classify + API OCR + Disk upload")
    cached = mailscan.load_mail_cache(mail_cache)
    log.info("Mail cache: %d attachments", len(cached))
    for att in cached:
        handle(att)

    ingest.save_manifest(records, work_dir)
    log.info("Done scanned=%d medical=%d uploaded=%d skipped=%d failed=%d",
             stats["scanned"], stats["medical"], stats["uploaded"],
             stats["skipped"], stats["failed"])
    print(json.dumps(stats, ensure_ascii=False))
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
