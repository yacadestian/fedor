#!/usr/bin/env python3
"""Scan mailbox (all time), OCR via vision API, upload medical docs + sidecar txt
to Yandex Disk next to each original.

    .venv/bin/python scripts/sync_mail_medical_to_yadisk.py
    .venv/bin/python scripts/sync_mail_medical_to_yadisk.py --cache-only
    .venv/bin/python scripts/sync_mail_medical_to_yadisk.py --process-only

Remote layout (same folder, same basename):
    {YANDEX_DISK_DIR}/mail_medical/{year}/{YYYY-MM-DD}_{title}_{hash8}.pdf
    {YANDEX_DISK_DIR}/mail_medical/{year}/{YYYY-MM-DD}_{title}_{hash8}.txt

OCR is API-only (OCR_BACKEND=vision_api). No local OCR.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
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


def _enforce_api_ocr() -> None:
    if os.getenv("OCR_BACKEND", "").strip().lower() == "tesseract":
        raise SystemExit("Локальный OCR запрещён: только API (OCR_BACKEND=vision_api)")
    os.environ["OCR_BACKEND"] = "vision_api"
    os.environ["OCR_TESSERACT"] = "0"


def process_cache(work_dir: Path, mail_cache: Path) -> dict:
    _enforce_api_ocr()
    if not yadisk.enabled():
        raise SystemExit("Yandex Disk не настроен")

    lock_path = work_dir / "sync.lock"
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_f = open(lock_path, "a+", encoding="utf-8")
    fcntl.flock(lock_f, fcntl.LOCK_EX)
    try:
        state_path = work_dir / STATE_NAME
        state = _load_state(state_path)
        uploaded: dict = state.setdefault("uploaded", {})

        records: list[ingest.IngestRecord] = []
        seen: set[str] = set()
        if (work_dir / "manifest.json").exists():
            for r in ingest.load_manifest(work_dir):
                err = r.error or ""
                ok = (not err) or err == "duplicate" or err.startswith("skip ")
                if ok:
                    records.append(r)
                    if r.file_hash and r.error != "duplicate":
                        seen.add(r.file_hash)

        stats = {"scanned": 0, "medical": 0, "uploaded": 0, "skipped": 0, "failed": 0}
        cached = mailscan.load_mail_cache(mail_cache)
        log.info("Process cache: %d attachments, %d already ingested hashes",
                 len(cached), len(seen))

        for att in cached:
            digest = hashlib.sha256(att.data).hexdigest()
            stats["scanned"] += 1
            rec = ingest.process_file(
                att.filename, att.data, "mail",
                f"{att.folder} | {att.subject} | {att.msg_date} | {digest[:12]}",
                work_dir, seen)
            records.append(rec)
            ingest.save_manifest(records, work_dir)
            if not rec.medical:
                stats["skipped"] += 1
                log.info("  skip non-medical %s | %s",
                         att.filename, (rec.error or rec.title or "")[:70])
                continue
            stats["medical"] += 1
            if digest in uploaded and uploaded[digest].get("orig") and uploaded[digest].get("txt"):
                stats["uploaded"] += 1
                log.info("  already on Disk %s", uploaded[digest]["orig"])
                continue

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
                dest_orig = (f"{os.getenv('YANDEX_DISK_DIR', '/healthbot').rstrip('/')}"
                             f"/{remote_subdir}/{orig.name}")
                dest_txt = (f"{os.getenv('YANDEX_DISK_DIR', '/healthbot').rstrip('/')}"
                            f"/{remote_subdir}/{txt.name}")
                if yadisk.exists(dest_orig) and yadisk.exists(dest_txt):
                    uploaded[digest] = {"orig": dest_orig, "txt": dest_txt,
                                        "title": rec.title, "date": rec.doc_date}
                    _save_state(state_path, state)
                    stats["uploaded"] += 1
                    log.info("  exists on Disk %s", dest_orig)
                    continue
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

        ingest.save_manifest(records, work_dir)
        log.info("Process done scanned=%d medical=%d uploaded=%d skipped=%d failed=%d",
                 stats["scanned"], stats["medical"], stats["uploaded"],
                 stats["skipped"], stats["failed"])
        return stats
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()


def cache_mail(work_dir: Path) -> int:
    mail_cache = work_dir / ".mail_cache"
    mail_cache.mkdir(parents=True, exist_ok=True)
    log.info("Phase 1: cache all PDF attachments (all time, API OCR later)")
    n_new = 0
    try:
        for _att in mailscan.iter_scan_mail(years=0, cache_dir=mail_cache):
            n_new += 1
    except Exception as exc:
        log.error("Mail scan failed (cache continues from disk): %s", exc)
        return n_new
    log.info("Phase 1 done, newly cached this run: %d", n_new)
    return n_new


def main() -> int:
    _setup()
    _enforce_api_ocr()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--process-only", action="store_true")
    ap.add_argument("--loop", action="store_true",
                    help="with --process-only: keep processing as cache grows")
    args = ap.parse_args()

    work_dir = Path("data/history_bundle")
    mail_cache = work_dir / ".mail_cache"

    if args.cache_only:
        cache_mail(work_dir)
        return 0
    if args.process_only:
        while True:
            stats = process_cache(work_dir, mail_cache)
            print(json.dumps(stats, ensure_ascii=False))
            if not args.loop:
                return 0 if stats["failed"] == 0 else 1
            time.sleep(30)
    # default: cache then process
    cache_mail(work_dir)
    stats = process_cache(work_dir, mail_cache)
    print(json.dumps(stats, ensure_ascii=False))
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
