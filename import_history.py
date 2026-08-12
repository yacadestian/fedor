#!/usr/bin/env python3
"""Bulk history import: Yandex Disk + mail → filtered medical archive.

Collect phase (any machine):
    .venv/bin/python import_history.py --yadisk-dir "/Здоровье" --mail \
        --out data/history_bundle

Load into the bot's ClickHouse (run on the bot's server):
    .venv/bin/python import_history.py --from-bundle data/history_bundle --to-db

Options:
    --yadisk-dir PATH   remote folder on Yandex Disk (recursive)
    --mail              scan mail (MAIL_* env / .env)
    --mail-limit N      stop after N attachments (for a trial run)
    --out DIR           bundle output dir (default data/history_bundle)
    --from-bundle DIR   skip collecting, use an existing bundle
    --to-db             import medical records into ClickHouse
    --owner ID          owner_id for --to-db (default: your Telegram chat_id)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from healthbot import ingest, mailscan, yadisk  # noqa: E402
from healthbot.config import LOGS_DIR  # noqa: E402


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / "import_history.log", encoding="utf-8"),
                  logging.StreamHandler()],
        force=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Import medical history")
    ap.add_argument("--yadisk-dir", default="")
    ap.add_argument("--mail", action="store_true")
    ap.add_argument("--mail-limit", type=int, default=None)
    ap.add_argument("--out", default="data/history_bundle")
    ap.add_argument("--from-bundle", default="")
    ap.add_argument("--to-db", action="store_true")
    ap.add_argument("--owner", default="")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger("health-bot")
    work_dir = Path(args.from_bundle or args.out)

    if args.from_bundle:
        records = ingest.load_manifest(work_dir)
        log.info("Loaded %d records from %s", len(records), work_dir)
    else:
        # Resume: keep successes; re-queue records that failed (rate limits etc.)
        records: list[ingest.IngestRecord] = []
        seen: set[str] = set()
        done_refs: set[str] = set()
        if (work_dir / "manifest.json").exists():
            prev = ingest.load_manifest(work_dir)
            kept = []
            for r in prev:
                # Successful recognition, duplicates, or intentional skips → keep
                err = r.error or ""
                ok = (not err) or err == "duplicate" or err.startswith("skip ")
                if ok:
                    kept.append(r)
                    if r.file_hash and r.error != "duplicate":
                        seen.add(r.file_hash)
                    if r.source_ref:
                        done_refs.add(r.source_ref)
            records = kept
            log.info("Resuming: %d ok kept, %d failed will be retried",
                     len(kept), len(prev) - len(kept))

        if args.yadisk_dir:
            log.info("=== Yandex Disk: %s ===", args.yadisk_dir)
            files = yadisk.list_files(args.yadisk_dir)
            log.info("Found %d files", len(files))
            for i, (remote_path, size) in enumerate(files, 1):
                name = remote_path.rsplit("/", 1)[-1]
                if remote_path in done_refs:
                    log.info("[%d/%d] SKIP (done) %s", i, len(files), name)
                    continue
                log.info("[%d/%d] %s (%d KB)", i, len(files), name, size // 1024)
                try:
                    local = yadisk.download(remote_path, work_dir / ".dl" / name)
                    data = local.read_bytes()
                    local.unlink(missing_ok=True)
                except Exception as exc:
                    log.warning("Download failed %s: %s", remote_path, exc)
                    continue
                rec = ingest.process_file(name, data, "yadisk", remote_path, work_dir, seen)
                records.append(rec)
                done_refs.add(remote_path)
                label = rec.error or (rec.title or "")[:60]
                log.info("  → medical=%s %s %s | text=%d chars",
                         rec.medical, rec.kind, label, rec.text_chars)
                ingest.save_manifest(records, work_dir)  # crash-safe progress
                # Soft throttle to stay under Gemini free-tier rate limits
                import time
                time.sleep(4.0)

        if args.mail:
            log.info("=== Mail scan ===")
            mail_cache = work_dir / ".mail_cache"

            def _ingest_mail_att(att: mailscan.MailAttachment) -> None:
                ref = f"{att.folder} | {att.subject} | {att.msg_date}"
                if ref in done_refs and att.filename:  # already processed this ref
                    # still allow content-hash dedup inside process_file
                    pass
                if ref in done_refs:
                    log.info("  SKIP mail (done) %s", (att.subject or "")[:60])
                    return
                rec = ingest.process_file(att.filename, att.data, "mail", ref,
                                          work_dir, seen)
                records.append(rec)
                done_refs.add(ref)
                label = rec.error or (rec.title or "")[:60]
                log.info("  → medical=%s %s %s", rec.medical, rec.kind, label)
                ingest.save_manifest(records, work_dir)

            # Resume from on-disk cache first (crash-safe), then live IMAP scan
            cached = mailscan.load_mail_cache(mail_cache)
            if cached:
                log.info("Mail cache: %d attachments, processing pending…", len(cached))
                for att in cached:
                    _ingest_mail_att(att)
            try:
                n_new = 0
                for att in mailscan.iter_scan_mail(limit=args.mail_limit,
                                                   cache_dir=mail_cache):
                    before = len(records)
                    _ingest_mail_att(att)
                    if len(records) > before:
                        n_new += 1
                log.info("Mail scan done, newly ingested this run: %d", n_new)
            except Exception as exc:
                log.error("Mail scan failed: %s", exc)

        manifest = ingest.save_manifest(records, work_dir)
        medical = [r for r in records if r.medical]
        log.info("Total: %d files, medical: %d", len(records), len(medical))

        consolidated = ingest.build_consolidated(records, work_dir)
        out_doc = work_dir / "consolidated.md"
        out_doc.write_text(consolidated, encoding="utf-8")
        log.info("Consolidated document: %s (%d KB)", out_doc, out_doc.stat().st_size // 1024)
        print(f"\nГотово: {len(medical)} медицинских документов из {len(records)} файлов")
        print(f"Единый документ: {out_doc}")
        print(f"Манифест: {manifest}")

    if args.to_db:
        owner = args.owner.strip()
        if not owner:
            print("Для --to-db нужен --owner (ваш Telegram chat_id из users.yaml)")
            return 2
        n_docs, n_biomarkers = ingest.import_bundle_to_db(work_dir, owner)
        print(f"В базу: {n_docs} документов, {n_biomarkers} биомаркеров")

    return 0


if __name__ == "__main__":
    sys.exit(main())
