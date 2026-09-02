#!/usr/bin/env python3
"""OCR every medical file under Yandex Disk /Анализы. and put a .txt sidecar
next to the original (same folder, same basename).

    .venv/bin/python scripts/ocr_analizy_to_yadisk.py
    .venv/bin/python scripts/ocr_analizy_to_yadisk.py --only 'общий.jpg'
    .venv/bin/python scripts/ocr_analizy_to_yadisk.py --force

OCR is API-only. Existing non-empty sidecars are skipped unless --force.
Telegram UI crumbs (tiny png / media_ / section_) are skipped.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from healthbot import ingest, ocr, yadisk  # noqa: E402
from healthbot.config import LOGS_DIR  # noqa: E402

log = logging.getLogger("health-bot")

REMOTE_ROOT = os.getenv("ANALIZY_DISK_DIR", "/Анализы.")
IMAGE_SFX = {".jpg", ".jpeg", ".png", ".webp"}
DOC_SFX = IMAGE_SFX | {".pdf", ".docx"}


def _setup() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "ocr_analizy.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def sidecar_path(remote_file: str) -> str:
    stem, _dot, _sfx = remote_file.rpartition(".")
    if not stem:
        return remote_file + ".txt"
    # keep compound names; only replace the last suffix
    return remote_file.rsplit(".", 1)[0] + ".txt"


def extract_local(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SFX:
        return ocr.ocr_image(path).text
    if suffix == ".pdf":
        return ingest.pdf_to_text(path)
    if suffix == ".docx":
        return ingest._docx_to_text(path.read_bytes())
    raise ValueError(f"unsupported {suffix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=REMOTE_ROOT)
    parser.add_argument("--only", default="", help="substring filter on remote path")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    _setup()

    if not yadisk.enabled():
        log.error("Yandex Disk не настроен")
        return 1

    files = yadisk.list_files(args.root)
    existing = {p for p, _ in files}
    todo = []
    for path, size in files:
        if args.only and args.only not in path:
            continue
        suffix = Path(path).suffix.lower()
        if suffix not in DOC_SFX:
            continue
        name = Path(path).name
        probe = b"\0" * min(size, 100_000)
        junk = ingest._is_junk_asset(name, probe)
        if junk:
            log.debug("skip junk %s (%s)", path, junk)
            continue
        txt = sidecar_path(path)
        if not args.force and txt in existing:
            # skip if a real sidecar already sits next to this file
            continue
        todo.append((path, size))

    log.info("To OCR: %d files (root=%s)", len(todo), args.root)
    done = 0
    errors = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for path, size in todo:
            if args.limit and done >= args.limit:
                break
            local = tmp_path / Path(path).name
            try:
                log.info("download %s (%d bytes)", path, size)
                yadisk.download(path, local)
                text = extract_local(local)
                if len((text or "").strip()) < 30:
                    log.warning("too little text: %s (%d chars)", path, len(text or ""))
                    errors += 1
                    continue
                header = (
                    f"{Path(path).name}\n\n"
                    f"Источник: Яндекс.Диск {path}\n"
                    "OCR: vision API (prepared upright/upscaled)\n\n"
                    "--- распознанный текст ---\n\n"
                )
                out = tmp_path / (local.stem + ".txt")
                out.write_text(header + text.strip() + "\n", encoding="utf-8")
                remote_txt = sidecar_path(path)
                yadisk.upload_file(out, remote_txt)
                log.info("uploaded %s (%d chars)", remote_txt, len(text))
                done += 1
            except Exception as exc:
                log.exception("fail %s: %s", path, exc)
                errors += 1
    log.info("done ok=%d errors=%d", done, errors)
    print(f"ok={done} errors={errors}")
    return 0 if errors == 0 or done > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
