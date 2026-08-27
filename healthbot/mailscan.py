"""IMAP scanner: find PDF attachments in mail (last N years).

Targets INBOX + user folders (e.g. "Чеки") — many labs email results as PDF.
Only attachment parts are downloaded (headers + BODYSTRUCTURE first), so even
a huge mailbox scans cheaply. Stdlib imaplib — no extra deps.

Env:
    MAIL_HOST          imap.yandex.ru | imap.gmail.com | imap.mail.ru | ...
                       (auto-detected from login domain when unset)
    MAIL_LOGIN         full address (user@yandex.ru)
    MAIL_PASSWORD      app password (Yandex/Gmail/Mail.ru all require one
                       for IMAP when 2FA is on)
    MAIL_SINCE_YEARS   how far back to scan (default 5)
    MAIL_FOLDERS       comma-separated, default "INBOX,Чеки"
"""
from __future__ import annotations

import imaplib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from email.header import decode_header

log = logging.getLogger("health-bot")

_HOST_BY_DOMAIN = {
    "yandex.ru": "imap.yandex.ru", "ya.ru": "imap.yandex.ru",
    "yandex.com": "imap.yandex.com",
    "gmail.com": "imap.gmail.com", "googlemail.com": "imap.gmail.com",
    "mail.ru": "imap.mail.ru", "bk.ru": "imap.mail.ru",
    "inbox.ru": "imap.mail.ru", "list.ru": "imap.mail.ru",
    "outlook.com": "outlook.office365.com", "hotmail.com": "outlook.office365.com",
}


@dataclass
class MailAttachment:
    folder: str
    msg_date: str
    subject: str
    sender: str
    filename: str
    data: bytes = field(repr=False, default=b"")


def _mail_config() -> tuple[str, str, str, int, list[str]]:
    login = os.getenv("MAIL_LOGIN", "").strip()
    password = os.getenv("MAIL_PASSWORD", "").strip()
    host = os.getenv("MAIL_HOST", "").strip()
    if not host and "@" in login:
        host = _HOST_BY_DOMAIN.get(login.rsplit("@", 1)[1].lower(), "")
    if not (login and password and host):
        raise RuntimeError(
            "Почта не настроена. Нужны MAIL_LOGIN + MAIL_PASSWORD (пароль приложения) "
            "в .env; MAIL_HOST определится по домену или задайте явно.")
    years = int(os.getenv("MAIL_SINCE_YEARS", "0"))
    folders = [f.strip() for f in os.getenv("MAIL_FOLDERS", "INBOX,Чеки").split(",") if f.strip()]
    return host, login, password, years, folders


def decode_mutf7(text: str) -> str:
    """Decode IMAP modified UTF-7 folder names (Cyrillic etc.)."""
    def _sub(m: re.Match) -> str:
        chunk = m.group(1).replace(",", "/")
        pad = "=" * (-len(chunk) % 4)
        import base64
        return base64.b64decode(chunk + pad).decode("utf-16-be", errors="replace")
    return re.sub(r"&([^-]*)-", _sub, text).replace("&-", "&")


def decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    parts = []
    for chunk, enc in decode_header(value):
        if isinstance(chunk, bytes):
            parts.append(chunk.decode(enc or "utf-8", errors="replace"))
        else:
            parts.append(chunk)
    return "".join(parts)


def match_folders(all_folders: list[str], wanted: list[str]) -> list[str]:
    """Match requested folder names against the server's list (decoded)."""
    decoded = {f: decode_mutf7(f) for f in all_folders}
    result = []
    for want in wanted:
        want_l = want.lower()
        for raw, dec in decoded.items():
            if raw in result:
                continue
            if dec.lower() == "inbox" and want_l == "inbox":
                result.append(raw)
            elif want_l != "inbox" and want_l in dec.lower():
                result.append(raw)
    return result


def _fetch_bodystructures(mail: imaplib.IMAP4_SSL,
                          msg_ids: list[bytes]) -> dict[bytes, str]:
    """One FETCH BODYSTRUCTURE roundtrip for a chunk of messages.
    Returns {msg_id: structure_text}; unparsable entries are skipped."""
    set_str = b",".join(msg_ids).decode()
    try:
        status, data = mail.fetch(set_str, "(BODYSTRUCTURE)")
    except Exception as exc:
        log.warning("Batch BODYSTRUCTURE failed: %s", exc)
        return {}
    if status != "OK":
        return {}
    out: dict[bytes, str] = {}
    for item in data:
        if isinstance(item, tuple):
            text = "".join(
                part.decode("utf-8", errors="replace") if isinstance(part, bytes) else str(part)
                for part in item if part)
        elif isinstance(item, bytes):
            text = item.decode("utf-8", errors="replace")
        else:
            continue
        m = re.match(r"(\d+)\s+\(BODYSTRUCTURE\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if m:
            out[m.group(1).encode()] = m.group(2)
    return out


def _list_folders(mail: imaplib.IMAP4_SSL) -> list[str]:
    status, data = mail.list()
    if status != "OK":
        return []
    folders = []
    for line in data:
        if not line:
            continue
        line = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        m = re.search(r'" (?P<name>.+)$| (?P<name2>[^ ]+)$', line)
        name = (m.group("name") or m.group("name2")).strip('"') if m else ""
        if name:
            folders.append(name)
    return folders


def _message_has_pdf(structure: str) -> list[int]:
    """Parse FETCH BODYSTRUCTURE response text → part numbers of PDF parts.

    Matches both "application/pdf" parts and application/octet-stream parts
    whose filename (possibly RFC2047-encoded, e.g. =?utf-8?q?...=2Epdf?=)
    ends with .pdf.
    """
    parts: list[int] = []
    depth = 0
    part_no = 0
    i = 0
    body_start = None
    while i < len(structure):
        ch = structure[i]
        if ch == "(":
            if depth == 1:
                part_no += 1
                body_start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 1 and body_start is not None:
                seg = structure[body_start:i]
                seg_l = seg.lower()
                if ('"application" "pdf"' in seg_l or '.pdf"' in seg_l
                        or ".pdf" in decode_mime_header(seg).lower()):
                    parts.append(part_no)
                body_start = None
        i += 1
    return parts


def _cache_paths(cache_dir: Path, digest: str) -> tuple[Path, Path]:
    return cache_dir / "pdf" / f"{digest}.pdf", cache_dir / "meta" / f"{digest}.json"


def _save_mail_cache(cache_dir: Path, digest: str, att: "MailAttachment") -> None:
    import json
    pdf_path, meta_path = _cache_paths(cache_dir, digest)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        pdf_path.write_bytes(att.data)
    meta_path.write_text(json.dumps({
        "folder": att.folder, "msg_date": att.msg_date, "subject": att.subject,
        "sender": att.sender, "filename": att.filename, "sha256": digest,
        "size": len(att.data),
    }, ensure_ascii=False), encoding="utf-8")


def load_mail_cache(cache_dir: Path) -> list[MailAttachment]:
    """Load previously cached PDF attachments (for crash-safe resume)."""
    import json
    meta_dir = cache_dir / "meta"
    if not meta_dir.is_dir():
        return []
    out: list[MailAttachment] = []
    for meta_path in sorted(meta_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pdf_path = cache_dir / "pdf" / f"{meta['sha256']}.pdf"
            if not pdf_path.exists():
                continue
            out.append(MailAttachment(
                folder=meta.get("folder", ""), msg_date=meta.get("msg_date", ""),
                subject=meta.get("subject", ""), sender=meta.get("sender", ""),
                filename=meta.get("filename") or f"{meta['sha256'][:12]}.pdf",
                data=pdf_path.read_bytes(),
            ))
        except Exception as exc:
            log.warning("Bad mail cache entry %s: %s", meta_path.name, exc)
    return out


def iter_scan_mail(years: int | None = None, folders: list[str] | None = None,
                   limit: int | None = None,
                   cache_dir: Path | None = None,
                   progress=lambda msg: log.info(msg)):
    """Yield PDF attachments one-by-one; optionally persist to cache_dir."""
    host, login, password, cfg_years, cfg_folders = _mail_config()
    years = cfg_years if years is None else years
    folders = folders or cfg_folders
    # years <= 0 → all mail, no SINCE filter
    since = None
    if years > 0:
        since = (datetime.now() - timedelta(days=365 * years)).strftime("%d-%b-%Y")

    seen_hashes: set[str] = set()
    if cache_dir:
        for att in load_mail_cache(cache_dir):
            import hashlib
            seen_hashes.add(hashlib.sha256(att.data).hexdigest())

    yielded = 0
    mail: imaplib.IMAP4_SSL | None = None

    def _connect() -> imaplib.IMAP4_SSL:
        last: Exception | None = None
        for attempt in range(1, 6):
            try:
                m = imaplib.IMAP4_SSL(host)
                m.login(login, password)
                return m
            except Exception as exc:
                last = exc
                log.warning("IMAP connect failed (%s), retry %d/5", exc, attempt)
                import time as _t
                _t.sleep(min(20, 2 * attempt))
        raise RuntimeError(f"IMAP недоступен: {last}")

    def _safe_logout(m: imaplib.IMAP4_SSL | None) -> None:
        if m is None:
            return
        try:
            m.logout()
        except Exception:
            try:
                m.shutdown()
            except Exception:
                pass

    try:
        mail = _connect()
        all_folders = _list_folders(mail)
        targets = match_folders(all_folders, folders)
        log.info("Mail folders matched: %s (of %s)", targets, all_folders)

        for folder in targets:
            retries_folder = 0
            msg_ids: list[bytes] = []
            selected = False
            while retries_folder < 5:
                try:
                    status, _ = mail.select(f'"{folder}"', readonly=True)
                    if status != "OK":
                        log.warning("Cannot select folder %s", folder)
                        break
                    query = f"(SINCE {since})" if since else "ALL"
                    status, data = mail.search(None, query)
                    if status != "OK":
                        break
                    msg_ids = data[0].split()
                    log.info("Folder %s: %d messages %s",
                             folder, len(msg_ids),
                             f"since {since}" if since else "(all time)")
                    selected = True
                    break
                except Exception as exc:
                    retries_folder += 1
                    log.warning("IMAP select/search %s failed (%s), reconnect %d",
                                folder, exc, retries_folder)
                    _safe_logout(mail)
                    import time as _t
                    _t.sleep(min(30, 3 * retries_folder))
                    mail = _connect()
            if not selected or mail is None:
                continue

            batch = int(os.getenv("MAIL_BATCH", "50"))
            chunk_start = 0
            while chunk_start < len(msg_ids):
                if limit and yielded >= limit:
                    break
                chunk = msg_ids[chunk_start:chunk_start + batch]
                try:
                    struct_map = _fetch_bodystructures(mail, chunk)
                except Exception as exc:
                    log.warning("BODYSTRUCTURE chunk %d failed: %s — reconnect",
                                chunk_start, exc)
                    _safe_logout(mail)
                    import time as _t
                    _t.sleep(2)
                    mail = _connect()
                    mail.select(f'"{folder}"', readonly=True)
                    try:
                        struct_map = _fetch_bodystructures(mail, chunk)
                    except Exception as exc2:
                        log.warning("BODYSTRUCTURE retry failed: %s, skip chunk", exc2)
                        chunk_start += batch
                        continue

                candidates = []
                for num in chunk:
                    struct_text = struct_map.get(num, "")
                    if "pdf" not in struct_text.lower():
                        continue
                    pdf_parts = _message_has_pdf(struct_text)
                    if pdf_parts:
                        candidates.append((num, pdf_parts))
                log.info("  %s: scanned %d/%d, %d with PDF so far",
                         folder, min(chunk_start + batch, len(msg_ids)),
                         len(msg_ids), len(candidates))

                for num, pdf_parts in candidates:
                    if limit and yielded >= limit:
                        break
                    try:
                        status, hdata = mail.fetch(
                            num, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
                    except Exception as exc:
                        log.warning("IMAP fetch header failed: %s, reconnect", exc)
                        _safe_logout(mail)
                        mail = _connect()
                        mail.select(f'"{folder}"', readonly=True)
                        try:
                            status, hdata = mail.fetch(
                                num, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
                        except Exception:
                            continue
                    header_text = ""
                    if status == "OK" and hdata and isinstance(hdata[0], tuple):
                        header_text = hdata[0][1].decode("utf-8", errors="replace")
                    subject = decode_mime_header(
                        re.search(r"Subject:\s*(.+?)(?:\r?\n\S|$)", header_text, re.S | re.I)
                        .group(1).strip() if re.search(r"Subject:", header_text, re.I) else "")
                    sender = decode_mime_header(
                        re.search(r"From:\s*(.+?)(?:\r?\n|$)", header_text, re.I)
                        .group(1).strip() if re.search(r"From:", header_text, re.I) else "")
                    msg_date = (re.search(r"Date:\s*(.+?)(?:\r?\n|$)", header_text, re.I)
                                .group(1).strip() if re.search(r"Date:", header_text, re.I) else "")

                    for part_no in pdf_parts:
                        try:
                            status, pdata = mail.fetch(num, f"(BODY.PEEK[{part_no}])")
                        except Exception as exc:
                            log.warning("IMAP fetch body failed: %s, reconnect", exc)
                            _safe_logout(mail)
                            mail = _connect()
                            mail.select(f'"{folder}"', readonly=True)
                            try:
                                status, pdata = mail.fetch(num, f"(BODY.PEEK[{part_no}])")
                            except Exception:
                                continue
                        if status != "OK" or not pdata or not isinstance(pdata[0], tuple):
                            continue
                        raw = pdata[0][1] or b""
                        import base64
                        try:
                            blob = base64.b64decode(raw, validate=True)
                        except Exception:
                            blob = raw
                        if not blob.startswith(b"%PDF"):
                            try:
                                blob = base64.b64decode(raw + b"=" * (-len(raw) % 4))
                            except Exception:
                                continue
                            if not blob.startswith(b"%PDF"):
                                continue
                        import hashlib
                        digest = hashlib.sha256(blob).hexdigest()
                        if digest in seen_hashes:
                            continue
                        seen_hashes.add(digest)
                        att = MailAttachment(
                            folder=decode_mutf7(folder), msg_date=msg_date,
                            subject=subject, sender=sender,
                            filename=f"mail_{num.decode()}_{part_no}.pdf",
                            data=blob,
                        )
                        if cache_dir is not None:
                            _save_mail_cache(cache_dir, digest, att)
                        progress(f"PDF: {subject[:60]} | {msg_date[:20]} | {len(blob)//1024} KB")
                        yielded += 1
                        yield att
                chunk_start += batch
    finally:
        _safe_logout(mail)


def scan_mail(years: int | None = None, folders: list[str] | None = None,
              limit: int | None = None,
              cache_dir: Path | None = None,
              progress=lambda msg: log.info(msg)) -> list[MailAttachment]:
    """Scan configured mailbox, return PDF attachments (deduped by content)."""
    return list(iter_scan_mail(years=years, folders=folders, limit=limit,
                               cache_dir=cache_dir, progress=progress))
