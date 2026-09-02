"""Yandex Disk sync via WebDAV.

Every received document (PDF / photo) and its recognized .txt mirror is
uploaded to Yandex Disk, so the raw archive survives server reinstalls.

Auth — either OAuth token or login + app password:
    YANDEX_DISK_TOKEN     OAuth token (https://yandex.ru/dev/disk/poligon)
    YANDEX_DISK_LOGIN + YANDEX_DISK_PASSWORD  (app password for 2FA accounts)
Path:
    YANDEX_DISK_DIR       remote folder (default /healthbot)
Failures are logged and never break the main flow.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

log = logging.getLogger("health-bot")

WEBDAV_URL = "https://webdav.yandex.ru"
_TIMEOUT = 30


def _auth():
    token = os.getenv("YANDEX_DISK_TOKEN", "").strip()
    if token:
        return {"headers": {"Authorization": f"OAuth {token}"}}
    login = os.getenv("YANDEX_DISK_LOGIN", "").strip()
    password = os.getenv("YANDEX_DISK_PASSWORD", "").strip()
    if login and password:
        return {"auth": (login, password)}
    return None


def enabled() -> bool:
    return _auth() is not None


def _mkdirs(remote_dir: str) -> None:
    """Create remote_dir and its parents (MKCOL is not recursive)."""
    auth = _auth()
    parts = [p for p in remote_dir.split("/") if p]
    current = ""
    for part in parts:
        current += "/" + part
        resp = requests.request("MKCOL", WEBDAV_URL + current, timeout=_TIMEOUT, **auth)
        if resp.status_code not in (201, 405, 409, 301):
            # 405/409 = already exists
            resp.raise_for_status()


def exists(remote_path: str) -> bool:
    """True if a file or folder already exists on Disk."""
    auth = _auth()
    if auth is None:
        return False
    resp = requests.request(
        "PROPFIND", WEBDAV_URL + remote_path, data=b"",
        headers={"Depth": "0", **auth.get("headers", {})},
        auth=auth.get("auth"), timeout=_TIMEOUT,
    )
    return resp.status_code in (207, 200)


def upload(local_path: str | Path, remote_subdir: str = "") -> str | None:
    """Upload a file to YANDEX_DISK_DIR[/remote_subdir]/filename.

    Returns the remote path on success, None when sync is disabled.
    Raises on HTTP errors (caller catches and logs).
    """
    if not enabled():
        return None
    local_path = Path(local_path)
    base = os.getenv("YANDEX_DISK_DIR", "/healthbot").rstrip("/")
    remote_dir = f"{base}/{remote_subdir.strip('/')}" if remote_subdir else base
    _mkdirs(remote_dir)
    remote_path = f"{remote_dir}/{local_path.name}"
    auth = _auth()
    with open(local_path, "rb") as f:
        resp = requests.put(WEBDAV_URL + remote_path, data=f,
                            timeout=max(_TIMEOUT, local_path.stat().st_size // 100_000),
                            **auth)
    resp.raise_for_status()
    log.info("Yandex Disk: uploaded %s → %s (%d bytes)",
             local_path.name, remote_path, local_path.stat().st_size)
    return remote_path


def upload_file(local_path: str | Path, remote_path: str) -> str | None:
    """Upload to an exact Disk path (e.g. /Анализы./2024/оак.txt)."""
    if not enabled():
        return None
    local_path = Path(local_path)
    remote_path = "/" + remote_path.lstrip("/")
    remote_dir = remote_path.rsplit("/", 1)[0] or "/"
    _mkdirs(remote_dir)
    auth = _auth()
    with open(local_path, "rb") as f:
        resp = requests.put(
            WEBDAV_URL + remote_path, data=f,
            timeout=max(_TIMEOUT, local_path.stat().st_size // 100_000),
            **auth,
        )
    resp.raise_for_status()
    log.info("Yandex Disk: uploaded %s → %s (%d bytes)",
             local_path.name, remote_path, local_path.stat().st_size)
    return remote_path


def sync_document(local_path: str | Path, txt_mirror: str | Path | None = None,
                  remote_subdir: str = "") -> None:
    """Best-effort sync of a document and its .txt mirror. Never raises."""
    if not enabled():
        return
    for path in [local_path, txt_mirror]:
        if not path:
            continue
        try:
            upload(path, remote_subdir)
        except Exception as exc:
            log.warning("Yandex Disk sync failed for %s: %s", path, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Reading: recursive listing + download (for history import)
# ─────────────────────────────────────────────────────────────────────────────
def _propfind(remote_dir: str) -> list[tuple[str, int, bool]]:
    """PROPFIND Depth:1 → list of (path, size_bytes, is_dir) for children."""
    auth = _auth()
    if auth is None:
        raise RuntimeError("Yandex Disk не настроен: задай YANDEX_DISK_TOKEN "
                           "или YANDEX_DISK_LOGIN + YANDEX_DISK_PASSWORD в .env")
    body = ('<?xml version="1.0" encoding="utf-8"?>'
            '<d:propfind xmlns:d="DAV:"><d:prop>'
            '<d:resourcetype/><d:getcontentlength/></d:prop></d:propfind>')
    resp = requests.request(
        "PROPFIND", WEBDAV_URL + remote_dir, data=body,
        headers={"Depth": "1", "Content-Type": "application/xml",
                 **auth.get("headers", {})},
        auth=auth.get("auth"), timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return parse_propfind(resp.text, remote_dir)


def parse_propfind(xml_text: str, remote_dir: str) -> list[tuple[str, int, bool]]:
    """Parse a WebDAV multistatus response into (path, size, is_dir) tuples."""
    import xml.etree.ElementTree as ET
    entries: list[tuple[str, int, bool]] = []
    root = ET.fromstring(xml_text)
    for response in root.iter("{DAV:}response"):
        href = response.findtext("{DAV:}href", "")
        # href may be URL-encoded
        from urllib.parse import unquote
        path = unquote(href)
        is_dir = response.find(".//{DAV:}resourcetype/{DAV:}collection") is not None
        size_text = response.findtext(".//{DAV:}getcontentlength") or "0"
        try:
            size = int(size_text)
        except ValueError:
            size = 0
        # Skip the queried directory itself
        if path.rstrip("/") != remote_dir.rstrip("/"):
            entries.append((path, size, is_dir))
    return entries


def list_files(remote_dir: str) -> list[tuple[str, int]]:
    """Recursively list all files under remote_dir → [(path, size_bytes)]."""
    out: list[tuple[str, int]] = []
    stack = [remote_dir.rstrip("/")]
    while stack:
        current = stack.pop()
        for path, size, is_dir in _propfind(current):
            if is_dir:
                stack.append(path)
            else:
                out.append((path, size))
    return sorted(out)


def download(remote_path: str, dest: str | Path) -> Path:
    """Download a remote file to a local path."""
    auth = _auth()
    if auth is None:
        raise RuntimeError("Yandex Disk не настроен (см. .env)")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(WEBDAV_URL + remote_path, timeout=120,
                        stream=True, **auth)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dest
