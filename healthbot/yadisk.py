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
