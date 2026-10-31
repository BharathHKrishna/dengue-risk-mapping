"""Raw data cache with pull dates.

Every raw download is stored under ``data/raw`` together with an entry in
``data/raw/manifest.jsonl`` recording the URL, local file, size and UTC pull time.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from dengue.utils.log import get_logger

log = get_logger(__name__)


def record_pull(raw_dir: Path, url: str, dest: Path, note: str = "") -> None:
    """Append a manifest line for a completed download."""
    entry = {
        "url": url,
        "file": str(dest.relative_to(raw_dir)).replace("\\", "/"),
        "bytes": dest.stat().st_size,
        "pulled_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": note,
    }
    with open(raw_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def download(
    url: str,
    dest: Path,
    raw_dir: Path,
    timeout: float = 120,
    retries: int = 3,
    force: bool = False,
    note: str = "",
) -> Path:
    """Download ``url`` to ``dest`` unless it is already cached.

    Retries with exponential backoff. Writes to a temp file first so a failed
    download never leaves a truncated cache entry.
    """
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            record_pull(raw_dir, url, dest, note)
            return dest
        except requests.RequestException as e:
            wait = 2**attempt * 5
            log.warning("download failed (%s), attempt %d, retry in %ds", e, attempt + 1, wait)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}")
