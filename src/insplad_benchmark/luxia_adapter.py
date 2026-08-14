from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LUXIA_DOCUMENT_AI_ENDPOINT = "https://bridge.luxiacloud.com/luxia/v1/document-ai"


def image_to_data_uri(image_path: Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported Luxia image format: {path.suffix}")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _extract_content(payload: Any) -> str:
    if isinstance(payload, dict):
        if isinstance(payload.get("content"), str):
            return payload["content"].strip()
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            return data["content"].strip()
    return ""


def call_luxia_document_ai(
    image_path: Path,
    api_key: str,
    *,
    endpoint: str = LUXIA_DOCUMENT_AI_ENDPOINT,
    timeout_s: int = 90,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("LUXIA_API_KEY is required.")

    body = json.dumps({"image": image_to_data_uri(Path(image_path))}).encode("utf-8")
    req = Request(
        endpoint,
        data=body,
        headers={"apikey": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urlopen(req, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Luxia HTTP {exc.code}: {raw[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Luxia request failed: {exc.reason}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Luxia returned non-JSON response: {raw[:1000]}") from exc

    return {
        "http_status": status,
        "latency_s": round(time.time() - started, 3),
        "content": _extract_content(payload),
        "response": payload,
    }
