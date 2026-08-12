from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_LUXIA_CHAT_URL = "https://bridge.luxiacloud.com/luxia/v1/chat"


def _mime_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/jpeg"


def to_data_url(path: Path) -> str:
    path = Path(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_type(path)};base64,{encoded}"


def build_multimodal_chat_payload(
    model_id: str,
    image_paths: Sequence[Path],
    prompt: str,
    *,
    max_completion_tokens: int = 80,
) -> dict[str, Any]:
    """Build the OpenAI-compatible multimodal chat body used for Luxia external models.

    Luxia documents the shared chat endpoint and `apikey` authentication. External
    model pricing exposes OpenAI/Gemini model IDs through the same platform. The
    bridge request is intentionally kept provider-neutral and can be pointed at a
    different Luxia external-chat URL through configuration if Saltlux publishes
    a dedicated route later.
    """
    content: list[dict[str, Any]] = []
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": to_data_url(Path(path))},
            }
        )
    content.append({"type": "text", "text": prompt})
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": 0,
        "max_completion_tokens": max_completion_tokens,
    }


def _extract_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    texts: list[str] = []
                    for part in content:
                        if not isinstance(part, Mapping):
                            continue
                        text = part.get("text")
                        if isinstance(text, str):
                            texts.append(text)
                    if texts:
                        return "\n".join(texts).strip()

    for key in ("content", "text", "output_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise RuntimeError(f"Luxia response did not contain assistant text: {payload}")


def infer_luxia_external(
    *,
    api_key: str,
    model_id: str,
    image_paths: Sequence[Path],
    prompt: str,
    endpoint: str = DEFAULT_LUXIA_CHAT_URL,
    timeout_s: float = 120.0,
    max_completion_tokens: int = 80,
) -> str:
    if not api_key:
        raise RuntimeError("LUXIA_API_KEY is not configured.")

    body = build_multimodal_chat_payload(
        model_id,
        image_paths,
        prompt,
        max_completion_tokens=max_completion_tokens,
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "apikey": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Luxia HTTP {exc.code} for model {model_id}: {details[:1000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Luxia request failed for model {model_id}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Luxia returned non-JSON response: {raw[:1000]}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Luxia returned unexpected payload: {payload!r}")
    return _extract_content(payload)
