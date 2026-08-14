from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LUXIA_DOCUMENT_AI_ENDPOINT = "https://bridge.luxiacloud.com/luxia/v1/document-ai"
LUXIA_CHAT_ENDPOINT = "https://bridge.luxiacloud.com/luxia/v1/chat"
LUXIA_CHAT_MODEL = "luxia3-llm-32b-0731"


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


def _extract_chat_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def _post_json(
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    *,
    timeout_s: int,
) -> tuple[int, dict[str, Any], float]:
    if not api_key:
        raise ValueError("LUXIA_API_KEY is required.")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Luxia returned non-JSON response: {raw[:1000]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Luxia returned unexpected JSON shape: {type(parsed).__name__}")
    return status, parsed, round(time.time() - started, 3)


def call_luxia_document_ai(
    image_path: Path,
    api_key: str,
    *,
    endpoint: str = LUXIA_DOCUMENT_AI_ENDPOINT,
    timeout_s: int = 90,
) -> dict[str, Any]:
    status, payload, latency_s = _post_json(
        endpoint,
        api_key,
        {"image": image_to_data_uri(Path(image_path))},
        timeout_s=timeout_s,
    )
    content = _extract_content(payload)
    if not content:
        raise RuntimeError(f"Luxia Document AI returned empty content: {payload}")
    return {
        "http_status": status,
        "latency_s": latency_s,
        "content": content,
        "response": payload,
    }


def call_luxia_chat(
    prompt: str,
    api_key: str,
    *,
    model: str = LUXIA_CHAT_MODEL,
    endpoint: str = LUXIA_CHAT_ENDPOINT,
    timeout_s: int = 90,
    max_completion_tokens: int = 160,
) -> dict[str, Any]:
    status, payload, latency_s = _post_json(
        endpoint,
        api_key,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
            "top_p": 1,
            "max_completion_tokens": max_completion_tokens,
        },
        timeout_s=timeout_s,
    )
    content = _extract_chat_content(payload)
    if not content:
        raise RuntimeError(f"Luxia Chat returned empty content: {payload}")
    return {
        "http_status": status,
        "latency_s": latency_s,
        "content": content,
        "response": payload,
        "model": model,
    }


def infer_luxia_pipeline(
    image_paths: Sequence[Path],
    prompt: str,
    api_key: str,
    *,
    chat_model: str = LUXIA_CHAT_MODEL,
) -> dict[str, Any]:
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("Luxia pipeline requires at least one image.")

    vision_results = [call_luxia_document_ai(path, api_key) for path in paths]
    descriptions = "\n\n".join(
        f"Image {index + 1} visual description from Luxia Vision:\n{result['content']}"
        for index, result in enumerate(vision_results)
    )
    chat_prompt = (
        "You are evaluating power-line asset images using visual descriptions produced by "
        "Luxia Vision. Follow the benchmark instruction exactly and return only the requested "
        "JSON. Do not invent visual details that are absent from the descriptions.\n\n"
        f"{descriptions}\n\nBenchmark instruction:\n{prompt}"
    )
    chat = call_luxia_chat(chat_prompt, api_key, model=chat_model)
    return {
        "output": chat["content"],
        "latency_s": round(
            sum(float(result["latency_s"]) for result in vision_results)
            + float(chat["latency_s"]),
            3,
        ),
        "vision_results": vision_results,
        "chat_result": chat,
        "chat_model": chat_model,
    }
