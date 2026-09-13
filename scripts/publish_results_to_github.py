#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FILES = (
    "build_summary.json",
    "master_counts.csv",
    "manual_review_final.csv",
    "leaderboard.csv",
    "metrics_by_asset.csv",
    "screening_summary.csv",
    "baseline_vs_reference.csv",
    "asset_baseline_vs_reference.csv",
    "run_status.json",
    "run_metadata.json",
)


def _request_json(url: str, token: str, *, method: str = "GET", payload: dict | None = None) -> dict | None:
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "lvlm-insplad-result-publisher",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {exc.code} for {url}: {detail[:1000]}") from exc
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _existing_sha(repo: str, branch: str, remote_path: str, token: str) -> str | None:
    encoded = urllib.parse.quote(remote_path, safe="/")
    url = f"https://api.github.com/repos/{repo}/contents/{encoded}?ref={urllib.parse.quote(branch)}"
    try:
        value = _request_json(url, token)
    except RuntimeError as exc:
        if "GitHub API 404" in str(exc):
            return None
        raise
    if isinstance(value, dict):
        sha = value.get("sha")
        return str(sha) if sha else None
    return None


def _put_file(repo: str, branch: str, remote_path: str, source: Path, token: str, message: str) -> dict:
    encoded = urllib.parse.quote(remote_path, safe="/")
    url = f"https://api.github.com/repos/{repo}/contents/{encoded}"
    payload = {
        "message": message,
        "branch": branch,
        "content": base64.b64encode(source.read_bytes()).decode("ascii"),
    }
    sha = _existing_sha(repo, branch, remote_path, token)
    if sha:
        payload["sha"] = sha
    result = _request_json(url, token, method="PUT", payload=payload)
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected GitHub response for {remote_path}")
    return result


def publish(
    out_dir: Path,
    *,
    repo: str,
    branch: str,
    prefix: str,
    token: str,
    filenames: tuple[str, ...] = DEFAULT_FILES,
    target_root: str = "results/latest",
) -> dict:
    out_dir = out_dir.resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(out_dir)

    uploaded: list[dict] = []
    missing: list[str] = []
    base = f"{target_root.rstrip('/')}/{prefix.strip('/')}"
    now = datetime.now(timezone.utc).isoformat()

    for name in filenames:
        source = out_dir / name
        if not source.is_file():
            missing.append(name)
            continue
        remote_path = f"{base}/{name}"
        result = _put_file(
            repo,
            branch,
            remote_path,
            source,
            token,
            message=f"Publish {prefix}: {name}",
        )
        content = result.get("content") or {}
        commit = result.get("commit") or {}
        uploaded.append(
            {
                "name": name,
                "remote_path": remote_path,
                "html_url": content.get("html_url"),
                "commit_sha": commit.get("sha"),
                "size_bytes": source.stat().st_size,
                "mime_type": mimetypes.guess_type(source.name)[0],
            }
        )

    manifest = {
        "schema_version": 1,
        "published_at_utc": now,
        "source_dir": str(out_dir),
        "repo": repo,
        "branch": branch,
        "prefix": prefix,
        "target_root": target_root,
        "uploaded": uploaded,
        "missing": missing,
    }
    manifest_local = out_dir / "github_publish_manifest.json"
    manifest_local.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_remote = f"{base}/github_publish_manifest.json"
    manifest_result = _put_file(
        repo,
        branch,
        manifest_remote,
        manifest_local,
        token,
        message=f"Publish {prefix}: manifest",
    )
    manifest_commit = (manifest_result.get("commit") or {}).get("sha") if isinstance(manifest_result, dict) else None
    return {**manifest, "manifest_remote_path": manifest_remote, "manifest_commit_sha": manifest_commit}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo", default="arnold6444/lvlm_insplad")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--target-root", default="results/latest")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise RuntimeError(f"environment variable {args.token_env} is missing")
    filenames = tuple(args.files) if args.files else DEFAULT_FILES
    result = publish(
        args.out_dir,
        repo=args.repo,
        branch=args.branch,
        prefix=args.prefix,
        token=token,
        filenames=filenames,
        target_root=args.target_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
