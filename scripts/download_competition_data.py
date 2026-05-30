#!/usr/bin/env python3
"""SIGNATE コンペの全データセットをダウンロードする。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

from signate import config as signate_config
from signate.cli import api_session, getApiToken

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_KEY = "37308d147238487c96551300b8e4cb76"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
META_DIR = PROJECT_ROOT / "meta"
TIMEOUT = (30, 3600)  # connect, read


def api_get(path: str) -> dict:
    api_session.cookies.set(signate_config.JWT_COOKIE_KEY, getApiToken())
    url = signate_config.COMPETITION_URL + path
    resp = api_session.get(
        url,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def download_file(task_key: str, file_key: str, file_name: str, dest: Path) -> None:
    api_session.cookies.set(signate_config.JWT_COOKIE_KEY, getApiToken())
    auth_resp = api_session.post(
        signate_config.COMPETITION_URL + "/dataset/auth",
        headers={"Content-Type": "application/json"},
        json={"public_key": task_key, "file_name": file_name},
        timeout=TIMEOUT,
    )
    auth_resp.raise_for_status()
    download_url = auth_resp.json()["url"]

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with api_session.get(download_url, stream=True, timeout=TIMEOUT) as file_resp:
        file_resp.raise_for_status()
        total = int(file_resp.headers.get("content-length", 0) or 0)
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in file_resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {dest.name}: {pct}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)
        print()

    tmp.replace(dest)


def main() -> None:
    print("コンペ:", COMPETITION_KEY)
    task_data = api_get(f"/task/{COMPETITION_KEY}/list")
    tasks = task_data.get("task_list", [])
    if not tasks:
        print("課題が見つかりません。", file=sys.stderr)
        sys.exit(1)

    manifest: dict = {"competition_key": COMPETITION_KEY, "tasks": []}

    for task in tasks:
        task_key = task["public_key"]
        task_name = task.get("task_name", task_key)
        print(f"\n課題: {task_name} ({task_key})")

        try:
            dataset = api_get(f"/dataset/{task_key}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                print(
                    "  → 404: ブラウザでコンペに「参加」してから再実行してください。",
                    file=sys.stderr,
                )
            raise

        files = dataset.get("public_file_list", [])
        task_entry = {"task_key": task_key, "task_name": task_name, "files": []}

        for item in files:
            file_key = item["public_key"]
            file_name = item["file_name"]
            title = item.get("title", file_name)
            size = item.get("file_size", 0)
            dest = OUTPUT_DIR / file_name

            print(f"  取得: {title} ({file_name}, {size} bytes)")
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  スキップ（既存）: {dest}")
            else:
                download_file(task_key, file_key, file_name, dest)

            task_entry["files"].append(
                {
                    "file_key": file_key,
                    "file_name": file_name,
                    "title": title,
                    "local_path": str(dest),
                }
            )

        manifest["tasks"].append(task_entry)

    META_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = META_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完了。データ: {OUTPUT_DIR}")
    print(f"一覧: {manifest_path}")


if __name__ == "__main__":
    main()
