"""SIGNATE への提出（非対話）。"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from signate import config as signate_config
from signate.cli import api_session, getApiToken

from pipeline.config import CompetitionConfig, load_config


def submit_to_signate(
    path: Path,
    memo: str,
    *,
    config: CompetitionConfig | None = None,
) -> None:
    config = config or load_config()
    path = path.resolve()
    file_name = path.name
    mime_type, _ = mimetypes.guess_type(file_name)
    mime_type = mime_type or "application/octet-stream"

    api_session.cookies.set(signate_config.JWT_COOKIE_KEY, getApiToken())
    auth_resp = api_session.post(
        signate_config.COMPETITION_URL + "/submission/auth",
        headers={"Content-Type": "application/json"},
        json={"public_key": config.task_key, "file_name": file_name},
        timeout=(30, 120),
    )
    auth_resp.raise_for_status()
    auth_data = auth_resp.json()

    with path.open("rb") as f:
        upload_resp = api_session.put(
            auth_data["url"],
            data=f,
            headers={
                "Content-Type": mime_type,
                "Content-Length": str(path.stat().st_size),
            },
            timeout=(30, 120),
        )
    upload_resp.raise_for_status()

    complete_resp = api_session.post(
        signate_config.COMPETITION_URL + "/submission",
        headers={"Content-Type": "application/json"},
        json={
            "public_key": config.task_key,
            "submission_public_key": auth_data["public_key"],
            "file_name": file_name,
            "memo": memo,
        },
        timeout=(30, 120),
    )
    complete_resp.raise_for_status()
    print("Submission completed successfully.")
