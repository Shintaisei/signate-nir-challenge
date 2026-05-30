#!/usr/bin/env python3
"""config.py の認証情報で SIGNATE API トークンを取得する。"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_credentials() -> tuple[str, str]:
    try:
        import config
    except ImportError as exc:
        raise SystemExit(
            "config.py がありません。\n"
            "  cp config.example.py config.py\n"
            "のあと、メールとパスワードを記入してください。"
        ) from exc

    email = getattr(config, "SIGNATE_EMAIL", "").strip()
    password = getattr(config, "SIGNATE_PASSWORD", "")

    if not email or email == "your_email@example.com":
        raise SystemExit("config.py の SIGNATE_EMAIL を設定してください。")
    if not password or password == "your_password_here":
        raise SystemExit("config.py の SIGNATE_PASSWORD を設定してください。")

    return email, password


def main() -> None:
    email, password = _load_credentials()

    from signate import config as signate_config
    from signate.cli import (
        api_session,
        getCookieTarget,
        getToken,
        setApiToken,
        signIn,
        signInOrganization,
        success,
    )

    print(f"SIGNATE にサインイン中: {email}")
    getToken()
    user_csrf = getCookieTarget("_user_csrf_cloud")
    signIn(user_csrf, email, password)
    signInOrganization(user_csrf)
    jwt = getCookieTarget(signate_config.JWT_COOKIE_KEY)
    setApiToken(jwt)
    success("The API Token has been downloaded successfully.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)
