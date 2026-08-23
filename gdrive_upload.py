#!/usr/bin/env python3
"""Upload one local file to Google Drive, then remove the local copy.

Requirements:
    pip install --break-system-packages google-api-python-client google-auth
    pip install --break-system-packages google-auth-oauthlib

The OAuth client secrets file is expected at ~/MoneyPrinterTurbo/credentials.json.
The target Drive folder must be accessible by the authenticated Google account.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path


CLIENT_SECRETS_FILE = Path("~/MoneyPrinterTurbo/credentials.json").expanduser()
TOKEN_FILE = Path("~/MoneyPrinterTurbo/token.json").expanduser()
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local file to a Google Drive folder."
    )
    parser.add_argument("local_file", type=Path, help="Path to the local file")
    parser.add_argument("folder_id", help="Google Drive folder ID")
    return parser.parse_args()


def get_credentials():
    """Load saved OAuth credentials, refreshing them silently when possible."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    if TOKEN_FILE.is_file():
        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_FILE), DRIVE_SCOPES
        )
        if credentials.expired:
            if not credentials.refresh_token:
                raise RuntimeError(
                    f"saved token is expired and has no refresh token: {TOKEN_FILE}"
                )
            credentials.refresh(Request())
            TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
        return credentials

    if not CLIENT_SECRETS_FILE.is_file():
        raise FileNotFoundError(
            f"OAuth client secrets file does not exist: {CLIENT_SECRETS_FILE}"
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRETS_FILE), DRIVE_SCOPES
    )
    credentials = flow.run_local_server(port=8080)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def upload_file(local_file: Path, folder_id: str) -> str:
    """Upload *local_file* and return its Drive file ID on success."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    if not local_file.is_file():
        raise FileNotFoundError(f"local file does not exist: {local_file}")
    if not folder_id.strip():
        raise ValueError("Google Drive folder ID must not be empty")

    credentials = get_credentials()
    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)

    mime_type = mimetypes.guess_type(local_file.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(local_file), mimetype=mime_type, resumable=True)
    metadata = {"name": local_file.name, "parents": [folder_id.strip()]}

    # A non-2xx Drive response raises HttpError from execute(). Only proceed to
    # local deletion after execute returns a response containing a file ID.
    uploaded = (
        drive.files()
        .create(body=metadata, media_body=media, fields="id")
        .execute()
    )
    file_id = uploaded.get("id") if isinstance(uploaded, dict) else None
    if not file_id:
        raise RuntimeError("Drive API returned no uploaded file ID")
    return str(file_id)


def main() -> int:
    args = parse_args()
    local_file = args.local_file.expanduser()

    try:
        file_id = upload_file(local_file, args.folder_id)
        os.remove(local_file)
    except Exception as exc:
        print(f"Upload failed: {exc}")
        return 1

    print(f"Uploaded file Drive ID: {file_id}")
    print(f"Success: deleted local file {local_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
