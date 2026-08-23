#!/usr/bin/env python3
"""Upload one local file to Google Drive, then remove the local copy.

Requirements:
    pip install --break-system-packages google-api-python-client google-auth

The service account key is expected at ~/MoneyPrinterTurbo/gdrive-key.json.
The target Drive folder must be shared with the service account.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path


SERVICE_ACCOUNT_KEY = Path("~/MoneyPrinterTurbo/gdrive-key.json").expanduser()
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local file to a Google Drive folder."
    )
    parser.add_argument("local_file", type=Path, help="Path to the local file")
    parser.add_argument("folder_id", help="Google Drive folder ID")
    return parser.parse_args()


def upload_file(local_file: Path, folder_id: str) -> str:
    """Upload *local_file* and return its Drive file ID on success."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    if not local_file.is_file():
        raise FileNotFoundError(f"local file does not exist: {local_file}")
    if not folder_id.strip():
        raise ValueError("Google Drive folder ID must not be empty")
    if not SERVICE_ACCOUNT_KEY.is_file():
        raise FileNotFoundError(
            f"service account key does not exist: {SERVICE_ACCOUNT_KEY}"
        )

    credentials = Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_KEY), scopes=DRIVE_SCOPES
    )
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
