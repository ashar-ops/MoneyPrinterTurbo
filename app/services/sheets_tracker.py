"""Optional Google Sheets tracking for completed and failed video jobs."""
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.config import config


def _sheet():
    settings = config.sheets
    if not settings.get("enabled") or not settings.get("tracking_id"):
        return None
    import gspread

    key_file = Path(str(settings.get("service_account_file", "sheets-key.json")))
    if not key_file.is_absolute():
        key_file = Path(config.root_dir) / key_file
    client = gspread.service_account(filename=str(key_file))
    return client.open_by_key(str(settings["tracking_id"])).sheet1


def _append(row):
    try:
        sheet = _sheet()
        if sheet is not None:
            sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as exc:
        # Tracking must never turn a successfully rendered video into a failed job.
        logger.warning(f"Google Sheets tracking skipped: {type(exc).__name__}: {exc}")


def track_video_success(topic, script, duration, drive_link):
    _append([
        datetime.now(timezone.utc).isoformat(), topic, script, duration, drive_link, "Success"
    ])


def track_video_failure(topic, error_message):
    _append([
        datetime.now(timezone.utc).isoformat(), topic, "", "", "", f"Failure: {error_message}"
    ])
