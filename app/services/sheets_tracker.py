"""Optional Google Sheets tracking for completed and failed video jobs."""
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from loguru import logger

from app.config import config

# API/WebUI 可以并发生成多条视频任务，成功与失败回调可能同时写表。
# 进程内串行化可以保证“先读表尾、再写下一空行”的两步操作不会互相踩踏。
_append_lock = threading.Lock()

# Google Sheets 单元格上限是 50000 字符。脚本全文会写入一列，这里提前截断，
# 避免超长文案导致整次写入被 API 以 400 拒绝。
_MAX_CELL_LENGTH = 45000

_sheet_handle = None


def _resolve_key_file() -> Path:
    key_file = Path(str(config.sheets.get("service_account_file", "sheets-key.json")))
    if not key_file.is_absolute():
        key_file = Path(config.root_dir) / key_file
    return key_file


def _sheet():
    """
    返回跟踪表的 Worksheet 句柄，未配置时返回 None。

    gspread 客户端每次构建都会读取服务账号 JSON 并请求 OAuth token，开销不小；
    句柄按进程缓存后，同一进程内的多次写入可以复用已授权的 HTTP 连接。
    配置在运行期被关闭时立即返回 None，不保留旧句柄。
    """
    global _sheet_handle

    if _sheet_handle is not None:
        return _sheet_handle

    import gspread

    client = gspread.service_account(filename=str(_resolve_key_file()))
    _sheet_handle = client.open_by_key(
        str(config.sheets["tracking_id"])
    ).sheet1
    return _sheet_handle


def reset_sheet_handle() -> None:
    """丢弃缓存的表句柄；测试和运行期改配置后可以强制重建连接。"""
    global _sheet_handle
    _sheet_handle = None


def _sanitize_cell(value: Any) -> Any:
    """清洗单个单元格：None 转空串、字符串截断到安全长度。"""
    if value is None:
        return ""
    if isinstance(value, str) and len(value) > _MAX_CELL_LENGTH:
        return value[:_MAX_CELL_LENGTH]
    return value


def _next_empty_row(sheet) -> int:
    """
    计算第一个完全空白的行号（从 1 开始）。

    不能依赖 ``append_row`` 的自动表格探测：当历史行存在尾部空单元格（例如
    失败记录的空列）时，API 会把新条目追加到错误的位置，出现“新条目的第一列
    从上一条最后一列之后开始”的漂移。这里显式读回当前表格内容并定位首个空行，
    再用绝对坐标写入 A 列，保证每条记录都从第 1 列开始。
    """
    existing_rows = sheet.get_all_values()
    return len(existing_rows) + 1


def _write_row(sheet, row_index: int, values: List[Any]) -> None:
    # gspread 6.x 默认 raw=True，按原文写入单元格，不会把时长、链接或以 "="
    # 开头的内容重新解释成数字/公式/日期，保证各列内容与位置稳定。
    sheet.update(
        values=[[_sanitize_cell(value) for value in values]],
        range_name=f"A{row_index}",
    )


def _append(row: List[Any]) -> None:
    label = str(row[0] or "")
    try:
        settings = config.sheets
        if not settings.get("enabled") or not settings.get("tracking_id"):
            return

        sheet = _sheet()
        if sheet is None:
            return

        with _append_lock:
            # RAW 模式原样写入文本，避免 USER_ENTERED 把时长、链接或以 "="
            # 开头的内容重新解释成数字/公式/日期，破坏列对齐。
            _write_row(sheet, _next_empty_row(sheet), row)
        logger.success(
            f"tracked entry in google sheets: timestamp={label}, row_topic={row[1]!r}"
        )
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
