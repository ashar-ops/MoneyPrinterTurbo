import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import sheets_tracker


class _FakeWorksheet:
    """模拟 gspread Worksheet：按绝对坐标写入，并记录每次 update 调用。"""

    def __init__(self, rows=None):
        self.rows = [list(row) for row in (rows or [])]
        self.updates = []

    def get_all_values(self):
        return [list(row) for row in self.rows]

    def update(self, values, range_name=None, **kwargs):
        if not range_name or not range_name.startswith("A"):
            raise AssertionError(f"writes must target column A, got: {range_name}")
        row_index = int(range_name[1:])
        while len(self.rows) < row_index:
            self.rows.append([])
        self.rows[row_index - 1] = list(values[0])
        self.updates.append({"row": row_index, "values": list(values[0])})


class SheetsTrackerTestBase(unittest.TestCase):
    def setUp(self):
        self.original_sheets = dict(config.sheets)
        config.sheets["enabled"] = True
        config.sheets["tracking_id"] = "sheet-id"
        config.sheets["service_account_file"] = "sheets-key.json"
        sheets_tracker.reset_sheet_handle()

    def tearDown(self):
        config.sheets.clear()
        config.sheets.update(self.original_sheets)
        sheets_tracker.reset_sheet_handle()


class TestSheetsTrackerPlacement(SheetsTrackerTestBase):
    def test_entry_written_at_column_a_of_next_empty_row(self):
        """
        回归测试：历史条目存在尾部空单元格时，append_row 的自动表格探测会让
        新条目从上一条最后一列之后开始。修复后必须显式定位到第一个空行的 A 列。
        """
        worksheet = _FakeWorksheet(
            rows=[
                ["2026-08-20T00:00:00+00:00", "topic-1", "script", 61, "link", "Success"],
                ["2026-08-21T00:00:00+00:00", "topic-2", "", "", "", "Failure: boom"],
            ]
        )

        with patch.object(sheets_tracker, "_sheet", return_value=worksheet):
            sheets_tracker.track_video_success("topic-3", "script-3", 45, "link-3")

        self.assertEqual(len(worksheet.updates), 1)
        update = worksheet.updates[0]
        self.assertEqual(update["row"], 3)
        # 第一列必须是时间戳：新条目必须从 A 列开始。
        self.assertEqual(update["values"][0], worksheet.rows[2][0])
        self.assertEqual(len(update["values"]), 6)

    def test_first_entry_lands_on_row_one(self):
        worksheet = _FakeWorksheet(rows=[])
        with patch.object(sheets_tracker, "_sheet", return_value=worksheet):
            sheets_tracker.track_video_failure("topic-x", "error-x")

        self.assertEqual(worksheet.updates[0]["row"], 1)
        self.assertEqual(worksheet.updates[0]["values"][-1], "Failure: error-x")
        # 失败记录的空列保持为空字符串，不产生 None 单元格。
        self.assertEqual(worksheet.updates[0]["values"][2], "")

    def test_entry_written_after_trailing_blank_rows(self):
        """测试在有尾部空行（如全空字符串行）时，新条目精准写在最后一个有效行之后。"""
        worksheet = _FakeWorksheet(
            rows=[
                ["2026-08-20T00:00:00+00:00", "topic-1", "script", 61, "link", "Success"],
                ["", "", "", "", "", ""],
                ["", "", "", "", "", ""],
            ]
        )

        with patch.object(sheets_tracker, "_sheet", return_value=worksheet):
            sheets_tracker.track_video_success("topic-2", "script-2", 45, "link-2")

        self.assertEqual(len(worksheet.updates), 1)
        update = worksheet.updates[0]
        self.assertEqual(update["row"], 2)
        self.assertEqual(update["values"][1], "topic-2")

    def test_long_script_is_truncated_to_safe_cell_length(self):
        worksheet = _FakeWorksheet()
        huge_script = "x" * (sheets_tracker._MAX_CELL_LENGTH + 5000)

        with patch.object(sheets_tracker, "_sheet", return_value=worksheet):
            sheets_tracker.track_video_success("topic", huge_script, 30, "")

        stored_script = worksheet.rows[0][2]
        self.assertEqual(len(stored_script), sheets_tracker._MAX_CELL_LENGTH)


class TestSheetsTrackerResilience(SheetsTrackerTestBase):
    def test_disabled_tracking_is_a_noop(self):
        config.sheets["enabled"] = False
        with patch.object(sheets_tracker, "_sheet") as sheet_factory:
            sheets_tracker.track_video_success("topic", "script", 10, "")

        sheet_factory.assert_not_called()

    def test_write_failure_never_propagates(self):
        worksheet = _FakeWorksheet()
        worksheet.update = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("API offline")
        )

        with patch.object(sheets_tracker, "_sheet", return_value=worksheet):
            # 不应抛异常：跟踪失败不能拖垮已经成功的视频任务。
            sheets_tracker.track_video_success("topic", "script", 10, "")

    def test_sheet_handle_is_reused_across_calls(self):
        worksheet = _FakeWorksheet()
        build_calls = []

        def _service_account(**kwargs):
            build_calls.append(kwargs)
            return SimpleNamespace(
                open_by_key=lambda key: SimpleNamespace(sheet1=worksheet)
            )

        with patch.dict("sys.modules", {"gspread": SimpleNamespace(service_account=_service_account)}):
            sheets_tracker.track_video_success("a", "s", 1, "")
            sheets_tracker.track_video_success("b", "s", 1, "")

        # 两次写入只允许构建一次 gspread 客户端（OAuth 初始化开销大）。
        self.assertEqual(len(build_calls), 1)
        self.assertEqual(len(worksheet.updates), 2)


if __name__ == "__main__":
    unittest.main()
