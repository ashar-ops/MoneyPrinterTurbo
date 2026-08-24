import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

from app.config import config
from app.models.schema import MaterialInfo
from app.services import vision_filter


def _png_blob(color=(200, 100, 50), size=(64, 96)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class VisionFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["gemini_api_key"] = "vision-key"
        vision_filter._verdict_cache.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        vision_filter._verdict_cache.clear()


class TestVerdictParsing(unittest.TestCase):
    def test_parse_standard_response(self):
        text = "ASSET_1: SAFE\nASSET_2: WOMAN_PRESENT\nASSET_3: SAFE"
        verdicts = vision_filter.parse_safety_verdicts(text, expected_count=3)

        self.assertEqual(verdicts, {1: True, 2: False, 3: True})

    def test_parse_tolerates_loose_formatting(self):
        text = (
            "Here is the analysis:\n"
            "- ASSET_2 - woman_present\n"
            "- asset 1 : safe\n"
            "ASSET_3: WOMAN PRESENT"
        )
        verdicts = vision_filter.parse_safety_verdicts(text, expected_count=3)

        self.assertEqual(verdicts, {1: True, 2: False, 3: False})

    def test_parse_ignores_out_of_range_indices_and_garbage(self):
        verdicts = vision_filter.parse_safety_verdicts(
            "ASSET_9: SAFE\nno verdicts here at all", expected_count=2
        )

        self.assertEqual(verdicts, {})


class TestCompositeGrid(VisionFilterTestBase):
    def test_grid_layout_matches_asset_count(self):
        blobs = [_png_blob() for _ in range(7)]

        grid = vision_filter.build_composite_grid(
            [(f"ASSET_{i + 1}", blob) for i, blob in enumerate(blobs)]
        )
        image = Image.open(io.BytesIO(grid))

        columns = min(vision_filter._GRID_COLUMNS, len(blobs))
        rows = -(-len(blobs) // columns)
        self.assertEqual(image.size, (columns * vision_filter._CELL_SIZE,
                                      rows * (vision_filter._CELL_SIZE + vision_filter._LABEL_BAR_HEIGHT)))
        # 输出必须是合法 PNG，供 genai inline part 直接使用。
        self.assertEqual(image.format, "PNG")


class TestScreenImageBlobs(VisionFilterTestBase):
    def test_unsafe_assets_are_rejected_and_cached(self):
        safe_blob, unsafe_blob = _png_blob(), _png_blob((10, 20, 30))
        responses = iter(["ASSET_1: SAFE\nASSET_2: WOMAN_PRESENT"])

        with patch.object(
            vision_filter, "_call_vision_model",
            side_effect=lambda grid, cfg: next(responses),
        ) as model_call:
            safe, unsafe = vision_filter.screen_image_blobs(
                {"a": safe_blob, "b": unsafe_blob}, context="city"
            )

        self.assertEqual(safe, {"a"})
        self.assertEqual(unsafe, {"b"})
        self.assertEqual(model_call.call_count, 1)

        # 第二次筛查相同图片时命中缓存，不再消耗视觉请求。
        with patch.object(vision_filter, "_call_vision_model") as cached_call:
            safe_again, unsafe_again = vision_filter.screen_image_blobs(
                {"a": safe_blob, "b": unsafe_blob}, context="city"
            )

        cached_call.assert_not_called()
        self.assertEqual(safe_again, {"a"})
        self.assertEqual(unsafe_again, {"b"})

    def test_model_failure_fails_open(self):
        with patch.object(
            vision_filter, "_call_vision_model",
            side_effect=RuntimeError("model offline"),
        ):
            safe, unsafe = vision_filter.screen_image_blobs(
                {"a": _png_blob(), "b": _png_blob((1, 2, 3))}, context="ocean"
            )

        # fail-open：请求失败时整批放行，绝不阻断素材下载主流程。
        self.assertEqual(safe, {"a", "b"})
        self.assertEqual(unsafe, set())

    def test_missing_verdict_keeps_asset(self):
        with patch.object(
            vision_filter, "_call_vision_model", return_value="ASSET_1: SAFE"
        ):
            safe, unsafe = vision_filter.screen_image_blobs(
                {"a": _png_blob(), "b": _png_blob((5, 5, 5))}, context="forest"
            )

        self.assertIn("b", safe)
        self.assertEqual(unsafe, set())


class TestFilterMaterialItems(VisionFilterTestBase):
    @staticmethod
    def _item(url: str, thumbnail_url: str) -> MaterialInfo:
        return MaterialInfo(
            provider="pexels",
            url=url,
            duration=10,
            source_info={"thumbnail_url": thumbnail_url},
        )

    def test_disabled_filter_returns_items_unchanged(self):
        config.app.pop("gemini_api_key", None)
        items = [self._item("https://cdn/a.mp4", "https://img/a.jpg")]

        with patch.object(vision_filter, "_call_vision_model") as model_call:
            kept = vision_filter.filter_material_items(items, context="river")

        model_call.assert_not_called()
        self.assertEqual(kept, items)

    def test_rejected_items_are_removed_from_candidates(self):
        items = [
            self._item("https://cdn/safe.mp4", "https://img/safe.jpg"),
            self._item("https://cdn/bad.mp4", "https://img/bad.jpg"),
        ]
        # _download_thumbnails 以 item.url 为键返回下载结果。
        blobs = {
            "https://cdn/safe.mp4": _png_blob(),
            "https://cdn/bad.mp4": _png_blob((9, 9, 9)),
        }

        with patch.object(vision_filter, "_download_thumbnails", return_value=blobs), \
             patch.object(
                 vision_filter, "_call_vision_model",
                 return_value="ASSET_1: SAFE\nASSET_2: WOMAN_PRESENT",
             ):
            kept = vision_filter.filter_material_items(items, context="market")

        self.assertEqual([item.url for item in kept], ["https://cdn/safe.mp4"])

    def test_items_without_thumbnail_bypass_screening(self):
        no_thumb = MaterialInfo(provider="pexels", url="https://cdn/x.mp4", duration=8)

        with patch.object(vision_filter, "_download_thumbnails", return_value={}), \
             patch.object(vision_filter, "_call_vision_model") as model_call:
            kept = vision_filter.filter_material_items([no_thumb], context="desert")

        model_call.assert_not_called()
        self.assertEqual(kept, [no_thumb])


if __name__ == "__main__":
    unittest.main()
