import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import material


def _pexels_video(video_id: int, slug: str, link: str, **extra) -> dict:
    payload = {
        "id": video_id,
        "url": f"https://www.pexels.com/video/{slug}-{video_id}/",
        "duration": 8,
        "user": {"id": 1, "name": "Creator One", "url": "https://www.pexels.com/@c1/"},
        "video_files": [
            {"id": 10, "width": 1080, "height": 1920, "link": link}
        ],
    }
    payload.update(extra)
    return payload


class FemaleFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)


class TestSearchTermSanitization(unittest.TestCase):
    def test_gendered_words_are_removed(self):
        cleaned = material.sanitize_search_terms(
            ["woman working laptop", "city girl nightlife", "female athlete running"]
        )

        self.assertEqual(cleaned, ["working laptop", "city nightlife", "athlete running"])

    def test_empty_terms_and_duplicates_are_dropped(self):
        cleaned = material.sanitize_search_terms(
            ["woman", "ocean waves", "", "Ocean Waves"]
        )

        self.assertEqual(cleaned, ["ocean waves"])

    def test_download_videos_sanitizes_terms_before_provider_search(self):
        config.app["pexels_api_keys"] = ["k"]
        captured_terms = []

        def fake_search(search_term, minimum_duration, video_aspect):
            captured_terms.append(search_term)
            return []

        with patch.object(material, "search_videos_pexels", side_effect=fake_search), \
             patch.object(material.vision_filter, "is_enabled", return_value=False):
            material.download_videos(
                "task-x",
                ["woman cooking pasta"],
                audio_duration=1,
            )

        self.assertEqual(captured_terms, ["cooking pasta"])


class TestPexelsMetadataFilter(FemaleFilterTestBase):
    def test_results_with_female_metadata_are_rejected(self):
        config.app["pexels_api_keys"] = ["k"]
        videos = [
            _pexels_video(1, "happy-dog-running", "https://cdn/dog.mp4"),
            _pexels_video(2, "woman-dancing-street", "https://cdn/woman.mp4"),
            {
                **_pexels_video(3, "street-market", "https://cdn/market.mp4"),
                "user": {"id": 2, "name": "girls-just-want-fun"},
            },
        ]
        fake_response = SimpleNamespace(json=lambda: {"videos": videos})

        with patch.object(material, "requests") as requests_mock, \
             patch.object(material, "_get_tls_verify", return_value=True):
            requests_mock.get.return_value = fake_response
            results = material.search_videos_pexels("street", minimum_duration=1)

        # 命中女性关键词的 URL slug 与作者名都必须被拒绝。
        self.assertEqual([item.source_info["asset_id"] for item in results], ["1"])

    def test_thumbnail_url_is_captured_for_vision_filter(self):
        config.app["pexels_api_keys"] = ["k"]
        videos = [
            _pexels_video(
                4,
                "mountain-sunrise",
                "https://cdn/mountain.mp4",
                image="https://images.pexels.com/videos/4/preview.jpg?auto=compress&h=630",
            )
        ]
        fake_response = SimpleNamespace(json=lambda: {"videos": videos})

        with patch.object(material, "requests") as requests_mock:
            requests_mock.get.return_value = fake_response
            results = material.search_videos_pexels("mountain", minimum_duration=1)

        self.assertEqual(
            results[0].source_info["thumbnail_url"],
            "https://images.pexels.com/videos/4/preview.jpg",
        )


class TestPixabayMetadataFilter(FemaleFilterTestBase):
    def test_results_with_female_tags_or_usernames_are_rejected(self):
        config.app["pixabay_api_keys"] = ["k"]
        hits = [
            {
                "id": 11,
                "pageURL": "https://pixabay.com/videos/ocean-waves-11/",
                "duration": 9,
                "tags": "ocean, waves, nature",
                "user": "naturecam",
                "picture_id": 555,
                "videos": {
                    "medium": {"width": 1080, "height": 1920, "url": "https://cdn/pix-ocean.mp4"}
                },
            },
            {
                "id": 12,
                "pageURL": "https://pixabay.com/videos/fashion-girl-12/",
                "duration": 9,
                "tags": "girl, fashion, model",
                "user": "shooter2",
                "picture_id": 556,
                "videos": {
                    "medium": {"width": 1080, "height": 1920, "url": "https://cdn/pix-girl.mp4"}
                },
            },
            {
                "id": 13,
                "pageURL": "https://pixabay.com/videos/city-night-13/",
                "duration": 9,
                "tags": "city, night",
                "user": "lady_films",
                "picture_id": 557,
                "videos": {
                    "medium": {"width": 1080, "height": 1920, "url": "https://cdn/pix-city.mp4"}
                },
            },
        ]
        fake_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            json=lambda: {"hits": hits},
        )

        with patch.object(material, "requests") as requests_mock:
            requests_mock.get.return_value = fake_response
            results = material.search_videos_pixabay("city", minimum_duration=1)

        # tags 含 “girl” 与作者名含 “lady” 的结果都要被拒绝。
        self.assertEqual([item.source_info["asset_id"] for item in results], ["11"])
        self.assertTrue(results[0].source_info["thumbnail_url"].startswith(
            "https://i.vimeocdn.com/video/555"
        ))


class TestCoverrMetadataFilter(FemaleFilterTestBase):
    def test_results_with_female_title_are_rejected_recursively(self):
        config.app["coverr_api_keys"] = ["k"]
        hits = [
            {
                "id": "cov-1",
                "title": "aerial coastline",
                "duration": 12,
                "max_width": 2160,
                "max_height": 3840,
                "urls": {
                    "mp4_download": "https://api.coverr.co/dl/cov-1.mp4?token=jwt",
                    "thumbnail": "https://img.coverr.co/cov-1.jpg",
                },
            },
            {
                "id": "cov-2",
                "title": "woman practicing yoga at sunrise",
                "duration": 12,
                "max_width": 2160,
                "max_height": 3840,
                "urls": {
                    "mp4_download": "https://api.coverr.co/dl/cov-2.mp4?token=jwt"
                },
            },
        ]
        fake_response = SimpleNamespace(json=lambda: {"hits": hits})

        with patch.object(material, "requests") as requests_mock:
            requests_mock.get.return_value = fake_response
            results = material.search_videos_coverr("coastline", minimum_duration=1)

        self.assertEqual([item.source_info["asset_id"] for item in results], ["cov-1"])
        self.assertEqual(
            results[0].source_info["thumbnail_url"],
            "https://img.coverr.co/cov-1.jpg",
        )


class TestWaveSpeedVisionCheck(FemaleFilterTestBase):
    def test_unsafe_generated_clip_is_deleted(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = os.path.join(tmp_dir, "generated.mp4")
            with open(video_path, "wb") as f:
                f.write(b"fake")

            with patch.object(
                material, "_extract_first_frame_png", return_value=b"frame"
            ), patch.object(
                material.vision_filter,
                "screen_image_blobs",
                return_value=(set(), {video_path}),
            ) as screen:
                passed = material._generated_clip_passes_vision_check(
                    video_path, "cooking pasta"
                )

            self.assertFalse(passed)
            self.assertFalse(os.path.exists(video_path))
            screen.assert_called_once()

    def test_frame_extraction_failure_fails_open(self):
        with patch.object(
            material, "_extract_first_frame_png", return_value=None
        ):
            self.assertTrue(
                material._generated_clip_passes_vision_check("whatever.mp4", "term")
            )


if __name__ == "__main__":
    unittest.main()
