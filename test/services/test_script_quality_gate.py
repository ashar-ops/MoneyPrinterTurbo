import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import llm


def _rating(average: float) -> dict:
    return {
        "hook_strength": average,
        "engagement": average,
        "conversational_tone": average,
        "viral_potential": average,
        "average": average,
        "feedback": "add a stronger hook",
    }


class TestAnalyzeScript(unittest.TestCase):
    def test_parses_fenced_json_response(self):
        raw = '```json\n{"hook_strength": 9, "engagement": 8.5, "conversational_tone": 9, "viral_potential": 8.5, "average": 8.75, "feedback": "tighten CTA"}\n```'

        with patch.object(llm, "_generate_response", return_value=raw):
            review = llm.analyze_script("script text")

        self.assertEqual(review["average"], 8.75)
        self.assertEqual(review["feedback"], "tighten CTA")
        self.assertEqual(review["hook_strength"], 9.0)

    def test_error_response_returns_neutral_rating(self):
        with patch.object(llm, "_generate_response", return_value="Error: quota exhausted"):
            review = llm.analyze_script("script text")

        self.assertEqual(review["average"], 0.0)
        self.assertEqual(review, dict(llm.UNRATED_SCRIPT_REVIEW))

    def test_garbage_response_returns_neutral_rating(self):
        with patch.object(llm, "_generate_response", return_value="not json at all"):
            review = llm.analyze_script("script text")

        self.assertEqual(review["average"], 0.0)

    def test_scores_are_clamped_to_ten_scale(self):
        raw = '{"hook_strength": 42, "engagement": -3, "conversational_tone": 8, "viral_potential": 8, "average": 99}'

        with patch.object(llm, "_generate_response", return_value=raw):
            review = llm.analyze_script("script text")

        self.assertEqual(review["hook_strength"], 10.0)
        self.assertEqual(review["engagement"], 0.0)
        # 越界的 average 同样被收敛，防止质量门被虚假高分绕过。
        self.assertEqual(review["average"], 10.0)


class TestScriptQualityGate(unittest.TestCase):
    def setUp(self):
        self.config = {
            "script_refinement_enabled": True,
            "script_refinement_min_rating": 8,
            "script_refinement_max_attempts": 3,
        }

    def test_accepts_first_candidate_above_threshold(self):
        with patch.object(llm, "generate_script", return_value="great script") as gen, \
             patch.object(llm, "analyze_script", return_value=_rating(8.7)) as analyzer:
            script = llm.generate_script_with_refinement(
                "subject", 1, "", "", app_config=self.config
            )

        self.assertEqual(script, "great script")
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(analyzer.call_count, 1)

    def test_score_equal_to_threshold_is_rejected(self):
        """
        验收标准是“严格高于阈值”：刚好等于 8 分的脚本必须继续重试，
        不能被直接采纳。
        """
        candidates = iter(["borderline", "better"])
        ratings = iter([_rating(8.0), _rating(9.1)])

        with patch.object(llm, "generate_script", side_effect=lambda *a, **k: next(candidates)), \
             patch.object(llm, "analyze_script", side_effect=lambda *a, **k: next(ratings)):
            script = llm.generate_script_with_refinement(
                "subject", 1, "", "", app_config=self.config
            )

        self.assertEqual(script, "better")

    def test_rejected_candidates_regenerate_with_feedback(self):
        candidates = iter(["weak", "strong"])
        ratings = iter([_rating(5.5), _rating(9.0)])
        captured_prompts = []

        def fake_generate(
            video_subject,
            language="",
            paragraph_number=1,
            video_script_prompt="",
            custom_system_prompt="",
            app_config=None,
        ):
            # 与 llm.generate_script 的真实签名保持一致：重试反馈通过
            # video_script_prompt 参数传入。
            captured_prompts.append(video_script_prompt)
            return next(candidates)

        with patch.object(llm, "generate_script", side_effect=fake_generate), \
             patch.object(llm, "analyze_script", side_effect=lambda *a, **k: next(ratings)):
            script = llm.generate_script_with_refinement(
                "subject", 1, "", "", app_config=self.config
            )

        self.assertEqual(script, "strong")
        # 第二次生成必须携带上一轮的评审反馈。
        self.assertIn("Previous attempt feedback", captured_prompts[1])
        self.assertIn("add a stronger hook", captured_prompts[1])

    def test_best_candidate_returned_when_threshold_never_reached(self):
        candidates = iter(["first", "second", "third"])
        ratings = iter([_rating(6.0), _rating(7.5), _rating(4.0)])

        with patch.object(llm, "generate_script", side_effect=lambda *a, **k: next(candidates)), \
             patch.object(llm, "analyze_script", side_effect=lambda *a, **k: next(ratings)):
            script = llm.generate_script_with_refinement(
                "subject", 1, "", "", app_config=self.config
            )

        self.assertEqual(script, "second")
        self.assertEqual(
            len(list(candidates)), 0, "should stop after configured attempts"
        )

    def test_disabled_gate_generates_single_candidate_without_scoring(self):
        disabled = dict(self.config, script_refinement_enabled=False)

        with patch.object(llm, "generate_script", return_value="plain") as gen, \
             patch.object(llm, "analyze_script") as analyzer:
            script = llm.generate_script_with_refinement(
                "subject", 1, "", "", app_config=disabled
            )

        self.assertEqual(script, "plain")
        self.assertEqual(gen.call_count, 1)
        analyzer.assert_not_called()

    def test_failed_analysis_keeps_pipeline_alive(self):
        """评分请求全部失败时仍应返回已生成的最佳脚本，而不是空文案。"""
        with patch.object(llm, "generate_script", return_value="solid draft"), \
             patch.object(llm, "analyze_script", return_value=dict(llm.UNRATED_SCRIPT_REVIEW)):
            script = llm.generate_script_with_refinement(
                "subject", 1, "", "", app_config=self.config
            )

        self.assertEqual(script, "solid draft")


class TestSampleScriptInjection(unittest.TestCase):
    def test_examples_are_injected_into_generation_prompt(self):
        examples = (
            "=== EXAMPLE 1 ===\nYou've been lied to...\n\n"
            "=== EXAMPLE 2 ===\nEver wonder why...\n"
        )
        captured = {}

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            return "Generated script body."

        with patch.object(llm, "get_script_examples", return_value=examples), \
             patch.object(llm, "_generate_response", side_effect=fake_generate_response):
            llm.generate_script("subject about money")

        prompt = captured["prompt"]
        self.assertIn("SCRIPT EXAMPLES:", prompt)
        self.assertIn("=== EXAMPLE 1 ===", prompt)
        self.assertIn("=== EXAMPLE 2 ===", prompt)
        # 样例必须出现在正式生成指令之前，作为风格参考上下文。
        self.assertLess(prompt.index("SCRIPT EXAMPLES:"), prompt.index("# Role:"))

    def test_missing_examples_do_not_break_generation(self):
        captured = {}

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            return "Generated script body."

        with patch.object(llm, "get_script_examples", return_value=""), \
             patch.object(llm, "_generate_response", side_effect=fake_generate_response):
            result = llm.generate_script("another subject")

        self.assertNotIn("SCRIPT EXAMPLES:", captured["prompt"])
        self.assertEqual(result, "Generated script body.")


if __name__ == "__main__":
    unittest.main()
