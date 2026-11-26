"""Displaying detected brush card regions must not abort the UI refresh."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from huahua_assistant.app import HuahuaApp
from huahua_assistant.models import BoardObservation, Rect, RegionKind, RegionObservation, SceneObservation


class RegionDisplayTests(unittest.TestCase):
    def test_all_detected_regions_render_including_both_brush_cards(self):
        fake = SimpleNamespace(
            region_text=Mock(), manual_correction_applied=False,
            last_capture_quality=None, last_rule_anomalies=[],
        )
        for name in (
            "_write_perf_debug", "_write_coordinate_debug", "_write_learning_stats",
            "_write_current_terrain_bucket_hint", "_write_backpack", "_write_level_ledger",
            "_write_install_plan", "_write_recommendations", "_write_brush_recommendations",
            "_write_special_actions",
        ):
            setattr(fake, name, Mock())
        scene = SceneObservation(
            board=BoardObservation(rect=Rect(332, 237, 575, 576), score=0.9, tiles=()),
            regions=tuple(
                RegionObservation(kind=kind, rect=Rect(980, 400, 520, 100), confidence=0.9, label="test")
                for kind in RegionKind
            ),
        )

        HuahuaApp._write_regions(fake, scene)

        rendered = "".join(call.args[1] for call in fake.region_text.insert.call_args_list)
        self.assertIn("笔触卡片 1", rendered)
        self.assertIn("笔触卡片 2", rendered)
        self.assertEqual(len(RegionKind) + 1, rendered.count("置信="))
        fake._write_special_actions.assert_called_once_with()
        fake.region_text.configure.assert_called_with(state="disabled")


if __name__ == "__main__":
    unittest.main()
