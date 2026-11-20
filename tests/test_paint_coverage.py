"""「家族 × 等级」覆盖台账的回归测试。

台账要回答的唯一问题是「还差哪几个组合」，所以它的正确性全在口径上：

1. **该有几个等级是按条目名算的**。黑潮那一块（``g008``）在 PlotItem 表里占 5 条记录，
   但头两条的名字是「玩法颜料-赛季」，真正属于黑潮的只有 Lv5~Lv7。硬编码等级范围
   迟早会跟官方数据对不上，所以按「条目名末段 == 家族名」过滤。
2. **去重键是 image+row+col**。同一张截图重扫不能把计数翻倍，换一张图拍到同一家族
   同一等级才是真样本。
3. **认出来之后要从 unknown 迁走**，并且已经划掉的格子永远不再回来。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from huahua_assistant import paint_coverage, paint_icon


class ExpectedLevelsTests(unittest.TestCase):
    def test_regular_family_has_all_seven_levels(self) -> None:
        self.assertEqual((1, 2, 3, 4, 5, 6, 7), paint_coverage.expected_levels("鼠人"))

    def test_season_only_family_shrinks_to_its_own_levels(self) -> None:
        """黑潮在数据里挂 5 条记录，但只有 Lv5~Lv7 的名字末尾是「黑潮」。"""
        self.assertEqual((5, 6, 7), paint_coverage.expected_levels("黑潮"))

    def test_unknown_family_falls_back_to_all_levels(self) -> None:
        """查不到的家族按 7 级算，避免因为拼错名字让缺口凭空消失。"""
        self.assertEqual(paint_coverage.LEVELS, paint_coverage.expected_levels("不存在的家族"))

    def test_expected_total_matches_official_record_count(self) -> None:
        # 13×7 + 6×7 + 7×(2 赛季 + 3 专属) = 168 条记录，去掉赛季那两条后是 156。
        self.assertEqual(156, paint_coverage.expected_total())


class RecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = paint_coverage.load(Path(tempfile.gettempdir()) / "no-such-ledger.json")

    def test_duplicate_scan_does_not_double_count(self) -> None:
        sample = ("a.png", 2, 3, "黑帮", 3)

        self.assertEqual(1, paint_coverage.record(self.store, [sample]))
        self.assertEqual(0, paint_coverage.record(self.store, [sample]))
        self.assertEqual(1, paint_coverage.matrix(self.store)["黑帮"][3])

    def test_same_cell_from_another_screenshot_counts_again(self) -> None:
        paint_coverage.record(self.store, [("a.png", 2, 3, "黑帮", 3)])
        paint_coverage.record(self.store, [("b.png", 2, 3, "黑帮", 3)])

        self.assertEqual(2, paint_coverage.matrix(self.store)["黑帮"][3])

    def test_recognized_cell_migrates_out_of_unknown(self) -> None:
        paint_coverage.record(self.store, [("a.png", 2, 3, None, None)])
        self.assertEqual(1, len(paint_coverage.unknown_cells(self.store)))

        paint_coverage.record(self.store, [("a.png", 2, 3, "邪影", 4)])

        self.assertEqual([], paint_coverage.unknown_cells(self.store))
        self.assertEqual(1, paint_coverage.matrix(self.store)["邪影"][4])

    def test_ignored_cell_never_enters_the_ledger(self) -> None:
        paint_coverage.ignore(self.store, "a.png", 2, 3, "阻挡地块")

        self.assertEqual(0, paint_coverage.record(self.store, [("a.png", 2, 3, "黑帮", 3)]))
        self.assertEqual({}, paint_coverage.matrix(self.store))

    def test_ignore_reports_whether_it_changed_anything(self) -> None:
        self.assertTrue(paint_coverage.ignore(self.store, "a.png", 2, 3, "阻挡地块"))
        self.assertFalse(
            paint_coverage.ignore(self.store, "a.png", 2, 3, "重复划"),
            "同一格重复划出应返回 False",
        )

    def test_unknown_level_goes_to_the_question_bucket(self) -> None:
        paint_coverage.record(self.store, [("a.png", 0, 0, None, 5)])

        self.assertIn("5", self.store["unknown"])
        self.assertEqual(1, len(paint_coverage.unknown_cells(self.store)))


class CorrectionTests(unittest.TestCase):
    """人工复核改判：台账只认「家族→等级→格子」，改判必须先把格子摘走。"""

    def setUp(self) -> None:
        self.store = paint_coverage.load(Path(tempfile.gettempdir()) / "no-such-ledger.json")

    def test_forget_reports_where_the_cell_used_to_live(self) -> None:
        paint_coverage.record(self.store, [("a.png", 2, 3, "宝箱", 5)])

        removed = paint_coverage.forget(self.store, "a.png", 2, 3)

        self.assertEqual([("宝箱", 5)], removed)
        self.assertEqual({}, paint_coverage.matrix(self.store))

    def test_forget_of_an_absent_cell_returns_nothing(self) -> None:
        self.assertEqual([], paint_coverage.forget(self.store, "a.png", 2, 3))

    def test_relabel_moves_the_cell_instead_of_copying_it(self) -> None:
        """实拍场景：宝箱 Lv5 那两格其实是征战之神。"""
        paint_coverage.record(self.store, [("a.png", 2, 5, "宝箱", 5)])

        moved = paint_coverage.relabel(self.store, "a.png", 2, 5, "征战之神", 5)

        self.assertEqual([("宝箱", 5)], moved)
        matrix = paint_coverage.matrix(self.store)
        self.assertEqual(1, matrix["征战之神"][5])
        self.assertNotIn(5, matrix.get("宝箱", {}), "改判之后旧家族不该还留着这一格")

    def test_relabel_does_not_double_count_a_repeat(self) -> None:
        paint_coverage.record(self.store, [("a.png", 2, 5, "宝箱", 5)])
        paint_coverage.relabel(self.store, "a.png", 2, 5, "征战之神", 5)
        paint_coverage.relabel(self.store, "a.png", 2, 5, "征战之神", 5)

        self.assertEqual(1, paint_coverage.matrix(self.store)["征战之神"][5])

    def test_unignore_lets_the_cell_come_back(self) -> None:
        paint_coverage.ignore(self.store, "a.png", 2, 3, "看走眼了")

        self.assertTrue(paint_coverage.unignore(self.store, "a.png", 2, 3))
        self.assertEqual(1, paint_coverage.record(self.store, [("a.png", 2, 3, "黑帮", 3)]))

    def test_unignore_reports_whether_it_changed_anything(self) -> None:
        self.assertFalse(
            paint_coverage.unignore(self.store, "a.png", 2, 3),
            "不在 ignore 名单里应返回 False",
        )


class CoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = paint_coverage.load(Path(tempfile.gettempdir()) / "no-such-ledger.json")

    def test_coverage_counts_distinct_combinations(self) -> None:
        paint_coverage.record(
            self.store,
            [
                ("a.png", 0, 0, "黑帮", 3),
                ("b.png", 0, 0, "黑帮", 3),
                ("a.png", 0, 1, "邪影", 4),
            ],
        )

        have, total, _missing = paint_coverage.coverage(self.store)

        self.assertEqual(2, have, "两个样本落在同一组合里，只算一个组合")
        self.assertEqual(paint_coverage.expected_total(), total)

    def test_matrix_is_empty_for_a_fresh_ledger(self) -> None:
        self.assertEqual({}, paint_coverage.matrix(self.store))

    def test_matrix_levels_stay_within_the_family_level_range(self) -> None:
        """没有样本的等级不出现；出现的等级必须是这个家族理论上会有的等级。"""
        paint_coverage.record(self.store, [("a.png", 0, 0, "黑潮", 5)])

        grid = paint_coverage.matrix(self.store)

        self.assertEqual({"黑潮": {5: 1}}, grid)
        self.assertLessEqual(set(grid["黑潮"]), set(paint_coverage.expected_levels("黑潮")))

    def test_report_lists_missing_combinations(self) -> None:
        paint_coverage.record(self.store, [("a.png", 0, 0, "黑帮", 3)])
        text = paint_coverage.report_text(self.store)

        self.assertIn("覆盖 1/156 组合", text)
        self.assertIn("黑帮", text)
        # 黑帮只差 Lv3 以外的几级，报告里应当逐级列出来。
        self.assertIn("Lv1", text)


class PersistenceTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            store = paint_coverage.load(ledger)
            paint_coverage.record(store, [("a.png", 2, 3, "黑帮", 3)])
            paint_coverage.ignore(store, "b.png", 1, 1, "空白")
            paint_coverage.save(store, ledger)

            reloaded = paint_coverage.load(ledger)

        self.assertEqual([["a.png", 2, 3]], reloaded["samples"]["黑帮"]["3"])
        self.assertEqual(1, len(reloaded["ignored"]))

    def test_save_drops_emptied_unknown_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            store = paint_coverage.load(ledger)
            paint_coverage.record(store, [("a.png", 2, 3, None, 4)])
            paint_coverage.record(store, [("a.png", 2, 3, "邪影", 4)])
            paint_coverage.save(store, ledger)

            payload = json.loads(ledger.read_text(encoding="utf-8"))

        self.assertEqual({}, payload["unknown"], "迁空之后不该留下空桶")

    def test_save_drops_emptied_sample_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            store = paint_coverage.load(ledger)
            paint_coverage.record(store, [("a.png", 2, 5, "宝箱", 5)])
            paint_coverage.relabel(store, "a.png", 2, 5, "征战之神", 5)
            paint_coverage.save(store, ledger)

            payload = json.loads(ledger.read_text(encoding="utf-8"))

        self.assertNotIn("宝箱", payload["samples"], "改判掏空的家族不该留下空目录")
        self.assertIn("5", payload["samples"]["征战之神"])

    def test_broken_ledger_reads_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            ledger.write_text("{not json", encoding="utf-8")

            store = paint_coverage.load(ledger)

        self.assertEqual({}, store["samples"])
        self.assertEqual([], store["ignored"])

    def test_missing_ledger_reads_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = paint_coverage.load(Path(tmp) / "absent.json")

        self.assertEqual({"version": 1, "ignored": [], "samples": {}, "unknown": {}}, store)


class SafeStemTests(unittest.TestCase):
    def test_chinese_and_dashes_survive(self) -> None:
        self.assertEqual("结束--地块全部无空白", paint_coverage._safe_stem("结束--地块全部无空白.png"))

    def test_windows_reserved_characters_are_replaced(self) -> None:
        self.assertEqual("a_b_c", paint_coverage._safe_stem('a<b>c.png'))

    def test_plain_name_is_unchanged(self) -> None:
        self.assertEqual("其他", paint_coverage._safe_stem("其他.png"))


class ExportSamplesTests(unittest.TestCase):
    """导出：每个样本格落成 ``家族/LvN__来源_rXcY.png``，并写清单与总览。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "samples"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _store(self) -> dict:
        store = paint_coverage.load(Path(self.tmp.name) / "ledger.json")
        # 用 jietu/ 里真实存在的截图名：export 会先确认文件在，假名字会整张被跳过。
        paint_coverage.record(
            store,
            [
                ("其他.png", 1, 2, "黑帮", 3),
                ("其他.png", 2, 5, "黑帮", 6),
                ("其他.png", 0, 0, "邪影", 4),
            ],
        )
        return store

    def test_files_are_named_by_family_level_and_source_cell(self) -> None:
        crops = {(row, col): np.zeros((60, 60, 3), dtype=np.uint8) for row, col in ((1, 2), (2, 5), (0, 0))}
        with (
            patch.object(paint_coverage, "_tile_crops", return_value=crops),
            patch.object(paint_coverage, "draw_sample_index", return_value=self.out / "index.png"),
        ):
            payload = paint_coverage.export_samples(self._store(), self.out, verbose=False)

        self.assertEqual(3, payload["written"])
        self.assertEqual(3, payload["combinations"], "黑帮 Lv3/Lv6 + 邪影 Lv4 共三个组合")
        self.assertTrue((self.out / "黑帮" / "Lv3__其他_r1c2.png").is_file())
        self.assertTrue((self.out / "黑帮" / "Lv6__其他_r2c5.png").is_file())
        self.assertTrue((self.out / "邪影" / "Lv4__其他_r0c0.png").is_file())

    def test_index_json_maps_family_level_to_files(self) -> None:
        crops = {(row, col): np.zeros((60, 60, 3), dtype=np.uint8) for row, col in ((1, 2), (2, 5), (0, 0))}
        with (
            patch.object(paint_coverage, "_tile_crops", return_value=crops),
            patch.object(paint_coverage, "draw_sample_index", return_value=self.out / "index.png"),
        ):
            paint_coverage.export_samples(self._store(), self.out, verbose=False)

        payload = json.loads((self.out / "index.json").read_text(encoding="utf-8"))

        self.assertEqual(["黑帮/Lv3__其他_r1c2.png"], payload["index"]["黑帮"]["3"])
        self.assertEqual(["黑帮/Lv6__其他_r2c5.png"], payload["index"]["黑帮"]["6"])

    def test_missing_tile_is_counted_as_skipped(self) -> None:
        with (
            patch.object(paint_coverage, "_tile_crops", return_value={(1, 2): np.zeros((60, 60, 3), np.uint8)}),
            patch.object(paint_coverage, "draw_sample_index", return_value=self.out / "index.png"),
        ):
            payload = paint_coverage.export_samples(self._store(), self.out, verbose=False)

        self.assertEqual(1, payload["written"])
        self.assertEqual(2, payload["skipped"])

    def test_unreadable_screenshot_is_skipped_whole(self) -> None:
        with (
            patch.object(paint_coverage, "_tile_crops", return_value={}),
            patch.object(paint_coverage, "draw_sample_index", return_value=self.out / "index.png"),
        ):
            payload = paint_coverage.export_samples(self._store(), self.out, verbose=False)

        self.assertEqual(0, payload["written"])
        self.assertEqual(3, payload["skipped"])


class VerifySamplesTests(unittest.TestCase):
    """复核以官方图标为准：冲突必须报出来，认不出只算没证据。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "samples"
        for family, filename in (("黑帮", "Lv3__a_r1c2.png"), ("邪影", "Lv4__a_r0c0.png")):
            folder = self.out / family
            folder.mkdir(parents=True, exist_ok=True)
            (folder / filename).write_bytes(self._png_bytes())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _png_bytes() -> bytes:
        ok, buffer = cv2.imencode(".png", np.zeros((40, 40, 3), dtype=np.uint8))
        if not ok:
            raise AssertionError("测试用的纯色 PNG 编码失败")
        return buffer.tobytes()

    @staticmethod
    def _match(family: str, good: int = 60) -> paint_icon.IconMatch:
        return paint_icon.IconMatch(family=family, share=0.2, good=good, runner_up=None, runner_good=0)

    def test_all_matching_samples_report_no_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "黑帮"
            folder.mkdir()
            (folder / "Lv3__a_r1c2.png").write_bytes(self._png_bytes())

            with patch.object(paint_coverage.paint_icon, "identify", return_value=self._match("黑帮")):
                payload = paint_coverage.verify_samples(Path(tmp), verbose=False)

        self.assertEqual(1, payload["consistent"])
        self.assertEqual([], payload["conflicts"])

    def test_conflicting_family_is_reported(self) -> None:
        with patch.object(
            paint_coverage.paint_icon, "identify", return_value=self._match("黑帮")
        ):
            payload = paint_coverage.verify_samples(self.out, verbose=False)

        # 目录里是 黑帮 / 邪影 各一张，图标都说「黑帮」，所以一真一假。
        self.assertEqual(1, payload["consistent"])
        self.assertEqual(1, len(payload["conflicts"]))
        self.assertEqual("邪影", payload["conflicts"][0]["expected"])
        self.assertEqual("黑帮", payload["conflicts"][0]["actual"])

    def test_unrecognized_sample_is_not_a_conflict(self) -> None:
        with patch.object(paint_coverage.paint_icon, "identify", return_value=None):
            payload = paint_coverage.verify_samples(self.out, verbose=False)

        self.assertEqual(0, len(payload["conflicts"]))
        self.assertEqual(2, payload["unclear"])


if __name__ == "__main__":
    unittest.main()
