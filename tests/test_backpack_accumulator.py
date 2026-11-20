"""背包颜料累计：两次识别读全 16 格，重复颜料不丢枚数。

实机背包 3 行 × 5 列 = 15 格、容量 16，所以单帧一定有一个被挤到屏外。
落一枚后整列前移，第二帧就能看到它。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from huahua_assistant import game_model as gm
from huahua_assistant.backpack import (
    PAGE_CAPACITY,
    SNAPSHOT_SCHEMA,
    BackpackAccumulator,
    paint_signature,
    snapshot_payload,
)
from huahua_assistant.models import PaintCategory, PaintObject, Rect

GAMEPLAY = PaintCategory.GAMEPLAY
MONSTER = PaintCategory.MONSTER


@dataclass
class FakeObs:
    """冒充视觉层的 BackpackPaintObservation。"""

    category: PaintCategory
    level: int
    subtype: str | None = None
    is_treasure: bool = False
    category_confidence: float = 0.9
    level_confidence: float = 0.9


def _paints(count: int, *, category: PaintCategory = GAMEPLAY) -> list[FakeObs]:
    """造 count 枚互不相同的颜料（靠等级+子类型区分）。"""
    out: list[FakeObs] = []
    for index in range(count):
        level = 1 + index % gm.MAX_LEVEL
        out.append(FakeObs(category=category, level=level, subtype=f"t{index}"))
    return out


class TwoScanCoverageTest(unittest.TestCase):
    def test_full_page_is_flagged_as_possibly_truncated(self) -> None:
        acc = BackpackAccumulator()
        acc.update(_paints(PAGE_CAPACITY), "rule-a")

        self.assertEqual(acc.total, PAGE_CAPACITY)
        self.assertTrue(acc.may_be_truncated, "满屏说明可能还有一格在屏外")

    def test_partial_page_is_not_flagged(self) -> None:
        acc = BackpackAccumulator()
        acc.update(_paints(9), "rule-a")

        self.assertEqual(acc.total, 9)
        self.assertFalse(acc.may_be_truncated)

    def test_second_scan_after_one_placement_reveals_the_hidden_slot(self) -> None:
        # 16 枚，一屏只看得到前 15 枚；落掉第一枚后，第 16 枚进入视野
        sixteen = _paints(16)
        acc = BackpackAccumulator()
        acc.update(sixteen[:PAGE_CAPACITY], "rule-a")
        self.assertEqual(acc.total, PAGE_CAPACITY)

        remaining = sixteen[1:]
        paints = acc.update(remaining, "rule-a", consumed=True)

        self.assertEqual(len(paints), 15)
        signatures = {paint_signature(paint) for paint in paints}
        hidden = gm.Paint(
            category=sixteen[15].category,
            level=sixteen[15].level,
            subtype=sixteen[15].subtype,
        )
        self.assertIn(paint_signature(hidden), signatures, "第 16 枚应该被读到")
        consumed = gm.Paint(
            category=sixteen[0].category,
            level=sixteen[0].level,
            subtype=sixteen[0].subtype,
        )
        self.assertNotIn(paint_signature(consumed), signatures, "已用掉的不该还在")


class MultiplicityTest(unittest.TestCase):
    def test_three_identical_paints_stay_three(self) -> None:
        acc = BackpackAccumulator()
        acc.update([FakeObs(GAMEPLAY, 1, "leaf")] * 3, "rule-a")

        self.assertEqual(acc.total, 3)
        self.assertEqual(acc.distinct, 1)
        self.assertEqual(len(acc.paints()), 3)

    def test_consuming_one_of_three_leaves_two(self) -> None:
        acc = BackpackAccumulator()
        acc.update([FakeObs(GAMEPLAY, 1, "leaf")] * 3, "rule-a")
        acc.update([FakeObs(GAMEPLAY, 1, "leaf")] * 2, "rule-a", consumed=True)

        self.assertEqual(acc.total, 2)

    def test_moving_slots_does_not_duplicate(self) -> None:
        acc = BackpackAccumulator()
        first = _paints(5)
        acc.update(first, "rule-a")
        # 同一批颜料，顺序变了（槽位前移）
        acc.update(list(reversed(first)), "rule-a")

        self.assertEqual(acc.total, 5)
        self.assertEqual(acc.distinct, 5)


#: 实机背包 3 行 × 5 列的格坐标（2026-11-17 那张 1920x1020 实拍）。
GRID = tuple(
    (1032 + 116 * col, 429 + 116 * row, 111, 111) for row in range(3) for col in range(5)
)


@dataclass
class SlotObs(FakeObs):
    """带格子坐标的观测——视觉层真实返回的就是这个形状。"""

    slot_index: int = 0

    @property
    def slot_rect(self) -> Rect:
        return Rect(*GRID[self.slot_index])


def _page(levels: list[int], *, subtype: str | None = None, category: PaintCategory = MONSTER) -> list[SlotObs]:
    return [
        SlotObs(category=category, level=level, subtype=subtype, slot_index=index)
        for index, level in enumerate(levels)
    ]


class SlotIdentityTest(unittest.TestCase):
    """同一个格子读数变了就是「换了一枚」或「认错了」，绝不留两份。

    这一组是 2026-11-17 实机「背包多出两枚」「拖动时冒出一大堆」的回归锁。
    旧实现按 ``(类别, 等级, 子类型, 宝箱)`` 累加，同一个 3 格背包会 3 → 6 → 9。
    """

    def test_background_family_does_not_double_the_page(self) -> None:
        acc = BackpackAccumulator()
        levels = [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 2]
        acc.update(_page(levels))
        self.assertEqual(acc.total, 15)
        # 后台线程把家族认出来了：只是补了名称，不是多了一枚
        acc.update(_page(levels, subtype="矮人"))
        self.assertEqual(acc.total, 15, "家族识别只是补名字，不该多出颜料")

    def test_level_flicker_does_not_duplicate(self) -> None:
        acc = BackpackAccumulator()
        levels = [2, 2, 2] + [1] * 12
        acc.update(_page(levels, subtype="矮人"))
        flick = list(levels)
        flick[0] = 3
        acc.update(_page(flick, subtype="矮人"))
        self.assertEqual(acc.total, 15, "一格从 2 抖成 3 不能凭空多一枚")
        self.assertEqual(acc.paints()[0].level, 3, "抖的那一帧以最新读数为准")

        acc.update(_page(levels, subtype="矮人"))
        self.assertEqual(acc.total, 15)

    def test_dragging_two_slots_does_not_duplicate(self) -> None:
        acc = BackpackAccumulator()
        before = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        acc.update(_page(before, subtype="矮人"))
        after = list(before)
        after[0], after[1] = before[1], before[0]
        acc.update(_page(after, subtype="矮人"))
        self.assertEqual(acc.total, 15, "拖动只是换位置，不是多了颜料")
        self.assertEqual(acc.paints()[0].level, 1)
        self.assertEqual(acc.paints()[1].level, 2)

    def test_same_page_flip_is_not_accumulated(self) -> None:
        """同一个排序目标页里翻页/拖动只保留最新一页，**不累计**。

        实机踩坑：跨页归档把上一页读数拼回来，界面冒出背包里没有的颜料
        （「多出两枚」「拖动时冒出一大堆」「几个 5 级机械之神」）。
        """
        acc = BackpackAccumulator()
        first = [1] * 15
        acc.update(_page(first, subtype="矮人"), page_key="monster")
        acc.update(_page(first, subtype="矮人"), page_key="monster")
        # 整页换血 = 翻页
        acc.update(_page([2] * 15, subtype="赛季", category=GAMEPLAY), page_key="monster")
        self.assertEqual(acc.total, 15, "同一页只保留最新的 15 格")
        self.assertEqual(acc.all_total, 15, "没换排序目标就不该出现第二页")
        acc.update(_page(first, subtype="矮人"), page_key="monster")
        self.assertEqual(acc.total, 15, "来回翻页也不累计")

    def test_target_switch_keeps_both_pages(self) -> None:
        """怪物目标页 + 玩法目标页各 15 格，合计 30 格（允许重复）。

        用户 2026-11-18：怪物初始颜料时背包按怪物排序，记 15 格；等玩法初始颜料
        出现，背包改按玩法排序，再记 15 格。两页都要留。
        """
        acc = BackpackAccumulator()
        acc.update(_page([2, 2, 1, 1, 1] * 3, subtype="矮人"), page_key="monster")
        self.assertEqual(acc.all_total, 15)
        acc.update(
            _page([4] * 15, subtype="赛季", category=GAMEPLAY), page_key="gameplay"
        )
        self.assertEqual(acc.total, 15, "当前页是玩法页")
        self.assertEqual(acc.all_total, 30, "两页各 15 格")
        self.assertEqual([key for key, _ in acc.pages()], ["monster", "gameplay"])
        self.assertTrue(all(paint.subtype == "赛季" for paint in acc.paints()))
        self.assertEqual(len(acc.all_paints()), 30)

        # 翻回怪物页：读回原来那 15 格，不是重新累计
        acc.update(_page([2, 2, 1, 1, 1] * 3, subtype="矮人"), page_key="monster")
        self.assertEqual(acc.total, 15)
        self.assertEqual(acc.all_total, 30, "两页都还在，且没有翻倍")

    def test_placement_drops_a_slot_on_current_page_only(self) -> None:
        acc = BackpackAccumulator()
        acc.update(_page([1] * 15), page_key="monster")
        acc.update(_page([2] * 15, category=GAMEPLAY), page_key="gameplay")
        self.assertEqual(acc.all_total, 30)

        acc.update(_page([2] * 14, category=GAMEPLAY), page_key="gameplay", consumed=True)
        self.assertEqual(acc.total, 14, "落笔后的那一帧就是新的真相")
        self.assertEqual(acc.all_total, 29, "另一页不受影响")

    def test_missed_slot_without_placement_is_kept(self) -> None:
        acc = BackpackAccumulator()
        acc.update(_page([1] * 15))
        acc.update(_page([1] * 15)[:14])
        self.assertEqual(acc.total, 15, "没落笔却少一格，是漏识别，该保留")

    def test_unknown_target_does_not_open_a_page(self) -> None:
        """目标读不出来时**不许开页**。

        实机 2026-11-18 抓到的 bug：选色面板刚弹出、``current_paint`` 还认不出，
        ``_scene_backpack_page_key`` 以前返回 ``"unknown"``，于是快照里多出一页
        ``unknown``、15 格全是空读数，把「这局有哪些颜料」那份记录污染成
        ``monster 15 + unknown 15``。现在返回空串 → 沿用当前页。
        """
        acc = BackpackAccumulator()
        acc.update(_page([2] * 15, subtype="矮人"), page_key="monster")
        for bad in ("", "unknown", None):
            with self.subTest(page_key=bad):
                acc.update(_page([1] * 15), page_key=bad)
                self.assertEqual(
                    [key for key, _ in acc.pages()],
                    ["monster"],
                    f"page_key={bad!r} 不该新增页",
                )
                self.assertEqual(acc.all_total, 15)


class MissedSlotRecoveryTest(unittest.TestCase):
    def test_missing_paint_without_a_placement_is_restored(self) -> None:
        acc = BackpackAccumulator()
        first = _paints(10)
        acc.update(first, "rule-a")
        # 这一帧漏识别了一枚，但没落笔
        acc.update(first[1:], "rule-a")

        self.assertEqual(acc.total, 10, "没落笔却少了一枚，是漏识别")

    def test_placement_really_shrinks_the_backpack(self) -> None:
        acc = BackpackAccumulator()
        first = _paints(10)
        acc.update(first, "rule-a")
        acc.update(first[1:], "rule-a", consumed=True)

        self.assertEqual(acc.total, 9)


class SessionTest(unittest.TestCase):
    def test_new_rule_clears_previous_paints(self) -> None:
        acc = BackpackAccumulator()
        acc.update(_paints(6), "rule-a")
        acc.update(_paints(4, category=MONSTER), "rule-b")

        self.assertEqual(acc.total, 4)
        self.assertTrue(all(paint.category is MONSTER for paint in acc.paints()))


class MatchingTest(unittest.TestCase):
    def test_matching_only_returns_target_category(self) -> None:
        acc = BackpackAccumulator()
        acc.update(
            [
                FakeObs(MONSTER, 2, "a"),
                FakeObs(GAMEPLAY, 3, "b"),
                FakeObs(GAMEPLAY, 1, "c"),
            ],
            "rule-a",
        )

        gameplay = acc.matching(PaintObject.GAMEPLAY)
        self.assertEqual(len(gameplay), 2)
        self.assertTrue(all(paint.category is GAMEPLAY for paint in gameplay))

    def test_matching_all_returns_everything(self) -> None:
        acc = BackpackAccumulator()
        acc.update([FakeObs(MONSTER, 2, "a"), FakeObs(GAMEPLAY, 3, "b")], "rule-a")

        self.assertEqual(len(acc.matching(PaintObject.ALL)), 2)
        self.assertEqual(len(acc.matching(None)), 2)


class SnapshotTest(unittest.TestCase):
    """落盘快照：复现这局时要知道当时背包里有哪些颜料（两个排序页共 30 格）。"""

    def test_snapshot_records_single_page_with_geometry(self) -> None:
        acc = BackpackAccumulator()
        acc.update(_page([2, 2, 1, 1, 1] * 3, subtype="矮人"), page_key="monster")
        payload = snapshot_payload(acc.pages(), session="s-1", target="monster", now=1700000000.0)

        self.assertEqual(payload["schema"], SNAPSHOT_SCHEMA)
        self.assertEqual(payload["session"], "s-1")
        self.assertEqual(payload["current_target"], "monster")
        self.assertEqual(payload["page_count"], 1)
        self.assertEqual(payload["total_slots"], 15)
        page = payload["pages"][0]
        self.assertEqual(page["target"], "monster")
        self.assertEqual(page["visible_total"], 15)
        first = page["slots"][0]
        self.assertEqual(first["subtype"], "矮人")
        self.assertEqual(first["level"], 2)
        self.assertEqual(first["slot"], list(GRID[0]))
        self.assertEqual(payload["ts"], 1700000000.0)

    def test_snapshot_records_both_target_pages(self) -> None:
        """怪物页 + 玩法页各 15 格、合计 30 格，允许重复。"""
        acc = BackpackAccumulator()
        acc.update(_page([2, 2, 1, 1, 1] * 3, subtype="矮人"), page_key="monster")
        acc.update(_page([4] * 15, subtype="赛季", category=GAMEPLAY), page_key="gameplay")
        payload = snapshot_payload(acc.pages())

        self.assertEqual(payload["page_count"], 2)
        self.assertEqual(payload["total_slots"], 30, "两页各 15 格")
        self.assertEqual([page["target"] for page in payload["pages"]], ["monster", "gameplay"])
        self.assertEqual(
            {slot["subtype"] for slot in payload["pages"][0]["slots"]}, {"矮人"}
        )
        self.assertEqual(
            {slot["subtype"] for slot in payload["pages"][1]["slots"]}, {"赛季"}
        )

    def test_snapshot_flags_full_page_as_truncated(self) -> None:
        acc = BackpackAccumulator()
        acc.update(_page([1] * 15), page_key="monster")
        payload = snapshot_payload(acc.pages())
        self.assertTrue(payload["pages"][0]["may_be_truncated"], "顶满 15 格可能还有一格在屏外")


if __name__ == "__main__":
    unittest.main()
