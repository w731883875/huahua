"""「作用范围提升」类笔触的目标判定与对账（用户 2026-11-15 实机口径）。

四条硬规则（第 2 条被用户当场纠正过一次，别改回去）：

1. **目标 = 作用范围里已有的颜料**，类别要过得了**笔触自己的** ``target_object``；
2. **初始颜料不吃这条**。它只吃「使自身等级提升 N 级」——作用范围按定义排除锚点，
   拿锚点兜底等于凭空给初始颜料加级；
3. 范围里没有颜料 → 这一笔什么也不涨。用户说的「范围全空白、放下第一枚背包颜料就提升它」
   是因为那一枚**落在范围内**，落笔后成了唯一候选，不是兜底；
4. 官方「随机 N 个」在候选多于 N 时只中一部分，对账必须按区间判，不能每笔都报红。

另外：调色（按目标类别升级）、笔触（范围 +1）、自身提升是**三个独立来源**，
各有各的范围与目标类别，不能互相顶替：
英灵 Lv4（``category=monster``）在调色目标是玩法时只吃笔触 +1 变成 5；
同一格换成玩法 Lv4 就是调色 +1 与笔触 +1 都吃，变 6。
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from huahua_assistant import game_model as gm
from huahua_assistant.codes import resolve_affix
from huahua_assistant.game_model import (
    BOARD_SIZE,
    CurrentPaintRule,
    Paint,
    PaintCategory,
    PaintDirection,
    PaintObject,
    PuzzleState,
    Tile,
    TileState,
    scope_upgrade_targets,
    tile_final_level,
)
from huahua_assistant.level_reconcile import reconcile

ANCHOR_ROW = 4
RANDOM_ONE = resolve_affix("对调色范围内的随机1个颜料额外提升1级")
RANDOM_TWO = resolve_affix("对调色范围内的随机2个颜料额外提升1级")
ALL_ONE = resolve_affix("范围内所有颜料提升1级")
SAME_PAINT_ONE = resolve_affix("对调色范围内与自身内容相同的颜料额外提升1级")


def _state(
    cells: dict[tuple[int, int], Paint] | None = None,
    *,
    panel: Paint | None = None,
    upgrade: int = 2,
    target: PaintObject = PaintObject.GAMEPLAY,
    anchor: tuple[int, int] = (ANCHOR_ROW, 0),
    affix=None,
    pending: bool = False,
) -> PuzzleState:
    tiles = [Tile(state=TileState.BLANK) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    for (row, col), paint in (cells or {}).items():
        tiles[row * BOARD_SIZE + col] = Tile(state=TileState.PAINT, paint=paint)
    rule = CurrentPaintRule(
        paint=panel or Paint(PaintCategory.MONSTER, 5, "伊吉"),
        direction=PaintDirection.HORIZONTAL,
        target_object=target,
        match_required=4,
        upgrade_amount=upgrade,
        match_current=0,
    )
    return PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        current_anchor=None if pending else anchor,
        current_rule_pending=pending,
        active_affix=affix,
        observed_scope_cells=tuple((ANCHOR_ROW, col) for col in range(6)),
    )


def _monster(level: int) -> Paint:
    return Paint(PaintCategory.MONSTER, level, "伊吉")


def _yingling(level: int) -> Paint:
    """英灵 —— ``paint_catalog.json`` 里 m013，``category=monster``。

    这正是用户那句「调色效果是玩法升级吃不到」的原因：调色目标如果是玩法，
    怪物颜料一分不加，4→5 只能来自笔触。
    """
    return Paint(PaintCategory.MONSTER, level, "英灵")


def _gameplay(level: int, name: str = "赛季") -> Paint:
    return Paint(PaintCategory.GAMEPLAY, level, name)


class ScopeUpgradeTargetTests(unittest.TestCase):
    def test_placed_paint_inside_scope_is_the_only_candidate(self) -> None:
        """范围里只有刚落下的那一枚 → 它就是候选，随机 1 个必然抽中它。

        这不是「兜底」：它本来就在范围里，落笔后自然成为范围内的唯一颜料。
        """
        state = _state({(ANCHOR_ROW, 1): _monster(5)}, affix=RANDOM_ONE)
        sure, maybe, amount = scope_upgrade_targets(state, RANDOM_ONE, placed=(ANCHOR_ROW, 1))
        self.assertEqual(sure, (ANCHOR_ROW * BOARD_SIZE + 1,))
        self.assertEqual(maybe, ())
        self.assertEqual(amount, 1)

    def test_placed_cell_inside_scope_is_counted_not_the_anchor(self) -> None:
        """锚点上是初始颜料、落点在别的范围格：候选是落点那枚，锚点永远不算。"""
        state = _state(
            {(ANCHOR_ROW, 0): _monster(5), (ANCHOR_ROW, 2): _monster(5)},
            affix=RANDOM_ONE,
            anchor=(ANCHOR_ROW, 0),
        )
        sure, _maybe, _amount = scope_upgrade_targets(state, RANDOM_ONE, placed=(ANCHOR_ROW, 2))
        self.assertEqual(sure, (ANCHOR_ROW * BOARD_SIZE + 2,))

    def test_placed_outside_scope_gets_nothing(self) -> None:
        """落点在作用范围外 → 这条笔触与它无关，一级都不该涨。

        锚点是 (4, 0)、方向横向，落点在 (2, 0)：几何上不在同一行，
        旧实现会无条件拿落点兜底，结果给范围外的格子白加一级。
        """
        state = _state(
            {(ANCHOR_ROW, 0): _monster(5), (2, 0): _monster(5)},
            affix=RANDOM_ONE,
            anchor=(ANCHOR_ROW, 0),
        )
        sure, maybe, _amount = scope_upgrade_targets(state, RANDOM_ONE, placed=(2, 0))
        self.assertEqual((sure, maybe), ((), ()))

    def test_anchor_itself_never_gets_scope_upgrade(self) -> None:
        """初始颜料自己不在作用范围里：范围空时也不许把它顶上去。"""
        state = _state({(ANCHOR_ROW, 0): _monster(5)}, affix=RANDOM_ONE)
        sure, maybe, _amount = scope_upgrade_targets(state, RANDOM_ONE, placed=(ANCHOR_ROW, 0))
        self.assertEqual((sure, maybe), ((), ()))

    def test_single_candidate_with_random_one_is_certain(self) -> None:
        state = _state({(ANCHOR_ROW, 2): _gameplay(4)}, affix=RANDOM_ONE)
        sure, maybe, _amount = scope_upgrade_targets(state, RANDOM_ONE, placed=(ANCHOR_ROW, 0))
        self.assertEqual(sure, (ANCHOR_ROW * BOARD_SIZE + 2,))
        self.assertEqual(maybe, ())

    def test_more_candidates_than_count_is_uncertain(self) -> None:
        state = _state(
            {
                (ANCHOR_ROW, 1): _gameplay(4),
                (ANCHOR_ROW, 3): _gameplay(4),
                (ANCHOR_ROW, 5): _gameplay(4),
            },
            affix=RANDOM_ONE,
        )
        sure, maybe, _amount = scope_upgrade_targets(state, RANDOM_ONE, placed=(ANCHOR_ROW, 0))
        self.assertEqual(sure, ())
        self.assertEqual(len(maybe), 3)

    def test_count_two_with_two_candidates_is_certain(self) -> None:
        state = _state(
            {(ANCHOR_ROW, 1): _gameplay(4), (ANCHOR_ROW, 3): _gameplay(4)},
            affix=RANDOM_TWO,
        )
        sure, maybe, _amount = scope_upgrade_targets(state, RANDOM_TWO, placed=(ANCHOR_ROW, 0))
        self.assertEqual(len(sure), 2)
        self.assertEqual(maybe, ())

    def test_all_paints_affix_targets_every_candidate(self) -> None:
        state = _state(
            {(ANCHOR_ROW, 1): _gameplay(4), (ANCHOR_ROW, 3): _gameplay(4)},
            affix=ALL_ONE,
        )
        sure, maybe, _amount = scope_upgrade_targets(state, ALL_ONE, placed=(ANCHOR_ROW, 0))
        self.assertEqual(len(sure), 2)
        self.assertEqual(maybe, ())

    def test_same_paint_affix_ignores_other_kinds(self) -> None:
        """「与自身内容相同」不碰异类：落笔是怪物，英灵不该被选中。"""
        state = _state(
            {(ANCHOR_ROW, 1): _gameplay(4), (ANCHOR_ROW, 3): _monster(4)},
            affix=SAME_PAINT_ONE,
        )
        sure, maybe, _amount = scope_upgrade_targets(
            state, SAME_PAINT_ONE, placed=(ANCHOR_ROW, 0)
        )
        self.assertEqual(sure, (ANCHOR_ROW * BOARD_SIZE + 3,))
        self.assertEqual(maybe, ())

    def test_brush_target_object_filters_categories(self) -> None:
        """笔触自己的目标类别要生效：写「怪物」时玩法颜料不吃这一级。

        官方文案没限类别时 ``target_object`` 为空，不能借调色规则的目标来筛；
        反过来，文案限了类别就必须筛，否则会把玩法也顶上去。
        """
        monster_only = replace(RANDOM_ONE, target_object=PaintObject.MONSTER)
        state = _state(
            {(ANCHOR_ROW, 1): _yingling(4), (ANCHOR_ROW, 3): _gameplay(4)},
            affix=monster_only,
        )
        sure, maybe, _amount = scope_upgrade_targets(state, monster_only, placed=(ANCHOR_ROW, 0))
        self.assertEqual(sure, (ANCHOR_ROW * BOARD_SIZE + 1,))
        self.assertEqual(maybe, ())

    def test_settle_buff_upgrades_the_paint_inside_scope(self) -> None:
        state = _state({(ANCHOR_ROW, 1): _monster(5)}, affix=RANDOM_ONE)
        settled, count = gm._settle_scope_buff(
            state, random.Random(0), placed=(ANCHOR_ROW, 1)
        )
        self.assertEqual(count, 1)
        tile = settled.tile(ANCHOR_ROW, 1)
        self.assertEqual(tile_final_level(tile), 6)

    def test_settle_buff_does_not_touch_the_anchor(self) -> None:
        """放初始颜料那一次：笔触一级都不给（旧实现会误加 1 级）。"""
        state = _state({(ANCHOR_ROW, 0): _monster(5)}, affix=RANDOM_ONE)
        settled, count = gm._settle_scope_buff(
            state, random.Random(0), placed=(ANCHOR_ROW, 0)
        )
        self.assertEqual(count, 0)
        self.assertEqual(tile_final_level(settled.tile(ANCHOR_ROW, 0)), 5)


class LevelSourceTests(unittest.TestCase):
    """来源表：调色 / 笔触 / 自身，各自带范围与目标类别（用户第 3 点）。"""

    def _sources(self, state, **kwargs):
        return gm.level_sources(state, affix=kwargs.pop("affix", RANDOM_ONE), **kwargs)

    def test_each_source_carries_its_own_target_and_scope(self) -> None:
        state = _state(
            {(ANCHOR_ROW, 1): _gameplay(4)}, target=PaintObject.GAMEPLAY, upgrade=1
        )
        sources = self._sources(state)
        self.assertEqual([source.key for source in sources], ["tint", "brush"])
        tint, brush = sources
        # 调色认玩法；笔触这条官方文案没限类别（空 = 所有颜料）。
        self.assertEqual(tint.target, PaintObject.GAMEPLAY)
        self.assertIsNone(brush.target)
        self.assertIn("玩法颜料", tint.describe())
        self.assertIn("所有颜料", brush.describe())
        # 范围各自独立：这里两者几何相同，但字段是分别算的。
        self.assertNotIn((ANCHOR_ROW, 0), tint.scope)
        self.assertNotIn((ANCHOR_ROW, 0), brush.scope)

    def test_gameplay_cell_takes_tint_and_brush_together(self) -> None:
        """玩法 Lv4：调色 +1 与笔触 +1 都吃 → 6。"""
        state = _state(
            {(ANCHOR_ROW, 1): _gameplay(4)}, target=PaintObject.GAMEPLAY, upgrade=1
        )
        base, parts = gm.tile_level_breakdown(
            state, ANCHOR_ROW, 1, self._sources(state), base_level=4
        )
        self.assertEqual(base, 4)
        self.assertEqual(parts, (("调色效果", 1), ("笔触效果", 1)))
        self.assertEqual(base + sum(amount for _label, amount in parts), 6)

    def test_monster_cell_takes_brush_only(self) -> None:
        """英灵 Lv4（``category=monster``）：调色目标是玩法 → 只吃笔触 +1 → 5。"""
        state = _state(
            {(ANCHOR_ROW, 1): _yingling(4)}, target=PaintObject.GAMEPLAY, upgrade=1
        )
        base, parts = gm.tile_level_breakdown(
            state, ANCHOR_ROW, 1, self._sources(state), base_level=4
        )
        self.assertEqual(parts, (("笔触效果", 1),))
        self.assertEqual(base + sum(amount for _label, amount in parts), 5)

    def test_self_source_covers_only_the_anchor(self) -> None:
        """「自身提升」只落在锚点那一格，且与范围/目标无关。"""
        state = _state(
            {(ANCHOR_ROW, 0): _monster(5), (ANCHOR_ROW, 1): _monster(5)},
            upgrade=1,
        )
        sources = self._sources(state, self_upgrade=2)
        self_source = [source for source in sources if source.key == "self"]
        self.assertEqual(len(self_source), 1)
        self.assertEqual(self_source[0].scope, ((ANCHOR_ROW, 0),))
        base, parts = gm.tile_level_breakdown(state, ANCHOR_ROW, 0, sources, base_level=5)
        self.assertEqual(parts, (("自身提升", 2),))
        self.assertEqual(base + sum(amount for _label, amount in parts), 7)
        # 旁边那枚不吃「自身提升」。
        _base, other = gm.tile_level_breakdown(state, ANCHOR_ROW, 1, sources, base_level=5)
        self.assertNotIn("自身提升", [label for label, _amount in other])


class BrushVsTintSeparationTests(unittest.TestCase):
    """笔触（范围 +1）与调色（按目标类别）互不顶替。"""

    def test_英灵_4_to_5_is_explained_by_brush_only(self) -> None:
        previous = _state(
            {(ANCHOR_ROW, 1): _yingling(4)}, upgrade=2, affix=RANDOM_ONE
        )
        current = _state(
            {(ANCHOR_ROW, 1): _yingling(5), (ANCHOR_ROW, 0): _monster(5)},
            upgrade=2,
            affix=RANDOM_ONE,
        )
        report = reconcile(previous, current, active_affix=RANDOM_ONE)
        self.assertTrue(report.ran)
        self.assertEqual(report.mismatches, (), report.summary())

    def test_tint_only_applies_to_matching_target(self) -> None:
        """调色目标是玩法时，怪物不吃调色，只能吃笔触 +1。"""
        previous = _state({(ANCHOR_ROW, 1): _monster(4)}, upgrade=2, affix=None)
        current = _state(
            {(ANCHOR_ROW, 1): _monster(4), (ANCHOR_ROW, 0): _monster(5)}, upgrade=2, affix=None
        )
        report = reconcile(previous, current)
        self.assertEqual(report.mismatches, (), report.summary())

    def test_extra_jump_beyond_brush_and_tint_is_flagged(self) -> None:
        """英灵一次涨 2 级（笔触只解释 +1）必须报出来。"""
        previous = _state({(ANCHOR_ROW, 1): _yingling(4)}, affix=RANDOM_ONE)
        current = _state(
            {(ANCHOR_ROW, 1): _yingling(6), (ANCHOR_ROW, 0): _monster(5)}, affix=RANDOM_ONE
        )
        report = reconcile(previous, current, active_affix=RANDOM_ONE)
        self.assertEqual(len(report.mismatches), 1)
        item = report.mismatches[0]
        self.assertEqual((item.row, item.col), (ANCHOR_ROW, 1))
        self.assertEqual(item.observed, 6)


class RandomTargetReconcileTests(unittest.TestCase):
    """随机 N 个时按区间判：落在区间内不报警，超出才报。"""

    def _previous(self):
        return _state(
            {
                (ANCHOR_ROW, 1): _yingling(4),
                (ANCHOR_ROW, 3): _yingling(4),
                (ANCHOR_ROW, 5): _yingling(4),
            },
            affix=RANDOM_ONE,
        )

    def test_whichever_cell_got_the_bonus_is_not_a_mismatch(self) -> None:
        for hit in ((ANCHOR_ROW, 1), (ANCHOR_ROW, 3), (ANCHOR_ROW, 5)):
            with self.subTest(hit=hit):
                cells = {
                    (ANCHOR_ROW, 1): _gameplay(5 if hit == (ANCHOR_ROW, 1) else 4),
                    (ANCHOR_ROW, 3): _gameplay(5 if hit == (ANCHOR_ROW, 3) else 4),
                    (ANCHOR_ROW, 5): _gameplay(5 if hit == (ANCHOR_ROW, 5) else 4),
                    (ANCHOR_ROW, 0): _monster(5),
                }
                report = reconcile(self._previous(), _state(cells, affix=RANDOM_ONE), active_affix=RANDOM_ONE)
                self.assertEqual(report.mismatches, (), report.summary())
                self.assertEqual(len(report.uncertain), 1)

    def test_two_bonused_cells_are_flagged(self) -> None:
        """「随机 1 个」却涨了两个 → 超出上限，必须报。"""
        cells = {
            (ANCHOR_ROW, 1): _yingling(5),
            (ANCHOR_ROW, 3): _yingling(5),
            (ANCHOR_ROW, 5): _yingling(4),
            (ANCHOR_ROW, 0): _monster(5),
        }
        report = reconcile(self._previous(), _state(cells, affix=RANDOM_ONE), active_affix=RANDOM_ONE)
        self.assertEqual(len(report.mismatches), 2)

    def test_no_bonus_at_all_is_also_within_range(self) -> None:
        """一次都没抽中（范围里有别的目标但随机没选），也不该报红。"""
        cells = {
            (ANCHOR_ROW, 1): _yingling(4),
            (ANCHOR_ROW, 3): _yingling(4),
            (ANCHOR_ROW, 5): _yingling(4),
            (ANCHOR_ROW, 0): _monster(5),
        }
        report = reconcile(self._previous(), _state(cells, affix=RANDOM_ONE), active_affix=RANDOM_ONE)
        self.assertEqual(report.mismatches, ())


class UnifiedScopeTests(unittest.TestCase):
    """调色与笔触必须用**同一份纯几何**范围（用户 2026-11-15 两次定调）。

    第一次：「要统一」；第二次追问后升级为「**不能**按画面金框收窄」——

    * 金框只亮能放颜料的格子（``vision.is_scope_highlight`` 要求空白/像空白），
      虚空（``S8_Nihility``）、失色（``S8_Ruins``）本来就不亮框；
    * 扩散能把虚空/失色转成空白（官方 3014/3015「全图随机 N 个虚空地块转变为空白地块」），
      转完之后它们就是范围内的普通格，落笔时理应吃加成 —— 按金框收窄会永久判它们「不在范围」；
    * 方向识别已经由 ``vision._correct_current_paint_from_scope`` 单独校正，
      收窄不增加信息，只会把识别漏掉的格子一起吞掉。

    所以下面这些用例里，``observed_scope_cells`` 一律只当「识别到的一半」看，
    断言的是**几何口径**：只要格子在锚点所在行/列上，就在范围内。
    """

    def _highlight(self, cells: dict[tuple[int, int], Paint], observed) -> PuzzleState:
        return replace(
            _state(cells, affix=RANDOM_ONE),
            observed_scope_cells=tuple(observed),
        )

    def test_brush_and_tint_scope_are_identical(self) -> None:
        state = self._highlight({}, [(ANCHOR_ROW, 1), (ANCHOR_ROW, 2)])
        self.assertEqual(
            gm._affix_scope_cells(state, RANDOM_ONE),
            gm._placement_scope_cells(state, state.current_rule),
        )

    def test_highlight_does_not_shrink_the_scope(self) -> None:
        """金框只亮 c2~c3，几何范围仍是整行去掉锚点（5 格）。"""
        state = self._highlight({}, [(ANCHOR_ROW, 1), (ANCHOR_ROW, 2)])
        scope = gm._affix_scope_cells(state, RANDOM_ONE)
        self.assertEqual(len(scope), BOARD_SIZE - 1)
        self.assertIn((ANCHOR_ROW, 5), scope)

    def test_paint_in_an_unhighlighted_cell_still_counts(self) -> None:
        """金框没亮的格子里有颜料 → 仍在范围内，照样是候选。

        就是「金框漏识别一格」的场景：不能因为识别漏了就把加成吃掉。
        """
        state = self._highlight(
            {(ANCHOR_ROW, 4): _yingling(4)},
            [(ANCHOR_ROW, 1), (ANCHOR_ROW, 2)],
        )
        sure, maybe, _amount = scope_upgrade_targets(state, RANDOM_ONE)
        self.assertEqual(sure, (ANCHOR_ROW * BOARD_SIZE + 4,))
        self.assertEqual(maybe, ())

    def test_void_turned_free_cell_gets_the_scope_buff(self) -> None:
        """扩散把虚空转成空白、再放上颜料 → 吃范围 buff。

        这一条是用户点名的场景：那一格从来不在金框里（当时是虚空），
        但转成空白后它就是范围内的普通格。按金框收窄会漏掉这级加成。
        """
        state = self._highlight(
            {(ANCHOR_ROW, 5): _yingling(4)},
            [(ANCHOR_ROW, 1), (ANCHOR_ROW, 2)],
        )
        sure, _maybe, _amount = scope_upgrade_targets(
            state, RANDOM_ONE, placed=(ANCHOR_ROW, 5)
        )
        self.assertEqual(sure, (ANCHOR_ROW * BOARD_SIZE + 5,))

    def test_placed_cell_inside_geometry_gets_the_buff(self) -> None:
        """落点在几何范围内、金框却没亮 → 仍然吃加成（同一条理由）。"""
        state = self._highlight(
            {(ANCHOR_ROW, 3): _monster(5)},
            [(ANCHOR_ROW, 1), (ANCHOR_ROW, 2)],
        )
        sure, maybe, _amount = scope_upgrade_targets(
            state, RANDOM_ONE, placed=(ANCHOR_ROW, 3)
        )
        self.assertEqual(sure, (ANCHOR_ROW * BOARD_SIZE + 3,))
        self.assertEqual(maybe, ())

    def test_highlight_does_not_leak_into_other_rows(self) -> None:
        """反向保护：收窄去掉了，不等于范围可以越界到别的行。"""
        state = self._highlight(
            {(ANCHOR_ROW + 1, 1): _yingling(4)},
            [(ANCHOR_ROW, 1), (ANCHOR_ROW + 1, 1)],
        )
        sure, maybe, _amount = scope_upgrade_targets(state, RANDOM_ONE)
        self.assertEqual((sure, maybe), ((), ()))

    def test_level_sources_report_geometry_scope(self) -> None:
        state = self._highlight({}, [(ANCHOR_ROW, 1), (ANCHOR_ROW, 2), (ANCHOR_ROW, 3)])
        sources = {source.key: source for source in gm.level_sources(state, affix=RANDOM_ONE)}
        self.assertIn("tint", sources)
        self.assertIn("brush", sources)
        self.assertEqual(sources["tint"].scope, sources["brush"].scope)
        self.assertEqual(len(sources["brush"].scope), BOARD_SIZE - 1)


if __name__ == "__main__":
    unittest.main()
