"""锚点别被识别抖动冲掉（2026-11-14 实机事故）。

用户现场：伊吉（怪物 Lv7）已经落在 r5c6，面板进度 2/4，本该「往调色范围里落背包颜料」，
但软件一直判「还没落初始颜料」，推荐落点是**空白格上的锚点**。

两个独立的误判都能造出这个结果：

1. ``_fast_current_paint_changed`` 把「作用对象从 所有 抖成 玩法」当成「换了新颜料」，
   调用方随即清掉锚点、把局面判成待落锚点；
2. ``_state_for_visible_brush_panel`` 只要看到笔触面板就强制清锚点 + 待落锚点，
   可面板在落子之后仍然开着（选词缀就发生在落子之后）。

还有一条隐蔽的同类：面板等级 5→7（选了「使自身等级提升2级」）也会被判成换颜料。
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from huahua_assistant.app import (
    _anchor_tile_holds_rule_paint,
    _fast_current_paint_changed,
    _merge_fast_rule_state,
    _same_initial_paint,
    _state_for_visible_brush_panel,
)
from huahua_assistant.game_model import (
    CurrentPaintRule,
    Paint,
    PuzzleState,
    Tile,
    build_affix_catalog,
)
from huahua_assistant.models import (
    BoardObservation,
    BrushOptionsObservation,
    CurrentPaintObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    Rect,
    SceneObservation,
    TileObservation,
    TileState,
)
from huahua_assistant.vision import _carry_confirmed_target

ANCHOR = (4, 5)  # r5c6：伊吉落的那一格


def _tile(index: int, state: TileState = TileState.BLANK, **overrides) -> TileObservation:
    values = {
        "row": index // 6,
        "col": index % 6,
        "rect": Rect((index % 6) * 10, (index // 6) * 10, 10, 10),
        "state": state,
        "confidence": 0.95,
        "features": {},
    }
    values.update(overrides)
    return TileObservation(**values)


def _board(paints: dict[int, TileObservation] | None = None) -> BoardObservation:
    paints = paints or {}
    return BoardObservation(
        rect=Rect(0, 0, 60, 60),
        score=0.9,
        tiles=tuple(paints.get(index, _tile(index)) for index in range(36)),
    )


def _iji_tile(level: int = 7) -> TileObservation:
    return _tile(
        ANCHOR[0] * 6 + ANCHOR[1],
        TileState.PAINT,
        paint_category=PaintCategory.MONSTER,
        paint_category_confidence=0.82,
        paint_level=level,
        paint_level_confidence=0.93,
        paint_subtype="伊吉",
        is_current_anchor=True,
        current_anchor_confidence=0.18,
    )


def _current_paint(
    *,
    level: int = 7,
    target: PaintObject = PaintObject.ALL,
    direction: PaintDirection = PaintDirection.HORIZONTAL,
    match_current: int = 2,
    match_required: int = 4,
    category: PaintCategory = PaintCategory.MONSTER,
) -> CurrentPaintObservation:
    return CurrentPaintObservation(
        panel_rect=Rect(1300, 200, 500, 160),
        icon_rect=Rect(1310, 210, 90, 90),
        text_rect=Rect(1420, 210, 360, 150),
        category=category,
        category_confidence=0.82,
        subtype="伊吉",
        is_treasure=False,
        level=level,
        level_confidence=0.93,
        match_current=match_current,
        match_required=match_required,
        target_object=target,
        direction=direction,
        upgrade_amount=2,
        keyword_scores={},
        raw_text_hint="probe",
    )


def _rule(*, level: int = 7, target: PaintObject = PaintObject.ALL) -> CurrentPaintRule:
    return CurrentPaintRule(
        paint=Paint(category=PaintCategory.MONSTER, level=level, subtype="伊吉"),
        direction=PaintDirection.HORIZONTAL,
        target_object=target,
        match_required=4,
        upgrade_amount=2,
        match_current=2,
    )


def _scene(*, current: CurrentPaintObservation | None = None, anchor_tile: TileObservation | None = None):
    return SceneObservation(
        board=_board({ANCHOR[0] * 6 + ANCHOR[1]: anchor_tile or _iji_tile()}),
        regions=(),
        current_paint=current or _current_paint(),
    )


def _state(*, anchor: tuple[int, int] | None = ANCHOR, rule: CurrentPaintRule | None = None) -> PuzzleState:
    return PuzzleState(
        tiles=tuple(Tile(TileState.BLANK) for _ in range(36)),
        current_rule=rule or _rule(),
        current_anchor=anchor,
        current_rule_pending=False,
    )


class AnchorIsPhysicalEvidenceTests(unittest.TestCase):
    def test_target_flip_does_not_count_as_new_paint_while_anchor_stands(self) -> None:
        """作用对象在「所有 / 玩法」之间翻，锚点还在 → 不是换颜料。"""
        scene = _scene(current=_current_paint(target=PaintObject.GAMEPLAY))
        self.assertFalse(_fast_current_paint_changed(scene, _state()))

    def test_level_jump_from_self_upgrade_does_not_count_as_new_paint(self) -> None:
        """「使自身等级提升2级」5→7：面板等级跳了，但棋盘上还是那枚伊吉。"""
        scene = _scene(current=_current_paint(level=7))
        self.assertFalse(_fast_current_paint_changed(scene, _state(rule=_rule(level=5))))

    def test_emptied_anchor_cell_lets_new_paint_be_detected(self) -> None:
        """锚点那格已经没有同类颜料（换了画纸）→ 允许判成新颜料。"""
        scene = _scene(current=_current_paint(target=PaintObject.GAMEPLAY), anchor_tile=_tile(ANCHOR[0] * 6 + ANCHOR[1]))
        self.assertTrue(_fast_current_paint_changed(scene, _state()))

    def test_unknown_category_on_anchor_cell_is_still_evidence(self) -> None:
        """锚点格颜料类别读不出来时按「还在」处理，宁可不动锚点。"""
        anchor_tile = _tile(
            ANCHOR[0] * 6 + ANCHOR[1],
            TileState.PAINT,
            paint_category=PaintCategory.UNKNOWN,
            is_current_anchor=True,
        )
        scene = _scene(current=_current_paint(target=PaintObject.GAMEPLAY), anchor_tile=anchor_tile)
        self.assertFalse(_fast_current_paint_changed(scene, _state()))

    def test_anchor_check_is_read_only_on_mismatched_category(self) -> None:
        anchor_tile = _tile(0, TileState.PAINT, paint_category=PaintCategory.GAMEPLAY)
        rule = _rule()
        self.assertFalse(_anchor_tile_holds_rule_paint(_scene(anchor_tile=anchor_tile), rule, (0, 0)))


class SameInitialPaintTests(unittest.TestCase):
    def test_level_change_keeps_identity(self) -> None:
        self.assertTrue(_same_initial_paint(_rule(level=7), _rule(level=5)))

    def test_target_change_keeps_identity(self) -> None:
        self.assertTrue(_same_initial_paint(_rule(target=PaintObject.GAMEPLAY), _rule(target=PaintObject.ALL)))

    def test_category_change_breaks_identity(self) -> None:
        other = replace(_rule(), paint=Paint(category=PaintCategory.GAMEPLAY, level=7))
        self.assertFalse(_same_initial_paint(other, _rule()))


class VisibleBrushPanelTests(unittest.TestCase):
    def _observation(self) -> BrushOptionsObservation:
        return BrushOptionsObservation(
            panel_rect=Rect(400, 300, 760, 250),
            stage1_options=("堆砌", "扩散"),
            stage1_scores={"堆砌": 0.9, "扩散": 0.9},
            chosen_stage1=None,
            stage2_options=(),
            stage2_scores={},
            keyword_scores={},
            confidence=0.9,
        )

    def test_panel_does_not_erase_a_placed_anchor(self) -> None:
        state = _state(anchor=ANCHOR)
        updated = _state_for_visible_brush_panel(state, self._observation())
        self.assertEqual(ANCHOR, updated.current_anchor)
        self.assertFalse(updated.current_rule_pending)

    def test_panel_still_clears_previous_affix(self) -> None:
        """面板开着 = 正在挑新词缀，上一条不该继续生效。"""
        affix = build_affix_catalog()[0]
        state = replace(_state(anchor=ANCHOR), active_affix=affix)
        self.assertIsNone(_state_for_visible_brush_panel(state, self._observation()).active_affix)


class TargetCarryTests(unittest.TestCase):
    """栅格模板把「所有」读成「玩法」，文字确认过的结论不能被它翻掉。"""

    def _previous(self, *, text_confirmed: bool, target: PaintObject = PaintObject.ALL):
        paint = _current_paint(target=target)
        scores = {"panel_text:target": 1.0} if text_confirmed else {}
        return replace(paint, keyword_scores=scores)

    def test_text_confirmed_target_survives_a_throttled_frame(self) -> None:
        merged = _carry_confirmed_target(
            self._previous(text_confirmed=True), {}, PaintObject.GAMEPLAY
        )
        self.assertEqual(PaintObject.ALL, merged)

    def test_unconfirmed_target_takes_the_fresh_read(self) -> None:
        merged = _carry_confirmed_target(
            self._previous(text_confirmed=False), {}, PaintObject.GAMEPLAY
        )
        self.assertEqual(PaintObject.GAMEPLAY, merged)

    def test_fresh_text_wins_over_the_carried_one(self) -> None:
        merged = _carry_confirmed_target(
            self._previous(text_confirmed=True),
            {"panel_text:target": 1.0},
            PaintObject.GAMEPLAY,
        )
        self.assertEqual(PaintObject.GAMEPLAY, merged)


class PendingStickinessTests(unittest.TestCase):
    """「待落初始颜料」一旦被棋盘证伪，就不该在同一条规则里翻回来。

    直接打 ``_merge_fast_rule_state``——就是快路径每帧真跑的那段决策。
    """

    def test_placed_anchor_survives_a_target_flip(self) -> None:
        previous = _state()
        # 本帧面板把作用对象读成「玩法」，棋盘上伊吉还在原位
        current = replace(previous, current_rule=_rule(target=PaintObject.GAMEPLAY))
        scene = _scene(current=_current_paint(target=PaintObject.GAMEPLAY))
        merged = _merge_fast_rule_state(current, previous, scene, ())
        self.assertEqual(ANCHOR, merged.current_anchor)
        self.assertFalse(merged.current_rule_pending)
        # 规则本身要用本帧的新值，否则界面会一直挂着旧目标
        self.assertEqual(PaintObject.GAMEPLAY, merged.current_rule.target_object)

    def test_pending_never_flip_back_once_board_shows_anchor(self) -> None:
        previous = replace(_state(), current_rule_pending=True)
        current = replace(previous, current_rule_pending=False)
        scene = _scene()
        merged = _merge_fast_rule_state(current, previous, scene, ())
        self.assertFalse(merged.current_rule_pending)

    def test_pending_stays_while_board_shows_nothing(self) -> None:
        previous = replace(_state(anchor=None), current_rule_pending=True)
        current = replace(previous, current_rule_pending=True)
        scene = _scene(anchor_tile=_tile(ANCHOR[0] * 6 + ANCHOR[1]))
        merged = _merge_fast_rule_state(current, previous, scene, ())
        self.assertTrue(merged.current_rule_pending)

    def test_new_paint_clears_anchor(self) -> None:
        previous = _state()
        current = replace(
            previous,
            current_rule=replace(_rule(), paint=Paint(category=PaintCategory.GAMEPLAY, level=3)),
        )
        scene = _scene(
            current=_current_paint(category=PaintCategory.GAMEPLAY),
            anchor_tile=_tile(ANCHOR[0] * 6 + ANCHOR[1]),
        )
        merged = _merge_fast_rule_state(current, previous, scene, ())
        self.assertIsNone(merged.current_anchor)
        self.assertTrue(merged.current_rule_pending)

    def test_dead_latched_anchor_hands_over_to_fresh_detection(self) -> None:
        """上一帧的锚点格已经空了（伊吉被炸掉/换画纸）→ 让位给本帧判定。"""
        previous = _state()
        current = replace(previous, current_anchor=(0, 0))
        scene = _scene(anchor_tile=_tile(ANCHOR[0] * 6 + ANCHOR[1]))
        merged = _merge_fast_rule_state(current, previous, scene, ())
        self.assertEqual((0, 0), merged.current_anchor)

    def test_fresh_anchor_fills_in_when_previous_had_none(self) -> None:
        previous = _state(anchor=None)
        current = replace(previous, current_anchor=ANCHOR)
        merged = _merge_fast_rule_state(current, previous, _scene(), ())
        self.assertEqual(ANCHOR, merged.current_anchor)


if __name__ == "__main__":
    unittest.main()
