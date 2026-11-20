"""地块加成的「闭环」：视觉看不见的两项数怎么进模型、怎么进录制。

这一层解决的是第 7 条前置条件。棋盘右下角的 ``LvN`` 角标显示的是**最终等级**
（颜料等级 + 地块加成 + 光环），而空地上叠的加成连像素都没有。所以：

- ``game_model.state_from_observation(..., previous=...)`` 把上一帧的
  ``bonus_level`` / ``aura_amount`` 按格子继承过来（buff 挂在地块上，不是挂在颜料上），
  再把观测到的最终等级反推回颜料自身等级；
- ``fast_vision.snapshot_from_scene(scene, model)`` 把这两项回填进快照，
  录制器才有东西可写。

这里锁定六件事：

1. 不传 ``previous`` 时行为与老代码完全一致（加成为 0，等级原样）；
2. 加成按**格子**继承，包括空地和失色地块；
3. 反推：观测最终等级 4 + 加成 3 → 颜料自身 1 级，``tile_final_level`` 仍是 4；
4. 封顶 7 级反推不唯一，保持原样；
5. 反推小于 1 级说明模型多算了，就地清掉加成自我纠正；
6. 快照回填 + 录制器出 ``tile_buff_changed`` 事件。
"""

from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant.fast_vision import FastVisionSnapshot, snapshot_from_scene
from huahua_assistant.models import (
    BoardObservation,
    PaintCategory,
    Rect,
    TileObservation,
    TileState,
)
from huahua_assistant.run_recorder import AutomaticRunRecorder


def _tile(
    index: int,
    state: TileState = TileState.BLANK,
    *,
    category: PaintCategory = PaintCategory.UNKNOWN,
    level: int | None = None,
    subtype: str | None = None,
    anchor: bool = False,
) -> TileObservation:
    return TileObservation(
        row=index // 6,
        col=index % 6,
        rect=Rect((index % 6) * 10, (index // 6) * 10, 10, 10),
        state=state,
        confidence=0.95,
        features={},
        paint_category=category,
        paint_category_confidence=0.9 if level is not None else 0.0,
        paint_level=level,
        paint_level_confidence=0.9 if level is not None else 0.0,
        paint_subtype=subtype,
        is_current_anchor=anchor,
        current_anchor_confidence=0.9 if anchor else 0.0,
    )


def _board(overrides: dict[int, TileObservation]) -> BoardObservation:
    tiles = [overrides.get(index, _tile(index)) for index in range(36)]
    return BoardObservation(rect=Rect(0, 0, 60, 60), score=0.9, tiles=tuple(tiles))


def _state_with_buffs(cells: dict[int, tuple[int, int]]) -> gm.PuzzleState:
    """造一个「已经算过光环」的状态：``aura_level`` 与 ``aura_amount`` 一致。"""
    tiles = []
    for index in range(36):
        bonus, aura = cells.get(index, (0, 0))
        tiles.append(
            gm.Tile(
                state=TileState.BLANK,
                bonus_level=bonus,
                aura_amount=aura,
                aura_level=aura,
            )
        )
    return gm.PuzzleState(tiles=tuple(tiles))


class CarryOverTest(unittest.TestCase):
    def test_without_previous_nothing_is_carried(self) -> None:
        """老路径一字不改：加成为 0，等级就是观测到的等级。"""
        board = _board({8: _tile(8, TileState.PAINT, category=PaintCategory.MONSTER, level=4)})
        state = gm.state_from_observation(board)
        tile = state.tile(1, 2)
        self.assertEqual(gm.tile_bonus_level(tile), 0)
        self.assertEqual(gm.tile_aura_level(tile), 0)
        self.assertEqual(tile.paint.level, 4)
        self.assertEqual(gm.tile_final_level(tile), 4)

    def test_bonus_follows_the_cell_not_the_paint(self) -> None:
        """buff 挂在地块上：颜料没了（失色）、格子空了（空白），加成还在原地。"""
        previous = _state_with_buffs({8: (3, 0), 9: (2, 0)})
        board = _board(
            {
                8: _tile(8, TileState.FADED),
                9: _tile(9, TileState.BLANK),
            }
        )
        state = gm.state_from_observation(board, previous=previous)
        self.assertEqual(gm.tile_bonus_level(state.tile(1, 2)), 3)
        self.assertEqual(gm.tile_bonus_level(state.tile(1, 3)), 2)

    def test_cells_without_history_stay_clean(self) -> None:
        previous = _state_with_buffs({8: (3, 0)})
        state = gm.state_from_observation(_board({}), previous=previous)
        self.assertEqual(gm.tile_bonus_level(state.tile(0, 0)), 0)


class RebaseObservedLevelTest(unittest.TestCase):
    def test_observed_final_level_is_split_into_base_plus_bonus(self) -> None:
        """角标是最终等级：Lv4 落在 +3 地块上，颜料自身是 1 级。"""
        previous = _state_with_buffs({8: (3, 0)})
        board = _board({8: _tile(8, TileState.PAINT, category=PaintCategory.MONSTER, level=4)})
        state = gm.state_from_observation(board, previous=previous)
        tile = state.tile(1, 2)
        self.assertEqual(tile.paint.level, 1, "颜料自身等级 = 最终 4 - 加成 3")
        self.assertEqual(gm.tile_bonus_level(tile), 3)
        self.assertEqual(gm.tile_final_level(tile), 4, "加起来必须还原成屏幕上看到的 4")

    def test_capped_level_is_left_alone(self) -> None:
        """7 级是封顶值，反推不唯一（可能是 4+3，也可能是 5+2），保持原样。"""
        previous = _state_with_buffs({8: (3, 0)})
        board = _board({8: _tile(8, TileState.PAINT, category=PaintCategory.MONSTER, level=7)})
        state = gm.state_from_observation(board, previous=previous)
        tile = state.tile(1, 2)
        self.assertEqual(tile.paint.level, 7)
        self.assertEqual(gm.tile_bonus_level(tile), 3)

    def test_impossible_buff_is_dropped_instead_of_making_level_zero(self) -> None:
        """观测只有 2 级却记着 +3：模型明显多算了，就地清掉而不是算出 -1 级。"""
        previous = _state_with_buffs({8: (3, 0)})
        board = _board({8: _tile(8, TileState.PAINT, category=PaintCategory.MONSTER, level=2)})
        state = gm.state_from_observation(board, previous=previous)
        tile = state.tile(1, 2)
        self.assertEqual(gm.tile_bonus_level(tile), 0)
        self.assertEqual(tile.paint.level, 2)
        self.assertEqual(gm.tile_final_level(tile), 2)

    def test_anchor_uses_the_panel_level_which_is_already_the_base(self) -> None:
        """锚点等级来自「当前颜料」面板（颜料自己的等级），不能再减一次。"""
        previous = _state_with_buffs({8: (3, 0)})
        board = _board(
            {8: _tile(8, TileState.PAINT, category=PaintCategory.MONSTER, level=4, anchor=True)}
        )
        current = _current_paint(level=4)
        state = gm.state_from_observation(board, current, (), previous=previous)
        tile = state.tile(1, 2)
        self.assertEqual(tile.paint.level, 4)
        self.assertEqual(gm.tile_final_level(tile), 7, "面板 4 + 地块 3，封顶 7")


class AuraCarryTest(unittest.TestCase):
    def test_aura_anchor_survives_the_rebuild(self) -> None:
        """光环记在锚点地块的 aura_amount 上，重建后要重新算到同种颜料身上。"""
        previous = _state_with_buffs({0: (0, 3)})
        board = _board(
            {
                0: _tile(
                    0, TileState.PAINT, category=PaintCategory.MONSTER, level=4, subtype="邪影", anchor=True
                ),
                7: _tile(7, TileState.PAINT, category=PaintCategory.MONSTER, level=1, subtype="邪影"),
            }
        )
        state = gm.state_from_observation(board, _current_paint(level=1), (), previous=previous)
        self.assertEqual(state.tile(0, 0).aura_amount, 3)
        self.assertEqual(gm.tile_aura_level(state.tile(1, 1)), 3, "同种颜料吃到光环")
        self.assertEqual(gm.tile_final_level(state.tile(1, 1)), 4, "1 + 3")

    def test_aura_does_not_leak_to_other_kinds(self) -> None:
        previous = _state_with_buffs({0: (0, 3)})
        board = _board(
            {
                0: _tile(
                    0, TileState.PAINT, category=PaintCategory.MONSTER, level=4, subtype="邪影", anchor=True
                ),
                7: _tile(7, TileState.PAINT, category=PaintCategory.GAMEPLAY, level=2),
            }
        )
        state = gm.state_from_observation(board, _current_paint(level=1), (), previous=previous)
        self.assertEqual(gm.tile_aura_level(state.tile(1, 1)), 0)
        self.assertEqual(gm.tile_final_level(state.tile(1, 1)), 2)


class SnapshotBackfillTest(unittest.TestCase):
    def _scene(self):
        from huahua_assistant.models import SceneObservation

        board = _board(
            {
                8: _tile(8, TileState.PAINT, category=PaintCategory.MONSTER, level=4),
                9: _tile(9, TileState.BLANK),
            }
        )
        return SceneObservation(board=board, regions=())

    def test_without_model_every_buff_is_zero(self) -> None:
        snapshot = snapshot_from_scene(self._scene())
        self.assertIsNotNone(snapshot)
        self.assertTrue(all(tile.bonus_level == 0 for tile in snapshot.tiles))
        self.assertTrue(all(tile.aura_level == 0 for tile in snapshot.tiles))

    def test_model_state_is_backfilled_into_the_snapshot(self) -> None:
        model = _state_with_buffs({8: (3, 1), 9: (2, 0)})
        snapshot = snapshot_from_scene(self._scene(), model)
        self.assertIsNotNone(snapshot)
        by_cell = {(tile.row, tile.col): tile for tile in snapshot.tiles}
        self.assertEqual((by_cell[(1, 2)].bonus_level, by_cell[(1, 2)].aura_level), (3, 1))
        self.assertEqual((by_cell[(1, 3)].bonus_level, by_cell[(1, 3)].aura_level), (2, 0))
        self.assertEqual((by_cell[(0, 0)].bonus_level, by_cell[(0, 0)].aura_level), (0, 0))
        self.assertEqual(by_cell[(1, 2)].paint_level, 4, "视觉读到的最终等级不变")

    def test_plain_mapping_is_accepted_too(self) -> None:
        snapshot = snapshot_from_scene(self._scene(), {(1, 2): (5, 0)})
        by_cell = {(tile.row, tile.col): tile for tile in snapshot.tiles}
        self.assertEqual(by_cell[(1, 2)].bonus_level, 5)


class RecorderBuffEventTest(unittest.TestCase):
    def test_buff_only_change_emits_its_own_event(self) -> None:
        """画面一格没动、只有加成变了，也要在日志里留痕。"""
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        base = _snapshot_from_model(_state_with_buffs({}), {8: 3})
        buffed = _snapshot_from_model(_state_with_buffs({8: (3, 0)}), {8: 3})

        recorder.process(base, observed_at=100.0)
        recorder.process(base, observed_at=100.1)
        result = recorder.process(buffed, observed_at=100.2)
        self.assertIn("tile_buff_changed", result.events)

        event = next(
            item for item in recorder._events if item["event"] == "tile_buff_changed"
        )
        self.assertEqual(event["payload"]["count"], 1)
        self.assertEqual(event["payload"]["bonus_delta"], 3)

    def test_board_payload_lists_buffed_cells(self) -> None:
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        snapshot = _snapshot_from_model(_state_with_buffs({8: (3, 2)}), {8: 3})
        recorder.process(snapshot, observed_at=100.0)
        board = recorder.initial_board
        self.assertEqual(
            [(item["row"], item["col"], item["bonus_level"], item["aura_level"]) for item in board["buffs"]],
            [(1, 2, 3, 2)],
        )


def _current_paint(level: int):
    from huahua_assistant.models import CurrentPaintObservation, PaintDirection, PaintObject

    return CurrentPaintObservation(
        panel_rect=Rect(0, 0, 100, 30),
        icon_rect=Rect(0, 0, 20, 20),
        text_rect=Rect(20, 0, 80, 20),
        category=PaintCategory.MONSTER,
        category_confidence=0.9,
        level=level,
        level_confidence=0.9,
        subtype="邪影",
        is_treasure=False,
        direction=PaintDirection.DIAGONAL,
        target_object=PaintObject.MONSTER,
        upgrade_amount=1,
        match_current=0,
        match_required=3,
        keyword_scores={},
        raw_text_hint="",
    )


def _snapshot_from_model(model, levels: dict[int, int]) -> FastVisionSnapshot:
    """按 ``levels`` 造一帧观测：格子 index → 屏幕上的最终等级。"""
    from huahua_assistant.models import SceneObservation

    overrides = {
        index: _tile(index, TileState.PAINT, category=PaintCategory.MONSTER, level=level)
        for index, level in levels.items()
    }
    board = _board(overrides)
    # 带一个「当前颜料」面板，录制器才会认成对局进行中并开一场。
    scene = SceneObservation(
        board=board,
        regions=(),
        current_paint=_current_paint(level=3),
    )
    snapshot = snapshot_from_scene(scene, model)
    assert snapshot is not None
    return snapshot


if __name__ == "__main__":
    unittest.main()
