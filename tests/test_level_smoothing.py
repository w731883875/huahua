"""棋盘等级时序稳定 + 结算后冻结。

用 2026-11-22 那局实测到的抖动序列做样本：点击地块看加成时浮窗遮挡，
读数在 1↔7、3↔6 之间反复跳。
"""
from __future__ import annotations

import unittest

from huahua_assistant.level_smoothing import LevelStabilizer
from huahua_assistant.models import (
    BoardObservation,
    PaintCategory,
    Rect,
    TileObservation,
    TileState,
)


def _tile(row: int, col: int, level: int | None, *, subtype: str = "梦魇") -> TileObservation:
    return TileObservation(
        row=row,
        col=col,
        rect=Rect(0, 0, 10, 10),
        state=TileState.PAINT,
        confidence=0.9,
        features={},
        paint_category=PaintCategory.GAMEPLAY if level is not None else None,
        paint_category_confidence=0.9,
        paint_level=level,
        paint_level_confidence=0.9,
        paint_subtype=subtype if level is not None else None,
        paint_is_treasure=False,
        is_scope_highlight=False,
        scope_highlight_confidence=0.0,
        is_current_anchor=False,
        current_anchor_confidence=0.0,
    )


def _board(*tiles: TileObservation) -> BoardObservation:
    return BoardObservation(rect=Rect(0, 0, 60, 60), tiles=tuple(tiles), score=0.9)


def _level_of(board: BoardObservation, row: int, col: int) -> int | None:
    for tile in board.tiles:
        if (tile.row, tile.col) == (row, col):
            return tile.paint_level
    raise AssertionError("格子不在盘上")


class StabilizerTests(unittest.TestCase):
    def test_first_sighting_is_accepted_immediately(self) -> None:
        """首帧没有基准，必须直接采纳，否则开局全是空盘。"""

        stab = LevelStabilizer()
        out = stab.observe(_board(_tile(4, 4, 7)))

        self.assertEqual(7, _level_of(out, 4, 4))

    def test_single_frame_jitter_is_ignored(self) -> None:
        """实测序列 [7, 1, 7]：中间那个 1 是遮挡误读，不能采纳。"""

        stab = LevelStabilizer()
        stab.observe(_board(_tile(4, 4, 7)))
        jittered = stab.observe(_board(_tile(4, 4, 1)))
        recovered = stab.observe(_board(_tile(4, 4, 7)))

        self.assertEqual(7, _level_of(jittered, 4, 4), "单帧遮挡不该把 7 改成 1")
        self.assertEqual(7, _level_of(recovered, 4, 4))

    def test_sustained_change_is_accepted_after_enough_frames(self) -> None:
        """真实变化（落子升级 3→6）连续两帧后必须采纳。"""

        stab = LevelStabilizer()
        stab.observe(_board(_tile(4, 4, 3)))
        first = stab.observe(_board(_tile(4, 4, 6)))
        second = stab.observe(_board(_tile(4, 4, 6)))

        self.assertEqual(3, _level_of(first, 4, 4), "第一帧先别急着改")
        self.assertEqual(6, _level_of(second, 4, 4), "连续两帧一致就采纳")

    def test_alternating_reads_never_settle_on_a_wrong_value(self) -> None:
        """实测 [3,1,7,1,7,1,7]：来回跳的读数不能把基准带跑。"""

        stab = LevelStabilizer()
        stab.observe(_board(_tile(5, 2, 3)))
        for level in (1, 7, 1, 7, 1, 7):
            stab.observe(_board(_tile(5, 2, level)))

        self.assertEqual(3, stab.committed_level(5, 2))

    def test_unread_frame_does_not_count_as_change(self) -> None:
        """首帧没读出内容（paint=None）不该被当成变化或新落笔。"""

        stab = LevelStabilizer()
        stab.observe(_board(_tile(0, 5, 6)))
        blank = stab.observe(_board(_tile(0, 5, None)))
        again = stab.observe(_board(_tile(0, 5, 6)))

        self.assertIsNone(_level_of(blank, 0, 5))
        self.assertEqual(6, _level_of(again, 0, 5))
        self.assertEqual(6, stab.committed_level(0, 5))

    def test_subtype_change_also_needs_confirmation(self) -> None:
        """种类变化同样要连续两帧：覆盖笔触会让种类跳变，但抖动也会。"""

        stab = LevelStabilizer()
        stab.observe(_board(_tile(3, 3, 7, subtype="梦魇")))
        out = stab.observe(_board(_tile(3, 3, 7, subtype="魔方")))

        self.assertEqual("梦魇", out.tiles[0].paint_subtype)

    def test_frozen_blocks_level_and_kind_changes(self) -> None:
        """结算后等级和种类不再变化——用户 2026-11-22 明确要求。"""

        stab = LevelStabilizer(stable_frames=1)
        stab.observe(_board(_tile(4, 3, 7)))
        stab.freeze()

        out = stab.observe(_board(_tile(4, 3, 2, subtype="黑潮")))

        self.assertEqual(7, _level_of(out, 4, 3))
        self.assertEqual("梦魇", out.tiles[0].paint_subtype)

    def test_reset_clears_history_and_unfreezes(self) -> None:
        stab = LevelStabilizer(stable_frames=1)
        stab.observe(_board(_tile(4, 3, 7)))
        stab.freeze()
        stab.reset()

        self.assertFalse(stab.frozen)
        stab.observe(_board(_tile(4, 3, 2)))
        self.assertEqual(2, stab.committed_level(4, 3))


if __name__ == "__main__":
    unittest.main()
