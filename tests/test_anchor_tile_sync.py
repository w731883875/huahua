"""「盘上唯一一枚颜料」不能被初始颜料面板无条件覆盖（2026-11-15 实机事故）。

现象：新画纸开局自带一枚 **赛季 Lv4**，玩家反馈「4 级赛季未识别成功」。
抓帧逐层拆下去，视觉层其实是对的：

* ``frame_level.read_frame_level`` → ``level=4, confidence=0.88``（紫框就是 Lv4）；
* ``paint_kind.read_kind`` → ``赛季 0.889``；
* ``_detect_board_tile_level`` → ``(4, 0.70)``。

真正把它改坏的是 ``_sync_single_anchor_tile_from_current_paint``：它假设
「盘上只有一枚颜料 == 这枚就是本局的初始颜料」，于是拿面板读数（当时是
``monster Lv3``）把整格盖掉——等级变 3，连家族都被写成 ``None``。

修法：没有锚点高亮、而这一格自己认出了家族、家族类别又和面板规则类别冲突时，
它们是两枚不同的颜料，面板没有资格覆盖。同时「面板没读出家族」时不要抹掉
棋盘已认出的家族。

2026-11-15 下午复现（同一局的后续帧），发现上面这条还漏了两处：

* ``paint_indices = marked_indices or [...]`` 一旦判出锚点就只看那一格，
  「盘上只有一枚颜料」这个前提被整体绕过 —— 盘上 5 枚颜料时照样拿面板盖；
* 判据只比**类别**。赛季与机械之神同属玩法类别，类别一样就放过了，
  真凶恰恰是这一对。

补的三道闸：盘上多枚颜料整段不动、面板读出的家族与棋盘不同就拒绝、
左上角带感叹号的初始颜料格不许被改（见 ``INITIAL_MARKER_*``）。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from dataclasses import replace

from huahua_assistant import vision
from huahua_assistant.models import (
    BoardObservation,
    CurrentPaintObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    Rect,
    TileObservation,
    TileState,
)


def _panel(
    *,
    category: PaintCategory = PaintCategory.MONSTER,
    level: int = 3,
    subtype: str | None = None,
    is_treasure: bool = False,
) -> CurrentPaintObservation:
    return CurrentPaintObservation(
        panel_rect=Rect(1400, 400, 400, 300),
        icon_rect=Rect(1420, 420, 80, 80),
        text_rect=Rect(1420, 520, 360, 160),
        category=category,
        category_confidence=0.78,
        subtype=subtype,
        is_treasure=is_treasure,
        level=level,
        level_confidence=0.92,
        match_current=0,
        match_required=4,
        target_object=PaintObject.MONSTER if category == PaintCategory.MONSTER else PaintObject.GAMEPLAY,
        direction=PaintDirection.ADJACENT,
        upgrade_amount=2,
        keyword_scores={},
        raw_text_hint="",
    )


def _board(
    *,
    subtype: str | None,
    level: int | None,
    category: PaintCategory = PaintCategory.GAMEPLAY,
    anchor: bool = False,
    level_confidence: float = 0.70,
) -> BoardObservation:
    """36 格棋盘，只有 (2, 2) 是颜料——也就是「盘上唯一一枚」。"""
    tiles: list[TileObservation] = []
    for index in range(36):
        row, col = divmod(index, 6)
        rect = Rect(col * 100, row * 100, 100, 100)
        if (row, col) == (2, 2):
            tiles.append(
                TileObservation(
                    row=row,
                    col=col,
                    rect=rect,
                    state=TileState.PAINT,
                    confidence=0.9,
                    features={},
                    paint_category=category,
                    paint_category_confidence=0.889,
                    paint_level=level,
                    paint_level_confidence=level_confidence,
                    paint_subtype=subtype,
                    paint_is_treasure=subtype == "宝箱",
                    is_current_anchor=anchor,
                    current_anchor_confidence=0.42 if anchor else 0.0,
                )
            )
        else:
            tiles.append(
                TileObservation(
                    row=row,
                    col=col,
                    rect=rect,
                    state=TileState.BLANK,
                    confidence=0.9,
                    features={},
                )
            )
    return BoardObservation(rect=Rect(0, 0, 600, 600), score=0.83, tiles=tuple(tiles))


class SinglePaintSyncTests(unittest.TestCase):
    def test_conflicting_family_is_not_overwritten(self) -> None:
        """赛季是玩法颜料，面板规则是怪物 → 不是同一枚，不许覆盖。"""
        board = _board(subtype="赛季", level=4)
        synced = vision._sync_single_anchor_tile_from_current_paint(board, _panel())
        tile = synced.tile(2, 2)
        self.assertEqual(tile.paint_level, 4)
        self.assertEqual(tile.paint_subtype, "赛季")
        self.assertEqual(tile.paint_category, PaintCategory.GAMEPLAY)

    def test_unknown_family_is_still_calibrated_by_panel(self) -> None:
        """认不出家族的格子仍走老路——这条是那个函数存在的理由。"""
        board = _board(subtype=None, level=1, category=PaintCategory.MONSTER)
        synced = vision._sync_single_anchor_tile_from_current_paint(board, _panel(level=3))
        tile = synced.tile(2, 2)
        self.assertEqual(tile.paint_level, 3)
        self.assertEqual(tile.paint_category, PaintCategory.MONSTER)

    def test_same_category_family_is_calibrated(self) -> None:
        """同一类别（伊吉 vs 面板怪物）时仍用面板校准等级：可能只是边框读歪了。"""
        board = _board(subtype="伊吉", level=1, category=PaintCategory.MONSTER)
        synced = vision._sync_single_anchor_tile_from_current_paint(board, _panel(level=7))
        tile = synced.tile(2, 2)
        self.assertEqual(tile.paint_level, 7)
        self.assertEqual(tile.paint_subtype, "伊吉")

    def test_anchor_tile_still_follows_panel(self) -> None:
        """真锚点（金色高亮）无条件采信面板——它本来就是初始颜料那一格。"""
        board = _board(
            subtype="赛季",
            level=4,
            category=PaintCategory.GAMEPLAY,
            anchor=True,
        )
        synced = vision._sync_single_anchor_tile_from_current_paint(board, _panel(level=5))
        tile = synced.tile(2, 2)
        self.assertEqual(tile.paint_level, 5)
        self.assertEqual(tile.paint_category, PaintCategory.MONSTER)

    def test_panel_without_family_keeps_board_family(self) -> None:
        """面板读不出家族时，别把棋盘认出来的家族写成空。"""
        board = _board(subtype="宝箱", level=5, category=PaintCategory.GAMEPLAY)
        synced = vision._sync_single_anchor_tile_from_current_paint(
            board, _panel(category=PaintCategory.GAMEPLAY, level=5, subtype=None)
        )
        tile = synced.tile(2, 2)
        self.assertEqual(tile.paint_subtype, "宝箱")
        self.assertTrue(tile.paint_is_treasure)

    def test_solid_frame_level_is_not_overwritten_even_when_category_matches(self) -> None:
        """边框色读得够硬（0.88）时，同类别也不拿面板改等级。

        赛季 Lv4 实测 ``frame_level`` 就是 ``level=4, confidence=0.88``；
        如果面板规则恰好也是玩法颜料（类别一致），只看类别就会再把等级改错。
        """
        board = _board(subtype="赛季", level=4, level_confidence=0.88)
        synced = vision._sync_single_anchor_tile_from_current_paint(
            board, _panel(category=PaintCategory.GAMEPLAY, level=2)
        )
        tile = synced.tile(2, 2)
        self.assertEqual(tile.paint_level, 4)
        self.assertEqual(tile.paint_level_confidence, 0.88)

    def test_weak_frame_level_still_falls_back_to_panel(self) -> None:
        """边框勉强能读（0.70）才退回面板等级。"""
        board = _board(subtype="伊吉", level=1, category=PaintCategory.MONSTER, level_confidence=0.70)
        synced = vision._sync_single_anchor_tile_from_current_paint(
            board, _panel(category=PaintCategory.MONSTER, level=6)
        )
        self.assertEqual(synced.tile(2, 2).paint_level, 6)

    def test_multiple_paints_are_left_alone(self) -> None:
        """盘上不止一枚颜料时这个函数本来就不该动。"""
        board = _board(subtype="赛季", level=4)
        tiles = list(board.tiles)
        tiles[0] = TileObservation(
            row=0,
            col=0,
            rect=Rect(0, 0, 100, 100),
            state=TileState.PAINT,
            confidence=0.9,
            features={},
            paint_subtype="伊吉",
        )
        board = BoardObservation(rect=board.rect, score=board.score, tiles=tuple(tiles))
        synced = vision._sync_single_anchor_tile_from_current_paint(board, _panel())
        self.assertEqual(synced.tile(2, 2).paint_level, 4)

class InitialMarkerTests(unittest.TestCase):
    """左上角感叹号 = 初始颜料的静态标记，与金框无关。"""

    FIXTURES = Path(__file__).resolve().parent / "fixtures"

    def _tile(self, name: str):
        import cv2
        import numpy as np

        path = self.FIXTURES / name
        image = cv2.imread(str(path)) if path.exists() else None
        if image is None:
            self.skipTest(f"缺固件 {path}")
        return image

    def test_stamped_initial_paint_is_detected(self) -> None:
        score = vision._tile_initial_marker_score(self._tile("initial_marker_tile.png"))
        self.assertEqual(score, 1.0, "带感叹号的初始颜料格没被识别出来")

    def test_plain_paint_cells_have_no_marker(self) -> None:
        for name in ("plain_paint_tile.png", "plain_paint_tile2.png"):
            with self.subTest(name=name):
                score = vision._tile_initial_marker_score(self._tile(name))
                self.assertEqual(score, 0.0, f"{name} 被误判成带感叹号")

    def test_marker_is_recorded_in_features(self) -> None:
        tile = vision._classify_tile_observation(
            self._tile("initial_marker_tile.png"), 2, 2, Rect(0, 0, 74, 74)
        )
        self.assertGreaterEqual(tile.features.get("initial_marker_score", 0.0), 0.5)
        self.assertIn("initial_marker_score", tile.features)


class PanelMustNotOverwriteExistingPaintTests(unittest.TestCase):
    """面板那枚还没落时，不许改盘上已有的颜料。"""

    def _paint(self, row: int, col: int, subtype: str, level: int, category: PaintCategory, markers: bool = False):
        return TileObservation(
            row=row,
            col=col,
            rect=Rect(col * 100, row * 100, 100, 100),
            state=TileState.PAINT,
            confidence=0.9,
            features={"initial_marker_score": 1.0} if markers else {},
            paint_category=category,
            paint_category_confidence=0.889,
            paint_level=level,
            paint_level_confidence=0.70,
            paint_subtype=subtype,
            paint_is_treasure=False,
        )

    def _board(self, paints: dict[tuple[int, int], TileObservation]) -> BoardObservation:
        tiles: list[TileObservation] = []
        for index in range(36):
            row, col = divmod(index, 6)
            tiles.append(paints.get((row, col)) or TileObservation(
                row=row, col=col, rect=Rect(col * 100, row * 100, 100, 100),
                state=TileState.BLANK, confidence=0.9, features={},
            ))
        return BoardObservation(rect=Rect(0, 0, 600, 600), score=0.888, tiles=tuple(tiles))

    def _mechanical_god_panel(self) -> CurrentPaintObservation:
        return CurrentPaintObservation(
            panel_rect=Rect(1400, 400, 400, 300),
            icon_rect=Rect(1420, 420, 80, 80),
            text_rect=Rect(1420, 520, 360, 160),
            category=PaintCategory.GAMEPLAY,
            category_confidence=0.80,
            subtype="机械之神",
            is_treasure=False,
            level=3,
            level_confidence=0.90,
            match_current=0,
            match_required=3,
            target_object=PaintObject.GAMEPLAY,
            direction=PaintDirection.VERTICAL,
            upgrade_amount=1,
            keyword_scores={},
            raw_text_hint="",
        )

    def test_full_board_with_anchor_is_left_alone(self) -> None:
        """实机复现：盘上 5 枚 + 锚点在 r2c2 + 面板「机械之神 Lv3」。"""
        board = self._board({
            (2, 2): replace(
                self._paint(2, 2, "赛季", 4, PaintCategory.GAMEPLAY, markers=True),
                is_current_anchor=True,
                current_anchor_confidence=0.42,
            ),
            (2, 1): self._paint(2, 1, "英灵", 4, PaintCategory.MONSTER),
            (2, 3): self._paint(2, 3, "鼠人", 6, PaintCategory.MONSTER),
            (1, 2): self._paint(1, 2, "邪影", 4, PaintCategory.MONSTER),
            (3, 2): self._paint(3, 2, "幽灵", 4, PaintCategory.MONSTER),
        })
        tile = vision._sync_single_anchor_tile_from_current_paint(
            board, self._mechanical_god_panel()
        ).tile(2, 2)
        self.assertEqual(tile.paint_subtype, "赛季")
        self.assertEqual(tile.paint_level, 4)
        self.assertEqual(tile.paint_category, PaintCategory.GAMEPLAY)

    def test_full_board_is_left_alone_even_when_family_matches(self) -> None:
        """只锁「盘上多枚颜料」这一道闸：面板家族故意设成与棋盘相同。

        上一版用例里面板是「机械之神」，家族冲突那道闸也会拦住它，
        于是删掉「多枚颜料」这道闸负例仍然绿——断言等于没锁。
        """
        board = self._board({
            (2, 2): replace(
                self._paint(2, 2, "赛季", 4, PaintCategory.GAMEPLAY, markers=True),
                is_current_anchor=True,
                current_anchor_confidence=0.42,
            ),
            (2, 1): self._paint(2, 1, "英灵", 4, PaintCategory.MONSTER),
            (2, 3): self._paint(2, 3, "鼠人", 6, PaintCategory.MONSTER),
        })
        panel = replace(self._mechanical_god_panel(), subtype="赛季", level=1)
        tile = vision._sync_single_anchor_tile_from_current_paint(board, panel).tile(2, 2)
        self.assertEqual(tile.paint_level, 4, "盘上已有 3 枚颜料时，面板不该改任何一格")
        self.assertEqual(tile.paint_subtype, "赛季")

    def test_same_category_other_family_is_not_overwritten(self) -> None:
        """赛季 vs 机械之神同属玩法类别，但家族不同就是两枚颜料。"""
        board = self._board({
            (2, 2): replace(
                self._paint(2, 2, "赛季", 4, PaintCategory.GAMEPLAY, markers=True),
                is_current_anchor=True,
                current_anchor_confidence=0.42,
            ),
        })
        tile = vision._sync_single_anchor_tile_from_current_paint(
            board, self._mechanical_god_panel()
        ).tile(2, 2)
        self.assertEqual(tile.paint_subtype, "赛季")
        self.assertEqual(tile.paint_level, 4)

    def test_panel_without_family_still_calibrates_level(self) -> None:
        """面板没读出家族时，这条校准路径仍然有效（函数存在的理由）。"""
        board = self._board({
            (2, 2): self._paint(2, 2, "赛季", 4, PaintCategory.GAMEPLAY, markers=True),
        })
        panel = replace(self._mechanical_god_panel(), subtype=None, category=PaintCategory.GAMEPLAY, level=5)
        tile = vision._sync_single_anchor_tile_from_current_paint(board, panel).tile(2, 2)
        self.assertEqual(tile.paint_level, 5)
        self.assertEqual(tile.paint_subtype, "赛季")


if __name__ == "__main__":
    unittest.main()
