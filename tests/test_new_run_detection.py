"""新一局要能自己开起来（2026-11-14 实机事故，2026-11-15 补齐判据）。

现象：一局画满之后，玩家已经点「入画」开了新画纸并落了好几手，
软件还挂着上一局的「初始颜料 Lv3 怪物｜目标 横向 所有 (3/4)」，
状态灯停在「本局已收尾」。

实机取证先把两个画面分清（这一步以前搞混过）：

* **终局/入画界面**：棋盘仍是画满的样子（19 枚颜料 + 17 个不可放置格，blank=0），
  右侧 ``完成地块进度 0/11``，右下角蓝色「入画」。**它不是新画纸**，
  只是上一局的棋盘还留在画布上——实测与终局帧的颜料位置逐格相同。
* **按下「入画」真正发牌**：实测 22:40 开局 `blank=12 / paint=7 / void=13 / blocked=4`。

三个独立缺陷叠在一起：

1. **识别层**：深蓝空板没有角标、没有家族、没有等级，但「背包框形状」分类器
   看它是蓝色框，给了 ``monster 0.78``（刚好越过 0.62），于是终局盘 17 格空板
   全变成「怪物颜料」，整盘 blank/void 归零；
2. **新局判据**：旧实现要求 ``paint <= 8``，而画纸开局自带一整盘 1~2 级颜料，
   有的局开局就有 12 枚，这条直接不成立；
3. **触发条件**：``_looks_like_new_run`` 只挂在 ``real_session_finished`` 上，
   而那个标志要玩家真去点「结束本局」才会置位——画满就入画的人永远等不到重置。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from huahua_assistant import capture, vision
from huahua_assistant.app import _looks_like_new_run, _scene_has_finish_evidence
from huahua_assistant.models import (
    BoardObservation,
    Rect,
    RegionKind,
    RegionObservation,
    SceneObservation,
    TileObservation,
    TileState,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: 终局/入画界面（实机 23:40 满铺）：void=13、blocked=4、paint=19、blank=0。
#: **不是新画纸**——按下「入画」之前画布上留着的还是上一局画完的棋盘。
ENTRY_SCREEN_STATES = "VVPPPVVPVPPVKPPKPVVPPKPPVVPPPPVPPVKV"
#: 真正发过牌的新画纸（实机 22:40 开局）：blank=12、paint=7、void=13、blocked=4。
NEW_DEAL_STATES = "VVBBBVVBVBBVKBBKBVVPPKPPVVBPPPVBBVKV"
#: 同一张画纸的另一种开局（发牌多）：void/blocked 布局与终局盘一模一样，
#: blank=7、paint=12 —— 专门用来证明「布局重排」判据接不住时新判据能接住。
SAME_LAYOUT_DEAL_STATES = "VVBBBVVBVBBVKBPKPVVPPKPPVVPPPPVPPVKV"
#: 换了一张画纸的终局盘：void 布局整体不同，但同样满铺（ok 用来守住第 3 条判据）。
OTHER_PAPER_TERMINAL_STATES = "PPPPPVVPVPPVVPPVPVVPPVPPVVPPPPVPPVVV"


def _board_payload(states: str) -> dict:
    """按 states 串还原出 ``_compact_board_summary`` 的形态。"""
    counted = {code: states.count(code) for code in set(states)}
    return {
        "states": states,
        "blank": counted.get("B", 0),
        "blocked": counted.get("K", 0),
        "void": counted.get("V", 0),
        "faded": counted.get("F", 0),
        "paint": counted.get("P", 0),
        "score": 0.8,
    }


def _states(*, void: int, blocked: int, blank: int, paint: int) -> str:
    """按数量拼一个 36 格 states 串（VOID/BLOCKED 都排在前面，位置固定）。"""
    assert void + blocked + blank + paint == 36, (void, blocked, blank, paint)
    return "V" * void + "K" * blocked + "B" * blank + "P" * paint


def _scene(states: str, regions: tuple[RegionObservation, ...] = ()) -> SceneObservation:
    mapping = {
        "V": TileState.VOID,
        "P": TileState.PAINT,
        "K": TileState.BLOCKED,
        "B": TileState.BLANK,
        "F": TileState.FADED,
    }
    tiles = tuple(
        TileObservation(
            row=index // 6,
            col=index % 6,
            rect=Rect((index % 6) * 10, (index // 6) * 10, 10, 10),
            state=mapping.get(code, TileState.UNKNOWN),
            confidence=0.9,
            features={},
        )
        for index, code in enumerate(states)
    )
    return SceneObservation(
        board=BoardObservation(rect=Rect(0, 0, 60, 60), score=0.8, tiles=tiles),
        regions=regions,
    )


class EmptyBoardTileTests(unittest.TestCase):
    """入画阶段的深蓝空板不是颜料。"""

    def test_empty_board_tile_is_not_paint(self) -> None:
        tile = capture.read_image(FIXTURES / "new_run_empty_tile.png")
        state, confidence, features = vision.classify_tile(tile)
        self.assertNotEqual(state, TileState.PAINT, f"深蓝空板被认成颜料（置信度 {confidence:.2f}）")
        self.assertEqual(state, TileState.VOID)
        # 空板的关键特征：整格几乎全是低亮度蓝，没有任何亮部。
        self.assertGreater(float(features["blue_ratio"]), 0.90)
        self.assertLess(float(features["bright_ratio"]), 0.05)

    def test_paint_tiles_are_still_paint(self) -> None:
        """别为了拦空板把真颜料一起拦掉——这两格来自同一帧。"""
        for name in ("new_run_paint_tile.png", "new_run_paint_tile2.png"):
            with self.subTest(fixture=name):
                tile = capture.read_image(FIXTURES / name)
                state, _confidence, _features = vision.classify_tile(tile)
                self.assertEqual(state, TileState.PAINT)


class NewRunDetectionTests(unittest.TestCase):
    def test_terminal_board_reopening_is_a_new_run(self) -> None:
        """满铺之后又出现空位：只有换画纸这一种解释。"""
        previous = {"board": _board_payload("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")}
        scene = _scene("VVBBVVPPVVBBVVPPVVBBVVPPVVBBVVPPVVBB")
        self.assertTrue(_looks_like_new_run(scene, previous))

    def test_new_deal_with_many_paints_is_a_new_run(self) -> None:
        """发牌后还剩 12 枚颜料 —— ``paint<=8`` 接不住，必须靠「空位回升 + 颜料减少」。

        这一条是真实场景：画纸开局自带多少枚颜料取决于画纸，有的局只发 7 枚，
        有的局发 12 枚。VOID 布局与终局盘完全一致（同一张画纸），所以「布局重排」
        也接不住，能认出来的只有「blank 0→7、paint 19→12」这个事实。
        """
        payload = _board_payload(SAME_LAYOUT_DEAL_STATES)
        self.assertEqual((payload["blank"], payload["paint"]), (7, 12))
        previous = {"board": _board_payload(ENTRY_SCREEN_STATES)}
        self.assertTrue(_looks_like_new_run(_scene(SAME_LAYOUT_DEAL_STATES), previous))

    def test_real_new_deal_is_a_new_run(self) -> None:
        """实机 22:40 真正发牌那一帧：blank=12 / paint=7，与终局盘布局也不同。"""
        previous = {"board": _board_payload(ENTRY_SCREEN_STATES)}
        self.assertTrue(_looks_like_new_run(_scene(NEW_DEAL_STATES), previous))

    def test_void_layout_reshuffle_is_a_new_run(self) -> None:
        """同一张画纸的 VOID 位置是固定的，换纸后重排（两局终局盘对照）。"""
        previous = {"board": _board_payload(ENTRY_SCREEN_STATES)}
        self.assertTrue(_looks_like_new_run(_scene(OTHER_PAPER_TERMINAL_STATES), previous))

    def test_identical_board_is_not_a_new_run(self) -> None:
        previous = {"board": _board_payload(ENTRY_SCREEN_STATES)}
        self.assertFalse(_looks_like_new_run(_scene(ENTRY_SCREEN_STATES), previous))

    def test_single_cell_flicker_is_not_a_new_run(self) -> None:
        """同局里一两格识别抖动不能当成换纸。"""
        states = list(ENTRY_SCREEN_STATES)
        states[31] = "B" if states[31] != "B" else "P"
        previous = {"board": _board_payload(ENTRY_SCREEN_STATES)}
        self.assertFalse(_looks_like_new_run(_scene("".join(states)), previous))

    def test_small_drift_is_not_a_new_run(self) -> None:
        """blank +1 / paint -1 还不够格：两个方向都要 ≥2 才算「重新发了牌」。"""
        previous = {"board": _board_payload(_states(void=13, blocked=4, blank=4, paint=15))}
        scene = _scene(_states(void=13, blocked=4, blank=5, paint=14))
        self.assertFalse(_looks_like_new_run(scene, previous))

    def test_opening_layout_still_recognised(self) -> None:
        """没有终局参照时，空位多 + 颜料少的开局盘仍按新局处理。"""
        previous = {"board": _board_payload("PPPPPPPPVVVVVVVVVVVVVVVVVVVVVVVVVVVV")}
        scene = _scene("BBBBBBBBVVVVVVVVPPVVPPVVVVVVVVVVVVVV")
        self.assertTrue(_looks_like_new_run(scene, previous))

    def test_no_previous_step_is_a_new_run(self) -> None:
        self.assertTrue(_looks_like_new_run(_scene(ENTRY_SCREEN_STATES), None))

    def test_full_board_without_open_cells_is_not_a_new_run(self) -> None:
        """结算画面上棋盘仍然是满的：这时不该重置，等新画纸出现再说。"""
        previous = {"board": _board_payload(ENTRY_SCREEN_STATES)}
        full = "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
        self.assertFalse(_looks_like_new_run(_scene(full), previous))


class EnterPaintingButtonTests(unittest.TestCase):
    """「入画」按钮常驻作画界面，不能当「本局已画完」。

    实机 00:18 重启后第一帧就判 `settlement_candidate`（按钮置信 0.792），随即
    `run_session_finished`，状态灯一直报「本局已收尾」，而玩家明明还在画。
    抓帧确认：``_find_enter_painting_button`` 每帧都命中右下角那颗蓝色按钮
    （x=1263 y=779，置信 0.92），而「完成绘制」按钮此时并不存在。
    """

    def _region(self, label: str, confidence: float = 0.92) -> RegionObservation:
        return RegionObservation(
            kind=RegionKind.FINISH_DRAWING,
            rect=Rect(1263, 779, 293, 53),
            confidence=confidence,
            label=label,
        )

    def test_enter_painting_label_is_recognised_with_fast_suffix(self) -> None:
        # 实时链路的 label 带 " fast" 后缀，按前缀比才不会漏。
        self.assertTrue(self._region("enter painting button fast").is_enter_painting)
        self.assertTrue(self._region("enter painting button").is_enter_painting)
        self.assertFalse(self._region("finish drawing button").is_enter_painting)
        self.assertFalse(self._region("finish ready banner").is_enter_painting)

    def test_enter_painting_button_is_not_finish_evidence(self) -> None:
        scene = _scene(ENTRY_SCREEN_STATES, (self._region("enter painting button fast"),))
        state = _state_with_blank()
        self.assertFalse(_scene_has_finish_evidence(scene, state))

    def test_real_finish_button_is_finish_evidence(self) -> None:
        scene = _scene(ENTRY_SCREEN_STATES, (self._region("finish drawing button", 0.86),))
        self.assertTrue(_scene_has_finish_evidence(scene, _state_with_blank()))


class NewRunTriggerTests(unittest.TestCase):
    """闸门：什么时候才轮到 ``_looks_like_new_run`` 被问。

    ``real_session_finished`` 只有玩家点「结束本局」才会置位，而实际玩法是画满就点
    「入画」开下一张，所以必须靠「本局画满过」这条事实来放行。放行条件写错会出两种事故：
    永远不放行（旧 bug），或者没有参照帧也放行（满铺画面上每帧重置，整局记不下来）。
    """

    def _app(self, *, finished: bool, full: bool, previous: dict | None):
        from huahua_assistant.app import HuahuaApp

        started: list[bool] = []

        class _Fake:
            real_session_finished = finished
            board_reached_full = full
            last_real_capture_step = previous
            last_finished_board_states = ""

            def start_new_real_session(self, record_event: bool = True) -> None:
                started.append(record_event)
                self.board_reached_full = False
                self.last_real_capture_step = None

        fake = _Fake()
        return fake, started, lambda scene: HuahuaApp._maybe_start_new_run_from_board(
            fake, scene, trigger="test"
        )

    def test_full_board_then_deal_starts_new_run(self) -> None:
        """画满过 + 新画纸发牌（blank 0→7、paint 19→12）→ 必须开新局。"""
        previous = {"board": _board_payload(ENTRY_SCREEN_STATES)}
        fake, started, call = self._app(finished=False, full=True, previous=previous)
        with patch("huahua_assistant.app.record_observation"):
            self.assertTrue(call(_scene(SAME_LAYOUT_DEAL_STATES)))
        self.assertEqual(started, [True])
        self.assertFalse(fake.board_reached_full)

    def test_full_seen_this_frame_does_not_reset_immediately(self) -> None:
        """第一次看到满铺只是记下事实，不能同一帧就重置（那会把整局清掉）。"""
        previous = {"board": _board_payload(_states(void=13, blocked=4, blank=0, paint=19))}
        fake, started, call = self._app(finished=False, full=False, previous=previous)
        with patch("huahua_assistant.app.record_observation"):
            self.assertFalse(call(_scene(ENTRY_SCREEN_STATES)))
        self.assertEqual(started, [])
        self.assertTrue(fake.board_reached_full)

    def test_without_previous_frame_never_resets(self) -> None:
        """没有参照帧时 ``_looks_like_new_run`` 恒 True —— 闸门必须挡住它。"""
        fake, started, call = self._app(finished=False, full=True, previous=None)
        with patch("huahua_assistant.app.record_observation"):
            self.assertFalse(call(_scene(ENTRY_SCREEN_STATES)))
        self.assertEqual(started, [])

    def test_finished_session_resets_without_full_board(self) -> None:
        """玩家真去点了「结束本局」：不需要满铺这条旁证。"""
        previous = {"board": _board_payload(_states(void=13, blocked=4, blank=3, paint=16))}
        fake, started, call = self._app(finished=True, full=False, previous=previous)
        with patch("huahua_assistant.app.record_observation"):
            self.assertTrue(call(_scene(NEW_DEAL_STATES)))
        self.assertEqual(started, [True])


def _state_with_blank():
    """棋盘上还有空位 → 结算证据只可能来自按钮。"""
    from huahua_assistant.game_model import BOARD_SIZE, PuzzleState, Tile

    tiles = [Tile(state=TileState.BLANK) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    tiles[0] = Tile(state=TileState.PAINT)
    return PuzzleState(tiles=tuple(tiles))


if __name__ == "__main__":
    unittest.main()
