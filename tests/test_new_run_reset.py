"""换局重置：日志「入画」开新局 + 背包会话键带局号。

这两个 bug 是同一个实机现象的两半——用户重绘开新局后界面还挂着上一局的东西：

1. 旧局状态只有「结算成功」才会清（``real_session_finished``），而重绘根本不
   走结算，于是 ``previous_state`` 一直被继承，调色目标/锚点/背包全是旧的。
2. 背包累积器的会话键只由调色目标构成，新局抽到同样的目标时键不变，上一局的
   颜料继续叠进来，背包越算越多。
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from huahua_assistant import app
from huahua_assistant.game_model import PaintCategory, PaintDirection, PaintObject
from huahua_assistant.level_smoothing import LevelStabilizer
from huahua_assistant.models import BoardObservation, Rect, SceneObservation, TileObservation, TileState


def _scene(direction=PaintDirection.VERTICAL, required=4, target=PaintObject.ALL):
    """造一个只有「当前调色目标」有意义的场景。"""

    paint = SimpleNamespace(
        category=PaintCategory.GAMEPLAY,
        level=3,
        direction=direction,
        target_object=target,
        match_required=required,
    )
    return SimpleNamespace(
        current_paint=paint,
        board=SimpleNamespace(tiles=()),
        # 必须给一条颜料：``_merge_detected_backpack`` 在「背包面板没露出来」时
        # 直接返回缓存，不会调用 update（那正是要区分的行为）。
        backpack=SimpleNamespace(
            paints=[
                SimpleNamespace(
                    category=PaintCategory.MONSTER,
                    level=1,
                    subtype=None,
                    is_treasure=False,
                    category_confidence=0.9,
                    level_confidence=0.9,
                )
            ]
        ),
    )


def _fake_app(session_id: str):
    return SimpleNamespace(
        real_session_id=session_id,
        last_backpack_paint_tiles=None,
        backpack_session_key=None,
        backpack_accumulator=Mock(),
    )


class BackpackSessionKeyTests(unittest.TestCase):
    def test_same_rule_in_a_new_run_still_resets_the_accumulator(self) -> None:
        """新局抽到一模一样的调色目标，也必须重置背包。"""

        scene = _scene()
        fake = _fake_app("run-1")
        app.HuahuaApp._merge_detected_backpack(fake, scene)
        self.assertEqual(1, fake.backpack_accumulator.reset.call_count)
        first_key = fake.backpack_accumulator.reset.call_args.args[0]

        fake.real_session_id = "run-2"
        app.HuahuaApp._merge_detected_backpack(fake, scene)
        self.assertEqual(2, fake.backpack_accumulator.reset.call_count)
        second_key = fake.backpack_accumulator.reset.call_args.args[0]

        self.assertNotEqual(first_key, second_key)

    def test_same_run_does_not_reset_on_every_frame(self) -> None:
        """同一局里连续几帧，不能每帧都清空背包。"""

        scene = _scene()
        fake = _fake_app("run-1")
        for _ in range(4):
            app.HuahuaApp._merge_detected_backpack(fake, scene)

        self.assertEqual(1, fake.backpack_accumulator.reset.call_count)

    def test_rule_change_inside_one_run_opens_a_new_page(self) -> None:
        """一局内换了调色目标：**不重置**，改为开新的一页。

        用户 2026-11-18 定调：怪物初始颜料时背包按怪物排序记 15 格，玩法初始颜料
        出现后按玩法排序再记 15 格，两页合计最多 30 格。换目标就清空等于把前一页
        丢了，所以只切页（``page_key``），累计器本身保留。
        """

        fake = _fake_app("run-1")
        app.HuahuaApp._merge_detected_backpack(
            fake, _scene(PaintDirection.VERTICAL, 4, PaintObject.MONSTER)
        )
        app.HuahuaApp._merge_detected_backpack(
            fake, _scene(PaintDirection.HORIZONTAL, 3, PaintObject.GAMEPLAY)
        )

        self.assertEqual(1, fake.backpack_accumulator.reset.call_count, "同一局不该重置")
        page_keys = [
            call.kwargs.get("page_key") for call in fake.backpack_accumulator.update.call_args_list
        ]
        self.assertEqual(["monster", "gameplay"], page_keys, f"换目标要换页：{page_keys}")


def _event(kind: str, name: str = "S8_HZXY_DrawPaper_Into"):
    return SimpleNamespace(
        kind=kind,
        name=name,
        captures=kind
        in {"pigment", "brush", "brush_click", "finish", "into", "tearup", "select", "unfold", "score"},
        game_time=None,
    )


def _log_consumer_app(events):
    """只补齐 ``_consume_game_log_events`` 真正会碰到的属性。"""

    stub = SimpleNamespace(
        game_log_watcher=SimpleNamespace(
            running=True,
            drain=lambda: list(events),
            revision=1,
            # 静默判定要用的三个量：日志写入距今、可识别事件距今、
            # 「在写但认不出」持续多久。都给「刚写过、不静默」，和加判定前行为一致。
            line_age=0.0,
            idle_seconds=0.0 if events else float("inf"),
            silent_seconds=0.0,
            lines_seen=0,
        ),
        auto_follow_pump=Mock(),
        last_game_log_events=(),
        auto_follow_last_event_at=0.0,
        auto_follow_log_revision=0,
        game_log_event_count=0,
        game_log_last_event=None,
        auto_follow_var=SimpleNamespace(get=lambda: False),
        auto_follow_burst_until=0.0,
        game_log_nudge_at=0.0,
        status_var=SimpleNamespace(set=lambda *_: None),
        pending_new_run=False,
        # 「等级动画窗口」：``_consume_game_log_events`` 里会先过一遍
        # ``_note_level_animation_events``，真实 ``__init__`` 的默认值就是 None
        # （没有待核对的对账就不开窗口），这里必须补齐，否则属性缺失。
        level_animation_window=None,
        level_reconcile=SimpleNamespace(ran=False),
        _sync_game_log_status=lambda *a, **k: None,
        _wake_auto_follow_pump=lambda: None,
        _nudge_auto_follow_after_event=lambda *a, **k: None,
        _auto_follow_idle_interval=lambda: 1.5,
        _auto_follow_base_interval=lambda: 1.5,
        auto_follow_log_silent=False,
        auto_follow_silence_notice_key=False,
        _set_notice=lambda *a, **k: None,
        after=lambda *a, **k: None,
    )
    # 走**真实**方法（窗口为 None 时立刻返回），别用空实现盖住它——
    # 那样以后这条链路再漏属性也发现不了。
    stub._note_level_animation_events = lambda events: app.HuahuaApp._note_level_animation_events(
        stub, events
    )
    # 同理：静默相关也走真实实现，别让空实现掩盖漏属性。
    stub._auto_follow_fallback_interval = lambda: app.HuahuaApp._auto_follow_fallback_interval(stub)
    stub._log_is_silent = lambda watcher: app.HuahuaApp._log_is_silent(stub, watcher)
    stub._note_log_silence = lambda watcher, now: app.HuahuaApp._note_log_silence(stub, watcher, now)
    stub._report_log_silence = lambda watcher: app.HuahuaApp._report_log_silence(stub, watcher)
    stub._report_log_resumed = lambda watcher: app.HuahuaApp._report_log_resumed(stub, watcher)
    return stub


class IntoEventDispatchTests(unittest.TestCase):
    def test_into_sets_pending_new_run(self) -> None:
        """「入画」必须被认成新局信号。

        实测时间线（两局各一次，且都紧跟在开局）::

            15.36.37 Draw_Finish
            15.37.17 Into → 15.37.18 TearUp → 15.37.19 Select → 15.37.20 Unfold
        """

        fake = _log_consumer_app([_event("into", "S8_HZXY_DrawPaper_Into")])
        with patch.object(app, "record_observation"):
            app.HuahuaApp._consume_game_log_events(fake)

        self.assertTrue(fake.pending_new_run)

    def test_ordinary_events_do_not_start_a_new_run(self) -> None:
        """落子/开笔锋/完成绘制都不该清局。"""

        for kind in ("pigment", "brush", "brush_click", "finish", "tearup", "select", "unfold", "score", "other"):
            fake = _log_consumer_app([_event(kind, f"S8_HZXY_DrawPaper_{kind}")])
            with patch.object(app, "record_observation"):
                app.HuahuaApp._consume_game_log_events(fake)
            self.assertFalse(fake.pending_new_run, f"{kind} 不该触发换局")

    def test_pending_flag_is_not_sticky_across_polls(self) -> None:
        """一次入画只该开一次新局，后续轮询不能反复置位。"""

        fake = _log_consumer_app([])
        with patch.object(app, "record_observation"):
            app.HuahuaApp._consume_game_log_events(fake)
        self.assertFalse(fake.pending_new_run)


class NewRunPaintIdentityTests(unittest.TestCase):
    """旧局冻结后，同位置的新怪物不能被旧玩法身份覆盖。"""

    def _app(self):
        fake = _log_consumer_app([_event("into")])
        fake.level_stabilizer = LevelStabilizer()
        for name in (
            "_clear_brush_choice", "_update_perf_panel", "_update_fast_snapshot",
            "_restore_app_window", "_set_last_capture_context", "_draw_empty_grid",
            "_write_regions",
        ):
            setattr(fake, name, Mock())
        fake.backpack_accumulator = Mock()
        fake.tracker = Mock()
        fake.board_var = Mock()
        fake.last_finished_board_states = ""
        fake.start_new_real_session = lambda **kw: app.HuahuaApp.start_new_real_session(fake, **kw)
        fake._stabilize_scene = lambda scene: app.HuahuaApp._stabilize_scene(fake, scene)
        return fake

    def _new_board(self):
        # 2026-11-25 实机新增的三个怪物及其位置（行列从 0 开始）。
        tiles = tuple(
            TileObservation(
                row=row, col=col, rect=Rect(col * 96, row * 96, 96, 96),
                state=TileState.PAINT, confidence=0.9, features={},
                paint_category=PaintCategory.MONSTER, paint_category_confidence=0.9,
                paint_subtype=subtype, paint_level=level, paint_level_confidence=0.9,
            )
            for row, col, subtype, level in ((2, 3, "邪影", 4), (3, 4, "黑帮", 3), (4, 5, "英灵", 3))
        )
        return BoardObservation(rect=Rect(0, 0, 576, 576), score=0.9, tiles=tiles)

    def test_every_new_run_entry_clears_old_identity_and_unfreezes(self):
        for entry in ("manual", "log_fast", "log_full", "board"):
            for frozen in (False, True):
                with self.subTest(entry=entry, frozen=frozen):
                    fake = self._app()
                    new_board = self._new_board()
                    old_board = replace(new_board, tiles=tuple(
                        replace(tile, paint_category=PaintCategory.GAMEPLAY,
                                paint_subtype="赛季", paint_level=7, paint_is_treasure=True)
                        for tile in new_board.tiles
                    ))
                    fake.level_stabilizer.observe(old_board)
                    if frozen:
                        fake.level_stabilizer.freeze()
                    fake.real_session_finished = frozen
                    fake.board_reached_full = True
                    fake.last_real_capture_step = {"board": {"states": "P" * 36, "blank": 0, "paint": 36}}
                    blank_scene = SceneObservation(board=replace(new_board, tiles=tuple(
                        replace(tile, state=TileState.BLANK, paint_category=PaintCategory.UNKNOWN,
                                paint_subtype=None, paint_level=None)
                        for tile in new_board.tiles
                    )), regions=())
                    with patch.object(app, "record_observation"):
                        if entry == "manual":
                            fake.start_new_real_session(record_event=False)
                        elif entry == "board":
                            self.assertTrue(app.HuahuaApp._maybe_start_new_run_from_board(
                                fake, blank_scene, trigger="test"))
                        else:
                            app.HuahuaApp._consume_game_log_events(fake)
                            self.assertTrue(fake.pending_new_run)
                            if entry == "log_fast":
                                app.HuahuaApp._render_fast_scene_result(fake, None, "test", 0.01, auto=True)
                            else:
                                app.HuahuaApp._render_result(fake, None, None, "test", 0.01, None, None)
                    self.assertFalse(fake.pending_new_run)
                    self.assertFalse(fake.level_stabilizer.frozen)
                    self.assertEqual(fake._stabilize_scene(blank_scene).board, blank_scene.board)
                    new_scene = SceneObservation(board=new_board, regions=())
                    self.assertEqual(fake._stabilize_scene(new_scene).board, new_board)
                    # 新局还要继续接收升级，不能只是第一帧看起来恢复了。
                    upgraded = replace(new_scene, board=replace(new_board, tiles=tuple(
                        replace(tile, paint_level=tile.paint_level + 1) for tile in new_board.tiles
                    )))
                    fake._stabilize_scene(upgraded)
                    self.assertEqual(fake._stabilize_scene(upgraded).board, upgraded.board)


if __name__ == "__main__":
    unittest.main()
