from __future__ import annotations

import hashlib
import time
import unittest
from dataclasses import replace

from huahua_assistant.fast_vision import (
    FastBackpackPaint,
    FastBrush,
    FastCurrentPaint,
    FastTile,
    FastVisionSnapshot,
)
from huahua_assistant.models import Rect, RegionKind, TileState
from huahua_assistant.run_recorder import (
    NO_BLANK_MIN_RUN_SECONDS,
    AutomaticRunRecorder,
    _board_payload,
)


def _paint_tile(index: int, category: str, level: int, *, subtype: str | None = None) -> FastTile:
    return FastTile(
        row=index // 6,
        col=index % 6,
        state=TileState.PAINT.value,
        rect=Rect((index % 6) * 10, (index // 6) * 10, 10, 10),
        paint_category=category,
        paint_level=level,
        paint_subtype=subtype,
        paint_is_treasure=False,
        confidence=0.95,
    )


def _blank_tile(index: int, state: str = TileState.BLANK.value) -> FastTile:
    return FastTile(
        row=index // 6,
        col=index % 6,
        state=state,
        rect=Rect((index % 6) * 10, (index // 6) * 10, 10, 10),
        paint_category="unknown",
        paint_level=None,
        paint_subtype=None,
        paint_is_treasure=False,
        confidence=0.95,
    )


def _backpack_paint(index: int, category: str, level: int) -> FastBackpackPaint:
    return FastBackpackPaint(
        slot_index=index,
        rect=Rect(index * 12, 70, 10, 10),
        category=category,
        level=level,
        subtype=None,
        is_treasure=False,
        fingerprint=f"slot-{index}-{category}-{level}",
        confidence=0.9,
    )


def _current() -> FastCurrentPaint:
    return FastCurrentPaint(
        panel_rect=Rect(80, 0, 100, 25),
        icon_rect=Rect(80, 0, 20, 20),
        text_rect=Rect(100, 0, 80, 20),
        category="gameplay",
        level=2,
        subtype=None,
        is_treasure=False,
        direction="horizontal",
        target_object="monster",
        upgrade_amount=1,
        match_current=0,
        match_required=3,
        confidence=0.9,
    )


def _snapshot(
    tiles: list[FastTile],
    *,
    current: FastCurrentPaint | None = None,
    brush: FastBrush | None = None,
    backpack: tuple[FastBackpackPaint, ...] = (),
    regions: tuple[str, ...] = (),
    page: str = "",
) -> FastVisionSnapshot:
    semantic = repr([(tile.state, tile.paint_category, tile.paint_level) for tile in tiles]).encode()
    signature = hashlib.sha1(semantic).hexdigest()[:16]
    counts = {state.value: 0 for state in TileState}
    for tile in tiles:
        counts[tile.state] += 1
    return FastVisionSnapshot(
        board_rect=Rect(0, 0, 60, 60),
        board_score=0.9,
        tiles=tuple(tiles),
        board_signature=signature,
        paint_signature=signature,
        terrain_counts=counts,
        backpack_rect=Rect(80, 30, 100, 50) if backpack else None,
        backpack=backpack,
        backpack_signature=repr([(paint.category, paint.level) for paint in backpack]),
        current=current,
        brush=brush,
        regions={name: Rect(0, 0, 10, 10) for name in regions},
        reliable=True,
        reasons=(),
        backpack_page=page,
    )


def _feed_stable(recorder: AutomaticRunRecorder, snapshot: FastVisionSnapshot, now: float) -> None:
    recorder.process(snapshot, observed_at=now)
    recorder.process(snapshot, observed_at=now + 0.1)


class AutomaticRunRecorderTests(unittest.TestCase):
    def test_stage2_click_resolves_exact_option_without_waiting_for_effect(self) -> None:
        recorder = AutomaticRunRecorder(path=None)
        tiles = [_blank_tile(index) for index in range(36)]
        stage2 = FastBrush(
            panel_rect=Rect(80, 30, 100, 50),
            stage1_options=("stack",),
            chosen_stage1="stack",
            stage2_options=("范围内随机1个颜料提升1级", "范围内颜料变成随机类型颜料"),
            confidence=0.9,
        )
        visible = _snapshot(tiles, current=_current(), brush=stage2)
        _feed_stable(recorder, visible, 900.0)

        click_result = recorder.process(visible, observed_at=900.3, clicks=((900.25, 120.0, 70.0),))
        hidden = _snapshot(tiles, current=_current())
        _feed_stable(recorder, hidden, 900.4)

        self.assertIn("affix_card_clicked", click_result.events)
        resolved = [event for event in recorder._events if event["event"] == "affix_choice_resolved"]
        self.assertEqual(1, len(resolved))
        self.assertEqual("范围内颜料变成随机类型颜料", resolved[0]["payload"]["selected"])
        self.assertEqual("captured_click", resolved[0]["payload"]["source"])

    def test_refresh_click_is_reported_as_geometry_guess_not_certainty(self) -> None:
        """刷新词缀区域是按几何估的，不能报 0.99 的"确定点到了"。

        2026-11-19：用户宠物 2 级，界面上根本没有刷新词缀按钮（该功能 5 级宠物解锁），
        但那一局的日志里出现了 6 次 ``affix_refresh_clicked`` —— 因为这块矩形
        （vision.py 里 x0 ~ panel_w*0.42、b.y+b.h*0.80）只是比例估算，
        低于 5 级时那个位置是别的东西。事件必须自曝来源，便于复盘时人工剔除。
        """
        recorder = AutomaticRunRecorder(path=None)
        tiles = [_blank_tile(index) for index in range(36)]
        stage2 = FastBrush(
            panel_rect=Rect(80, 30, 100, 50),
            stage1_options=("transform", "stack"),
            chosen_stage1="transform",
            stage2_options=("范围内颜料提升1级", "范围内颜料提升2级"),
            confidence=0.9,
        )
        snapshot = _snapshot(
            tiles,
            current=_current(),
            brush=stage2,
            regions=(RegionKind.REFRESH_AFFIX.value,),
        )
        # 先给一帧没有笔触面板的画面，再让阶段二出现——`_visible_stage2` 只在
        # 面板**变化**时更新（见 `_record_brush_transition`）。
        recorder.process(_snapshot(tiles, current=_current()), observed_at=949.0)
        _feed_stable(recorder, snapshot, 950.0)
        result = recorder.process(snapshot, observed_at=950.3, clicks=((950.25, 5.0, 5.0),))

        self.assertIn("affix_refresh_clicked", result.events)
        event = [e for e in recorder._events if e["event"] == "affix_refresh_clicked"][0]
        self.assertEqual("geometry_estimate", event["payload"]["origin"])
        self.assertLess(event["payload"]["confidence"], 0.9)
        self.assertIn("5 级", event["payload"]["note"])
        self.assertEqual([0, 0, 10, 10], event["payload"]["rect"])

    def test_refresh_click_is_ignored_when_stage2_panel_is_closed(self) -> None:
        """阶段二面板没开时，点在同一片像素上不算刷新词缀。

        2026-11-20：上一局日志里 11 次 ``affix_refresh_clicked``，而那一局宠物 2 级、
        界面根本没有这个按钮。那块矩形是一条 530x63 的横条，几乎盖住面板一整行，
        点哪都会命中。刷新按钮只存在于阶段二（词缀三选一）面板里，面板没开就不可能是它。
        """
        recorder = AutomaticRunRecorder(path=None)
        tiles = [_blank_tile(index) for index in range(36)]
        snapshot = _snapshot(
            tiles, current=_current(), regions=(RegionKind.REFRESH_AFFIX.value,)
        )
        _feed_stable(recorder, snapshot, 960.0)
        result = recorder.process(snapshot, observed_at=960.3, clicks=((960.25, 5.0, 5.0),))

        self.assertNotIn("affix_refresh_clicked", result.events)

    def test_full_run_records_actions_transforms_force_and_finish(self) -> None:
        recorder = AutomaticRunRecorder(path=None)
        tiles = [_blank_tile(index) for index in range(36)]
        start = _snapshot(tiles, regions=(RegionKind.START_DRAWING.value,))
        recorder.process(start, observed_at=1000.0)

        stage1 = FastBrush(
            panel_rect=Rect(80, 30, 100, 50),
            stage1_options=("transform", "stack"),
            chosen_stage1=None,
            stage2_options=(),
            confidence=0.9,
        )
        _feed_stable(recorder, _snapshot(tiles, current=_current(), brush=stage1), 1000.2)

        stage2 = FastBrush(
            panel_rect=Rect(80, 30, 100, 50),
            stage1_options=("stack",),
            chosen_stage1="stack",
            stage2_options=("范围内随机1个颜料提升1级", "范围内颜料变成随机类型颜料"),
            confidence=0.9,
        )
        _feed_stable(recorder, _snapshot(tiles, current=_current(), brush=stage2), 1000.5)

        after_anchor = list(tiles)
        after_anchor[0] = _paint_tile(0, "gameplay", 2)
        after_anchor[1] = _paint_tile(1, "monster", 1)
        _feed_stable(recorder, _snapshot(after_anchor, current=_current()), 1000.8)

        upgraded_anchor = list(after_anchor)
        upgraded_anchor[1] = _paint_tile(1, "monster", 2)
        _feed_stable(recorder, _snapshot(upgraded_anchor, current=_current()), 1001.1)

        inventory = (
            _backpack_paint(0, "monster", 2),
            _backpack_paint(1, "gameplay", 1),
        )
        _feed_stable(recorder, _snapshot(upgraded_anchor, current=_current(), backpack=inventory), 1001.4)

        installed = list(upgraded_anchor)
        installed[2] = _paint_tile(2, "monster", 2)
        installed[1] = _paint_tile(1, "gameplay", 2)
        remaining = (_backpack_paint(0, "gameplay", 1),)
        _feed_stable(recorder, _snapshot(installed, current=_current(), backpack=remaining), 1001.7)

        forced = list(installed)
        forced[3] = _blank_tile(3, TileState.FADED.value)
        _feed_stable(recorder, _snapshot(forced, current=_current(), backpack=remaining), 1002.0)

        finish = _snapshot(
            forced,
            current=_current(),
            backpack=remaining,
            regions=(RegionKind.FINISH_DRAWING.value,),
        )
        first = recorder.process(finish, observed_at=1002.3)
        second = recorder.process(finish, observed_at=1002.4)

        event_names = [event["event"] for event in recorder._events]
        self.assertIn("run_started", event_names)
        self.assertIn("run_active", event_names)
        self.assertIn("brush_stage1_selected", event_names)
        self.assertIn("affix_choice_observed", event_names)
        self.assertIn("affix_choice_resolved", event_names)
        self.assertIn("initial_paint_placed", event_names)
        self.assertIn("paint_installed", event_names)
        self.assertIn("paint_upgraded", event_names)
        self.assertIn("paint_category_changed", event_names)
        self.assertIn("force_complete", event_names)
        self.assertNotIn("run_finished", event_names)
        self.assertNotIn("run_finished", first.events)
        self.assertIn("run_finished", second.events)
        self.assertEqual("finished", recorder.phase)
        self.assertEqual(1, recorder._force_count)
        self.assertEqual(1, sum(recorder._installed_paints.values()))
        self.assertGreaterEqual(sum(recorder._initial_paints.values()), 1)

        conversion = next(event for event in recorder._events if event["event"] == "paint_category_changed")
        self.assertEqual({"monster_to_gameplay": 1}, conversion["payload"]["directions"])
        resolution = next(event for event in recorder._events if event["event"] == "affix_choice_resolved")
        self.assertEqual("范围内随机1个颜料提升1级", resolution["payload"]["selected"])

    def test_full_board_from_previous_run_does_not_finish_a_fresh_session(self) -> None:
        """新局刚开的头几帧还挂着上一局那张满铺棋盘，不能因此把新局结算掉。

        2026-11-22 20:27 实测：新 session 起来 0.02 秒就读到 ``blank=0``
        （画面是上一局的 ``VVVVVVPVPPVPPPPPPPPPPVVPPVPPPPPVVVVP``），0.55 秒后
        被 ``auto_settlement`` 结算。那一局前后开了 4 次 session
        （20:27:18 / 20:27:32 / 20:27:39 / 20:27:41），用户看到的就是
        「无法完成，只能重绘」。
        """

        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        # 1) 开始画面 → 起一局
        recorder.process(_snapshot(tiles, regions=(RegionKind.START_DRAWING.value,)), observed_at=1000.0)
        self.assertEqual("opening", recorder.phase)

        # 2) 紧接着一帧「上一局那种满铺棋盘」（没有空白格）
        full = [_paint_tile(index, "monster", 2) for index in range(36)]
        snapshot = _snapshot(full, current=_current())
        first = recorder.process(snapshot, observed_at=1001.0)
        second = recorder.process(snapshot, observed_at=1001.5)

        self.assertNotIn("run_finished", first.events)
        self.assertNotIn("run_finished", second.events, "开局 1.5 秒内不得因无空白格结束")
        self.assertNotEqual("finished", recorder.phase)

        # 3) 过了下限之后同样的画面才可以结案
        third = recorder.process(snapshot, observed_at=1000.0 + NO_BLANK_MIN_RUN_SECONDS + 1.0)
        self.assertIn("run_finished", third.events)

    def test_finish_button_still_settles_immediately(self) -> None:
        """「结束」按钮是强判据，不受「开局满铺」那条下限影响。"""

        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        recorder.process(_snapshot(tiles, regions=(RegionKind.START_DRAWING.value,)), observed_at=1000.0)
        full = [_paint_tile(index, "monster", 2) for index in range(36)]
        result = recorder.process(
            _snapshot(full, current=_current(), regions=(RegionKind.FINISH_DRAWING.value,)),
            observed_at=1001.0,
        )
        self.assertIn("run_finished", result.events)

    def test_initial_single_paint_is_not_mid_run_attach(self) -> None:
        """开局自带 1 块初始颜料，看到 1 块就接入不算「中途接入」。"""
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        tiles[1] = _paint_tile(1, "monster", 3)
        snapshot = _snapshot(tiles, current=_current())

        recorder.process(snapshot, observed_at=3000.0)

        started = next(event for event in recorder._events if event["event"] == "run_started")
        self.assertEqual(started["payload"]["reason"], "active_scene")
        self.assertFalse(started["payload"]["attached_mid_run"])

    def test_extra_paints_are_mid_run_attach(self) -> None:
        """没看到开始画面，且颜料多于开局自带数量，才叫中途接入。"""
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        for index in (1, 7, 8):
            tiles[index] = _paint_tile(index, "gameplay", 3)
        snapshot = _snapshot(tiles, current=_current())

        recorder.process(snapshot, observed_at=3000.0)

        started = next(event for event in recorder._events if event["event"] == "run_started")
        self.assertTrue(started["payload"]["attached_mid_run"])

    def test_current_effect_text_is_recorded_with_the_change(self) -> None:
        """面板第三行「调色效果」必须进 current_paint_changed。

        2026-11-18 实机复盘：同一类别/等级/方向、只差一条笔触的两枚颜料，
        旧记录里完全区分不出来，``effect`` 一项都没有 → 用户核对时对不上
        「第 5 个是转锋·进度-2」这类条目。现在 effect 既是身份的一部分
        （进 `_current_key`），也写进事件载荷。
        """
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        first = _current()
        second = FastCurrentPaint(
            panel_rect=first.panel_rect,
            icon_rect=first.icon_rect,
            text_rect=first.text_rect,
            category=first.category,
            level=first.level,
            subtype=first.subtype,
            is_treasure=first.is_treasure,
            direction=first.direction,
            target_object=first.target_object,
            upgrade_amount=first.upgrade_amount,
            match_current=first.match_current,
            match_required=first.match_required,
            confidence=first.confidence,
            effect_text="调色目标的进度要求减少2",
        )
        recorder.process(_snapshot(tiles, current=first), observed_at=5000.0)
        recorder.process(_snapshot(tiles, current=second), observed_at=5000.2)

        changes = [event for event in recorder._events if event["event"] == "current_paint_changed"]
        self.assertEqual(1, len(changes))
        self.assertIsNone(changes[0]["payload"]["before"]["effect"])
        self.assertEqual("调色目标的进度要求减少2", changes[0]["payload"]["after"]["effect"])
        self.assertEqual([], changes[0]["payload"]["after"]["effect_id"])
        self.assertEqual([1011], changes[0]["payload"]["after"]["brush_effect_id"])

    def test_current_payload_records_plot_effect_id_separately(self) -> None:
        """地块调色效果原文要记成 ``effect_id``，笔触表那份留空。

        实机 2026-11-19：面板第三行读到的是「使相邻的玩法颜料提升1级」（调色表 1108），
        旧实现只查笔触表 → 那一局 9 段全空。两张表 19 个 ID 撞号，必须分字段。
        """
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        first = _current()
        second = FastCurrentPaint(
            panel_rect=first.panel_rect,
            icon_rect=first.icon_rect,
            text_rect=first.text_rect,
            category=first.category,
            level=first.level,
            subtype=first.subtype,
            is_treasure=first.is_treasure,
            direction=first.direction,
            target_object=first.target_object,
            upgrade_amount=first.upgrade_amount,
            match_current=first.match_current,
            match_required=first.match_required,
            confidence=first.confidence,
            effect_text="使相邻的玩法颜料提升1级",
        )
        recorder.process(_snapshot(tiles, current=first), observed_at=5100.0)
        recorder.process(_snapshot(tiles, current=second), observed_at=5100.2)

        after = [
            event for event in recorder._events if event["event"] == "current_paint_changed"
        ][0]["payload"]["after"]
        self.assertEqual([1108], after["effect_id"])
        self.assertEqual([], after["brush_effect_id"])

    def test_initial_inventory_family_filled_after_background_worker(self) -> None:
        """开局库存的家族是后台线程补的，事后必须能补进 initial_inventory。

        实机（2026-11-18）：开局那一帧 15 格 ``subtype`` 全 None，
        ``run_finished.initial_inventory`` 里家族也全空 → 用户核对背包时
        只能看「怪物 Lv2 ×3 / Lv1 ×9」，认不出是矮人还是英灵。
        """
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        unknown = tuple(_backpack_paint(index, "monster", 2 if index < 3 else 1) for index in range(4))
        recorder.process(_snapshot(tiles, current=_current(), backpack=unknown), observed_at=6000.0)
        self.assertTrue(all(item["subtype"] is None for item in recorder.initial_inventory))

        def named(index: int, subtype: str) -> FastBackpackPaint:
            base = unknown[index]
            return FastBackpackPaint(
                slot_index=base.slot_index,
                rect=base.rect,
                category=base.category,
                level=base.level,
                subtype=subtype,
                is_treasure=False,
                fingerprint=base.fingerprint,
                confidence=base.confidence,
            )

        resolved = (named(0, "矮人"), unknown[1], unknown[2], unknown[3])
        recorder.process(_snapshot(tiles, current=_current(), backpack=resolved), observed_at=6000.4)

        self.assertEqual("矮人", recorder.initial_inventory[0]["subtype"])
        self.assertIsNone(recorder.initial_inventory[1]["subtype"])
        filled = [event for event in recorder._events if event["event"] == "initial_inventory_filled"]
        self.assertEqual(1, len(filled))
        self.assertEqual(1, filled[0]["payload"]["filled"])

    def test_initial_inventory_captured_when_start_frame_has_no_backpack(self) -> None:
        """开局那一帧没有背包读数时，用第一帧读到的库存补拍并标注延迟。"""
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        recorder.process(_snapshot(tiles, current=_current()), observed_at=7000.0)
        self.assertEqual([], recorder.initial_inventory)

        inventory = (_backpack_paint(0, "gameplay", 4),)
        recorder.process(_snapshot(tiles, current=_current(), backpack=inventory), observed_at=7012.5)

        self.assertEqual(1, len(recorder.initial_inventory))
        captured = [event for event in recorder._events if event["event"] == "initial_inventory_captured"]
        self.assertEqual(1, len(captured))
        self.assertAlmostEqual(12.5, captured[0]["payload"]["delay"], places=1)

    def test_initial_inventory_keeps_both_sort_pages(self) -> None:
        """背包两页各 15 格，合计 30 枚，第二页不能把第一页挤掉。

        用户 2026-11-18 定口径：看初始颜料的作用对象切两次排序，
        怪物一次、玩法一次，30 个颜料够用。
        """
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        monster_page = tuple(_backpack_paint(index, "monster", 3) for index in range(15))
        gameplay_page = tuple(_backpack_paint(index, "gameplay", 4) for index in range(15))

        recorder.process(
            _snapshot(tiles, current=_current(), backpack=monster_page, page="monster"),
            observed_at=8000.0,
        )
        recorder.process(
            _snapshot(tiles, current=_current(), backpack=gameplay_page, page="gameplay"),
            observed_at=8010.0,
        )

        self.assertEqual(30, len(recorder.initial_inventory))
        self.assertEqual({"monster", "gameplay"}, {item["page"] for item in recorder.initial_inventory})
        self.assertEqual(15, sum(1 for item in recorder.initial_inventory if item["page"] == "monster"))

        added = [event for event in recorder._events if event["event"] == "initial_inventory_page_added"]
        self.assertEqual(1, len(added))
        self.assertEqual("gameplay", added[0]["payload"]["page"])

    def test_same_page_does_not_collect_slots_that_scroll_in(self) -> None:
        """同一页里放掉一枚后第 16 个槽会滑进来，那不是新库存，不能收。"""
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        first = tuple(_backpack_paint(index, "monster", 3) for index in range(15))
        recorder.process(_snapshot(tiles, current=_current(), backpack=first, page="monster"), observed_at=8100.0)

        shifted = tuple(_backpack_paint(index, "monster", 3) for index in range(16))
        recorder.process(
            _snapshot(tiles, current=_current(), backpack=shifted, page="monster"), observed_at=8101.0
        )

        self.assertEqual(15, len(recorder.initial_inventory))

    def test_scene_swap_frame_with_paint_breakage_is_dropped_as_bad_frame(self) -> None:
        """入画/结算画面：成片 blocked→void **且**颜料跟着崩 → 整帧作废。

        官方扩散 18 条里阻挡只能变成空白，没有任何一条能让阻挡变虚空。
        2026-11-18 实测：末帧 4 个阻挡格同时变虚空，连带两枚颜料被读成怪物 Lv3
        （用户当场指出「赛季变怪物不成立」）。

        注意判据是**两条同时满足**：格数够多（≥1/3 盘）或伴随颜料崩盘。只靠格数分不开
        逐格抖动和入画画面——前者 4~6 格、后者也 4~6 格。
        """
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        for index in (7, 8, 27, 31):
            tiles[index] = _blank_tile(index, TileState.BLOCKED.value)
        tiles[19] = _paint_tile(19, "gameplay", 2, subtype="赛季")
        recorder.process(_snapshot(tiles, current=_current()), observed_at=9000.0)
        good = recorder._last_snapshot

        broken = list(tiles)
        for index in (7, 8, 27, 31):
            broken[index] = _blank_tile(index, TileState.VOID.value)
        # 颜料也被重读成别的类别：入画画面是整盘重画，颜料不会安然无恙。
        broken[19] = _paint_tile(19, "monster", 3, subtype="赛季")
        result = recorder.process(_snapshot(broken, current=_current()), observed_at=9001.0)

        self.assertIn("unreliable_frame_dropped", result.events)
        self.assertIs(recorder._last_snapshot, good)

        # 同一张坏画面再来一帧，不能重复报
        again = recorder.process(_snapshot(broken, current=_current()), observed_at=9002.0)
        self.assertNotIn("unreliable_frame_dropped", again.events)

    def test_local_terrain_jitter_does_not_freeze_the_run(self) -> None:
        """局部地形误读只按旧值压住，**基准照常推进**，整局记录不能停摆。

        2026-11-20 实机：开局把 4 个虚空格读成阻挡（终局 0 阻挡/16 虚空证明了这一点），
        之后每帧都读到 void，于是相对出错的开局基准帧帧都是 blocked→void。旧逻辑
        见到 ≥2 格就整帧丢，而丢帧不推进 `_last_snapshot` → 26.6s 起连续 8 次全丢，
        整局 19 次落笔、10 次笔触一条都没记下来。

        现在：这几格先按旧值上报（`terrain_conflict_held`），连续两帧同向才改判
        （`terrain_conflict_resolved`），期间**别的格子照样记**。
        """
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        for index in (7, 8, 27, 31):
            tiles[index] = _blank_tile(index, TileState.BLOCKED.value)
        recorder.process(_snapshot(tiles, current=_current()), observed_at=9100.0)

        jitter = list(tiles)
        for index in (7, 8, 27, 31):
            jitter[index] = _blank_tile(index, TileState.VOID.value)
        first = recorder.process(_snapshot(jitter, current=_current()), observed_at=9101.0)

        # 第一帧：不丢帧，四格按旧值压住
        self.assertNotIn("unreliable_frame_dropped", first.events)
        self.assertEqual(4, first.events.count("terrain_conflict_held"))
        held_states = {(tile.row, tile.col): tile.state for tile in recorder._last_snapshot.tiles}
        for index in (7, 8, 27, 31):
            self.assertEqual(
                TileState.BLOCKED.value,
                held_states[(index // 6, index % 6)],
                "压住期间基准上必须是旧值",
            )

        # 第二帧仍是 void：连续两帧同向 → 改判，并把这次地形变化记下来
        second = recorder.process(_snapshot(jitter, current=_current()), observed_at=9102.0)
        self.assertNotIn("unreliable_frame_dropped", second.events)
        self.assertEqual(4, second.events.count("terrain_conflict_resolved"))
        settled = {(tile.row, tile.col): tile.state for tile in recorder._last_snapshot.tiles}
        for index in (7, 8, 27, 31):
            self.assertEqual(TileState.VOID.value, settled[(index // 6, index % 6)])

        # 关键：改判之后再落一笔，照常记录（旧实现这里整局都记不到）
        played = list(jitter)
        played[0] = _paint_tile(0, "gameplay", 1)
        third = recorder.process(_snapshot(played, current=_current()), observed_at=9103.0)
        self.assertTrue(
            {"initial_paint_placed", "paint_installed"} & set(third.events),
            f"落笔必须被记下来，实际事件：{third.events}",
        )

    def test_repeated_scene_swap_frames_eventually_force_the_baseline(self) -> None:
        """连续多帧都被判"整盘替换"时必须放弃否决，否则记录会永久停摆。

        入画画面只出现在结束那一刻，不需要靠这条；连续 5 帧还在"整盘替换"，更像是
        识别器彻底坏了或者玩家真把半盘地形改了。此时继续否决 = 静默冻结整局，
        宁可推进基准并留下 `unreliable_frame_forced`，让复盘时看得见。
        """
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        for index in (7, 8, 27, 31):
            tiles[index] = _blank_tile(index, TileState.BLOCKED.value)
        tiles[19] = _paint_tile(19, "gameplay", 2)
        recorder.process(_snapshot(tiles, current=_current()), observed_at=9200.0)
        frozen = recorder._last_snapshot

        broken = list(tiles)
        for index in (7, 8, 27, 31):
            broken[index] = _blank_tile(index, TileState.VOID.value)
        broken[19] = _paint_tile(19, "monster", 3)
        events: list[str] = []
        for step in range(5):
            result = recorder.process(
                _snapshot(broken, current=_current()), observed_at=9201.0 + step
            )
            events.extend(result.events)
            if step < 4:
                self.assertIs(frozen, recorder._last_snapshot, "前四帧必须仍然否决")

        self.assertIn("unreliable_frame_forced", events)
        self.assertIsNot(frozen, recorder._last_snapshot, "强制采信后基准必须推进")

    def test_per_frame_record_processing_stays_well_under_budget(self) -> None:
        recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        snapshot = _snapshot(tiles, current=_current())
        durations = []
        for index in range(1000):
            started = time.perf_counter()
            result = recorder.process(snapshot, observed_at=2000.0 + index * 0.1)
            durations.append(max(result.elapsed, time.perf_counter() - started))

        self.assertLess(max(durations), 0.02)
        self.assertLess(sum(durations) / len(durations), 0.002)


class LevelSettleTests(unittest.TestCase):
    """等级变化只去抖，**不封方向**。

    历史：2026-11-18 之前这里写的是「只升不降 + 每格已见最高等级」，用来压住
    Lv7 掉档（r2c3 黑帆真实 Lv7，边框逐帧读成 6→4→7→5→1）。用户当场纠正：
    **随机颜料 / 与自身内容相同的颜料覆盖时，高级确实会变成低级**，不能写死。

    现在改成：
    * 升、降都要 ``LEVEL_CHANGE_CONFIRM_FRAMES`` 帧一致才生效（去抖）；
    * 换颜料（类别/家族/宝箱变）或清格立刻重置；
    * 「已见最高等级」只写进 ``final_board.paints[].peak_level`` 当旁注，不改上报值。
    """

    @staticmethod
    def _paint(
        board: list[FastTile], level: int, *, subtype: str = "黑帆", category: str = "gameplay"
    ) -> list[FastTile]:
        tiles = list(board)
        tiles[15] = _paint_tile(15, category, level, subtype=subtype)
        return tiles

    @staticmethod
    def _level(recorder: AutomaticRunRecorder) -> int | None:
        return recorder._last_snapshot.tiles[15].paint_level

    @staticmethod
    def _start(recorder: AutomaticRunRecorder) -> None:
        """先给一张开始画面，让录制器进入对局；否则 ``_last_snapshot`` 一直是 None。"""

        blank = [_blank_tile(index) for index in range(36)]
        recorder.process(
            _snapshot(blank, current=_current(), regions=(RegionKind.START_DRAWING.value,)),
            observed_at=990.0,
        )

    def _feed(
        self,
        recorder: AutomaticRunRecorder,
        board: list[FastTile],
        level: int,
        now: float,
        *,
        subtype: str = "黑帆",
        category: str = "gameplay",
    ) -> None:
        _feed_stable(
            recorder,
            _snapshot(self._paint(board, level, subtype=subtype, category=category), current=_current()),
            now,
        )

    def _feed_once(
        self,
        recorder: AutomaticRunRecorder,
        board: list[FastTile],
        level: int,
        now: float,
        *,
        subtype: str = "黑帆",
        category: str = "gameplay",
    ) -> None:
        """只喂一帧：用来验证「单帧跳变不采信」。"""

        recorder.process(
            _snapshot(self._paint(board, level, subtype=subtype, category=category), current=_current()),
            observed_at=now,
        )

    def test_a_confirmed_downgrade_is_accepted(self) -> None:
        """用户明确要求的场景：内容被覆盖，Lv7 变 Lv5 必须能记成 5。"""

        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 7, 1000.0)
        self.assertEqual(7, self._level(recorder))
        self._feed(recorder, board, 5, 1002.0)
        self.assertEqual(5, self._level(recorder), "连续两帧都是 Lv5 就该认")
        changed = [event for event in recorder._events if event["event"] == "level_changed"]
        self.assertEqual({"from": 7, "to": 5, "direction": "down"}, {k: changed[-1]["payload"][k] for k in ("from", "to", "direction")})

    def test_a_single_frame_flap_does_not_change_the_level(self) -> None:
        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 7, 1000.0)
        self._feed_once(recorder, board, 5, 1002.0)
        self.assertEqual(7, self._level(recorder), "单帧读到 5 只是噪声，不该掉档")
        self._feed(recorder, board, 7, 1003.0)
        self.assertEqual(7, self._level(recorder))

    def test_an_upgrade_also_needs_two_frames(self) -> None:
        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 3, 1000.0)
        self._feed_once(recorder, board, 5, 1002.0)
        self.assertEqual(3, self._level(recorder))
        self._feed(recorder, board, 5, 1003.0)
        self.assertEqual(5, self._level(recorder))

    def test_a_different_paint_takes_effect_immediately(self) -> None:
        """换家族就是另一枚颜料：黑帆 Lv7 之后放上玩偶 Lv3，立刻记 3。"""

        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 7, 1000.0)
        self._feed(recorder, board, 3, 1002.0, subtype="玩偶")
        self.assertEqual(3, self._level(recorder))

    def test_clearing_the_cell_resets_the_state(self) -> None:
        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 7, 1000.0)
        emptied = list(board)
        emptied[15] = _blank_tile(15, TileState.VOID.value)
        _feed_stable(recorder, _snapshot(emptied, current=_current()), 1002.0)
        self._feed(recorder, board, 2, 1004.0)
        self.assertEqual(2, self._level(recorder))

    def test_peak_level_is_recorded_as_a_note_only(self) -> None:
        """降级之后终局记 5，但 `peak_level` 要留 7 供复盘。"""

        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 7, 1000.0)
        self._feed(recorder, board, 5, 1002.0)
        self._feed(recorder, board, 5, 1004.0)
        self.assertEqual(5, self._level(recorder))
        # 15 号格 = r2c3（15 // 6 = 2、15 % 6 = 3）
        self.assertEqual(7, recorder._level_peak[(2, 3)][1])
        payload = _board_payload(recorder._last_snapshot, recorder._level_peak)
        cell = next(item for item in payload["paints"] if (item["row"], item["col"]) == (2, 3))
        self.assertEqual(5, cell["level"])
        self.assertEqual(7, cell["peak_level"])
        self.assertTrue(cell["level_below_peak"])

    def test_a_cross_family_blip_does_not_pin_the_cell(self) -> None:
        """r3c2 实测：赛季 Lv3 有一帧被读成「幽灵 Lv4」，不能据此把整格改掉。"""

        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 3, 1000.0, subtype="赛季")
        self._feed_once(recorder, board, 4, 1001.0, subtype="幽灵", category="monster")
        self.assertEqual(3, self._level(recorder))
        self._feed(recorder, board, 3, 1002.0, subtype="赛季")
        self.assertEqual(3, self._level(recorder))

    def test_two_families_on_one_cell_are_flagged_for_review(self) -> None:
        """家族反复变 = 我们没认准。不猜哪个对，标 ``needs_review`` 让人复核。

        这正是 r3c2 的处境（708s 还读「赛季」、729s 起一路读「幽灵」），
        用户事后也无法确认当时放的是哪一枚，所以记录必须自己承认不确定。
        """

        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 3, 1000.0, subtype="赛季")
        self._feed(recorder, board, 3, 1002.0, subtype="赛季")
        self._feed(recorder, board, 4, 1004.0, subtype="幽灵", category="monster")
        self._feed(recorder, board, 4, 1006.0, subtype="幽灵", category="monster")
        payload = _board_payload(recorder._last_snapshot, recorder._level_peak, recorder._subtype_votes)
        cell = next(item for item in payload["paints"] if (item["row"], item["col"]) == (2, 3))
        self.assertTrue(cell["needs_review"])
        self.assertEqual({"赛季", "幽灵"}, set(cell["subtype_votes"]))

    def test_a_single_family_is_not_flagged(self) -> None:
        recorder = AutomaticRunRecorder(path=None)
        self._start(recorder)
        board = [_blank_tile(index) for index in range(36)]
        self._feed(recorder, board, 3, 1000.0, subtype="赛季")
        self._feed(recorder, board, 4, 1002.0, subtype="赛季")
        payload = _board_payload(recorder._last_snapshot, recorder._level_peak, recorder._subtype_votes)
        cell = next(item for item in payload["paints"] if (item["row"], item["col"]) == (2, 3))
        self.assertNotIn("needs_review", cell)

if __name__ == "__main__":
    unittest.main()
