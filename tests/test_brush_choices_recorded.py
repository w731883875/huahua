"""笔触「阶段一选大类 → 阶段二选词缀」这条链，日志里必须能完整读出来。

要能回答的问题：
* 面板上出现过哪几个大类？（``brush_stage1_options_observed``）
* 最终选了哪个大类？是靠**点到了**还是靠**选中高亮**读出来的？
  （``brush_stage1_selected`` 的 ``source``）
* 阶段二出现过哪几条词缀？最后选了哪条？是点到还是看效果反推的？
  （``affix_choice_observed`` / ``affix_card_clicked`` / ``affix_choice_resolved``）
"""

from __future__ import annotations

import hashlib
import unittest

from huahua_assistant.fast_vision import (
    FastBrush,
    FastCurrentPaint,
    FastTile,
    FastVisionSnapshot,
)
from huahua_assistant.models import Rect, TileState
from huahua_assistant.run_recorder import AutomaticRunRecorder


def _blank_tile(index: int) -> FastTile:
    return FastTile(
        row=index // 6,
        col=index % 6,
        state=TileState.BLANK.value,
        rect=Rect((index % 6) * 10, (index // 6) * 10, 10, 10),
        paint_category="unknown",
        paint_level=None,
        paint_subtype=None,
        paint_is_treasure=False,
        confidence=0.95,
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
    brush: FastBrush | None = None,
    regions: dict[str, Rect] | None = None,
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
        backpack_rect=None,
        backpack=(),
        backpack_signature="",
        current=_current(),
        brush=brush,
        regions=dict(regions or {}),
        reliable=True,
        reasons=(),
    )


def _brush(**overrides) -> FastBrush:
    base = dict(
        panel_rect=Rect(100, 40, 300, 200),
        stage1_options=(),
        chosen_stage1=None,
        stage2_options=(),
        confidence=0.9,
    )
    base.update(overrides)
    return FastBrush(**base)


def _feed(recorder: AutomaticRunRecorder, snapshot: FastVisionSnapshot, now: float) -> None:
    recorder.process(snapshot, observed_at=now)
    recorder.process(snapshot, observed_at=now + 0.1)


class BrushChoiceRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = AutomaticRunRecorder(path=None, stable_frames=1)
        tiles = [_blank_tile(index) for index in range(36)]
        self.recorder.process(_snapshot(tiles), observed_at=100.0)

    def _events(self, name: str) -> list[dict]:
        return [event for event in self.recorder._events if event["event"] == name]

    def test_stage1_options_are_recorded_once_per_panel(self) -> None:
        tiles = [_blank_tile(index) for index in range(36)]
        _feed(self.recorder, _snapshot(tiles, brush=_brush(stage1_options=("transform", "stack", "spread"))), 200.0)
        _feed(self.recorder, _snapshot(tiles, brush=_brush(stage1_options=("transform", "stack", "spread"))), 200.4)

        observed = self._events("brush_stage1_options_observed")
        self.assertEqual(1, len(observed))
        payload = observed[0]["payload"]
        self.assertEqual(["transform", "stack", "spread"], payload["options"])
        self.assertEqual(["转锋", "堆砌", "扩散"], payload["names"])
        self.assertEqual(["1", "2", "3"], payload["codes"])

    def test_clicked_stage1_card_is_recorded_with_its_index(self) -> None:
        tiles = [_blank_tile(index) for index in range(36)]
        panel = _brush(stage1_options=("transform", "stack", "spread"))
        _feed(self.recorder, _snapshot(tiles, brush=panel), 300.0)

        # 阶段一的卡片是**上下排列**（用户 2026-11-22 纠正；不是横排三张）。
        # 面板 y 范围 40~240 三等分，点 140 落在第二张（堆砌）。
        result = self.recorder.process(
            _snapshot(tiles, brush=panel),
            observed_at=300.3,
            clicks=((300.25, 250.0, 140.0),),
        )

        self.assertIn("brush_stage1_card_clicked", result.events)
        clicked = self._events("brush_stage1_card_clicked")[0]["payload"]
        self.assertEqual("stack", clicked["selected"])
        self.assertEqual("堆砌", clicked["selected_name"])
        self.assertEqual(1, clicked["card_index"])

    def test_bottom_stage1_card_is_reachable(self) -> None:
        """点下面那张卡必须能记到。

        旧实现只认「面板顶部 34% 之内」，下面那张卡（relative_y ≈ 0.75）永远
        记不到——用户 2026-11-22 反馈「之前都可以读取点击哪条，你读取不了」。
        """

        tiles = [_blank_tile(index) for index in range(36)]
        panel = _brush(stage1_options=("transform", "stack"))
        _feed(self.recorder, _snapshot(tiles, brush=panel), 320.0)

        self.recorder.process(
            _snapshot(tiles, brush=panel),
            observed_at=320.3,
            clicks=((320.25, 250.0, 190.0),),
        )

        clicked = self._events("brush_stage1_card_clicked")[0]["payload"]
        self.assertEqual("stack", clicked["selected"])
        self.assertEqual(1, clicked["card_index"])

    def test_confirm_button_click_is_not_a_card_click(self) -> None:
        """点「确认」不能被记成点了下面那张词缀卡。

        确认按钮几何落在面板下半部（``board_rect.h * 0.86``），按 Y 分带会被算成
        ``card_index=1``。用户 2026-11-22 实机记录里 22:08:29 / 22:08:31 连续两次
        ``affix_card_clicked`` 就是这种污染。
        """

        tiles = [_blank_tile(index) for index in range(36)]
        panel = _brush(
            stage1_options=("stack",),
            stage2_options=("自身提升1级", "范围内颜料变成随机类型颜料"),
        )
        _feed(self.recorder, _snapshot(tiles, brush=panel), 340.0)

        confirm = Rect(100, 200, 120, 40)  # 落在面板 (100,40,300,200) 的下半部
        self.recorder.process(
            _snapshot(tiles, brush=panel, regions={"confirm_button": confirm}),
            observed_at=340.3,
            clicks=((340.25, 160.0, 220.0),),
        )

        self.assertEqual([], self._events("affix_card_clicked"))

    def test_stage1_choice_read_from_highlight_is_marked_as_such(self) -> None:
        tiles = [_blank_tile(index) for index in range(36)]
        _feed(self.recorder, _snapshot(tiles, brush=_brush(stage1_options=("transform", "stack"))), 400.0)
        _feed(
            self.recorder,
            _snapshot(tiles, brush=_brush(stage1_options=("transform", "stack"), chosen_stage1="stack")),
            400.5,
        )

        selected = self._events("brush_stage1_selected")
        self.assertEqual(1, len(selected))
        payload = selected[0]["payload"]
        self.assertEqual("stack", payload["selected"])
        self.assertEqual("堆砌", payload["selected_name"])
        self.assertEqual("card_highlight", payload["source"])
        self.assertIsNone(payload["card_index"])

    def test_stage2_options_carry_stage1_context_and_inferred_source(self) -> None:
        tiles = [_blank_tile(index) for index in range(36)]
        stage1 = _brush(stage1_options=("stack", "spread"))
        _feed(self.recorder, _snapshot(tiles, brush=stage1), 500.0)

        with_affixes = _brush(
            stage1_options=("stack", "spread"),
            chosen_stage1="stack",
            stage2_options=("范围内随机1个颜料提升1级", "范围内颜料变成随机类型颜料"),
        )
        _feed(self.recorder, _snapshot(tiles, brush=with_affixes), 500.5)
        _feed(self.recorder, _snapshot(tiles, brush=_brush(chosen_stage1="stack")), 501.0)

        observed = self._events("affix_choice_observed")
        self.assertEqual(1, len(observed))
        payload = observed[0]["payload"]
        self.assertEqual("stack", payload["brush"])
        self.assertEqual("堆砌", payload["brush_name"])
        self.assertEqual(["stack", "spread"], payload["stage1_options"])
        self.assertEqual(2, len(payload["options"]))

        # 没有抓到点击时，靠效果反推的那条必须标明来源，别和点击混淆。
        resolved = [event for event in self.recorder._events if event["event"] == "affix_choice_resolved"]
        for event in resolved:
            self.assertEqual("effect_inference", event["payload"]["source"])

    def test_stage2_click_keeps_click_source_and_affix_code(self) -> None:
        tiles = [_blank_tile(index) for index in range(36)]
        stage1 = _brush(stage1_options=("stack",), chosen_stage1="stack")
        _feed(self.recorder, _snapshot(tiles, brush=stage1), 600.0)
        with_affixes = _brush(
            stage1_options=("stack",),
            chosen_stage1="stack",
            stage2_options=("范围内随机1个颜料提升1级", "范围内颜料变成随机类型颜料"),
        )
        _feed(self.recorder, _snapshot(tiles, brush=with_affixes), 600.5)

        # 面板 y 范围 40~240；点 60（相对 0.1 < 0.52）→ 第 1 张。
        self.recorder.process(
            _snapshot(tiles, brush=with_affixes),
            observed_at=600.9,
            clicks=((600.85, 150.0, 60.0),),
        )
        _feed(self.recorder, _snapshot(tiles, brush=_brush(chosen_stage1="stack")), 601.2)

        clicked = self._events("affix_card_clicked")
        self.assertEqual(1, len(clicked))
        self.assertEqual("堆砌", clicked[0]["payload"]["brush_name"])

        resolved = self._events("affix_choice_resolved")
        self.assertEqual(1, len(resolved))
        payload = resolved[0]["payload"]
        self.assertEqual("captured_click", payload["source"])
        self.assertEqual("范围内随机1个颜料提升1级", payload["selected"])
        self.assertIsNotNone(payload["code"])


    def test_click_before_stage2_is_read_is_backfilled(self) -> None:
        """点击早于词缀 OCR 时要回填，不能整局 0 条点击。

        2026-11-22 16:42 实测：阶段二两条词缀在 16:42:32 才读出来，而 16:42:47
        才靠 ``effect_inference`` 反推，全session ``affix_card_clicked`` = 0 条。
        用户确认第二阶段是「点词缀卡 + 点确认」两步，说明那一下点击确实存在，
        只是当时 ``stage2_options`` 还是空。
        """

        tiles = [_blank_tile(index) for index in range(36)]
        stage1 = _brush()  # 只读到面板，卡片还没 OCR 出来
        _feed(self.recorder, _snapshot(tiles, brush=stage1), 700.0)

        # 面板在、但这一帧连阶段一卡片都还没读出来：这一下点击以前会被丢掉。
        bare = _brush()
        self.recorder.process(
            _snapshot(tiles, brush=bare),
            observed_at=700.3,
            clicks=((700.25, 150.0, 190.0),),  # 相对 0.75 → 第二条
        )
        self.assertEqual([], self._events("affix_card_clicked"))

        # 词缀读出来了：应把刚才那一下回填成「第二条」。
        with_affixes = _brush(
            stage1_options=("stack",),
            chosen_stage1="stack",
            stage2_options=("范围内随机1个颜料提升1级", "范围内颜料变成随机类型颜料"),
        )
        _feed(self.recorder, _snapshot(tiles, brush=with_affixes), 700.6)

        clicked = self._events("affix_card_clicked")
        self.assertEqual(1, len(clicked))
        payload = clicked[0]["payload"]
        self.assertEqual("buffered_click", payload["source"])
        self.assertEqual(1, payload["card_index"])
        self.assertEqual("范围内颜料变成随机类型颜料", payload["selected"])

    def test_stale_buffered_click_is_discarded(self) -> None:
        """放太久的缓存点击不能再回填——面板早换了一屏，坐标不对应了。"""

        tiles = [_blank_tile(index) for index in range(36)]
        stage1 = _brush()  # 只读到面板
        _feed(self.recorder, _snapshot(tiles, brush=stage1), 800.0)

        self.recorder.process(
            _snapshot(tiles, brush=stage1),
            observed_at=800.3,
            clicks=((800.25, 150.0, 190.0),),
        )

        # 隔了 20 秒才读出词缀：超过 8 秒上限，必须丢掉。
        with_affixes = _brush(
            stage1_options=("stack",),
            chosen_stage1="stack",
            stage2_options=("范围内随机1个颜料提升1级", "范围内颜料变成随机类型颜料"),
        )
        _feed(self.recorder, _snapshot(tiles, brush=with_affixes), 820.0)

        self.assertEqual([], self._events("affix_card_clicked"))


if __name__ == "__main__":
    unittest.main()
