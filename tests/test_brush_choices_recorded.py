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

    def test_click_while_awaiting_affix_is_not_a_stage1_click(self) -> None:
        """等词缀定案时，面板上的点击归阶段二，不能算成阶段一选卡。

        实机 2026-11-22 20:36：+131.7s 阶段二弹出（扩散两条词缀），+136.6s
        面板刚关闭那一帧又落了一次点击。那帧 ``stage2_options`` 已空、
        ``stage1_options`` 还是 ``['spread']``，点击被阶段一支路抢走，
        真正那条词缀点击丢失，最后只能落一条 ``affix_choice_unresolved``。
        """

        tiles = [_blank_tile(index) for index in range(36)]
        stage2 = ("横向随机阻挡地块变为空白地块", "当前颜料可放置在虚空地块但失去自身内容")
        # 阶段二可见
        _feed(
            self.recorder,
            _snapshot(
                tiles,
                brush=_brush(stage1_options=("spread",), chosen_stage1="spread", stage2_options=stage2),
            ),
            800.0,
        )
        # 面板关闭（stage2 从有变无）→ 产生 pending
        _feed(
            self.recorder,
            _snapshot(tiles, brush=_brush(stage1_options=("spread",), chosen_stage1="spread")),
            802.0,
        )
        self.assertIsNotNone(self.recorder._pending_affix, "面板关闭后应当进入 waiter 状态")

        # 再落一次点击（面板刚关那一帧）
        self.recorder.process(
            _snapshot(tiles, brush=_brush(stage1_options=("spread",), chosen_stage1="spread")),
            observed_at=803.0,
            clicks=((802.95, 160.0, 70.0),),
        )

        stage1_clicks = self._events("brush_stage1_card_clicked")
        self.assertEqual([], stage1_clicks, "这一下不能被记成阶段一选卡")
        self.assertTrue(
            any(abs(at - 802.95) < 0.01 for _x, _y, at in self.recorder._buffered_panel_clicks)
            or self.recorder._clicked_affix is not None,
            "这一下应当被留作阶段二定案用",
        )

    def test_narrowing_to_one_card_settles_the_choice(self) -> None:
        """用户 2026-11-22 19:40 那局的四次选择，按「窄化」全中。

        用户按「阶段一位次」报的答案：
            第2次 #2、第3次 #1、第4次 #2
        对应的面板变化：
            [转锋,扩散] → [扩散]   → 扩散（第 2 张）
            [转锋,堆砌] → [转锋]   → 转锋（第 1 张）
            [转锋,堆砌] → [堆砌]   → 堆砌（第 2 张）

        同一局第 3 次里，按点击坐标会算成 idx=1（堆砌）——那一下其实是「确认」。
        窄化比坐标硬。
        """

        cases = [
            (("transform", "spread"), "spread", 1),
            (("transform", "stack"), "transform", 0),
            (("transform", "stack"), "stack", 1),
        ]
        tiles = [_blank_tile(index) for index in range(36)]
        for offset, (options, expected, index) in enumerate(cases):
            with self.subTest(options=options, expected=expected):
                base = 500.0 + offset * 100
                _feed(self.recorder, _snapshot(tiles, brush=_brush(stage1_options=options)), base)
                self.recorder.process(
                    _snapshot(tiles, brush=_brush(stage1_options=(expected,), chosen_stage1=expected)),
                    observed_at=base + 2.0,
                )

                selected = [
                    event
                    for event in self._events("brush_stage1_selected")
                    if event["payload"]["selected"] == expected
                ]
                self.assertTrue(selected, f"应当记到选中 {expected}")
                payload = selected[-1]["payload"]
                self.assertEqual(index, payload["card_index"])
                self.assertEqual("narrowed", payload["source"])

    def test_narrowing_overrides_a_wrong_confirm_click(self) -> None:
        """确认按钮被算成「点了下面那张卡」时，窄化必须把它纠正回来。

        实机 19:40:24 点了转锋（idx=0），19:40:26 点确认被算成 idx=1（堆砌），
        随后列表窄化成 [转锋]。正确答案是转锋。
        """

        tiles = [_blank_tile(index) for index in range(36)]
        panel = _brush(stage1_options=("transform", "stack"))
        _feed(self.recorder, _snapshot(tiles, brush=panel), 700.0)

        # 先点上面那张（转锋）
        self.recorder.process(
            _snapshot(tiles, brush=panel),
            observed_at=700.3,
            clicks=((700.25, 160.0, 70.0),),
        )
        # 再点「确认」，被几何算成下面那张（堆砌）
        self.recorder.process(
            _snapshot(tiles, brush=panel),
            observed_at=702.0,
            clicks=((701.95, 160.0, 220.0),),
        )
        # 面板窄化成只剩转锋
        self.recorder.process(
            _snapshot(tiles, brush=_brush(stage1_options=("transform",), chosen_stage1="transform")),
            observed_at=702.4,
        )

        settled = self._events("brush_stage1_selected")
        self.assertTrue(settled)
        self.assertEqual("transform", settled[-1]["payload"]["selected"])
        self.assertEqual("narrowed", settled[-1]["payload"]["source"])

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
        """同一条词缀卡连点两下（点卡片 + 点确认）只算一次选择。

        为什么不按「确认按钮矩形」硬排除：那个矩形是**几何估算**的
        （``y+0.80h ~ 0.91h``），而下面那张词缀卡正好也占到 0.9h，硬排除会把
        「点下面那张卡」一起吃掉——2026-11-22 17:21 那局就是这样整局 0 条
        ``affix_card_clicked``。
        """

        tiles = [_blank_tile(index) for index in range(36)]
        panel = _brush(
            stage1_options=("stack",),
            stage2_options=("自身提升1级", "范围内颜料变成随机类型颜料"),
        )
        _feed(self.recorder, _snapshot(tiles, brush=panel), 340.0)

        # 连点两下同一条卡（下面那条），间隔 2 秒。
        for offset in (0.3, 2.3):
            self.recorder.process(
                _snapshot(tiles, brush=panel),
                observed_at=340.0 + offset,
                clicks=((340.0 + offset - 0.05, 160.0, 220.0),),
            )

        clicked = self._events("affix_card_clicked")
        self.assertEqual(1, len(clicked), "点卡片 + 点确认只该记一次")
        self.assertEqual("范围内颜料变成随机类型颜料", clicked[0]["payload"]["selected"])

    def test_bottom_affix_card_is_not_swallowed(self) -> None:
        """点下面那张词缀卡必须记到。

        这条是为了锁住上面那个教训：确认按钮的估算矩形和下面那张卡重叠，
        一旦回到「按矩形硬排除」，这条会立刻红。
        """

        tiles = [_blank_tile(index) for index in range(36)]
        panel = _brush(
            stage1_options=("stack",),
            stage2_options=("自身提升1级", "范围内颜料变成随机类型颜料"),
        )
        _feed(self.recorder, _snapshot(tiles, brush=panel), 360.0)

        confirm = Rect(100, 200, 120, 40)  # 与下面那张卡重叠的估算位置
        self.recorder.process(
            _snapshot(tiles, brush=panel, regions={"confirm_button": confirm}),
            observed_at=360.3,
            clicks=((360.25, 160.0, 220.0),),
        )

        clicked = self._events("affix_card_clicked")
        self.assertEqual(1, len(clicked))
        self.assertEqual(1, clicked[0]["payload"]["card_index"])

    def test_click_below_the_cards_is_a_confirm_not_a_card(self) -> None:
        """确认按钮的点击不能算成「选了最后一张卡」。

        实机 2026-11-22 22:37 实测：``CARD_SLOTS`` 里两张卡占
        ``0.108~0.488`` / ``0.520~0.900``，确认按钮在其下（0.92~0.98）。
        旧映射 ``int(relative_y * count)`` 把 0.94 算成 ``int(1.88) = 1``，
        于是同一次确认先落一条 ``brush_stage1_card_clicked idx=1``（stack），
        而窄化出来的真值是卡 1（transform），两处对不上。
        """

        tiles = [_blank_tile(index) for index in range(36)]
        # panel_rect = (100, 40, 300, 200)：卡 2 底边 0.900 → y=220；确认在 y>222。
        panel = _brush(stage1_options=("transform", "stack"))
        _feed(self.recorder, _snapshot(tiles, brush=panel), 400.0)

        self.recorder.process(
            _snapshot(tiles, brush=panel),
            observed_at=400.5,
            clicks=((400.45, 160.0, 228.0),),  # relative_y = 0.94
        )

        self.assertEqual([], self._events("brush_stage1_card_clicked"), "确认按钮不是选卡")

    def test_both_stage1_cards_are_reachable(self) -> None:
        """两张卡的点击都要记到正确的序号（分带边界按实测版式）。"""

        tiles = [_blank_tile(index) for index in range(36)]
        panel = _brush(stage1_options=("transform", "stack"))
        _feed(self.recorder, _snapshot(tiles, brush=panel), 420.0)

        # 卡 1 中心 relative_y≈0.30 → y=100；卡 2 中心 relative_y≈0.71 → y=182
        self.recorder.process(
            _snapshot(tiles, brush=panel),
            observed_at=420.5,
            clicks=((420.45, 160.0, 100.0),),
        )
        self.recorder.process(
            _snapshot(tiles, brush=panel),
            observed_at=428.5,
            clicks=((428.45, 160.0, 182.0),),
        )

        clicked = self._events("brush_stage1_card_clicked")
        self.assertEqual(2, len(clicked))
        self.assertEqual(0, clicked[0]["payload"]["card_index"])
        self.assertEqual("transform", clicked[0]["payload"]["selected"])
        self.assertEqual(1, clicked[1]["payload"]["card_index"])
        self.assertEqual("stack", clicked[1]["payload"]["selected"])

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
