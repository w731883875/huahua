#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型建议的「钉帧」与「卡片匹配」回归测试。

两类错误一定要挡住：
* **旧建议画到新画面上**（换了局/换了阶段/超出有效期还在指）；
* **把陌生词硬对到某张卡上**（默认第 0 张、子串误命中、同名卡随便挑一个）。
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from huahua_assistant.llm import binding as b
from huahua_assistant.llm.types import (
    ACTION_BACKPACK,
    ACTION_BRUSH1,
    ACTION_BRUSH2,
    ACTION_FINISH,
    ACTION_PLACE,
    AdvisorAdvice,
)
from huahua_assistant.models import (
    BoardObservation,
    PaintCategory,
    Rect,
    RegionKind,
    TileObservation,
    TileState,
)


def _scene(*, stage1=(), stage2=(), chosen=None, paints=(), regions=(), paint=True):
    options = None
    if stage1 or stage2:
        options = SimpleNamespace(
            panel_rect=Rect(0, 0, 100, 100),
            stage1_options=tuple(stage1),
            stage2_options=tuple(stage2),
            chosen_stage1=chosen,
        )
    backpack = None
    if paints is not None and paints != ():
        backpack = SimpleNamespace(paints=tuple(paints))
    current_paint = SimpleNamespace(category=PaintCategory.MONSTER) if paint else None
    return SimpleNamespace(
        brush_options=options,
        current_paint=current_paint,
        board=_board(),
        backpack=backpack,
        region=lambda kind: next((r for r in regions if r.kind == kind), None),
    )


def _board() -> BoardObservation:
    tiles = tuple(
        TileObservation(
            row=index // 6,
            col=index % 6,
            rect=Rect(index % 6 * 10, index // 6 * 10, 10, 10),
            state=TileState.BLANK,
            confidence=0.9,
            features={},
        )
        for index in range(36)
    )
    return BoardObservation(rect=Rect(0, 0, 60, 60), score=0.9, tiles=tiles)


class StageTest(unittest.TestCase):
    def test_stage_priority(self) -> None:
        self.assertEqual(b.stage_of(None), b.STAGE_NONE)
        self.assertEqual(b.stage_of(_scene(paint=False)), b.STAGE_IDLE)
        self.assertEqual(b.stage_of(_scene()), b.STAGE_PLACING, '手上拿着颜料 = 在放置阶段')
        self.assertEqual(b.stage_of(_scene(stage1=('转锋', '堆砌'))), b.STAGE_BRUSH1)
        self.assertEqual(b.stage_of(_scene(stage1=('转锋',), stage2=('扩散',))), b.STAGE_BRUSH2)

    def test_candidates_follow_stage(self) -> None:
        scene = _scene(stage1=('转锋', '堆砌'), stage2=('扩散', '固化'))
        self.assertEqual(b.stage_candidates(scene), ('扩散', '固化'), '二阶段要给二阶段的卡')
        scene1 = _scene(stage1=('转锋', '堆砌'))
        self.assertEqual(b.stage_candidates(scene1), ('转锋', '堆砌'))

    def test_signature_ignores_unrelated_fields(self) -> None:
        sig_a = b.scene_signature(_scene(stage1=('转锋',)), session=3, window=9)
        sig_b = b.scene_signature(_scene(stage1=('转锋',)), session=3, window=9)
        self.assertEqual(sig_a.key, sig_b.key)
        sig_c = b.scene_signature(_scene(stage1=('堆砌',)), session=3, window=9)
        self.assertNotEqual(sig_a.key, sig_c.key)


class MatchOptionTest(unittest.TestCase):
    def test_unique_hit(self) -> None:
        labels = ('转锋', '堆砌', '扩散')
        self.assertEqual(b.match_option_index('扩散', labels), 2)
        self.assertEqual(b.match_option_index('  扩散 ', labels), 2)

    def test_duplicate_cards_give_up(self) -> None:
        """两张同名卡：不猜哪一张。"""
        self.assertIsNone(b.match_option_index('扩散', ('扩散', '扩散')))

    def test_unknown_word_does_not_default_to_first(self) -> None:
        """旧实现 index 初值 0，陌生词会静默指到第 0 张。这里必须是 None。"""
        self.assertIsNone(b.match_option_index('完全没听过的词缀', ('转锋', '堆砌', '扩散')))

    def test_substring_is_not_a_match(self) -> None:
        """旧实现用 ``in`` 双向子串，'扩散' 会吃掉 '扩散强化'。"""
        self.assertIsNone(b.match_option_index('扩散', ('扩散强化',)))
        self.assertIsNone(b.match_option_index('扩散强化', ('扩散',)))

    def test_empty_inputs(self) -> None:
        self.assertIsNone(b.match_option_index(None, ('扩散',)))
        self.assertIsNone(b.match_option_index('', ('扩散',)))
        self.assertIsNone(b.match_option_index('扩散', ()))
        self.assertIsNone(b.match_option_index('扩散', ('', '')))

    def test_aliases_resolve_to_same_slot(self) -> None:
        """卡片自己登记的别名命中同一个位置。

        当前词缀库里的条目都没有登记别名（``affix_aliases`` 全空），所以这里
        直接把别名表注入进去验机制本身——等哪天补上别名，这段逻辑不用再改。
        """
        from unittest import mock

        def fake_aliases(label):
            return ('扩散强化',) if label == '扩散' else ()

        labels = ('转锋', '堆砌', '扩散')
        with mock.patch.object(b, 'affix_aliases', fake_aliases):
            self.assertEqual(b.match_option_index('扩散强化', labels), 2)
            # 别名命中同样要求唯一
            with mock.patch.object(b, 'affix_aliases', lambda label: ('同名',)):
                self.assertIsNone(b.match_option_index('同名', labels))


class BindingLifetimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sig = b.SceneSignature(stage=b.STAGE_BRUSH2, candidates=('扩散',), session=1, window=7)
        self.binding = b.AdviceBinding(signature=self.sig, captured_at=100.0, generation=2)

    def test_ok_within_ttl(self) -> None:
        self.assertIsNone(
            self.binding.why_not_current(now=100.5, current=self.sig, generation=2)
        )

    def test_expires_by_capture_time(self) -> None:
        self.assertEqual(
            self.binding.why_not_current(
                now=100.0 + b.ADVICE_TTL_SECONDS + 0.1, current=self.sig, generation=2
            ),
            'expired',
        )

    def test_deadline_does_not_move_on_rerender(self) -> None:
        """同一份建议读多少次，截止时刻都一样——这是"重绘不续期"的最小证明。"""
        deadlines = {self.binding.expires_at for _ in range(5)}
        self.assertEqual(len(deadlines), 1)
        self.assertEqual(deadlines.pop(), 100.0 + b.ADVICE_TTL_SECONDS)

    def test_generation_change(self) -> None:
        self.assertEqual(
            self.binding.why_not_current(now=101.0, current=self.sig, generation=3),
            'generation_changed',
        )

    def test_signature_drift(self) -> None:
        cases = {
            'session_changed': b.SceneSignature(b.STAGE_BRUSH2, ('扩散',), session=2, window=7),
            'window_changed': b.SceneSignature(b.STAGE_BRUSH2, ('扩散',), session=1, window=8),
            'stage_changed': b.SceneSignature(b.STAGE_PLACING, ('扩散',), session=1, window=7),
            'candidates_changed': b.SceneSignature(b.STAGE_BRUSH2, ('固化',), session=1, window=7),
            'stage1_changed': b.SceneSignature(b.STAGE_BRUSH2, ('扩散',), '转锋', 1, 7),
        }
        for expected, current in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    self.binding.why_not_current(now=101.0, current=current, generation=2),
                    expected,
                )

    def test_no_scene(self) -> None:
        self.assertEqual(
            self.binding.why_not_current(now=101.0, current=None, generation=2), 'no_scene'
        )


class TargetValidityTest(unittest.TestCase):
    def _state(self, *, blank=True, raise_at=None):
        class _State:
            def tile(self, row, col):
                if raise_at == (row, col):
                    raise IndexError('no tile')
                return SimpleNamespace(state=TileState.BLANK if blank else TileState.PAINT)

        return _State()

    def test_place_ok(self) -> None:
        advice = AdvisorAdvice(ok=True, action=ACTION_PLACE, row=1, col=2)
        self.assertIsNone(
            b.why_target_invalid(advice, scene=_scene(stage1=('转锋',)), state=self._state())
        )

    def test_place_on_occupied_cell_is_rejected(self) -> None:
        """模型看到的是旧画面：它想放的格子现在已经有颜料了。"""
        advice = AdvisorAdvice(ok=True, action=ACTION_PLACE, row=1, col=2)
        self.assertEqual(
            b.why_target_invalid(advice, scene=_scene(stage1=('转锋',)), state=self._state(blank=False)),
            'cell_not_placeable',
        )

    def test_place_outside_board(self) -> None:
        advice = AdvisorAdvice(ok=True, action=ACTION_PLACE, row=9, col=9)
        self.assertEqual(
            b.why_target_invalid(advice, scene=_scene(), state=None), 'cell_out_of_board'
        )

    def test_place_without_cell(self) -> None:
        advice = AdvisorAdvice(ok=True, action=ACTION_PLACE)
        self.assertEqual(b.why_target_invalid(advice, scene=_scene(), state=None), 'no_cell')

    def test_brush_stage_mismatch(self) -> None:
        advice = AdvisorAdvice(ok=True, action=ACTION_BRUSH2, option='扩散')
        self.assertEqual(
            b.why_target_invalid(advice, scene=_scene(stage1=('转锋',)), state=None),
            'stage_mismatch',
        )

    def test_brush_option_unmatched(self) -> None:
        advice = AdvisorAdvice(ok=True, action=ACTION_BRUSH1, option='没听过的词')
        self.assertEqual(
            b.why_target_invalid(advice, scene=_scene(stage1=('转锋', '堆砌')), state=None),
            'option_unmatched',
        )

    def test_brush_ok(self) -> None:
        advice = AdvisorAdvice(ok=True, action=ACTION_BRUSH1, option='堆砌')
        self.assertIsNone(
            b.why_target_invalid(advice, scene=_scene(stage1=('转锋', '堆砌')), state=None)
        )

    def test_backpack_hidden(self) -> None:
        advice = AdvisorAdvice(ok=True, action=ACTION_BACKPACK, option='3')
        self.assertEqual(b.why_target_invalid(advice, scene=_scene(), state=None), 'backpack_hidden')

    def test_backpack_slot_out_of_range(self) -> None:
        paints = tuple(SimpleNamespace(slot_rect=Rect(0, 0, 5, 5)) for _ in range(3))
        advice = AdvisorAdvice(ok=True, action=ACTION_BACKPACK, option='4')
        self.assertEqual(
            b.why_target_invalid(advice, scene=_scene(paints=paints), state=None),
            'slot_not_visible',
        )

    def test_backpack_ok(self) -> None:
        paints = tuple(SimpleNamespace(slot_rect=Rect(0, 0, 5, 5)) for _ in range(3))
        advice = AdvisorAdvice(ok=True, action=ACTION_BACKPACK, option='第2格')
        self.assertIsNone(
            b.why_target_invalid(advice, scene=_scene(paints=paints), state=None)
        )

    def test_finish_requires_region(self) -> None:
        advice = AdvisorAdvice(ok=True, action=ACTION_FINISH)
        self.assertEqual(
            b.why_target_invalid(advice, scene=_scene(), state=None), 'finish_region_missing'
        )
        with_region = _scene(
            regions=(
                SimpleNamespace(kind=RegionKind.FINISH_DRAWING, rect=Rect(0, 0, 9, 9)),
            )
        )
        self.assertIsNone(b.why_target_invalid(advice, scene=with_region, state=None))

    def test_failed_advice_is_not_pointable(self) -> None:
        self.assertEqual(
            b.why_target_invalid(AdvisorAdvice(ok=False, error='boom'), scene=_scene(), state=None),
            'advice_not_ok',
        )


class TargetPointWiringTest(unittest.TestCase):
    """``HuahuaApp._llm_target_point`` 的接线：不装 tkinter，用替身对象直接调。"""

    def _app(self, advice, scene, state=None):
        return SimpleNamespace(
            llm_advice=advice,
            llm_advice_binding=None,
            llm_reject_reason=None,
            _llm_generation=0,
            current_scene=scene,
            current_state=state,
            _llm_current_signature=lambda: b.scene_signature(scene),
            _screen_point_for_board_cell=lambda row, col: (100 + int(row), 200 + int(col)),
            _screen_point_for_capture_rect=lambda rect, *args: (int(rect.x) + 1, int(rect.y) + 2),
        )

    def test_brush_point_works_and_does_not_crash(self) -> None:
        """旧实现读 ``brush_options.affixes``（这个字段不存在）→ AttributeError。"""
        from huahua_assistant import app as app_mod

        scene = _scene(stage1=('转锋', '堆砌'), stage2=('扩散', '固化'))
        fake = self._app(
            AdvisorAdvice(ok=True, action=ACTION_BRUSH2, option='固化'), scene
        )
        point = app_mod.HuahuaApp._llm_target_point(fake)
        self.assertIsNotNone(point)
        self.assertIsNone(fake.llm_reject_reason)

    def test_unknown_card_gives_no_point(self) -> None:
        from huahua_assistant import app as app_mod

        scene = _scene(stage1=('转锋',), stage2=('扩散',))
        fake = self._app(
            AdvisorAdvice(ok=True, action=ACTION_BRUSH2, option='模型瞎编的卡'), scene
        )
        self.assertIsNone(app_mod.HuahuaApp._llm_target_point(fake))
        self.assertEqual(fake.llm_reject_reason, 'option_unmatched')

    def test_place_point_maps_to_cell(self) -> None:
        from huahua_assistant import app as app_mod

        scene = _scene(paint=False)
        fake = self._app(AdvisorAdvice(ok=True, action=ACTION_PLACE, row=2, col=3), scene, state=None)
        self.assertEqual(app_mod.HuahuaApp._llm_target_point(fake), (102, 203))

    def test_place_on_occupied_cell_gives_no_point(self) -> None:
        from huahua_assistant import app as app_mod

        scene = _scene(paint=False)
        state = SimpleNamespace(tile=lambda row, col: SimpleNamespace(state=TileState.PAINT))
        fake = self._app(AdvisorAdvice(ok=True, action=ACTION_PLACE, row=2, col=3), scene, state=state)
        self.assertIsNone(app_mod.HuahuaApp._llm_target_point(fake))
        self.assertEqual(fake.llm_reject_reason, 'cell_not_placeable')

    def test_stale_binding_gives_no_point(self) -> None:
        from huahua_assistant import app as app_mod

        scene = _scene(stage1=('转锋',), stage2=('扩散',))
        advice = AdvisorAdvice(ok=True, action=ACTION_BRUSH2, option='扩散')
        fake = self._app(advice, scene)
        # 这条建议是"另一局、上一次开托管"时问的，截止时刻早已过去。
        fake.llm_advice_binding = b.AdviceBinding(
            signature=b.SceneSignature(b.STAGE_BRUSH2, ('固化',), session=99, window=1),
            captured_at=0.0,
            generation=0,
        )
        self.assertIsNone(app_mod.HuahuaApp._llm_target_point(fake))
        self.assertIn(
            fake.llm_reject_reason,
            {'expired', 'session_changed', 'candidates_changed', 'stage1_changed'},
        )


if __name__ == '__main__':
    unittest.main()
