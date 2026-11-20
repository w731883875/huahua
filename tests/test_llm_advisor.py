"""大模型托管层的回归测试（全部离线，不真调网络）。

锁住四件事：

1. 模型输出怎么脏都能解析成结构化动作（```json 包裹、前后带解释、字符串数字）；
2. 配置没开 / 缺 key 时**不报错**，只是 ``ok=False``，上层照常走本地链路；
3. 节流窗口内复用上次结果，不会把请求排队堆起来；
4. 调用失败不覆盖上一次的好结果。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from huahua_assistant.llm import Advisor
from huahua_assistant.llm.client import LlmError
from huahua_assistant.llm.prompt import parse_advice
from huahua_assistant.llm.types import (
    ACTION_PLACE,
    ACTION_UNKNOWN,
    AdvisorAdvice,
    AdvisorConfig,
)


def _config(**overrides) -> AdvisorConfig:
    base = dict(
        enabled=True,
        provider="openai_compat",
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="test-vl",
        timeout=5.0,
        min_interval=0.0,
    )
    base.update(overrides)
    return AdvisorConfig(**base)


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


class ParseAdviceTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        advice = parse_advice('{"action":"place","row":1,"col":3,"option":null,"reason":"范围收益高"}')
        self.assertTrue(advice.ok)
        self.assertEqual(ACTION_PLACE, advice.action)
        self.assertEqual((1, 3), advice.cell)
        self.assertEqual("范围收益高", advice.reason)

    def test_fenced_json_with_chatter(self) -> None:
        text = (
            "好的，我的建议是：\n```json\n"
            '{"action":"brush_stage1","row":null,"col":null,"option":"堆砌","reason":"冲级"}\n'
            "```\n希望有帮助。"
        )
        advice = parse_advice(text)
        self.assertTrue(advice.ok)
        self.assertEqual("brush_stage1", advice.action)
        self.assertEqual("堆砌", advice.option)

    def test_string_numbers_are_accepted(self) -> None:
        advice = parse_advice('{"action":"place","row":"2","col":"5","option":null,"reason":"x"}')
        self.assertEqual((2, 5), advice.cell)

    def test_unknown_action_is_not_ok(self) -> None:
        advice = parse_advice('{"action":"unknown","row":null,"col":null,"option":null,"reason":"看不清"}')
        self.assertFalse(advice.ok)
        self.assertEqual(ACTION_UNKNOWN, advice.action)
        self.assertEqual("看不清", advice.reason)

    def test_place_without_coordinates_is_rejected(self) -> None:
        advice = parse_advice('{"action":"place","row":null,"col":null,"option":null,"reason":"x"}')
        self.assertFalse(advice.ok)
        self.assertIn("row/col", advice.error)

    def test_out_of_board_coordinates_are_rejected(self) -> None:
        advice = parse_advice('{"action":"place","row":6,"col":0,"option":null,"reason":"x"}')
        self.assertFalse(advice.ok)
        self.assertIn("越界", advice.error)

    def test_garbage_output_is_rejected(self) -> None:
        advice = parse_advice("我觉得应该放在左上角")
        self.assertFalse(advice.ok)
        self.assertIn("JSON", advice.error)


class AdvisorGatingTest(unittest.TestCase):
    def test_disabled_config_never_calls_the_model(self) -> None:
        advisor = Advisor(_config(enabled=False))
        with patch("huahua_assistant.llm.advisor.chat_vision") as chat:
            advice = advisor.advise(_frame())
        chat.assert_not_called()
        self.assertFalse(advice.ok)
        self.assertIn("未开启", advice.error)

    def test_missing_api_key_is_not_an_exception(self) -> None:
        advisor = Advisor(_config(api_key=""))
        advice = advisor.advise(_frame())
        self.assertFalse(advice.ok)
        self.assertIsNotNone(advice.error)

    def test_ollama_does_not_need_api_key(self) -> None:
        config = _config(provider="ollama", api_key="", base_url="http://127.0.0.1:11434/v1")
        self.assertTrue(config.ready)


class AdvisorThrottleTest(unittest.TestCase):
    def test_second_call_inside_the_window_reuses_the_first(self) -> None:
        advisor = Advisor(_config(min_interval=60.0))
        with patch(
            "huahua_assistant.llm.advisor.chat_vision",
            return_value='{"action":"place","row":0,"col":0,"option":null,"reason":"第一次"}',
        ) as chat:
            first = advisor.advise(_frame())
            second = advisor.advise(_frame())
        self.assertEqual(1, chat.call_count, "节流窗口内不该再调模型")
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(first.action, second.action)
        self.assertEqual((0, 0), second.cell)

    def test_force_bypasses_the_throttle(self) -> None:
        advisor = Advisor(_config(min_interval=60.0))
        with patch(
            "huahua_assistant.llm.advisor.chat_vision",
            return_value='{"action":"finish","row":null,"col":null,"option":null,"reason":"结束了"}',
        ) as chat:
            advisor.advise(_frame())
            advisor.advise(_frame(), force=True)
        self.assertEqual(2, chat.call_count)


class AdvisorFailureTest(unittest.TestCase):
    def test_network_error_degrades_without_raising(self) -> None:
        advisor = Advisor(_config(min_interval=0.0))
        with patch(
            "huahua_assistant.llm.advisor.chat_vision",
            side_effect=LlmError("连不上"),
        ):
            advice = advisor.advise(_frame())
        self.assertFalse(advice.ok)
        self.assertIn("连不上", advice.error)

    def test_failure_keeps_the_previous_good_result(self) -> None:
        advisor = Advisor(_config(min_interval=0.0))
        good = '{"action":"place","row":2,"col":2,"option":null,"reason":"好位置"}'
        with patch("huahua_assistant.llm.advisor.chat_vision", return_value=good):
            first = advisor.advise(_frame())
        self.assertTrue(first.ok)

        with patch("huahua_assistant.llm.advisor.chat_vision", side_effect=LlmError("超时")):
            second = advisor.advise(_frame())
        self.assertFalse(second.ok)
        # 上一次的好结果仍然留着，上层可以继续用它
        self.assertEqual((2, 2), advisor.last().cell)

    def test_rate_limit_enters_cooldown_and_reuses_last_good(self) -> None:
        """被限流后不要在冷却期里继续撞墙，直接复用上一次的好结果。"""
        from huahua_assistant.llm.client import RateLimitedError

        advisor = Advisor(_config(min_interval=0.0))
        good = '{"action":"place","row":1,"col":1,"option":null,"reason":"还行"}'
        with patch("huahua_assistant.llm.advisor.chat_vision", return_value=good):
            advisor.advise(_frame())

        with patch(
            "huahua_assistant.llm.advisor.chat_vision",
            side_effect=RateLimitedError("HTTP 429：访问量过大"),
        ) as mocked:
            limited = advisor.advise(_frame())
            self.assertFalse(limited.ok)
            self.assertIn("429", limited.error)
            # 冷却期内再问一次：不该真的再调模型
            again = advisor.advise(_frame())
        self.assertEqual(1, mocked.call_count)
        self.assertTrue(again.cached)
        self.assertEqual((1, 1), again.cell)


class ContextDescriptionTest(unittest.TestCase):
    """喂给模型的文字上下文：不能崩、不能把未知信息编出来。"""

    def test_empty_scene_gives_empty_text(self) -> None:
        from huahua_assistant.llm.context import describe_scene

        self.assertEqual("", describe_scene(None))

    def test_board_rect_and_paint_are_described(self) -> None:
        from types import SimpleNamespace

        from huahua_assistant.llm.context import describe_scene

        scene = SimpleNamespace(
            board=SimpleNamespace(rect=SimpleNamespace(x=100, y=200, w=600, h=600)),
            current_paint=SimpleNamespace(
                category=SimpleNamespace(value="红色"),
                level=4,
                subtype=None,
                direction=SimpleNamespace(value="横向"),
                target_object=SimpleNamespace(value="怪物"),
                match_current=1,
                match_required=3,
            ),
            brush_options=None,
            regions=[SimpleNamespace(kind=SimpleNamespace(value="finish_drawing"))],
        )
        text = describe_scene(scene)
        self.assertIn("(100,200,600,600)", text)
        self.assertIn("每格约 100 像素", text)
        self.assertIn("红色 Lv4", text)
        self.assertIn("1/3", text)
        self.assertIn("finish_drawing", text)

    def test_finish_button_is_stated_explicitly(self) -> None:
        """「入画」按钮在缩略图上太小，模型看不见，必须由本地明确说出来。"""
        from types import SimpleNamespace

        from huahua_assistant.llm.context import describe_scene

        scene = SimpleNamespace(
            board=None,
            current_paint=None,
            brush_options=None,
            regions=[SimpleNamespace(kind=SimpleNamespace(value="finish_drawing"))],
        )
        self.assertIn("本地已检测到「入画」按钮", describe_scene(scene))

    def test_brush_panel_uses_stage_options(self) -> None:
        """笔锋面板没有 brush_types 字段，别拿它去拼字符串（曾经就是这么崩的）。"""
        from types import SimpleNamespace

        from huahua_assistant.llm.context import describe_scene

        scene = SimpleNamespace(
            board=None,
            current_paint=None,
            brush_options=SimpleNamespace(
                stage1_options=("转锋", "堆砌"),
                stage2_options=("提升2级",),
                chosen_stage1="转锋",
            ),
            regions=(),
        )
        text = describe_scene(scene)
        self.assertIn("转锋、堆砌", text)
        self.assertIn("提升2级", text)
        self.assertIn("已选中的方向=转锋", text)

    def test_allowed_placement_cells_are_listed(self) -> None:
        """模型在缩略图上数格子会数错，高亮格必须由本地摊开给它。"""
        from types import SimpleNamespace

        from huahua_assistant.llm.context import describe_scene

        def tile(row, col, highlight):
            return SimpleNamespace(row=row, col=col, is_scope_highlight=highlight)

        scene = SimpleNamespace(
            board=SimpleNamespace(
                rect=SimpleNamespace(x=0, y=0, w=600, h=600),
                tiles=[tile(2, 3, True), tile(3, 2, True), tile(0, 0, False)],
            ),
            current_paint=None,
            brush_options=None,
            regions=(),
        )
        text = describe_scene(scene)
        self.assertIn("高亮格：(2,3)、(3,2)。", text)
        self.assertIn("place 只能从这些格子里选", text)

    def test_backpack_slots_are_described(self) -> None:
        from types import SimpleNamespace

        from huahua_assistant.llm.context import describe_scene

        def paint(category, level, selected=False):
            return SimpleNamespace(
                category=SimpleNamespace(value=category),
                level=level,
                is_selected=selected,
            )

        scene = SimpleNamespace(
            board=None,
            current_paint=None,
            brush_options=None,
            regions=(),
            backpack=SimpleNamespace(paints=[paint("monster", 3), paint("gameplay", 1, True)]),
        )
        text = describe_scene(scene)
        self.assertIn("背包共 2 格", text)
        self.assertIn("#1 monster Lv3", text)
        self.assertIn("#2 gameplay Lv1（已选中）", text)
        # 最后一行等级读不准这件事必须说出来，否则模型会当真值用
        self.assertIn("读不准", text)


class SlotIndexParseTest(unittest.TestCase):
    def test_common_shapes(self) -> None:
        from huahua_assistant.app import _parse_slot_index

        self.assertEqual(3, _parse_slot_index("3"))
        self.assertEqual(3, _parse_slot_index("第3格"))
        self.assertEqual(7, _parse_slot_index("#7"))
        self.assertEqual(12, _parse_slot_index("第 12 个"))

    def test_unparseable_gives_none(self) -> None:
        from huahua_assistant.app import _parse_slot_index

        for value in (None, "", "第三个", "unknown"):
            self.assertIsNone(_parse_slot_index(value))


class ConfigReadinessTest(unittest.TestCase):
    def test_configured_ignores_enabled_switch(self) -> None:
        """工具栏勾选框就是 enabled 开关本身，所以 configured 不能看 enabled。"""
        config = _config(enabled=False)
        self.assertFalse(config.ready)
        self.assertTrue(config.configured)

    def test_ready_needs_both(self) -> None:
        self.assertTrue(_config(enabled=True).ready)
        self.assertFalse(_config(enabled=True, api_key="").ready)


class ImageEncodingTest(unittest.TestCase):
    def test_large_frames_are_downscaled(self) -> None:
        from huahua_assistant.llm.client import encode_image

        payload, mime = encode_image(np.zeros((1080, 1920, 3), dtype=np.uint8), 640)
        self.assertEqual("image/jpeg", mime)
        self.assertGreater(len(payload), 100)

    def test_empty_frame_is_rejected(self) -> None:
        from huahua_assistant.llm.client import encode_image

        with self.assertRaises(LlmError):
            encode_image(np.zeros((0, 0, 3), dtype=np.uint8), 640)


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


class RequestPayloadTest(unittest.TestCase):
    """请求体长什么样：免费模型的私有开关、以及限流重试。"""

    def _call(self, config, *, side_effect=None, response="", out=None):
        from huahua_assistant.llm import client

        captured = out if out is not None else {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = request.data
            captured["calls"] = captured.get("calls", 0) + 1
            if side_effect is not None:
                raise side_effect
            return _FakeResponse(response)

        with patch.object(client.urllib.request, "urlopen", fake_urlopen):
            with patch.object(client.time, "sleep", lambda _s: None):
                return client.chat_vision(
                    config,
                    system_prompt="sys",
                    user_text="u",
                    image_b64="AAA",
                )

    def test_extra_body_is_merged_into_payload(self) -> None:
        import json

        captured = {}
        self._call(
            _config(extra_body={"thinking": {"type": "disabled"}}),
            response='{"choices":[{"message":{"content":"ok"}}]}',
            out=captured,
        )
        payload = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual({"type": "disabled"}, payload["thinking"])
        self.assertEqual("test-vl", payload["model"])
        self.assertEqual(2, len(payload["messages"]))

    def test_no_extra_body_keeps_payload_clean(self) -> None:
        import json

        captured = {}
        self._call(_config(), response='{"choices":[{"message":{"content":"ok"}}]}', out=captured)
        payload = json.loads(captured["body"].decode("utf-8"))
        self.assertNotIn("thinking", payload)

    def test_server_error_payload_is_reported_clearly(self) -> None:
        """平台过载时会用 HTTP 200 包一个 error 体，别报成「解析失败」。"""
        captured = {}
        with self.assertRaises(LlmError) as ctx:
            self._call(
                _config(),
                response='{"error":{"message":"Request method \'GET\' not supported","code":"500"}}',
                out=captured,
            )
        self.assertIn("GET", str(ctx.exception))

    def test_rate_limit_payload_becomes_rate_limited_error(self) -> None:
        from huahua_assistant.llm.client import RateLimitedError

        captured = {}
        with self.assertRaises(RateLimitedError):
            self._call(
                _config(),
                response='{"error":{"code":"1305","message":"该模型当前访问量过大，请您稍后再试"}}',
                out=captured,
            )

    def test_429_is_retried_then_reported_as_rate_limit(self) -> None:
        import urllib.error

        from huahua_assistant.llm.client import RateLimitedError

        err = urllib.error.HTTPError("u", 429, "too many", {"Retry-After": "0"}, None)
        captured = {}
        with self.assertRaises(RateLimitedError):
            self._call(_config(), side_effect=err, out=captured)
        # 两次退避 + 最后一次失败，共 3 次
        self.assertEqual(3, captured["calls"])

    def test_401_is_not_retried(self) -> None:
        import urllib.error

        err = urllib.error.HTTPError("u", 401, "unauthorized", {}, None)
        captured = {}
        with self.assertRaises(LlmError):
            self._call(_config(), side_effect=err, out=captured)
        self.assertEqual(1, captured["calls"])

    def test_max_tokens_is_sent(self) -> None:
        """推理型模型不设 max_tokens 会把额度全烧在思考上，正文一个字都不剩。"""
        import json

        captured = {}
        self._call(_config(max_tokens=777), response='{"choices":[{"message":{"content":"ok"}}]}', out=captured)
        payload = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(777, payload["max_tokens"])

    def test_remote_disconnect_is_retried(self) -> None:
        """中转站/网关负载高时会直接掐连接，这类抖动重试一次基本就好。"""
        import http.client

        from huahua_assistant.llm import client

        captured = {"calls": 0}

        def flaky(request, timeout=None):
            captured["calls"] += 1
            if captured["calls"] == 1:
                raise http.client.RemoteDisconnected("Remote end closed connection without response")
            return _FakeResponse('{"choices":[{"message":{"content":"ok"}}]}')

        with patch.object(client.urllib.request, "urlopen", flaky):
            with patch.object(client.time, "sleep", lambda _s: None):
                text = client.chat_vision(
                    _config(), system_prompt="sys", user_text="u", image_b64="AAA"
                )
        self.assertEqual("ok", text)
        self.assertEqual(2, captured["calls"])

    def test_persistent_disconnect_reports_clearly(self) -> None:
        import http.client

        err = http.client.RemoteDisconnected("Remote end closed connection without response")
        captured = {}
        with self.assertRaises(LlmError) as ctx:
            self._call(_config(), side_effect=err, out=captured)
        self.assertIn("连接被对端断开", str(ctx.exception))
        self.assertEqual(3, captured["calls"])

    def test_empty_content_falls_back_to_reasoning(self) -> None:
        """思考没关掉的推理模型：正文空、答案全在 reasoning_content 里。"""
        captured = {}
        text = self._call(
            _config(),
            response='{"choices":[{"message":{"content":"","reasoning_content":"答案是 A"},"finish_reason":"length"}]}',
            out=captured,
        )
        self.assertEqual("答案是 A", text)

    def test_empty_everything_reports_finish_reason(self) -> None:
        captured = {}
        with self.assertRaises(LlmError) as ctx:
            self._call(
                _config(),
                response='{"choices":[{"message":{"content":""},"finish_reason":"length"}]}',
                out=captured,
            )
        self.assertIn("length", str(ctx.exception))


class HostingToggleTest(unittest.TestCase):
    """关掉托管之后，还在飞的那次调用不能把旧建议写回来。

    实机反馈：取消勾选后界面要愣好几秒才刷新地块。根因有两个——
    工作线程直接用 ``after()`` 回调主线程（tkinter 不是线程安全的），
    以及结果回来时没人按「代」过滤，旧建议又被挂回去，还要再活 6 秒。
    """

    def _app(self, *, enabled: bool, generation: int = 1):
        """不走 __init__，直接拼一个够用的壳，避开 tkinter 窗口。"""
        import queue

        from huahua_assistant.app import HuahuaApp

        app = object.__new__(HuahuaApp)
        app._llm_results = queue.Queue()
        app._llm_generation = generation
        app.llm_var = _FakeVar(enabled)
        app.llm_advice = None
        app.llm_busy = True
        app.applied = []
        app._apply_llm_advice = lambda advice, binding=None: app.applied.append(advice)
        return app

    def test_result_from_old_generation_is_dropped(self) -> None:
        app = self._app(enabled=True, generation=3)
        app._llm_results.put((2, AdvisorAdvice(ok=True, action="place", row=1, col=2), None))
        app._llm_drain()
        self.assertEqual([], app.applied)

    def test_result_after_disable_is_dropped(self) -> None:
        app = self._app(enabled=False, generation=2)
        app._llm_results.put((2, AdvisorAdvice(ok=True, action="finish"), None))
        app._llm_drain()
        self.assertEqual([], app.applied)

    def test_current_result_is_applied(self) -> None:
        app = self._app(enabled=True, generation=4)
        app._llm_results.put((4, AdvisorAdvice(ok=True, action="place", row=0, col=1), None))
        app._llm_drain()
        self.assertEqual(1, len(app.applied))

    def test_disabling_releases_busy_flag(self) -> None:
        """关闭时必须把 busy 放掉，否则下次勾上时 _llm_tick 会直接 return，永远起不来。"""
        from huahua_assistant.config import save_strategy_config

        app = self._app(enabled=True, generation=1)
        app._llm_pending_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        app.llm_after_id = None
        app._set_notice = lambda *_a, **_k: None
        app.status_var = _FakeVar("")
        app.llm_var.set(False)

        with patch("huahua_assistant.config.save_strategy_config", lambda *_a, **_k: None):
            with patch("huahua_assistant.llm.get_advisor") as advisor:
                advisor.return_value.reload = lambda: None
                from huahua_assistant.app import HuahuaApp

                HuahuaApp.toggle_llm_hosting(app)

        self.assertFalse(app.llm_busy)
        self.assertIsNone(app.llm_advice)
        self.assertEqual(2, app._llm_generation)


class _FakeVar:
    """够用的 BooleanVar/StringVar 替身。"""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


if __name__ == "__main__":
    unittest.main()
