"""游戏日志事件解析与增量监听。

自动跟随原来是固定节拍盲截图：棋盘在两笔之间根本不动，那些帧全是白跑；
落子后动画没走完就去读，又容易读到中间态。日志里每一次落子/开笔锋/结算/入画
都写得很清楚，时间戳精确到毫秒，按它触发就行。

这些用例里的日志行都是本机 ``UE_game.log`` 里原样摘的。
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from huahua_assistant.game_log import (
    GameLogWatcher,
    find_game_log,
    log_beside,
    parse_line,
    parse_ui_scale,
    summarize,
)

#: 真正的播放事件。
PIGMENT = (
    "[2026.11.04-14.26.47:122][449]TLLua: Display: [Game] AudioMgr@PlayAudio  "
    "AudioEnum  == S8_HZXY_DrawPaper_Pigment_Puzzle"
)
#: 紧跟着的一条查表失败回声，事件名一样但必须跳过。
PIGMENT_ECHO = (
    "[2026.11.04-14.26.47:128][449]LogDataTable: Warning: UDataTable::FindRow : 'AudioMgr' "
    "requested row 'Play_UI_S8_HZXY_DrawPaper_Pigment_Puzzle' not in DataTable "
    "'/Game/Audio/BluePrint/CD/AudioCDTable.AudioCDTable'."
)
BRUSH = (
    "[2026.11.04-14.26.20:922][878]TLLua: Display: [Game] AudioMgr@PlayAudio  "
    "AudioEnum  == S8_HZXY_DrawPaper_Brushwork"
)
FINISH = (
    "[2026.11.04-14.27.00:525][239]TLLua: Display: [Game] AudioMgr@PlayAudio  "
    "AudioEnum  == S8_HZXY_DrawPaper_Draw_Finish"
)
SCALE = (
    "[2026.11.04-14.43.20:175][ 72]TLLua: Display: [Game] PageView@PageViewSetUIScale "
    "PageId = 2019_ConfirmV2Ctrl, PageScale = 0.80896353602409 0.809 0.80896353602409"
)

# --- 以下都是本机 UE_game.log 里原样摘的动画行（2026-11-21 那局）------------------
HUIZHI = "[2026.11.21-14.07.05:494][898]TLLua: Display: [Game] 1296_S8SnowField Anim_Huizhi Start"
HUIZHI_END = "[2026.11.21-14.07.06:340][948]TLLua: Display: [Game] 1296_S8SnowField Anim_HuizhiEnd"
FANGZHI = "[2026.11.21-14.07.05:494][898]TLLua: Display: [Game] 1296_S8SnowField Anim_Fangzhi Start"
SHENGJI = (
    "[2026.11.21-14.06.52:529][177]TLLua: Display: [Game] 1244_S8SnowField Anim_PutongShengji Start"
)
CAI_SHENGJI = (
    "[2026.11.21-14.50.30:730][ 73]TLLua: Display: [Game] 1302_S8SnowField Anim_CaiShengji Start"
)
TASK_SHOW = "[2026.11.21-14.06.56:129][389]TLLua: Display: [Game] 1345_S8TaskSucc Anim_Show Start"


def _wait_for(predicate, timeout: float = 2.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class ParseLineTests(unittest.TestCase):
    def test_pigment_is_recognised(self) -> None:
        event = parse_line(PIGMENT)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual("pigment", event.kind)
        self.assertTrue(event.captures)

    def test_audio_table_echo_is_skipped(self) -> None:
        """每个事件都会配一条 requested row 警告，不跳过就会数成两次。"""

        self.assertIsNone(parse_line(PIGMENT_ECHO))

    def test_brush_and_finish(self) -> None:
        self.assertEqual("brush", parse_line(BRUSH).kind)
        self.assertEqual("finish", parse_line(FINISH).kind)

    def test_unrelated_audio_is_ignored(self) -> None:
        line = (
            "[2026.11.04-14.26.47:122][449]TLLua: Display: [Game] AudioMgr@PlayAudio  "
            "AudioEnum  == UI_S16_Premium_Quality"
        )
        self.assertIsNone(parse_line(line), "别的玩法的事件不该触发本工具的截图")

    def test_unknown_s8_event_still_counts_as_a_change(self) -> None:
        line = (
            "[2026.11.04-14.27.07:656][652]TLLua: Display: [Game] AudioMgr@PlayAudio  "
            "AudioEnum  == S8_HZXY_DrawPaper_SomethingNew"
        )
        event = parse_line(line)

        assert event is not None
        self.assertEqual("other", event.kind)
        self.assertIn("other", {event.kind} & {"other"}, "未登记的事件仍要知道画面变了")

    def test_game_time_is_local_and_millisecond_precise(self) -> None:
        event = parse_line(PIGMENT)

        assert event is not None
        self.assertIsNotNone(event.game_time)
        assert event.game_time is not None
        self.assertEqual(122000, event.game_time.microsecond)
        self.assertIsNotNone(event.game_time.tzinfo)

    def test_score_animation_is_recognised(self) -> None:
        line = "[2026.11.04-14.27.03:172][393]TLLua: Display: [Game] 1782_S8Paint Anim_Shuzi Start"
        event = parse_line(line)

        assert event is not None
        self.assertEqual("score", event.kind)
        self.assertEqual("anim", event.source)
        self.assertEqual("S8Paint", event.widget)
        self.assertEqual(1782, event.object_id)


class AnimLineTests(unittest.TestCase):
    """UI 控件动画行。音频表改动/资源缺失时音效那条会消失，动画回调不会。"""

    def test_draw_is_recognised(self) -> None:
        event = parse_line(HUIZHI)

        assert event is not None
        self.assertEqual("draw", event.kind)
        self.assertTrue(event.captures)
        self.assertEqual(("anim", "S8SnowField", 1296), (event.source, event.widget, event.object_id))

    def test_animation_end_is_not_a_new_change(self) -> None:
        """``Anim_XXXEnd`` 只是动画走完的回调，画面在此之前就变了。"""

        self.assertIsNone(parse_line(HUIZHI_END))

    def test_place_upgrade_and_rainbow_upgrade(self) -> None:
        self.assertEqual("place", parse_line(FANGZHI).kind)
        self.assertEqual("upgrade", parse_line(SHENGJI).kind)
        self.assertEqual("rainbow_upgrade", parse_line(CAI_SHENGJI).kind)

    def test_one_upgrade_per_tile(self) -> None:
        """实测一次落笔升级 3 个格子会连打 3 条，对象序号各不相同——可用来数格子。"""

        lines = [
            "[2026.11.21-14.50.30:730][ 73]TLLua: Display: [Game] 1244_S8SnowField Anim_PutongShengji Start",
            "[2026.11.21-14.50.32:164][ 73]TLLua: Display: [Game] 1258_S8SnowField Anim_PutongShengji Start",
            "[2026.11.21-14.50.33:752][ 73]TLLua: Display: [Game] 1268_S8SnowField Anim_PutongShengji Start",
        ]
        events = [parse_line(line) for line in lines]

        self.assertEqual(["upgrade"] * 3, [event.kind for event in events if event])
        self.assertEqual({1244, 1258, 1268}, {event.object_id for event in events if event})

    def test_popup_animations_are_ignored(self) -> None:
        """``Anim_Show`` 任务成功/选能力都会播，认了就会每回合多截好几帧。"""

        self.assertIsNone(parse_line(TASK_SHOW))
        self.assertIsNone(
            parse_line("[2026.11.21-14.06.24:763][588]TLLua: Display: [Game] 1127_S8DrawMap Anim_Show Start")
        )

    def test_unknown_animation_on_snowfield_is_ignored(self) -> None:
        line = "[2026.11.21-14.06.24:763][588]TLLua: Display: [Game] 1127_S8SnowField Anim_SomethingNew Start"

        self.assertIsNone(parse_line(line))

    def test_audio_and_anim_coexist(self) -> None:
        """同一次落笔音效、动画各一条，两条都要收（谁先到谁触发截图）。"""

        kinds = [parse_line(PIGMENT).kind, parse_line(HUIZHI).kind]

        self.assertEqual(["pigment", "draw"], kinds)


class UiScaleTests(unittest.TestCase):
    def test_scale_is_read(self) -> None:
        self.assertAlmostEqual(0.80896, parse_ui_scale(SCALE), places=4)

    def test_absurd_scale_is_rejected(self) -> None:
        self.assertIsNone(parse_ui_scale("PageViewSetUIScale PageId = x, PageScale = 42.0"))
        self.assertIsNone(parse_ui_scale("PageViewSetUIScale PageId = x, PageScale = 0.0"))

    def test_line_without_scale(self) -> None:
        self.assertIsNone(parse_ui_scale(PIGMENT))


class LocateTests(unittest.TestCase):
    def test_log_beside_walks_up_from_the_exe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "UE_game" / "TorchLight" / "Saved" / "Logs" / "UE_game.log"
            log.parent.mkdir(parents=True)
            log.write_text("", encoding="utf-8")
            exe = root / "UE_game" / "Binaries" / "Win64" / "torchlight_infinite.exe"
            exe.parent.mkdir(parents=True)
            exe.write_text("", encoding="utf-8")

            self.assertEqual(log, log_beside(exe))

    def test_missing_install_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "somewhere" / "game.exe"
            exe.parent.mkdir(parents=True)
            exe.write_text("", encoding="utf-8")
            self.assertIsNone(log_beside(exe))

    def test_configured_path_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "UE_game.log"
            log.write_text("", encoding="utf-8")
            self.assertEqual(log, find_game_log(str(log)))


class WatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "UE_game.log"
        self.path.write_text("old line\n", encoding="utf-8")
        self.watcher = GameLogWatcher(self.path, poll_interval=0.02)
        self.watcher.start()

    def tearDown(self) -> None:
        self.watcher.stop()
        self._tmp.cleanup()

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def test_history_is_not_replayed(self) -> None:
        """接上时跳到文件末尾，不解析几 MB 的历史。"""

        time.sleep(0.1)
        self.assertEqual([], self.watcher.drain())

    def test_new_event_arrives(self) -> None:
        self._append(PIGMENT + "\n")

        self.assertTrue(_wait_for(lambda: self.watcher.revision > 0))
        events = self.watcher.drain()
        self.assertEqual(["pigment"], [event.kind for event in events])

    def test_drain_empties_the_queue(self) -> None:
        self._append(PIGMENT + "\n")
        self.assertTrue(_wait_for(lambda: self.watcher.revision > 0))
        self.watcher.drain()
        self.assertEqual([], self.watcher.drain())

    def test_several_events_in_one_chunk(self) -> None:
        self._append(BRUSH + "\n" + FINISH + "\n")

        kinds: set[str] = set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and kinds != {"brush", "finish"}:
            kinds |= {event.kind for event in self.watcher.drain()}
            time.sleep(0.02)
        self.assertEqual({"brush", "finish"}, kinds)

    def test_partial_line_is_buffered(self) -> None:
        """日志是边写边刷的，一行可能被切成两段。"""

        half = PIGMENT[:60]
        self._append(half)
        time.sleep(0.1)
        self.assertEqual([], self.watcher.drain(), "半行不该被当成事件")
        self._append(PIGMENT[60:] + "\n")

        self.assertTrue(_wait_for(lambda: self.watcher.revision > 0))
        self.assertEqual(["pigment"], [event.kind for event in self.watcher.drain()])

    def test_truncated_file_is_read_again(self) -> None:
        """游戏重启会把日志轮转成备份、重新开一个，必须从头读。"""

        self._append(PIGMENT + "\n")
        self.assertTrue(_wait_for(lambda: self.watcher.revision > 0))
        self.watcher.drain()

        self.path.write_text("", encoding="utf-8")
        time.sleep(0.08)
        self._append(FINISH + "\n")

        self.assertTrue(_wait_for(lambda: self.watcher.revision > 1))
        self.assertEqual(["finish"], [event.kind for event in self.watcher.drain()])

    def test_ui_scale_is_picked_up(self) -> None:
        self._append(SCALE + "\n")

        self.assertTrue(_wait_for(lambda: self.watcher.ui_scale is not None))
        self.assertAlmostEqual(0.809, self.watcher.ui_scale or 0, places=3)

    def test_missing_file_does_not_crash_the_thread(self) -> None:
        self.watcher.stop()
        missing = GameLogWatcher(Path(self._tmp.name) / "nope.log", poll_interval=0.02)
        missing.start()
        time.sleep(0.1)
        self.assertTrue(missing.running)
        self.assertEqual("log_missing", missing.last_error)
        missing.stop()

    def test_idle_seconds_tracks_events(self) -> None:
        self._append(PIGMENT + "\n")
        self.assertTrue(_wait_for(lambda: self.watcher.last_event_at > 0))
        self.assertLess(self.watcher.idle_seconds, 5.0)


class SummarizeTests(unittest.TestCase):
    def test_counts_repeated_kinds(self) -> None:
        events = [parse_line(PIGMENT), parse_line(PIGMENT), parse_line(BRUSH)]
        self.assertEqual("pigment×2 brush", summarize([e for e in events if e]))

    def test_empty(self) -> None:
        self.assertEqual("", summarize([]))


if __name__ == "__main__":
    unittest.main()
