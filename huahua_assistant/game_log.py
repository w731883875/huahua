"""读游戏自己写的日志，用来「知道画面什么时候变了」。

为什么需要它：自动跟随原来是按固定节拍盲截图。棋盘在两笔之间根本不动，那些
截图全是白跑；而落笔后画面动画还没结束就去读，又容易读到中间态。游戏其实把
每一次落子、开笔锋、结算、入画都写进了 ``UE_game.log``，时间戳精确到毫秒。
按日志事件去截图，既不漏也不空转。

参照 TartarusSolver 的做法（``GameLogWatcher``：150ms 增量轮询，检测到文件变短
视为新日志重置）。这里只取画中雪原用得到的部分。

日志位置（本机实测）：

    <安装目录>\\UE_game\\TorchLight\\Saved\\Logs\\UE_game.log

两种行都要认（音频与动画并存，互为兜底）：

    [2026.11.04-14.26.47:122][449]TLLua: Display: [Game] AudioMgr@PlayAudio  AudioEnum  == S8_HZXY_DrawPaper_Pigment_Puzzle
    [2026.11.21-14.06.52:529][177]TLLua: Display: [Game] 1244_S8SnowField Anim_PutongShengji Start

方括号里是 UTC 时间，末尾是事件名。同一次播放会再打一行 ``LogDataTable`` 的
``requested row 'Play_UI_S8_...'`` 警告，那是查表失败的回声，必须跳过，否则
每个事件都会数成两次。

**为什么不只认音频**：音效名会因资源缺失打出 ``DataTable FindRow Warning``，
一旦音频表被改动就再也听不到事件，而 ``Anim_*`` 是 UI 控件自己的动画回调，
只要动画播了就一定有。实测同一段时间里动画行（202 条）比音频行更密，而且
``Anim_PutongShengji`` **一条对一个地块**——一次落笔升级 3 个格子就打出 3 条，
这是音频给不了的信息（音效一次落笔只响一声）。两者都收，谁先到谁触发截图。
"""

from __future__ import annotations

import ctypes
import json
import re
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOG_RELATIVE = ("UE_game", "TorchLight", "Saved", "Logs", "UE_game.log")
CACHE_PATH = Path("runtime/game_log.json")

#: 轮询周期。150ms 是 TartarusSolver 用的值：比一次截图+识别短，又不会空转太多。
POLL_INTERVAL = 0.15

#: 日志多久没新行就算「游戏没在写」。游戏常驻心跳约 1 秒一条，5 秒足够宽松。
LOG_ALIVE_SECONDS = 5.0

#: 事件名 → 语义。没列出来的 ``S8_HZXY_`` 事件一律记成 ``other``，仍然算「画面变了」。
EVENT_KINDS = {
    "S8_HZXY_DrawPaper_Pigment_Puzzle": "pigment",
    "S8_HZXY_DrawPaper_Brushwork_Click": "brush_click",
    "S8_HZXY_DrawPaper_Brushwork": "brush",
    "S8_HZXY_DrawPaper_Draw_Finish": "finish",
    "S8_HZXY_DrawPaper_Into": "into",
    "S8_HZXY_DrawPaper_TearUp": "tearup",
    "S8_HZXY_DrawPaper_Select": "select",
    "S8_HZXY_DrawPaper_Unfold": "unfold",
}

#: UI 控件动画 → 语义。这批名字是 2026-11-21 从本机 ``UE_game.log`` 实测统计出来的，
#: 只出现在 ``S8SnowField``（棋盘本体）与 ``S8Paint``（颜料数字）上。
#:
#: * ``Huizhi``（绘制）：**整盘重绘**。实测 36 个地块对象在两毫秒内各播一条，
#:   只出现在进入/重绘画纸时，不是单次落笔——别拿它当"落子"用。
#: * ``Fangzhi``（放置）：把一枚颜料放上棋盘，之后紧跟该颜料的调色效果升级。
#: * ``PutongShengji``（普通升级）：**一个地块一条**，同一次放置升级几张就几条，
#:   而且是**排队播放**（实测同一批间隔 0 / 1.8s / 2.9s），所以别靠同时性判批。
#: * ``CaiShengji``（彩升级）：Lv6→Lv7 那一下。
#: * ``Shuzi``（数字）：结算数字滚动。
ANIM_KINDS = {
    "Huizhi": "draw",
    "Fangzhi": "place",
    "PutongShengji": "upgrade",
    "CaiShengji": "rainbow_upgrade",
    "Shuzi": "score",
}

#: 只认这几个控件上的动画。``Anim_Show`` 之类弹窗动画不认——任务成功
#: （``S8TaskSucc``）、选能力（``S8ChooseAblityMini``）都会 Show，认了就全是噪声。
ANIM_WIDGETS = frozenset({"S8SnowField", "S8Paint"})

#: 值得为此截一帧的事件。``brush`` 只是打开笔锋面板，画面同样变了，所以也算；
#: ``score`` 是结算数字滚动——一局结束时正是最该读一帧的时候。
CAPTURE_KINDS = frozenset(EVENT_KINDS.values()) | frozenset(ANIM_KINDS.values())

_LINE_RE = re.compile(r"^\[(\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}:\d{3})\]")
_AUDIO_RE = re.compile(r"AudioEnum\s*==\s*(\S+)")
# 例：``[Game] 1244_S8SnowField Anim_PutongShengji Start`` / ``... Anim_PutongShengjiEnd``
# 结尾的 ``End`` 是结束回调，不带 ``Start``；只认 ``Start``。
_ANIM_RE = re.compile(r"\[Game\]\s+(\d+)_(\w+)\s+Anim_(\w+?)(End)?\s*(Start)?\s*$")
_SCALE_RE = re.compile(r"PageViewSetUIScale.*?PageScale\s*=\s*([0-9.]+)")


@dataclass(frozen=True)
class GameLogEvent:
    """一条来自游戏日志的画面变化事件。"""

    name: str
    kind: str
    at: float
    """本机 monotonic 时刻，用来算「距离上次事件多久」。"""
    game_time: datetime | None
    """日志里写的游戏时间（已转本地时区），仅用于展示/排查。"""
    source: str = "audio"
    """来源：``audio``（音效播放）还是 ``anim``（UI 控件动画）。"""
    widget: str = ""
    """动画所在控件，如 ``S8SnowField``。音频事件为空串。"""
    object_id: int = 0
    """控件对象序号。同一次落笔升级 N 个地块就是 N 条同 ``kind``、不同 ``object_id`` 的事件。"""

    @property
    def captures(self) -> bool:
        return self.kind in CAPTURE_KINDS


def find_game_log(configured: str | Path | None = None, pids: tuple[int, ...] = ()) -> Path | None:
    """按「配置 → 运行中的游戏进程 → 上次找到的位置」的顺序定位日志。"""

    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return candidate

    for pid in pids:
        exe = process_image_path(pid)
        if not exe:
            continue
        found = log_beside(exe)
        if found is not None:
            remember_game_log(found)
            return found

    cached = _read_cache()
    if cached is not None and cached.is_file():
        return cached

    return None


def log_beside(game_exe: str | Path) -> Path | None:
    """从游戏 exe 往上找日志。安装目录层级不固定，所以逐级试。"""

    directory = Path(game_exe)
    if directory.is_file():
        directory = directory.parent
    for parent in (directory, *directory.parents):
        candidate = parent.joinpath(*LOG_RELATIVE)
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def process_image_path(pid: int) -> str | None:
    """取进程的完整 exe 路径。拿不到就返回 None（权限/进程已退出）。"""

    if pid <= 0:
        return None
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value
        return None
    finally:
        kernel32.CloseHandle(handle)


def remember_game_log(path: Path) -> None:
    """记住位置，下次窗口还没起来也能直接接上。"""

    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps({"path": str(path)}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _read_cache() -> Path | None:
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    path = data.get("path")
    return Path(path) if path else None


def parse_line(line: str) -> GameLogEvent | None:
    """把一行日志变成事件；不是画中雪原的事件就返回 None。

    先按音效解析（老路径，行为不变），再按 UI 动画解析（新增，音效表坏了也能收到）。
    两种行不会重叠——一条日志行只会命中其中一种。
    """

    if "requested row" in line:
        # 音频查表失败的回声，和真正的播放事件成对出现，不能重复计数。
        return None

    match = _AUDIO_RE.search(line)
    if match is not None:
        name = match.group(1)
        if not name.startswith("S8_"):
            return None
        return GameLogEvent(
            name=name,
            kind=EVENT_KINDS.get(name, "other"),
            at=time.monotonic(),
            game_time=_game_time(line),
        )

    return _parse_anim(line)


def _parse_anim(line: str) -> GameLogEvent | None:
    """``[Game] 1244_S8SnowField Anim_PutongShengji Start`` 这类动画行。"""

    match = _ANIM_RE.search(line)
    if match is None:
        return None
    if match.group(4):
        # ``Anim_XXXEnd``：动画走完的回调。画面在这之前就已经变了，不重复触发。
        return None

    object_id, widget, anim = match.group(1), match.group(2), match.group(3)
    if widget not in ANIM_WIDGETS:
        # 弹窗类 Anim_Show 一律不认：任务成功、选能力都会 Show，认了就是噪声。
        return None
    kind = ANIM_KINDS.get(anim)
    if kind is None:
        return None

    return GameLogEvent(
        name=f"{widget}.Anim_{anim}",
        kind=kind,
        at=time.monotonic(),
        game_time=_game_time(line),
        source="anim",
        widget=widget,
        object_id=int(object_id),
    )


def parse_ui_scale(line: str) -> float | None:
    match = _SCALE_RE.search(line)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if 0.2 <= value <= 3.0 else None


def _game_time(line: str) -> datetime | None:
    match = _LINE_RE.match(line)
    if match is None:
        return None
    try:
        stamp = datetime.strptime(match.group(1), "%Y.%m.%d-%H.%M.%S:%f")
    except ValueError:
        return None
    # 日志写的是 UTC，转成本地时间方便和用户看到的时间对上。
    return stamp.replace(tzinfo=timezone.utc).astimezone()


class GameLogWatcher:
    """增量盯着日志文件，把新事件收进队列。

    只读新写入的内容：第一次接上时直接跳到文件末尾，不去解析几 MB 的历史。
    文件被截短（游戏重启会轮转日志）时从头重读。
    """

    def __init__(self, path: Path, poll_interval: float = POLL_INTERVAL) -> None:
        self.path = Path(path)
        self.poll_interval = max(0.01, float(poll_interval))
        self._lock = threading.Lock()
        self._events: list[GameLogEvent] = []
        self._revision = 0
        self._ui_scale: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._position = 0
        self._buffer = b""
        self._identity: tuple[int, int] | None = None
        self.last_error: str | None = None
        self.last_event_at: float = 0.0
        #: 最近一次**读到新行**的时间（monotonic）。注意它跟 :attr:`last_event_at` 不是一回事：
        #: 游戏一直在写心跳和网络日志，行一直在涨，但可能**一行都认不出来**。
        #: 2026-11-22 实测：第二局 64 秒里日志写了 174 行，解析出的画中雪原事件是 0 条，
        #: 全靠这个字段才能把「日志没在写」和「在写但认不出」分开。
        self.last_line_at: float = 0.0
        self.lines_seen = 0

    # --- 生命周期 ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._attach()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="HuahuaGameLog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.6)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- 读给上层 ---------------------------------------------------------

    def drain(self) -> list[GameLogEvent]:
        """取走还没消费的事件。主线程调用。"""

        with self._lock:
            events = self._events
            self._events = []
        return events

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def ui_scale(self) -> float | None:
        with self._lock:
            return self._ui_scale

    @property
    def idle_seconds(self) -> float:
        if not self.last_event_at:
            return float("inf")
        return time.monotonic() - self.last_event_at

    @property
    def line_age(self) -> float:
        """距最近一次读到新行过了多久。日志停了会变大，日志在写就一直很小。"""

        if not self.last_line_at:
            return float("inf")
        return time.monotonic() - self.last_line_at

    @property
    def silent_seconds(self) -> float:
        """日志在写、但一条可识别事件都没有，已经持续多久。

        返回 ``0.0`` 表示「没有静默」（要么最近有事件，要么日志根本没在写）。
        这就是「用户明明在落子，工具却一动不动」的量化指标。
        """

        if self.line_age > LOG_ALIVE_SECONDS:
            return 0.0
        idle = self.idle_seconds
        if idle == float("inf"):
            # 接上之后一条事件都没来过：从接上那刻算起，避免一上来就报静默。
            return 0.0
        return idle

    # --- 内部 -------------------------------------------------------------

    def _attach(self) -> None:
        """跳到文件末尾，从「现在」开始听。"""

        try:
            stat = self.path.stat()
        except OSError as exc:
            self.last_error = f"log_missing:{type(exc).__name__}"
            return
        self._identity = (stat.st_ino, stat.st_dev)
        self._position = stat.st_size
        self._buffer = b""
        self.last_error = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # 读日志失败不该拖垮整个工具
                self.last_error = f"log_read_error:{type(exc).__name__}"
            self._stop.wait(self.poll_interval)

    def _poll_once(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            self.last_error = "log_missing"
            return

        identity = (stat.st_ino, stat.st_dev)
        if self._identity is not None and identity != self._identity:
            # 换了个文件（游戏重开），从头读。
            self._position = 0
            self._buffer = b""
        elif stat.st_size < self._position:
            # 同一个文件被截短：日志轮转了。
            self._position = 0
            self._buffer = b""
        self._identity = identity

        if stat.st_size == self._position:
            return

        with self.path.open("rb") as handle:
            handle.seek(self._position)
            chunk = handle.read(stat.st_size - self._position)
        self._position += len(chunk)
        self.last_error = None

        data = self._buffer + chunk
        lines = data.split(b"\n")
        self._buffer = lines.pop() if lines else b""
        if not lines:
            return
        # 不管认不认得出事件，先把「日志在动」记下来：区分「游戏没写」和「写了但认不出」。
        self.last_line_at = time.monotonic()

        events: list[GameLogEvent] = []
        scale: float | None = None
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self.lines_seen += 1
            if scale is None and "PageScale" in line:
                scale = parse_ui_scale(line)
            event = parse_line(line)
            if event is not None:
                events.append(event)

        if not events and scale is None:
            return
        with self._lock:
            if events:
                self._events.extend(events)
                self._revision += 1
                self.last_event_at = time.monotonic()
            if scale is not None:
                self._ui_scale = scale


def summarize(events: list[GameLogEvent]) -> str:
    """给状态栏用的一行摘要。"""

    if not events:
        return ""
    kinds = [event.kind for event in events]
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return " ".join(f"{kind}×{count}" if count > 1 else kind for kind, count in counts.items())
