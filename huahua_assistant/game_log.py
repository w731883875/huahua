"""读游戏自己写的日志，用来「知道画面什么时候变了」。

为什么需要它：自动跟随原来是按固定节拍盲截图。棋盘在两笔之间根本不动，那些
截图全是白跑；而落笔后画面动画还没结束就去读，又容易读到中间态。游戏其实把
每一次落子、开笔锋、结算、入画都写进了 ``UE_game.log``，时间戳精确到毫秒。
按日志事件去截图，既不漏也不空转。

参照 TartarusSolver 的做法（``GameLogWatcher``：150ms 增量轮询，检测到文件变短
视为新日志重置）。这里只取画中雪原用得到的部分。

日志位置（本机实测）：

    <安装目录>\\UE_game\\TorchLight\\Saved\\Logs\\UE_game.log

日志行形如：

    [2026.11.04-14.26.47:122][449]TLLua: Display: [Game] AudioMgr@PlayAudio  AudioEnum  == S8_HZXY_DrawPaper_Pigment_Puzzle

方括号里是 UTC 时间，末尾是事件名。同一次播放会再打一行 ``LogDataTable`` 的
``requested row 'Play_UI_S8_...'`` 警告，那是查表失败的回声，必须跳过，否则
每个事件都会数成两次。
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

#: 值得为此截一帧的事件。``brush`` 只是打开笔锋面板，画面同样变了，所以也算；
#: ``score`` 是结算数字滚动——一局结束时正是最该读一帧的时候。
CAPTURE_KINDS = frozenset(EVENT_KINDS.values()) | {"score"}

_LINE_RE = re.compile(r"^\[(\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}:\d{3})\]")
_AUDIO_RE = re.compile(r"AudioEnum\s*==\s*(\S+)")
_SCALE_RE = re.compile(r"PageViewSetUIScale.*?PageScale\s*=\s*([0-9.]+)")
_SHIZI_RE = re.compile(r"(\d+_S8Paint)\s+Anim_Shuzi\s*Start")


@dataclass(frozen=True)
class GameLogEvent:
    """一条来自游戏日志的画面变化事件。"""

    name: str
    kind: str
    at: float
    """本机 monotonic 时刻，用来算「距离上次事件多久」。"""
    game_time: datetime | None
    """日志里写的游戏时间（已转本地时区），仅用于展示/排查。"""

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
    """把一行日志变成事件；不是画中雪原的事件就返回 None。"""

    if "requested row" in line:
        # 音频查表失败的回声，和真正的播放事件成对出现，不能重复计数。
        return None

    name = None
    match = _AUDIO_RE.search(line)
    if match is not None:
        name = match.group(1)
    else:
        match = _SHIZI_RE.search(line)
        if match is not None:
            return GameLogEvent(
                name=match.group(1),
                kind="score",
                at=time.monotonic(),
                game_time=_game_time(line),
            )
    if name is None:
        return None
    if not name.startswith("S8_"):
        return None

    return GameLogEvent(
        name=name,
        kind=EVENT_KINDS.get(name, "other"),
        at=time.monotonic(),
        game_time=_game_time(line),
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
