from __future__ import annotations

import atexit
import json
import queue
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .codes import BRUSH_NAMES, affix_code, brush_code, resolve_affix
from .fast_vision import FastCurrentPaint, FastTile, FastVisionSnapshot
from .game_model import AffixEffect, AffixRule, BrushType
from .models import RegionKind, TileState


def _brush_name(brush: str | None) -> str | None:
    """笔触值 → 中文名（``stack`` → ``堆砌``）。

    日志里存英文值便于程序读，但复盘是人看的；两个都给，省得再拿字典去翻。
    """
    if not brush:
        return None
    try:
        return BRUSH_NAMES.get(BrushType(brush))
    except (TypeError, ValueError):
        return None


AUTO_RUN_LOG = Path("runtime") / "auto_run_records.jsonl"

# 这里原来有个 ``STAGE1_CLICK_HEIGHT_RATIO = 0.34``，用于「阶段一卡片只在面板
# 顶部 34% 内」的点击判定。2026-11-22 用户纠正：**两个阶段的卡片都是上下排列**，
# 阶段一不是横排三张。那个常量把下半张卡的点击全挡掉了，已随判定改成 Y 分带一并删除。

#: 缓存的「面板还没读出卡片时点下的那些点击」最多留几条。
#: 正常一次操作只有「点卡片 + 点确认」两下，留 4 条足够，再多多半是误点。
PENDING_CLICK_BUFFER_MAX = 4

#: 缓存的点击最多等这么久（秒）。超过就丢——面板早就换了一屏，坐标不再对应。
PENDING_CLICK_BUFFER_SECONDS = 8.0

#: 「点卡片 + 点确认」两下算一次选择：同一条卡在这个秒数内连点两下，只记第一下。
CONFIRM_DEDUPE_SECONDS = 3.0

#: 开局时棋盘上自带的颜料块数。
#:
#: 游戏开局就送 1 块初始颜料，它不是"上一局的残留"。所以接入时看到 1 块
#: **不能**算中途接入。之前用 ``reason == "active_scene"`` 直接当"中途接入"，
#: 而 ``active_scene`` 只表示"没看到开始画面就已经在对局里"，于是每一局
#: 都会被标 ``attached_mid_run=true``——包括从开局录起的完整对局。
INITIAL_PAINTS_AT_START = 1

#: 同一格同一枚颜料的等级**变化**要连续这么多帧读到同一新值才算数。
#:
#: 方向**不设限**：升和降都要确认。原因是游戏里确实会让高级变低级——
#: 随机颜料 / 与自身内容相同的颜料覆盖上来时，那格就是一枚新颜料，等级从 1
#: 起算（用户 2026-11-18 明确纠正「只升不降也是错误的」）。所以这里只做
#: **去抖**（单帧跳变不采信），不做**方向封锁**。
#:
#: 曾经写过「只升不降 + 每格已见最高等级」，用来压 Lv7 掉档；那是拿规则去补
#: 识别缺陷，遇到真实覆盖就会把整格钉死。Lv7 已改在 ``frame_level`` 用几何判据
#: （百线菱形 + 彩色辉光）直接认出来，不再需要这里兜。
LEVEL_CHANGE_CONFIRM_FRAMES = 2

#: 官方规则里**不可能**出现的地形对调：阻挡 ↔ 虚空。
#:
#: 扩散 18 条里阻挡只能变**空白**（3001~3009），虚空也只能变空白（3014/3015），
#: 没有任何一条能让两者互换。所以看到这对状态，一定是这两格里有读错的。
TERRAIN_CONFLICT_PAIRS = frozenset({("blocked", "void"), ("void", "blocked")})

#: 逐格地形冲突要连续这么多帧读到同一个新值才改判。
#:
#: 与等级去抖同源：一格在「阻挡」和「虚空」之间来回跳，是分类器在冰雪纹理边缘
#: 抖动，不是游戏在变。单帧跳变一律按旧值上报，连续两帧同向才认为真变了。
TERRAIN_CONFLICT_CONFIRM_FRAMES = 2

#: 一帧里地形对调达到这么多格，判为「画面被整体替换」（入画/结算画面），整帧作废。
#:
#: 36 格的 1/3。定这么高是因为**局部误读和入画画面在格数上分不开**：2026-11-20
#: 那局开局有 4~6 格 blocked↔void 抖动，2026-11-18 末帧入画画面也是 4~6 格。
#: 数量区分不了，只能靠「伴随颜料崩盘」这条补充判据（见 _frame_looks_replaced）。
SCENE_SWAP_CONFLICT_TILES = 12

#: 连续这么多帧都被判成"画面被整体替换"时，放弃否决、强制推进基准。
#:
#: 真正的入画画面出现在结束那一刻，不需要靠这里；连续 5 帧还在变，说明是识别器
#: 彻底坏了或者玩家真把半盘地形改了。此时继续否决只会让整局记录停摆 —— 宁可推进
#: 基准并留下 `unreliable_frame_forced`，也不要静默冻结（2026-11-20 实测：一次
#: 开局误读让基准冻死，整局 19 次落笔一条都没记下来）。
SCENE_SWAP_ESCAPE_FRAMES = 5

#: 同一帧里出现「颜料格崩盘」（颜料消失或类别被改）至少这么多格，配合地形对调
#: 就判为入画画面。转锋类笔触确实会改颜料类别，但**不会同时**让地形对调。
SCENE_SWAP_PAINT_BREAKAGE = 1


@dataclass(frozen=True)
class RecorderFrameResult:
    elapsed: float
    phase: str
    session_id: str | None
    events: tuple[str, ...]


@dataclass(frozen=True)
class RecorderStatus:
    """录制器对外的一次性状态快照（供界面上的「记录状态」灯）。

    存在的理由：``phase`` / ``session_id`` / ``_events`` 都在采集线程里被改，
    界面线程如果直接读属性，会读到「本局已换、phase 还没跟上」的中间态，
    表现就是灯乱跳。只读一份锁内快照，判断才稳。
    """

    phase: str
    session_id: str | None
    event_count: int
    duration: float
    frames: int
    missing_frames: int
    last_event_age: float | None
    path: str | None
    #: 写盘侧的状态。默认值等于"没有写盘线程/一切正常"，历史调用点不用改。
    writer_pending: int = 0
    writer_written_sequence: int | None = None
    writer_dropped: int = 0
    writer_failures: int = 0
    writer_error: str | None = None


@dataclass
class _PendingAffixChoice:
    options: tuple[str, ...]
    brush: str | None
    observed_at: float
    event_sequence: int
    #: 这组词缀是在哪个大类的面板里出现的（阶段一的三选一）。
    #: 词缀候选本身决定不了它是「转锋里的」还是「扩散里的」，必须把上下文带上，
    #: 否则复盘时只能看到两条孤立的词缀名。
    stage1_options: tuple[str, ...] = ()
    #: 阶段一读到/点到的选择，跟着词缀一起记，保证一次选择是一条完整的链。
    stage1_selected: str | None = None


@dataclass(frozen=True)
class WriterStatus:
    """写盘线程的一份只读快照。

    ``queued_*`` 是**最后入队**的那条，``written_*`` 是**最后确认落盘**的那条。
    两者不一致 = 还有东西没写完。以前只有"入队成功"这一个信号，于是磁盘慢、
    写入抛异常、关程序时队列被丢掉，界面都显示"已保存"——玩家只能在回放列表里
    发现那一局不见了。
    """

    queued_session: str | None = None
    queued_sequence: int | None = None
    written_session: str | None = None
    written_sequence: int | None = None
    pending: int = 0
    dropped: int = 0
    failures: int = 0
    last_error: str | None = None
    stopped: bool = False

    @property
    def settled(self) -> bool:
        """入队的最后一条是否已经确认写入。"""
        if self.queued_session is None and self.queued_sequence is None:
            return True
        return (self.queued_session, self.queued_sequence) == (
            self.written_session,
            self.written_sequence,
        )

    @property
    def degraded(self) -> bool:
        """丢过事件、写失败过、或者关闭超时。界面据此提示"记录降级"。"""
        return bool(self.dropped or self.failures or self.last_error)


class _AsyncJsonlWriter:
    """后台追加写 JSONL。**入队 ≠ 落盘**，两者必须分开报告。"""

    #: 队列上限。录制是一路追加写的，磁盘慢时无界队列会把内存吃光；
    #: 满了就丢最晚的事件并记 ``dropped``——宁可少一条，也不能让采集线程卡住。
    QUEUE_LIMIT = 4096

    def __init__(self, path: Path, *, limit: int | None = None) -> None:
        self.path = path
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=max(1, int(limit or self.QUEUE_LIMIT))
        )
        self._lock = threading.Lock()
        self._last_queued: tuple[str | None, int | None] | None = None
        self._last_written: tuple[str | None, int | None] | None = None
        self._dropped = 0
        self._failures = 0
        self._last_error: str | None = None
        self._stopped = False
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="HuahuaRunRecordWriter", daemon=True)
        self._thread.start()

    # ---- 写入 --------------------------------------------------------
    def submit(self, payload: dict[str, Any]) -> bool:
        """入队。返回 ``True`` 只表示**排上队了**，不代表已经落盘。"""
        with self._lock:
            if self._stopped:
                return False
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            with self._lock:
                self._dropped += 1
                self._last_error = "queue_full"
            return False
        with self._lock:
            self._last_queued = (payload.get("session"), payload.get("sequence"))
        return True

    def _run(self) -> None:
        handle = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            return
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if self._stopped:
                        # 停止接收 + 队列已空 = 排空完毕
                        return
                    continue
                if item is None:
                    return
                self._write(handle, item)
        finally:
            try:
                handle.close()
            except OSError:
                pass

    def _write(self, handle, item: dict[str, Any]) -> None:
        try:
            self._write_line(handle, item)
        except (OSError, ValueError, TypeError) as exc:
            # 单条写坏不能让整个录制线程死掉：记下来、继续写下一条，
            # 界面据 ``failures/last_error`` 显示"记录降级"。
            with self._lock:
                self._failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            self._last_written = (item.get("session"), item.get("sequence"))

    def _write_line(self, handle, item: dict[str, Any]) -> None:
        """真正落一行。测试可以覆盖它来模拟慢盘/写失败。"""
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    # ---- 状态 --------------------------------------------------------
    def status(self) -> WriterStatus:
        with self._lock:
            queued = self._last_queued
            written = self._last_written
            return WriterStatus(
                queued_session=queued[0] if queued else None,
                queued_sequence=queued[1] if queued else None,
                written_session=written[0] if written else None,
                written_sequence=written[1] if written else None,
                pending=int(self._queue.qsize()),
                dropped=int(self._dropped),
                failures=int(self._failures),
                last_error=self._last_error,
                stopped=bool(self._stopped),
            )

    def close(self, timeout: float = 1.0) -> WriterStatus:
        """停止接收 → 排空 → 确认写入。

        返回关闭后的快照；``status.settled`` 为 ``False`` 就是**没写完**
        （超时，或写失败），调用方必须把它说出来，不能说"已保存"。
        """
        with self._lock:
            self._stopped = True
            already = self._closed
            self._closed = True
        if not already and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))
        if not already and self._thread.is_alive():
            with self._lock:
                if not self._last_error:
                    self._last_error = "close_timeout"
        return self.status()


class AutomaticRunRecorder:
    """Convert stable fast-vision snapshots into a complete automatic run log."""

    def __init__(self, path: Path | None = AUTO_RUN_LOG, *, stable_frames: int = 2) -> None:
        self._lock = threading.RLock()
        self._writer = _AsyncJsonlWriter(path) if path is not None else None
        self._stable_frames = max(1, int(stable_frames))
        self.phase = "waiting"
        self.session_id: str | None = None
        self.started_at = 0.0
        self.sequence = 0
        self.initial_board: dict[str, Any] | None = None
        self.initial_inventory: list[dict[str, Any]] = []
        self._last_snapshot: FastVisionSnapshot | None = None
        self._candidate_snapshot: FastVisionSnapshot | None = None
        self._candidate_key: tuple | None = None
        self._candidate_frames = 0
        self._finish_frames = 0
        self._missing_frames = 0
        #: 上一条被判「整帧读错」的签名，同一张坏画面只报一次，别每帧刷一条。
        self._dropped_signature: str | None = None
        #: 连续被判「画面被整体替换」的帧数，达到
        #: :data:`SCENE_SWAP_ESCAPE_FRAMES` 就强制推进基准，见那条注释。
        self._scene_swap_frames = 0
        #: 当前这一帧是否被判成不可信（整帧作废）。结算时用它决定 final_board
        #: 用当前帧还是退回最后一张可信基准。
        self._frame_unreliable = False
        #: 逐格地形冲突 ``{(row, col): (observed_state, base_state, frames)}``。
        #: 「阻挡 ↔ 虚空」这种官方规则里不存在的对调先按旧值压住，连续
        #: :data:`TERRAIN_CONFLICT_CONFIRM_FRAMES` 帧同向才改判。**基准照常推进**
        #: （写回旧值），所以一次误读不会把整局锁死。
        self._terrain_conflict: dict[tuple[int, int], tuple[str, str, int]] = {}
        #: 已经报过 `terrain_conflict_held` 的 (格子, 观察值)，同一对只报一次。
        self._terrain_conflict_reported: set[tuple] = set()
        #: 每格等级状态 ``{(row, col): (kind, level, pending_level, pending_count)}``。
        #: 见 :data:`LEVEL_CHANGE_CONFIRM_FRAMES`——**升和降一视同仁**，都要连续
        #: 若干帧读到同一新值才生效。绝不写死「只升不降」：随机颜料 / 同内容颜料
        #: 覆盖时，高级确实会变成低级（用户 2026-11-18 明确纠正）。
        self._level_state: dict[tuple[int, int], tuple[tuple | None, int, int | None, int]] = {}
        #: 每格「已见最高等级」``{(row, col): (kind, level)}``。**只做旁注**，
        #: 写进 ``final_board.paints[].peak_level``，用于复盘「这一格曾经到过几级」。
        #: 它不参与上报值——否则又变成「只升不降」。
        self._level_peak: dict[tuple[int, int], tuple[tuple | None, int]] = {}
        #: 已经上报过的等级变化（``(row, col, from, to)``），同一对只报一次。
        self._level_reported: set[tuple] = set()
        #: 每格家族读数的投票箱 ``{(row, col): Counter(subtype -> 帧数)}``。
        #: 家族不像等级那样有「唯一正确值」——同一格前后读出两个家族，说明这一格
        #: **我们其实没认准**（r3c2 就是「赛季 Lv3」与「幽灵 Lv4」交替）。这种情况
        #: 不猜，写进 ``final_board.paints[].needs_review`` 交给用户复核。
        self._subtype_votes: dict[tuple[int, int], Counter] = {}
        self._pending_affix: _PendingAffixChoice | None = None
        self._visible_stage2: tuple[str, ...] = ()
        self._clicked_affix: tuple[str, float] | None = None
        #: 阶段一（转锋/堆砌/扩散）抓到的那次点击：``(名称, 时间, 卡片序号)``。
        #: 只靠「选中高亮」读不出用户到底点了哪个，点击是唯一确凿来源。
        self._clicked_stage1: tuple[str, float, int] | None = None
        #: 落在笔触面板里、但**当时还没读到「有哪几张卡」**的点击：``(x, y, 时间)``。
        #:
        #: 用户 2026-11-22 确认：两个阶段都是「先点卡片、再点确认」两步。而点击
        #: 往往比词缀文字 OCR 更早到达——stage2 还没读出来时点下的那一下，以前
        #: 直接被丢掉，于是那一局 0 条 ``affix_card_clicked``，只能退化成
        #: ``effect_inference`` 猜（16:42/16:43 两局实测就是这样）。
        #: 现在先按坐标存住，等卡片读出来再按 Y 分带回填。
        self._buffered_panel_clicks: list[tuple[float, float, float]] = []
        #: 最后一次看到的笔触面板矩形。点「确认」那一帧面板往往已经从视觉里消失，
        #: 没有这个兜底就没法判断「这一下点在面板上」。
        self._last_panel_rect: Rect | None = None
        #: 已经上报过选择的阶段一候选组。同一组候选只报一次 `brush_stage1_selected`，
        #: 免得每帧都补一条；候选变了（换面板/刷新）就重新允许上报。
        self._stage1_reported: tuple[str, ...] | None = None
        self._pending_force_clicked_at: float | None = None
        self._events: list[dict[str, Any]] = []
        self._installed_paints: Counter[tuple] = Counter()
        self._initial_paints: Counter[tuple] = Counter()
        self._force_count = 0
        self._last_finished_signature: str | None = None
        self._closed = False
        #: 界面状态灯要用的三个量：盘上路径、累计帧数、上一次出事件的时间。
        self._path = str(path) if path is not None else None
        self._frames = 0
        self._last_event_at: float | None = None
        self._finished_at: float | None = None
        atexit.register(self.close)

    def process(
        self,
        snapshot: FastVisionSnapshot | None,
        *,
        observed_at: float | None = None,
        clicks: tuple[tuple[float, float, float], ...] = (),
    ) -> RecorderFrameResult:
        started = time.perf_counter()
        now = time.time() if observed_at is None else float(observed_at)
        emitted: list[str] = []
        with self._lock:
            if snapshot is None:
                self._missing_frames += 1
                return RecorderFrameResult(
                    elapsed=time.perf_counter() - started,
                    phase=self.phase,
                    session_id=self.session_id,
                    events=(),
                )
            self._missing_frames = 0
            self._frames += 1
            start_visible = RegionKind.START_DRAWING.value in snapshot.regions
            finish_visible = RegionKind.FINISH_DRAWING.value in snapshot.regions

            if start_visible:
                if self.phase == "playing":
                    emitted.extend(self._finish_run(snapshot, now, reason="next_start_screen"))
                if self.phase in {"waiting", "finished"}:
                    emitted.extend(self._start_run(snapshot, now, reason="start_screen"))
                    self.phase = "opening"
            elif self.phase == "opening" and _has_active_gameplay(snapshot):
                self.phase = "playing"
                emitted.append(self._emit("run_active", {"board": _board_payload(snapshot)}, now))
            elif self.phase == "waiting" and _has_active_gameplay(snapshot):
                emitted.extend(self._start_run(snapshot, now, reason="active_scene"))
                self.phase = "playing"
            elif self.phase == "finished" and _looks_like_new_run(snapshot, self._last_finished_signature):
                emitted.extend(self._start_run(snapshot, now, reason="board_reset"))
                self.phase = "playing"

            if self.phase in {"opening", "playing"}:
                # 地形先结算：把「阻挡 ↔ 虚空」这种官方规则里不存在的逐格抖动压在旧值上，
                # 否则 `_accept_stable_snapshot` 会把它当成"这一帧瞎了"整帧否决，
                # 而**基准又不推进**，于是一次开局误读就把整局冻住（2026-11-20 实测：
                # 开局 4 格误读 → 之后 8 次连续丢帧 → 19 次落笔一条都没记下来）。
                snapshot, terrain_events = self._settle_terrain(snapshot, now)
                emitted.extend(terrain_events)
                snapshot, level_events = self._settle_levels(snapshot, now)
                emitted.extend(level_events)
                emitted.extend(self._record_control_clicks(snapshot, clicks))
                emitted.extend(self._refresh_initial_inventory(snapshot, now))
                emitted.extend(self._accept_stable_snapshot(snapshot, now))
                no_blank = int(snapshot.terrain_counts.get(TileState.BLANK.value, 0) or 0) == 0
                self._finish_frames = self._finish_frames + 1 if finish_visible or no_blank else 0
                if self.phase == "playing" and self._finish_frames >= self._stable_frames:
                    reason = "finish_button" if finish_visible else "no_blank_tiles"
                    # 结算盘面退回最后一张**可信**基准：这一帧若是入画画面（整帧作废），
                    # 拿它当 final_board 就等于把入画读数当成终局成绩（2026-11-18 实测）。
                    final_snapshot = (
                        self._last_snapshot
                        if self._frame_unreliable and self._last_snapshot is not None
                        else snapshot
                    )
                    emitted.extend(self._finish_run(final_snapshot, now, reason=reason))

        return RecorderFrameResult(
            elapsed=time.perf_counter() - started,
            phase=self.phase,
            session_id=self.session_id,
            events=tuple(emitted),
        )

    def status(self, *, now: float | None = None) -> RecorderStatus:
        """锁内取一份只读快照，界面线程只认这个。"""
        moment = time.time() if now is None else float(now)
        with self._lock:
            last_event_age = None
            if self._last_event_at is not None:
                last_event_age = max(0.0, moment - self._last_event_at)
            duration = 0.0
            if self.started_at:
                stop = self._finished_at if self._finished_at is not None else moment
                duration = max(0.0, stop - self.started_at)
            writer = self._writer
            writer_status = writer.status() if writer is not None else None
            return RecorderStatus(
                phase=self.phase,
                session_id=self.session_id,
                event_count=len(self._events),
                duration=duration,
                frames=self._frames,
                missing_frames=self._missing_frames,
                last_event_age=last_event_age,
                path=self._path,
                writer_pending=writer_status.pending if writer_status else 0,
                writer_written_sequence=(
                    writer_status.written_sequence if writer_status else None
                ),
                writer_dropped=writer_status.dropped if writer_status else 0,
                writer_failures=writer_status.failures if writer_status else 0,
                writer_error=writer_status.last_error if writer_status else None,
            )

    def close(self) -> None:
        """停录并确认落盘。返回值留给 :meth:`close_report` 查询。

        顺序是死的：**停止接收 → 排空队列 → 确认写入 → 报告超时**。
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            writer = self._writer
            self._writer = None
        if writer is None:
            self._close_report = {"settled": True, "error": None, "pending": 0}
            return
        status = writer.close()
        self._close_report = {
            "settled": bool(status.settled) and not status.failures,
            "error": status.last_error,
            "pending": status.pending,
            "dropped": status.dropped,
            "failures": status.failures,
            "last_written_sequence": status.written_sequence,
            "last_written_session": status.written_session,
        }

    def close_report(self) -> dict[str, Any]:
        """上一次 :meth:`close` 的确认结果。没关过就是一个空报告。"""
        report = getattr(self, "_close_report", None)
        if report is None:
            return {"settled": None, "error": None, "pending": 0}
        return dict(report)

    def _start_run(self, snapshot: FastVisionSnapshot, now: float, *, reason: str) -> list[str]:
        self.session_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + uuid.uuid4().hex[:6]
        self.started_at = now
        self.sequence = 0
        self._finished_at = None
        self.initial_board = _board_payload(snapshot)
        self.initial_inventory = _backpack_payload(snapshot)
        self._last_snapshot = snapshot
        self._candidate_snapshot = None
        self._candidate_key = None
        self._candidate_frames = 0
        self._finish_frames = 0
        self._pending_affix = None
        #: 新一局：等级状态与「已上报」全部作废，否则上一局的读数会污染这一局。
        self._level_state = {}
        self._level_peak = {}
        self._level_reported = set()
        #: 地形冲突同理：上一局压住的格子不能带到新一局，否则新局那块地会被
        #: 上一局的旧读数压住（棋盘已经重排，同一坐标根本不是同一块地）。
        self._terrain_conflict = {}
        self._terrain_conflict_reported = set()
        self._scene_swap_frames = 0
        self._frame_unreliable = False
        self._subtype_votes = {}
        self._visible_stage2 = ()
        self._clicked_affix = None
        self._clicked_stage1 = None
        self._buffered_panel_clicks = []
        self._last_panel_rect = None
        self._stage1_reported = None
        self._pending_force_clicked_at = None
        self._events = []
        self._installed_paints = Counter()
        self._initial_paints = Counter(
            _paint_key(tile)
            for tile in snapshot.tiles
            if tile.state == TileState.PAINT.value
        )
        self._force_count = 0
        self.phase = "opening" if RegionKind.START_DRAWING.value in snapshot.regions else "playing"
        # 「中途接入」只认一件事：没看到开始画面，且棋盘上的颜料**多于**开局自带
        # 的那一块。多出来的那些只能是这一局已经落下的，才叫中途接入。
        attached_mid_run = (
            reason == "active_scene"
            and sum(self._initial_paints.values()) > INITIAL_PAINTS_AT_START
        )
        return [
            self._emit(
                "run_started",
                {
                    "reason": reason,
                    "initial_board": self.initial_board,
                    "initial_inventory": self.initial_inventory,
                    "attached_mid_run": attached_mid_run,
                },
                now,
            )
        ]

    def _refresh_initial_inventory(
        self, snapshot: FastVisionSnapshot, now: float
    ) -> list[str]:
        """把开局库存补全成最终版。

        开局那一帧几乎必然读不到**家族**：背包认家族是后台线程做的
        （一格 40~60ms，不能放在帧里），`_start_run` 抓的那一帧
        ``subtype`` 一定还是 None。实测 2026-11-18 那局 15 格家族全空，
        复盘时只能看到「怪物 Lv2 ×3 / Lv1 ×9」这种粒度，和用户的实际库存对不上。

        两条补全路径，都只补不猜：

        1. ``initial_inventory`` 是空的（开局那一帧背包面板还没出来）→ 用
           第一次读到背包的那一帧当库存，并单独记一条
           ``initial_inventory_captured``，把「这是补拍、可能已经少了几枚」
           写清楚。
        2. 已有库存 → 按 ``(页, slot)`` 对齐，**类别和等级都对得上**才把空着的
           ``subtype``/``treasure`` 填上去。对不上说明这一格已经被消耗或
           被重排，绝不改写。

        **按页分开存**（用户 2026-11-18 定口径）：背包一页只有 3 排 15 格，
        整包 30 枚要看两次——怪物初始颜料时按怪物排一页、玩法初始颜料时按玩法
        排一页。没有 ``backpack_page`` 时两页读数会互相覆盖，第二页一来就把
        第一页整页挤掉，30 枚只能记到 15 枚。
        """
        paints = [paint for paint in snapshot.backpack if getattr(paint, "slot_index", None) is not None]
        if not paints:
            return []
        page = str(getattr(snapshot, "backpack_page", "") or "")
        payload = _backpack_payload(snapshot)
        if not self.initial_inventory:
            self.initial_inventory = payload
            return [
                self._emit(
                    "initial_inventory_captured",
                    {
                        "page": page,
                        "slots": len(self.initial_inventory),
                        "delay": round(max(0.0, now - self.started_at), 1) if self.started_at else None,
                        "note": "开局那一帧没有背包读数，这是第一帧读到的库存；此时可能已经放过颜料。",
                        "inventory": self.initial_inventory,
                    },
                    now,
                )
            ]
        known = {(item.get("page") or "", item.get("slot")): item for item in self.initial_inventory}
        known_pages = {item.get("page") or "" for item in self.initial_inventory}
        # 只有**这个页还一条都没有**时才整页收下。已经存在的页不再补新 slot：
        # 放掉一枚以后整列会上移，第 16 个 slot 是「下一枚被挤进来」而不是新库存，
        # 照收会让 30 格变成 40+。
        collect_whole_page = page not in known_pages
        new_page_added = 0
        filled = 0
        for paint, item in zip(paints, payload):
            existing = known.get((page, paint.slot_index))
            if existing is None:
                if not collect_whole_page:
                    continue
                self.initial_inventory.append(item)
                known[(page, paint.slot_index)] = item
                new_page_added += 1
                continue
            if existing.get("category") != paint.category or existing.get("level") != paint.level:
                continue
            if not existing.get("subtype") and paint.subtype:
                existing["subtype"] = paint.subtype
                filled += 1
            if not existing.get("treasure") and paint.is_treasure:
                existing["treasure"] = True
        emitted: list[str] = []
        if new_page_added:
            emitted.append(
                self._emit(
                    "initial_inventory_page_added",
                    {"page": page, "slots": new_page_added, "total": len(self.initial_inventory)},
                    now,
                )
            )
        if filled:
            emitted.append(
                self._emit(
                    "initial_inventory_filled",
                    {"page": page, "filled": filled, "total": len(self.initial_inventory)},
                    now,
                )
            )
        return emitted

    def _finish_run(self, snapshot: FastVisionSnapshot, now: float, *, reason: str) -> list[str]:
        if self.session_id is None or self.phase == "finished":
            return []
        emitted: list[str] = []
        if self._pending_affix is not None:
            emitted.append(
                self._emit(
                    "affix_choice_unresolved",
                    {
                        "options": list(self._pending_affix.options),
                        "brush": self._pending_affix.brush,
                        "brush_name": _brush_name(self._pending_affix.brush),
                        "stage1_options": list(self._pending_affix.stage1_options),
                        "confidence": 0.35,
                    },
                    now,
                )
            )
            self._pending_affix = None
        summary = {
            "reason": reason,
            "duration": round(max(0.0, now - self.started_at), 3),
            "initial_board": self.initial_board,
            "final_board": _board_payload(snapshot, self._level_peak, self._subtype_votes),
            "initial_inventory": self.initial_inventory,
            "inferred_initial_paints": _counter_payload(self._initial_paints),
            "installed_paints": _counter_payload(self._installed_paints),
            "force_complete_count": self._force_count,
            "event_count": len(self._events),
            "events": list(self._events),
        }
        emitted.append(self._emit("run_finished", summary, now, include_in_session=False))
        self._last_finished_signature = snapshot.board_signature
        self.phase = "finished"
        self._finished_at = now
        self._finish_frames = 0
        return emitted

    def _settle_terrain(
        self, snapshot: FastVisionSnapshot, now: float
    ) -> tuple[FastVisionSnapshot, list[str]]:
        """压住「阻挡 ↔ 虚空」的逐格抖动，**并且让基准照常推进**。

        为什么需要这一层（2026-11-20 实机取证）：

        - 那局开局把 4 个虚空格读成阻挡 ``(1,3) (2,5) (3,0) (3,4)``；终局盘面
          ``0 阻挡 / 16 虚空``，证明开局那 4 个 K 是误读。
        - 之后每一帧都稳定地读到 void，于是相对出错的开局基准帧帧都是
          ``blocked→void``。旧逻辑见到 ≥2 格这种对调就**整帧丢弃**，而丢帧时
          ``_last_snapshot`` **不推进** → 基准永远停在开局那一帧 → 26.6s 起连续
          8 次全丢，整局 19 次落笔、10 次笔触一条都没进日志。
        - 判据本身没错（官方扩散 18 条里阻挡只能变空白、虚空也只能变空白，两者
          不能互换），错在"一刀切丢整帧 + 基准不推进"。

        现在的处理：这些格**按旧值上报**（写回 settled 状态），基准照常推进到修正后
        的快照上；只有当同一格连续 :data:`TERRAIN_CONFLICT_CONFIRM_FRAMES` 帧都读到
        同一个新值时才改判，并额外记一条 ``terrain_conflict_resolved``。

        整帧级别的否决仍然保留，但门槛抬到 :data:`SCENE_SWAP_CONFLICT_TILES`，
        并且要求"伴随颜料崩盘"（见 :func:`_frame_looks_replaced`）——因为局部误读
        和入画画面的格数完全重叠，只能靠这条补充判据分开。
        """
        previous = self._last_snapshot
        if previous is None:
            return snapshot, []
        before = {(tile.row, tile.col): tile for tile in previous.tiles}
        conflicted: list[tuple[FastTile, FastTile]] = []
        for tile in snapshot.tiles:
            old = before.get((tile.row, tile.col))
            if old is None:
                continue
            if (str(old.state), str(tile.state)) in TERRAIN_CONFLICT_PAIRS:
                conflicted.append((tile, old))
        if not conflicted:
            # 这一帧没有任何地形对调：冲突计数全部作废（格子已被颜料覆盖/换成别的东西）。
            self._terrain_conflict.clear()
            self._terrain_conflict_reported.clear()
            return snapshot, []
        if _frame_looks_replaced(previous, snapshot, len(conflicted)):
            # 整盘重画（入画/结算画面）：不逐格压，交给 `_accept_stable_snapshot` 整帧否决。
            return snapshot, []

        conflict_cells = {(tile.row, tile.col) for tile, _ in conflicted}
        tiles: list[FastTile] = []
        events: list[str] = []
        for tile in snapshot.tiles:
            key = (tile.row, tile.col)
            if key not in conflict_cells:
                self._terrain_conflict.pop(key, None)
                tiles.append(tile)
                continue
            old = before[key]
            base = str(old.state)
            observed = str(tile.state)
            record = self._terrain_conflict.get(key)
            if record is not None and record[0] == observed and record[1] == base:
                frames = record[2] + 1
            else:
                frames = 1
            if frames >= TERRAIN_CONFLICT_CONFIRM_FRAMES:
                # 连续同向：认了。这一帧原样放行，`_record_transition` 会把它记成地形变化。
                self._terrain_conflict.pop(key, None)
                tiles.append(tile)
                events.append(
                    self._emit(
                        "terrain_conflict_resolved",
                        {
                            "row": tile.row,
                            "col": tile.col,
                            "from": base,
                            "to": observed,
                            "frames": frames,
                            "note": "连续多帧读到同一个新值，判定先前的读数有误，按新值记录。",
                        },
                        now,
                    )
                )
                continue
            self._terrain_conflict[key] = (observed, base, frames)
            tiles.append(replace(tile, state=old.state))
            marker = (key, observed)
            if marker not in self._terrain_conflict_reported:
                self._terrain_conflict_reported.add(marker)
                events.append(
                    self._emit(
                        "terrain_conflict_held",
                        {
                            "row": tile.row,
                            "col": tile.col,
                            "held": base,
                            "observed": observed,
                            "note": "官方规则里阻挡与虚空不能互换，暂按上一帧的地形记录，等下一帧复核。",
                        },
                        now,
                    )
                )
        return replace(
            snapshot,
            tiles=tuple(tiles),
            terrain_counts=_terrain_counts(tuple(tiles)),
        ), events

    def _settle_levels(
        self, snapshot: FastVisionSnapshot, now: float
    ) -> tuple[FastVisionSnapshot, list[str]]:
        """把单帧跳变的等级读数压住，**但不封方向**。

        - 同一格同一枚颜料（类别 + 家族 + 宝箱都相同）的等级**变了**：要连续
          ``LEVEL_CHANGE_CONFIRM_FRAMES`` 帧读到同一新值才认，升、降一视同仁。
          随机颜料/同内容颜料覆盖导致的高级→低级，两帧之后照常落地。
        - 换颜料（kind 变）或被清空的格子：状态直接重置，新读数立刻生效。
        - 每格「已见最高等级」只记进 ``_level_peak`` 当旁注，**不参与上报值**。

        实机依据：2026-11-18 那局 r2c3 黑帆真实 Lv7，边框逐帧读成 6 → 4 → 7 → 5
        → 1。以前靠「只升不降」把它按 7 记；现在改由 ``frame_level`` 的几何判据
        （白色菱形细线 + 彩色辉光，形状不随动画变）在源头上读出 7，这里只负责
        把单帧噪声挡掉，所以可以对称地允许降级。
        """
        tiles: list[FastTile] = []
        changes: list[dict[str, Any]] = []
        for tile in snapshot.tiles:
            key = (tile.row, tile.col)
            if str(tile.state) != TileState.PAINT.value:
                # 格子被清成空白/虚空 → 这枚颜料没了，状态与峰值一起作废。
                self._level_state.pop(key, None)
                self._level_peak.pop(key, None)
                tiles.append(tile)
                continue
            level = tile.paint_level
            if level is None:
                # 这一帧没读出等级：不动状态，也不改读数（上层按低置信处理）。
                tiles.append(tile)
                continue
            kind = _tile_kind(tile)
            if tile.paint_subtype:
                self._subtype_votes.setdefault(key, Counter())[tile.paint_subtype] += 1
            record = self._level_state.get(key)
            if record is None or record[0] != kind:
                # 首次见到 / 换了颜料：读数直接生效（新颜料本来就该按新等级记）。
                self._level_state[key] = (kind, level, None, 0)
                self._level_peak[key] = (kind, level)
                tiles.append(tile)
                continue
            settled = record[1]
            if level == settled:
                self._level_state[key] = (kind, settled, None, 0)
                tiles.append(tile)
                continue
            pending = record[2]
            count = record[3] + 1 if pending == level else 1
            if count >= LEVEL_CHANGE_CONFIRM_FRAMES:
                self._level_state[key] = (kind, level, None, 0)
                peak = self._level_peak.get(key)
                if peak is None or peak[0] != kind or level > peak[1]:
                    self._level_peak[key] = (kind, level)
                changes.append(
                    {
                        "row": tile.row,
                        "col": tile.col,
                        "from": settled,
                        "to": level,
                        "category": tile.paint_category,
                        "subtype": tile.paint_subtype,
                        "direction": "down" if level < settled else "up",
                    }
                )
                tiles.append(tile)
                continue
            # 单帧跳变：按已稳定值上报，只暂存待确认值。
            self._level_state[key] = (kind, settled, level, count)
            tiles.append(replace(tile, paint_level=settled))
        if not changes:
            return snapshot, []
        events: list[str] = []
        for item in changes:
            marker = (item["row"], item["col"], item["from"], item["to"])
            if marker in self._level_reported:
                continue
            self._level_reported.add(marker)
            events.append(self._emit("level_changed", item, now))
        return replace(snapshot, tiles=tuple(tiles)), events

    def _accept_stable_snapshot(self, snapshot: FastVisionSnapshot, now: float) -> list[str]:
        key = _semantic_snapshot_key(snapshot)
        if key == self._candidate_key:
            self._candidate_frames += 1
        else:
            self._candidate_key = key
            self._candidate_snapshot = snapshot
            self._candidate_frames = 1
        if self._candidate_frames < self._stable_frames or self._candidate_snapshot is None:
            return []
        stable = self._candidate_snapshot
        self._candidate_frames = 0
        if self._last_snapshot is not None and _semantic_snapshot_key(self._last_snapshot) == key:
            self._frame_unreliable = False
            return []
        previous = self._last_snapshot
        if previous is not None:
            changes = _tile_changes(previous, stable)
            illegal = _illegal_terrain_changes(changes)
            # 逐格抖动已经在 `_settle_terrain` 里压掉了，能走到这里的成片对调只可能是
            # 画面被整体替换（入画/结算画面把阻挡与虚空画成同一种暗色）。门槛按整盘
            # 比例算，并且要求同时有颜料崩盘——两者都满足才否掉整帧。
            if _frame_looks_replaced(previous, stable, len(illegal)):
                self._candidate_frames = 0
                self._frame_unreliable = True
                self._scene_swap_frames += 1
                if self._scene_swap_frames >= SCENE_SWAP_ESCAPE_FRAMES:
                    # 逃生阀：连着这么多帧判"整盘被替换"，更像识别器彻底失灵或玩家真把
                    # 半盘地形改了。再否决下去整局记录就停摆了，宁可推进基准并留证据。
                    forced = self._scene_swap_frames
                    self._scene_swap_frames = 0
                    self._frame_unreliable = False
                    self._last_snapshot = stable
                    return [
                        self._emit(
                            "unreliable_frame_forced",
                            {
                                "cells": illegal,
                                "count": len(illegal),
                                "frames": forced,
                                "reason": "scene_swap_escape",
                                "note": (
                                    "连续多帧都被判成整盘替换，已放弃否决并推进基准。"
                                    "这一局可能从中途开始就记错了，请人工复核盘面。"
                                ),
                            },
                            now,
                        )
                    ]
                if stable.board_signature == self._dropped_signature:
                    return []
                self._dropped_signature = stable.board_signature
                return [
                    self._emit(
                        "unreliable_frame_dropped",
                        {
                            "cells": illegal,
                            "count": len(illegal),
                            "reason": "scene_swap",
                            "note": (
                                "官方规则里阻挡地块只能变成空白，不能变成虚空；"
                                "成片对调且伴随颜料崩盘，说明这一帧读到的是入画/结算画面。"
                            ),
                        },
                        now,
                    )
                ]
        self._scene_swap_frames = 0
        self._frame_unreliable = False
        self._last_snapshot = stable
        if previous is None:
            return []
        return self._record_transition(previous, stable, now)

    def _record_transition(
        self,
        previous: FastVisionSnapshot,
        current: FastVisionSnapshot,
        now: float,
    ) -> list[str]:
        emitted: list[str] = []
        changes = _tile_changes(previous, current)
        brush_events = self._record_brush_transition(previous, current, now)
        emitted.extend(brush_events)

        if changes:
            emitted.append(
                self._emit(
                    "board_transition",
                    {
                        "changes": changes,
                        "before_signature": previous.board_signature,
                        "after_signature": current.board_signature,
                        "board": _board_payload(current),
                    },
                    now,
                )
            )

        added = [item for item in changes if item["from"]["state"] != TileState.PAINT.value and item["to"]["state"] == TileState.PAINT.value]
        upgraded = [
            item
            for item in changes
            if item["from"]["state"] == item["to"]["state"] == TileState.PAINT.value
            and _level(item["to"]) > _level(item["from"])
        ]
        converted = [
            item
            for item in changes
            if item["from"]["state"] == item["to"]["state"] == TileState.PAINT.value
            and item["from"].get("category") not in {None, "unknown"}
            and item["to"].get("category") not in {None, "unknown"}
            and item["from"].get("category") != item["to"].get("category")
        ]
        faded = [item for item in changes if item["to"]["state"] == TileState.FADED.value]
        terrain = [
            item
            for item in changes
            if item["from"]["state"] != item["to"]["state"]
            and item not in added
            and item not in faded
        ]
        consumed = _consumed_backpack_paints(previous, current)
        buff_changed = _buff_only_changes(changes)

        if added:
            initial_source = bool(
                self._pending_affix is not None
                or previous.brush is not None
                or not previous.backpack
            ) and not consumed
            kind = "initial_paint_placed" if initial_source else "paint_installed"
            # 一律用**地块读数**的键累加。以前「有背包消耗」时改用背包差值的键，
            # 那是 (类别, 等级, 家族, 宝箱)，家族还没算出来的帧会给出 None，
            # 于是同一枚颜料在台账里一会儿有家族一会儿没有。地块读数的家族更准。
            for item in added:
                key = _paint_key_from_payload(item["to"])
                if initial_source:
                    self._initial_paints[key] += 1
                else:
                    self._installed_paints[key] += 1
            emitted.append(
                self._emit(
                    kind,
                    {
                        "cells": added,
                        "count": len(added),
                        "consumed_from_backpack": _counter_payload(consumed),
                        "source": "initial_paint_inferred" if initial_source else "backpack_inferred",
                        # 判据摊开写，复盘时能看出这条分类到底靠什么：
                        # 只有「背包没扣」这一条是真证据，另两条是场景旁证。
                        "evidence": {
                            "backpack_consumed": bool(consumed),
                            "backpack_visible": bool(previous.backpack),
                            "brush_panel_visible": previous.brush is not None,
                            "affix_pending": self._pending_affix is not None,
                        },
                    },
                    now,
                )
            )
        if upgraded:
            emitted.append(self._emit("paint_upgraded", {"cells": upgraded, "count": len(upgraded)}, now))
        if converted:
            directions = Counter(
                f"{item['from'].get('category')}_to_{item['to'].get('category')}"
                for item in converted
            )
            emitted.append(
                self._emit(
                    "paint_category_changed",
                    {"cells": converted, "count": len(converted), "directions": dict(directions)},
                    now,
                )
            )
        if terrain:
            emitted.append(self._emit("terrain_changed", {"cells": terrain, "count": len(terrain)}, now))
        if buff_changed:
            # 画面上一格都没动，但地块加成变了：这是「叠 buff」唯一能留痕的地方。
            # 视觉看不到这两项，只有模型回填时才可能出现，出现就说明规则推演补上了一步。
            emitted.append(
                self._emit(
                    "tile_buff_changed",
                    {
                        "cells": buff_changed,
                        "count": len(buff_changed),
                        "bonus_delta": sum(
                            int(item["to"].get("bonus_level") or 0)
                            - int(item["from"].get("bonus_level") or 0)
                            for item in buff_changed
                        ),
                    },
                    now,
                )
            )
        if faded:
            self._force_count += 1
            emitted.append(
                self._emit(
                    "force_complete",
                    {
                        "ordinal": self._force_count,
                        "faded_cells": faded,
                        "faded_count": len(faded),
                        "clicked_at": self._pending_force_clicked_at,
                        "current_before": _current_payload(previous.current),
                    },
                    now,
                )
            )
            self._pending_force_clicked_at = None

        current_changed = _current_key(previous.current) != _current_key(current.current)
        if current_changed:
            emitted.append(
                self._emit(
                    "current_paint_changed",
                    {"before": _current_payload(previous.current), "after": _current_payload(current.current)},
                    now,
                )
            )

        if self._pending_affix is not None:
            resolution = _resolve_pending_affix(self._pending_affix, previous, current, changes)
            if resolution is not None:
                emitted.append(self._emit("affix_choice_resolved", resolution, now))
                self._pending_affix = None
        return emitted

    def _record_brush_transition(
        self,
        previous: FastVisionSnapshot,
        current: FastVisionSnapshot,
        now: float,
    ) -> list[str]:
        emitted: list[str] = []
        before = previous.brush
        after = current.brush
        before_stage2 = before.stage2_options if before is not None else ()
        after_stage2 = after.stage2_options if after is not None else ()
        before_stage1 = before.stage1_options if before is not None else ()
        after_stage1 = after.stage1_options if after is not None else ()

        # ---- 阶段一：先记「出现了哪几个大类」，再记「选了哪个 ----
        if after_stage1 and after_stage1 != before_stage1:
            self._stage1_reported = None
            emitted.append(
                self._emit(
                    "brush_stage1_options_observed",
                    {
                        "options": list(after_stage1),
                        "names": [_brush_name(option) for option in after_stage1],
                        "codes": [brush_code(option) for option in after_stage1],
                        "chosen": after.chosen_stage1 if after is not None else None,
                        "stage2_visible": bool(after_stage2),
                    },
                    now,
                )
            )
        chosen_before = before.chosen_stage1 if before is not None else None
        chosen_after = after.chosen_stage1 if after is not None else None
        # 「列表从多张窄化成一张」= 那张就是确认后的选择。
        #
        # 用户 2026-11-22 19:40 那局按「阶段一位次」报的答案，与三条独立记录对照：
        #   [转锋,扩散] → [扩散]  : 用户第 2 次报 #2 = 扩散 ✓
        #   [转锋,堆砌] → [转锋]  : 用户第 3 次报 #1 = 转锋 ✓
        #   [转锋,堆砌] → [堆砌]  : 用户第 4 次报 #2 = 堆砌 ✓
        # 三次全中。而同一局第 3 次按点击坐标会算成 idx=1（堆砌）——那一下其实是
        # 「确认」按钮落在下面那张卡的位置上。所以窄化比坐标硬，用它定案。
        narrowed = (
            len(before_stage1) > 1
            and len(after_stage1) == 1
            and after_stage1[0] in before_stage1
        )
        if narrowed:
            self._clicked_stage1 = (after_stage1[0], now, before_stage1.index(after_stage1[0]))
        settled = after_stage1[0] if narrowed else chosen_after
        if (
            settled
            and settled != chosen_before
            and self._stage1_reported != after_stage1
        ):
            clicked = self._clicked_stage1
            from_click = clicked is not None and clicked[0] == settled
            self._stage1_reported = after_stage1
            emitted.append(
                self._emit(
                    "brush_stage1_selected",
                    {
                        "selected": settled,
                        "selected_name": _brush_name(settled),
                        "code": brush_code(settled),
                        "options": list(after_stage1 or before_stage1),
                        "source": (
                            "narrowed" if narrowed else ("captured_click" if from_click else "card_highlight")
                        ),
                        "card_index": clicked[2] if from_click else None,
                        "stage2_options": list(after_stage2),
                    },
                    now,
                )
            )
            if from_click:
                self._clicked_stage1 = None

        if after_stage2 and after_stage2 != before_stage2:
            # 卡片刚读出来：看看之前有没有「面板还没读出来时就点下去」的一下。
            # 有就按 Y 分带回填成具体的词缀选择——这比事后靠等级变化反推可靠得多。
            self._resolve_buffered_panel_click(after, after_stage2, emitted, now)
            if before_stage2:
                emitted.append(
                    self._emit(
                        "affix_options_refreshed",
                        {"before": list(before_stage2), "after": list(after_stage2), "brush": after.chosen_stage1},
                        now,
                    )
                )
            elif before is not None:
                # 进到阶段二 = 阶段一这一步已经落定。若选中高亮/点击都没读到，
                # 这里补一条，保证「选了哪一类」在日志里一定存在。
                if self._stage1_reported != (after_stage1 or before_stage1):
                    self._stage1_reported = after_stage1 or before_stage1
                    emitted.append(
                        self._emit(
                            "brush_stage1_selected",
                            {
                                "selected": chosen_after,
                                "selected_name": _brush_name(chosen_after),
                                "code": brush_code(chosen_after),
                                "options": list(before_stage1),
                                "source": "stage2_entry",
                                "card_index": None,
                                "stage2_options": list(after_stage2),
                            },
                            now,
                        )
                    )
            self._visible_stage2 = after_stage2

        if before_stage2 and not after_stage2:
            self._pending_affix = _PendingAffixChoice(
                options=before_stage2,
                brush=before.chosen_stage1 if before is not None else None,
                observed_at=now,
                event_sequence=self.sequence,
                stage1_options=before_stage1,
                stage1_selected=before.chosen_stage1 if before is not None else None,
            )
            emitted.append(
                self._emit(
                    "affix_choice_observed",
                    {
                        "options": list(before_stage2),
                        "brush": self._pending_affix.brush,
                        "brush_name": _brush_name(self._pending_affix.brush),
                        "brush_code": brush_code(self._pending_affix.brush),
                        "stage1_options": list(before_stage1),
                        "status": "awaiting_effect_inference",
                    },
                    now,
                )
            )
            self._visible_stage2 = ()
            # 面板消失这一帧再回填一次：点「确认」关闭面板时，视觉里已经看不到
            # 卡片了，但点击坐标还留在缓存里。这是最容易漏掉的一次机会。
            if self._clicked_affix is None and before is not None:
                self._resolve_buffered_panel_click(before, before_stage2, emitted, now)
            clicked = self._clicked_affix
            if clicked is not None and clicked[0] in before_stage2:
                selected = resolve_affix(clicked[0])
                emitted.append(
                    self._emit(
                        "affix_choice_resolved",
                        {
                            "selected": clicked[0],
                            "code": affix_code(selected) if selected is not None else None,
                            "brush": selected.brush_type.value if selected is not None else self._pending_affix.brush,
                            "brush_name": _brush_name(
                                selected.brush_type.value if selected is not None else self._pending_affix.brush
                            ),
                            "effect": selected.effect.value if selected is not None else None,
                            "options": list(before_stage2),
                            "stage1_options": list(before_stage1),
                            "confidence": 0.99,
                            "source": "captured_click",
                            "clicked_at": round(clicked[1], 3),
                        },
                        now,
                    )
                )
                self._pending_affix = None
            self._clicked_affix = None
        if not after_stage1 and after_stage2:
            # 面板刷新后可能只读到词缀、读不到大类：别把上报状态挂死。
            self._stage1_reported = None
        return emitted

    def _resolve_buffered_panel_click(
        self,
        brush: FastBrush,
        stage2_options: tuple[str, ...],
        emitted: list[str],
        now: float,
    ) -> None:
        """把「卡片还没读出来时点下的那一下」回填成具体的词缀选择。

        用户 2026-11-22 确认：第二阶段是**点词缀卡 + 点确认**两步，而且点完不
        立即生效，要装到地块才生效。问题在于点击通常比词缀文字 OCR 更早到达，
        ``stage2_options`` 还是空的时候点下的那一下以前直接丢掉，整局就 0 条
        ``affix_card_clicked``，只能退化成 ``effect_inference`` 猜。

        回填后写进 ``self._clicked_affix``，交给面板关闭时那段现成的逻辑收尾——
        那边会标 ``source="captured_click"``、置信度 0.99，比反推硬气得多。
        """

        if self._clicked_affix is not None:
            return
        if brush is None or not stage2_options:
            return
        fresh = [(x, y, at) for x, y, at in self._buffered_panel_clicks if now - at <= PENDING_CLICK_BUFFER_SECONDS]
        self._buffered_panel_clicks = fresh
        if not fresh:
            return
        # 取**最后一次**落在面板里的点击。卡片点击之后紧跟着的是「确认」，两下都
        # 落在卡片区域内，所以「最后一下」和「第一下」只要序号相同就是同一次操作。
        chosen: tuple[float, float, float] | None = None
        for entry in fresh:
            if _point_in_rect(entry[0], entry[1], brush.panel_rect):
                chosen = entry
        if chosen is None:
            return
        x, y, clicked_at = chosen
        relative_y = (y - brush.panel_rect.y) / max(1, brush.panel_rect.h)
        index = min(len(stage2_options) - 1, max(0, int(relative_y * len(stage2_options))))
        name = stage2_options[index]
        self._clicked_affix = (name, clicked_at)
        self._buffered_panel_clicks = []
        emitted.append(
            self._emit(
                "affix_card_clicked",
                {
                    "selected": name,
                    "card_index": index,
                    "options": list(stage2_options),
                    "brush": brush.chosen_stage1,
                    "brush_name": _brush_name(brush.chosen_stage1),
                    "confidence": 0.90,
                    "source": "buffered_click",
                    "clicked_at": round(clicked_at, 3),
                    "note": "点击早于词缀文字 OCR，按点击纵坐标回填卡片序号",
                },
                clicked_at,
            )
        )

    def _record_control_clicks(
        self,
        snapshot: FastVisionSnapshot,
        clicks: tuple[tuple[float, float, float], ...],
    ) -> list[str]:
        if not clicks:
            return []
        emitted: list[str] = []
        brush = snapshot.brush or (self._last_snapshot.brush if self._last_snapshot is not None else None)
        if brush is not None:
            self._last_panel_rect = brush.panel_rect
        for clicked_at, x, y in clicks:
            if brush is not None and brush.stage2_options and _point_in_rect(x, y, brush.panel_rect):
                relative_y = (y - brush.panel_rect.y) / max(1, brush.panel_rect.h)
                index = 0 if relative_y < 0.52 else 1
                if 0 <= index < len(brush.stage2_options):
                    name = brush.stage2_options[index]
                    # 同一条卡连点两下 = 「点卡片 + 点确认」，只算一次选择。
                    #
                    # 为什么不去排除确认按钮的矩形：那个矩形是**几何估算**的
                    # （``y+0.80h ~ 0.91h``），而下面那张词缀卡正好也占到 0.9h，
                    # 硬排除会把「点下面那张卡」一起吃掉了——2026-11-22 17:21
                    # 那一局就是这样，整局 0 条 affix_card_clicked。
                    # 改按「同序号去重」判断，既不依赖那个不准的矩形，也符合
                    # 「确认不会改变选择」这个事实。
                    if (
                        self._clicked_affix is not None
                        and self._clicked_affix[0] == name
                        and clicked_at - self._clicked_affix[1] <= CONFIRM_DEDUPE_SECONDS
                    ):
                        continue
                    self._clicked_affix = (name, clicked_at)
                    emitted.append(
                        self._emit(
                            "affix_card_clicked",
                            {
                                "selected": name,
                                "card_index": index,
                                "options": list(brush.stage2_options),
                                "brush": brush.chosen_stage1,
                                "brush_name": _brush_name(brush.chosen_stage1),
                                "confidence": 0.99,
                            },
                            clicked_at,
                        )
                    )
                    continue
            if brush is not None and brush.stage1_options and _point_in_rect(x, y, brush.panel_rect):
                # 阶段一同样是**上下排列**（用户 2026-11-22 明确纠正：不是横排三张）。
                # 之前这里按「面板顶部 34% 之内 + X 坐标等分」反推，两个后果：
                #   * 点下面那张卡（relative_y > 0.34）永远记不到；
                #   * 点上面那张卡时用 X 反推索引，与真实位置无关，等于随机。
                # 现在改成按 Y 分带，和阶段二同一套几何。
                relative_y = (y - brush.panel_rect.y) / max(1, brush.panel_rect.h)
                count = len(brush.stage1_options)
                index = min(count - 1, max(0, int(relative_y * count)))
                name = brush.stage1_options[index]
                # 同上面阶段二：点卡片后紧跟着的「确认」会落在同一张卡上，
                # 同序号且时间相邻就不重复记，只保留第一次。
                if (
                    self._clicked_stage1 is not None
                    and self._clicked_stage1[0] == name
                    and clicked_at - self._clicked_stage1[1] <= CONFIRM_DEDUPE_SECONDS
                ):
                    continue
                self._clicked_stage1 = (name, clicked_at, index)
                emitted.append(
                    self._emit(
                        "brush_stage1_card_clicked",
                        {
                            "selected": name,
                            "selected_name": _brush_name(name),
                            "code": brush_code(name),
                            "card_index": index,
                            "options": list(brush.stage1_options),
                            "names": [_brush_name(option) for option in brush.stage1_options],
                            "confidence": 0.99,
                        },
                        clicked_at,
                    )
                )
                continue
            # 落到这里 = 点在笔触面板里，但这一帧还没读出「有哪几张卡」。
            # 先按坐标存住，等卡片读出来后由 ``_resolve_buffered_panel_click`` 回填。
            panel_rect = brush.panel_rect if brush is not None else self._last_panel_rect
            if panel_rect is not None and _point_in_rect(x, y, panel_rect):
                self._buffered_panel_clicks.append((x, y, clicked_at))
                del self._buffered_panel_clicks[:-PENDING_CLICK_BUFFER_MAX]
                continue
            force_rect = snapshot.regions.get(RegionKind.FORCE_COMPLETE.value)
            if force_rect is None and self._last_snapshot is not None:
                force_rect = self._last_snapshot.regions.get(RegionKind.FORCE_COMPLETE.value)
            if force_rect is not None and _point_in_rect(x, y, force_rect):
                self._pending_force_clicked_at = clicked_at
                emitted.append(self._emit("force_complete_clicked", {"confidence": 0.99}, clicked_at))
                continue
            refresh_rect = snapshot.regions.get(RegionKind.REFRESH_AFFIX.value)
            if (
                refresh_rect is not None
                and self._visible_stage2
                and _point_in_rect(x, y, refresh_rect)
            ):
                # ⚠️ 这块矩形不是"检测到的按钮"，而是按棋盘几何估出来的
                # （vision 里 x0 ~ panel_w*0.42、b.y+b.h*0.80）。宠物低于 5 级时界面
                # **没有刷新按钮**（用户 2026-11-19 确认，其宠物 2 级），那块位置是别的
                # 东西，点哪都会被记成一次刷新——上一局就这样凭空多出 6 次"刷新词缀"。
                #
                # 两道闸：
                # 1. 阶段二（词缀三选一）必须**当前可见**。刷新按钮只在那个面板里存在，
                #    面板没开时点到同一片像素，不可能是刷新。
                # 2. 即便面板开着，宠物没到 5 级仍然没有这个按钮，所以不谎报 0.99，
                #    载荷里写明是几何推断，复盘时按 `origin` 剔除。
                emitted.append(
                    self._emit(
                        "affix_refresh_clicked",
                        {
                            "confidence": 0.55,
                            "origin": "geometry_estimate",
                            "stage2_visible": True,
                            "rect": [refresh_rect.x, refresh_rect.y, refresh_rect.w, refresh_rect.h],
                            "note": "该区域来自几何估算；宠物未达 5 级时此按钮不存在，需人工复核",
                        },
                        clicked_at,
                    )
                )
        return emitted

    def _emit(
        self,
        kind: str,
        payload: dict[str, Any],
        now: float,
        *,
        include_in_session: bool = True,
    ) -> str:
        event = {
            # v1 只增字段、不删字段。⚠️ 2026-11-19 起 `current_paint_changed.payload.after`
            # 里的 `effect_id` 改了含义：现在是**调色效果表**(S8_Gameplay_ItemEffect)的 ID，
            # 笔触表 ID 挪到 `brush_effect_id`。看到没有 `brush_effect_id` 的老日志，
            # 说明它的 `effect_id` 还是笔触 ID，别拿两张表混着聚合。
            "schema": "huahua-auto-record-v1",
            "ts": round(now, 3),
            "session": self.session_id,
            "sequence": self.sequence,
            "t": round(max(0.0, now - self.started_at), 3) if self.started_at else 0.0,
            "event": kind,
            "payload": payload,
        }
        self.sequence += 1
        self._last_event_at = now
        if include_in_session:
            self._events.append(event)
        if self._writer is not None:
            self._writer.submit(event)
        return kind


def _semantic_snapshot_key(snapshot: FastVisionSnapshot) -> tuple:
    return (
        tuple(_tile_key(tile) for tile in snapshot.tiles),
        tuple((paint.category, paint.level, paint.subtype, paint.is_treasure) for paint in snapshot.backpack),
        _current_key(snapshot.current),
        (
            snapshot.brush.stage1_options,
            snapshot.brush.chosen_stage1,
            snapshot.brush.stage2_options,
        )
        if snapshot.brush is not None
        else None,
    )


def _tile_kind(tile: FastTile) -> tuple:
    """「同一枚颜料」的判据：类别 + 家族 + 宝箱标记。

    **不含等级**——等级会随时间变化，正是 :meth:`AutomaticRunRecorder._settle_levels`
    要跟踪的量。换颜料（类别/家族变）时状态必须重置，否则新颜料会被旧读数压住。
    """
    return (tile.paint_category, tile.paint_subtype, bool(tile.paint_is_treasure))


def _tile_key(tile: FastTile) -> tuple:
    return (
        tile.state,
        tile.paint_category,
        tile.paint_level,
        tile.paint_subtype,
        tile.paint_is_treasure,
        int(getattr(tile, "bonus_level", 0) or 0),
        int(getattr(tile, "aura_level", 0) or 0),
    )


def _paint_key(tile: FastTile) -> tuple:
    return (
        tile.paint_category,
        int(tile.paint_level or 0),
        tile.paint_subtype,
        bool(tile.paint_is_treasure),
    )


def _paint_key_from_payload(payload: dict[str, Any]) -> tuple:
    return (
        payload.get("category") or "unknown",
        int(payload.get("level") or 0),
        payload.get("subtype"),
        bool(payload.get("treasure")),
    )


def _tile_payload(tile: FastTile) -> dict[str, Any]:
    is_paint = tile.state == TileState.PAINT.value
    return {
        "state": tile.state,
        "category": tile.paint_category if is_paint else None,
        # level 是屏幕上显示的**最终等级**（颜料等级 + 地块等级加成），
        # 和 game_model.tile_final_level 同义。
        "level": tile.paint_level if is_paint else None,
        # 地块加成单独记一份：它挂在**地块**上，空地上叠的加成视觉上完全看不到，
        # 只能从规则推演得到。回放时要靠它区分「Lv1 落在 +3 地块」和「Lv4 落在空地」。
        "bonus_level": int(getattr(tile, "bonus_level", 0) or 0),
        # 同种颜料光环带来的等级（同样只能模型回填）。
        "aura_level": int(getattr(tile, "aura_level", 0) or 0),
        "subtype": tile.paint_subtype if is_paint else None,
        "treasure": bool(tile.paint_is_treasure) if is_paint else False,
    }


def _tile_changes(previous: FastVisionSnapshot, current: FastVisionSnapshot) -> list[dict[str, Any]]:
    before = {(tile.row, tile.col): tile for tile in previous.tiles}
    changes: list[dict[str, Any]] = []
    for tile in current.tiles:
        old = before.get((tile.row, tile.col))
        if old is None or _tile_key(old) == _tile_key(tile):
            continue
        changes.append(
            {
                "row": tile.row,
                "col": tile.col,
                "from": _tile_payload(old),
                "to": _tile_payload(tile),
            }
        )
    return changes


def _terrain_counts(tiles: tuple[FastTile, ...]) -> dict[str, int]:
    """按格子重算地形计数。

    ``FastVisionSnapshot.terrain_counts`` 是构造时算好的独立字段，格子的
    ``state`` 一旦被改（``_settle_terrain`` 会把抖动的格子写回旧值），它就过期了。
    而 ``process`` 用 ``terrain_counts['blank'] == 0`` 判"没空白了 → 该结算"，
    拿过期计数会提前/推迟结算，所以这里跟着重算一遍。
    """
    counts = {state.value: 0 for state in TileState}
    for tile in tiles:
        counts[tile.state] = counts.get(tile.state, 0) + 1
    return counts


def _frame_looks_replaced(
    previous: FastVisionSnapshot,
    current: FastVisionSnapshot,
    illegal_count: int,
) -> bool:
    """这一帧是不是「整盘被替换」（入画/结算画面）？

    只看地形对调格数**不够**：2026-11-20 那局开局有 4~6 格 blocked↔void 的逐格
    抖动，2026-11-18 末帧的入画画面也是 4~6 格，两者在数量上完全重叠。能分开它们
    的只有第二条证据——入画/结算画面是**整盘重画**，颜料会跟着一起崩（那次的
    r3c1/r3c2 两枚颜料被读成「怪物 Lv3」，用户当场指出「赛季变怪物不成立」）。

    而逐格抖动只动地形：玩家没落笔时颜料格是稳定的。转锋类笔触确实会改颜料类别，
    但**不会同时**让地形对调，所以只要同时满足两条，就归入画画面。
    """
    if illegal_count < 2:
        return False
    if illegal_count >= SCENE_SWAP_CONFLICT_TILES:
        return True
    breakage = _paint_breakage(previous, current)
    return breakage >= SCENE_SWAP_PAINT_BREAKAGE


def _paint_breakage(previous: FastVisionSnapshot, current: FastVisionSnapshot) -> int:
    """同一格「颜料」崩掉的格数：颜料消失，或颜料类别被告知换成了另一种。

    只数**已经读出类别**的格，``unknown`` 不算——那只是没认出来，不是崩了。
    """
    before = {(tile.row, tile.col): tile for tile in previous.tiles}
    count = 0
    for tile in current.tiles:
        old = before.get((tile.row, tile.col))
        if old is None or old.state != TileState.PAINT.value:
            continue
        if tile.state != TileState.PAINT.value:
            count += 1
            continue
        old_category = old.paint_category
        new_category = tile.paint_category
        if old_category in {None, "", "unknown"} or new_category in {None, "", "unknown"}:
            continue
        if old_category != new_category:
            count += 1
    return count


def _illegal_terrain_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """官方规则里不可能出现的地形转移。

    扩散 18 条里「阻挡地块」只能变成**空白**（3001~3009），虚空也只能变成空白
    （3014/3015），没有任何一条能让阻挡与虚空互换。所以成片出现 ``blocked→void``
    只可能是这一帧把入画/结算画面的暗色地块读错了。

    2026-11-18 实测：末帧 4 个阻挡格同时变虚空、连带把 r3c1/r3c2 两枚颜料读成
    「怪物 Lv3」，``final_board`` 因此被污染（用户当场指出「赛季变怪物不成立」）。
    """
    weird = {("blocked", "void"), ("void", "blocked")}
    return [
        item
        for item in changes
        if (str(item["from"]["state"]), str(item["to"]["state"])) in weird
    ]


def _buff_only_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """挑出「棋盘画面没变、只有地块加成变了」的格子。

    加成（``bonus_level`` / ``aura_level``）视觉上完全看不见，只有模型回填才会变。
    这类变化必须单独记一条事件，否则「叠 buff 到 7 级」这条主线在日志里毫无痕迹。
    """
    picked: list[dict[str, Any]] = []
    for item in changes:
        before, after = item["from"], item["to"]
        if before["state"] != after["state"] or before.get("category") != after.get("category"):
            continue
        if _level(before) != _level(after):
            continue
        if (
            int(before.get("bonus_level") or 0) == int(after.get("bonus_level") or 0)
            and int(before.get("aura_level") or 0) == int(after.get("aura_level") or 0)
        ):
            continue
        picked.append(item)
    return picked


def _buff_payload(snapshot: FastVisionSnapshot) -> list[dict[str, Any]]:
    """把当前所有带加成的地块列出来（只列非零的）。"""
    return [
        {
            "row": tile.row,
            "col": tile.col,
            "bonus_level": int(getattr(tile, "bonus_level", 0) or 0),
            "aura_level": int(getattr(tile, "aura_level", 0) or 0),
            "state": tile.state,
        }
        for tile in snapshot.tiles
        if int(getattr(tile, "bonus_level", 0) or 0) or int(getattr(tile, "aura_level", 0) or 0)
    ]


def _board_payload(
    snapshot: FastVisionSnapshot,
    peaks: dict[tuple[int, int], tuple[tuple | None, int]] | None = None,
    votes: dict[tuple[int, int], Counter] | None = None,
) -> dict[str, Any]:
    state_codes = {
        TileState.BLANK.value: "B",
        TileState.BLOCKED.value: "K",
        TileState.VOID.value: "V",
        TileState.FADED.value: "F",
        TileState.PAINT.value: "P",
        TileState.UNKNOWN.value: "?",
    }
    paints: list[dict[str, Any]] = []
    for tile in snapshot.tiles:
        if tile.state != TileState.PAINT.value:
            continue
        item = {"row": tile.row, "col": tile.col, **_tile_payload(tile)}
        # 「这一格曾经到过几级」只做旁注，不覆盖上报值：等级确实会被随机/同内容
        # 颜料覆盖而下降，所以 peak_level 与 level 不同时把差异标出来供复盘。
        peak = (peaks or {}).get((tile.row, tile.col))
        if peak is not None and peak[0] == _tile_kind(tile):
            item["peak_level"] = peak[1]
            if tile.paint_level is not None and peak[1] > tile.paint_level:
                item["level_below_peak"] = True
        tally = (votes or {}).get((tile.row, tile.col))
        if tally and len(tally) > 1:
            # 同一格读出过不止一个家族 → 我们没认准。不猜，标出来让人复核。
            item["needs_review"] = True
            item["subtype_votes"] = dict(tally.most_common())
        paints.append(item)
    return {
        "signature": snapshot.board_signature,
        "states": "".join(state_codes.get(tile.state, "?") for tile in snapshot.tiles),
        "terrain_counts": dict(snapshot.terrain_counts),
        "paints": paints,
        # 带加成的地块（含空地与失色地块）。视觉读不到，只有模型回填时才有值。
        "buffs": _buff_payload(snapshot),
        "score": snapshot.board_score,
    }


def _backpack_payload(snapshot: FastVisionSnapshot) -> list[dict[str, Any]]:
    page = str(getattr(snapshot, "backpack_page", "") or "")
    return [
        {
            "page": page,
            "slot": paint.slot_index,
            "category": paint.category,
            "level": paint.level,
            "subtype": paint.subtype,
            "treasure": paint.is_treasure,
        }
        for paint in snapshot.backpack
    ]


def _backpack_counter(snapshot: FastVisionSnapshot) -> Counter[tuple]:
    return Counter(
        (paint.category, int(paint.level), paint.subtype, bool(paint.is_treasure))
        for paint in snapshot.backpack
    )


def _consumed_backpack_paints(previous: FastVisionSnapshot, current: FastVisionSnapshot) -> Counter[tuple]:
    if not previous.backpack or not current.backpack:
        return Counter()
    before = _backpack_counter(previous)
    after = _backpack_counter(current)
    return Counter({key: count for key, count in (before - after).items() if count > 0})


def _counter_payload(counter: Counter[tuple]) -> list[dict[str, Any]]:
    return [
        {
            "category": key[0],
            "level": key[1],
            "subtype": key[2],
            "treasure": key[3],
            "count": count,
        }
        for key, count in sorted(counter.items(), key=lambda item: tuple(str(part) for part in item[0]))
        if count > 0
    ]


def _current_key(current: FastCurrentPaint | None) -> tuple | None:
    if current is None:
        return None
    return (
        current.category,
        current.level,
        current.subtype,
        current.is_treasure,
        current.direction,
        current.target_object,
        current.upgrade_amount,
        current.match_current,
        current.match_required,
        # 面板「调色效果」原文也是这枚颜料的身份：类别/等级/方向全同、只差一条效果
        # 的两枚颜料必须算两次变化，否则复盘看不到「这一手换了笔触」。
        current.effect_text,
    )


def _current_payload(current: FastCurrentPaint | None) -> dict[str, Any] | None:
    if current is None:
        return None
    return {
        "category": current.category,
        "level": current.level,
        "subtype": current.subtype,
        "treasure": current.is_treasure,
        "direction": current.direction,
        "target": current.target_object,
        "upgrade": current.upgrade_amount,
        "match_current": current.match_current,
        "match_required": current.match_required,
        # 笔触条目原文（面板第三行）。None = 那一帧没读到，不代表没有效果。
        "effect": current.effect_text,
        # 这句话在**地块调色效果表**（S8_Gameplay_ItemEffect，27 条）里的记录 ID。
        # 上线实测：面板第三行多数时候显示的是「使纵向的玩法颜料提升2级」这类调色效果，
        # 早先拿笔触表反查，结果 9 段全空。
        "effect_id": _plot_effect_ids(current.effect_text),
        # 同一句话在**笔触表**（S8_Gameplay_SelectableEffect，59 条）里的记录 ID。
        # 两张表 ID 撞号但语义不同（1101 在两张表里完全是两回事），所以分开记。
        "brush_effect_id": _brush_effect_ids(current.effect_text),
    }


def _plot_effect_ids(text: str | None) -> list[int]:
    if not text:
        return []
    # 延迟导入：codes 会拉 game_model，录制层平时不需要它。
    from .codes import plot_effect_ids

    return list(plot_effect_ids(text))


def _brush_effect_ids(text: str | None) -> list[int]:
    if not text:
        return []
    from .codes import brush_effect_ids

    return list(brush_effect_ids(text))


def _resolve_pending_affix(
    pending: _PendingAffixChoice,
    previous: FastVisionSnapshot,
    current: FastVisionSnapshot,
    changes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [affix for name in pending.options if (affix := resolve_affix(name)) is not None]
    if not candidates:
        return None
    ranked = sorted(
        ((_affix_evidence_score(affix, previous, current, changes), affix) for affix in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score < 2.0 or (len(ranked) > 1 and best_score - second_score < 1.0):
        return None
    confidence = min(0.98, 0.58 + best_score * 0.055)
    return {
        "selected": best.name,
        "code": affix_code(best),
        "brush": best.brush_type.value,
        "brush_name": _brush_name(best.brush_type.value),
        "effect": best.effect.value,
        "options": list(pending.options),
        "stage1_options": list(pending.stage1_options),
        "confidence": round(confidence, 3),
        "evidence_score": round(best_score, 2),
        "resolution_delay": round(max(0.0, time.time() - pending.observed_at), 3),
        # 这条是「看效果反推」出来的，不是抓到点击。复盘时必须能区分——
        # 反推有错的可能，点击没有。以前缺这个字段，两种来源混在一起分不清。
        "source": "effect_inference",
    }


def _affix_evidence_score(
    affix: AffixRule,
    previous: FastVisionSnapshot,
    current: FastVisionSnapshot,
    changes: list[dict[str, Any]],
) -> float:
    score = 0.0
    before = previous.current
    after = current.current
    effect = affix.effect
    if before is not None and after is not None:
        if effect == AffixEffect.CHANGE_DIRECTION and affix.direction is not None and after.direction == affix.direction.value:
            score += 5.0 if before.direction != after.direction else 2.0
        if effect == AffixEffect.CHANGE_TARGET_OBJECT and before.target_object != after.target_object:
            score += 4.0
        if effect == AffixEffect.MATCH_REQUIRED_DELTA and before.match_required != after.match_required:
            score += 5.0
        if effect in {AffixEffect.RANDOMIZE_CURRENT_LEVEL, AffixEffect.UPGRADE_SELF} and before.level != after.level:
            score += 4.0
        if effect == AffixEffect.RANDOMIZE_CURRENT_DIRECTION and before.direction != after.direction:
            score += 3.5
        if effect == AffixEffect.RANDOMIZE_CURRENT_CONTENT and before.category != after.category:
            score += 3.5
        if effect == AffixEffect.REROLL_CURRENT_RULE and _current_key(before) != _current_key(after):
            score += 2.5

    upgrades = [item for item in changes if _level(item["to"]) > _level(item["from"])]
    conversions = [
        item
        for item in changes
        if item["from"].get("category") not in {None, "unknown"}
        and item["to"].get("category") not in {None, "unknown"}
        and item["from"].get("category") != item["to"].get("category")
    ]
    cleared = [
        item
        for item in changes
        if item["from"]["state"] in {TileState.BLOCKED.value, TileState.VOID.value, TileState.FADED.value}
        and item["to"]["state"] == TileState.BLANK.value
    ]
    placed = [item for item in changes if item["to"]["state"] == TileState.PAINT.value and item["from"]["state"] != TileState.PAINT.value]
    if upgrades and effect in {
        AffixEffect.UPGRADE_IN_SCOPE,
        AffixEffect.UPGRADE_SCOPE_SAME_PAINT,
        AffixEffect.UPGRADE_GLOBAL_CATEGORY,
        AffixEffect.UPGRADE_GLOBAL_SAME_PAINT,
        AffixEffect.UPGRADE_SELF,
    }:
        score += 3.0
        if affix.amount and all(_level(item["to"]) - _level(item["from"]) == affix.amount for item in upgrades):
            score += 1.5
    if conversions and effect in {
        AffixEffect.COPY_SELF_TO_SCOPE,
        AffixEffect.RANDOMIZE_SCOPE_CATEGORY,
        AffixEffect.RANDOMIZE_SCOPE_ANY,
    }:
        score += 4.0
    if cleared and effect in {AffixEffect.CLEAR_TILE_IN_SCOPE, AffixEffect.CLEAR_RANDOM_TILE_GLOBAL}:
        score += 4.0
    if placed:
        from_states = {item["from"]["state"] for item in placed}
        if effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED and TileState.BLOCKED.value in from_states:
            score += 5.0
        if effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF and TileState.VOID.value in from_states:
            score += 5.0
        if effect == AffixEffect.ALLOW_PLACE_ON_FADED and TileState.FADED.value in from_states:
            score += 5.0
        if effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF and any(item["from"]["state"] == TileState.PAINT.value for item in placed):
            score += 5.0
    return score


def _level(payload: dict[str, Any]) -> int:
    return int(payload.get("level") or 0)


def _has_active_gameplay(snapshot: FastVisionSnapshot) -> bool:
    return bool(snapshot.current is not None or snapshot.brush is not None or snapshot.backpack)


def _point_in_rect(x: float, y: float, rect) -> bool:
    return rect.x <= x <= rect.right and rect.y <= y <= rect.bottom


def _looks_like_new_run(snapshot: FastVisionSnapshot, previous_signature: str | None) -> bool:
    if RegionKind.START_DRAWING.value in snapshot.regions:
        return True
    if previous_signature is None or snapshot.board_signature == previous_signature:
        return False
    blank = int(snapshot.terrain_counts.get(TileState.BLANK.value, 0) or 0)
    paint = int(snapshot.terrain_counts.get(TileState.PAINT.value, 0) or 0)
    return blank >= 6 and paint <= 4 and _has_active_gameplay(snapshot)


#: 界面状态灯的五种配色等级。文案由 :func:`describe_record_status` 产出。
RECORD_STATUS_LEVELS = ("off", "error", "waiting", "recording", "finished")


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def describe_record_status(
    status: RecorderStatus | None,
    *,
    auto_follow: bool,
    capture_error: str | None = None,
    frame_age: float | None = None,
    game_window: bool | None = None,
    paused: str | None = None,
    stall_after: float = 8.0,
) -> tuple[str, str]:
    """把录制器状态翻成界面上那盏灯的文字 + 配色等级。

    纯函数，不碰 tkinter：灯亮什么、什么颜色，全在这里定死，测试可以直接
    锁住每种组合。以前「到底在不在录」只能去翻 ``auto_run_records.jsonl``
    有没有变长，就是因为缺这一处权威判断。

    等级含义：``off`` 没开自动跟随；``error`` 抓不到画面/没锁定游戏窗口；
    ``waiting`` 有画面但还没进对局；``recording`` 正在写；``finished`` 本局刚收尾。

    ``game_window=False`` 是这次特意补的一态：游戏没开时，采集会**回退截整个屏幕**，
    帧照常进来、快照里只是没有颜料面板，于是 phase 一直是 ``waiting``。
    光看 phase 会显示成「待开局」——等于告诉用户「我在等你开局」，而他其实
    连游戏都没开。这种假安慰正是要消灭的东西。
    """
    if not auto_follow:
        return "记录 关（自动跟随未开启）", "off"
    if capture_error:
        return "记录 等待游戏", "error"
    if paused:
        return f"记录 暂停（{paused}）", "waiting"
    if status is None:
        return "记录 等待首帧", "waiting"
    if game_window is False:
        return "记录 未锁定游戏窗口", "error"
    if frame_age is not None and frame_age > stall_after:
        return f"记录 无画面 {frame_age:.0f}s", "error"
    if status.phase == "playing":
        text = f"记录 录制中 {status.event_count} 事件 {_format_duration(status.duration)}"
        return _with_writer_warning(text, status), "recording"
    if status.phase == "opening":
        return _with_writer_warning("记录 开局中", status), "waiting"
    if status.phase == "finished":
        return _with_writer_warning(
            f"记录 本局已收尾（{status.event_count} 事件）", status
        ), "finished"
    return "记录 待开局", "waiting"


def _with_writer_warning(text: str, status: RecorderStatus) -> str:
    """写盘出过问题就把话说出来：丢了几条 / 写失败几次 / 最后写到哪。

    只影响文案，不改配色等级——灯还是在录，只是"记得可能不全"。
    """
    problems: list[str] = []
    if status.writer_dropped:
        problems.append(f"丢{status.writer_dropped}条")
    if status.writer_failures:
        problems.append(f"写失败{status.writer_failures}次")
    if status.writer_error:
        problems.append(str(status.writer_error)[:24])
    if not problems:
        return text
    return f"{text} · 写盘降级（{'，'.join(problems)}）"
