from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import random
import re
import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import ImageTk

from .backpack import BackpackAccumulator, paint_from_observation, paint_signature, snapshot_payload
from .capture import (
    CAPTURE_KIND_PAUSED,
    CAPTURE_KIND_SCREEN,
    CAPTURE_KIND_WINDOW_REGION,
    DEFAULT_WINDOW_TITLE,
    TRUSTED_CAPTURE_KINDS,
    WindowInfo,
    capture_diagnostics,
    capture_exclusion_status,
    capture_kind,
    capture_target,
    close_capture_backends,
    exclude_tool_windows_from_capture,
    find_window,
    find_capture_window,
    format_capture_diagnostics,
    remote_session_reason,
    reset_wgc_state,
    window_content_capture_available,
    window_content_capture_ready,
    windows_excluded_from_capture,
    wgc_status,
    write_image,
)
from .config import config_value
from .brush_recording import BrushEvidenceService, EvidenceStore
from .codebook import Codebook, board_signature, paint_id
from .codes import (
    affix_code,
    affix_display_from_code,
    brush_code,
    brush_payload_codes,
    canonical_affix_id,
    compact_paint_with_code,
    compact_rule_with_code,
    resolve_affix,
)
from .fast_events import FastVisionEvent, detect_fast_events
from .fast_vision import FastVisionSnapshot, backpack_page_of, snapshot_from_scene
from .frame_pump import FramePump, default_auto_capture, same_window_geometry
from .game_log import LOG_ALIVE_SECONDS, GameLogEvent, GameLogWatcher, find_game_log

#: 日志说画面变了之后的「连拍」节拍：不额外等待，跑完一帧接着跑下一帧。
LOG_BURST_INTERVAL = 0.1

#: 棋盘/颜料/词缀编号字典。和 observations.jsonl 配套：记录里写 ``B12``，
#: 这个文件负责 ``B12`` ↔ 完整局面定义的双向映射。
CODEBOOK_PATH = Path("runtime/codebook.json")
#: 当前可见的 3 排 15 格背包快照。用户要求「识别出来的颜料要能保存，
#: 这局打完后复现时要知道当时背包里有哪些颜料」，所以每次读数变化就落一份。
BACKPACK_SNAPSHOT_PATH = Path("runtime/backpack_snapshot.json")
#: 没有事件时的节拍。棋盘在两笔之间不会动，按 100ms 盲截是纯浪费。
LOG_IDLE_INTERVAL = 1.5
#: 日志在写、但一条可识别事件都没有时的兜底节拍。
#:
#: 捕获的唤醒绑死在日志词汇上：认不出事件就永远等不到唤醒。2026-11-22 实测第二局，
#: 用户大量安装颜料（落在初始颜料生效范围外，任务未完成），游戏只播通用按钮音
#: ``ECommonBtnType``，``S8_*`` 一条都没有，于是 64 秒里零帧。兜底节拍只在
#: 「日志确实在增长」时才启用，所以不会在游戏关掉时白截。
LOG_FALLBACK_INTERVAL = 1.8
#: 日志在写、却连续这么久没有可识别事件，就判定为「静默」并启用兜底节拍 + 界面提示。
LOG_SILENT_SECONDS = 6.0
#: 采集线程还在跑、但这么久没往轨迹里写任何步骤，就落一条 ``capture_gap`` 证据。
CAPTURE_GAP_SECONDS = 20.0
#: 事件发生后多久再截：落子/开面板都有动画，抢在动画中间截会读到中间态。
LOG_SETTLE_MS = 220
#: 最后一次事件之后还保持连拍多久。
LOG_BURST_WINDOW = 0.9
#: 日志监听自己的节拍。它和自动跟随是两条线：自动跟随没开也要看得见「日志接上了、刚刚落了子」，
#: 否则用户放完颜料只能看到一片安静，分不清是没接上还是本来就不动。
GAME_LOG_TICK_MS = 250
#: 落笔后等「升级动画」播完的窗口。实机一条 ``Anim_PutongShengji`` 一格，
#: 一次落笔 1~4 条是**排队**播的（实测 0 / 1.8s / 2.9s），所以窗口要留够，
#: 否则最后几条还没播就被当成「模型多算了」。
LEVEL_ANIMATION_WINDOW = 6.0
#: 采集线程多久不出帧就当它卡住了，重开一个。正常节拍最慢 1.5s 一帧，
#: 但**拖窗口后的首次重新标定**要在缩略图上全搜索 + 36 格重分类，
#: 1765x965 实测 7.6s；阈值留在 6s 会把它误判成卡死并重开 pump，
#: 反而把刚要落盘的标定打断。留一倍余量。
AUTO_FOLLOW_STALL_SECONDS = 12.0
#: 完整识别会 withdraw 主窗口，超过这个毫秒数还没回来就强制恢复，别让工具"消失"。
CAPTURE_WINDOW_RESTORE_MS = 12000
from .llm.types import (
    ACTION_BACKPACK,
    ACTION_BRUSH1,
    ACTION_BRUSH2,
    ACTION_FINISH,
    ACTION_PLACE,
    AdvisorAdvice,
)
from .game_model import (
    BOARD_SIZE,
    DEFAULT_PET_LEVEL,
    MAX_FORCE_TIMES,
    MAX_LEVEL,
    PET_REFRESH_BY_LEVEL,
    AffixRule,
    AffixEffect,
    BrushChoiceScenario,
    BrushType,
    CurrentPaintRule,
    InstallPlan,
    Paint,
    PlacementRecommendation,
    PuzzleState,
    ResourceState,
    apply_affix_to_state,
    build_affix_catalog,
    current_scope_cells,
    current_rule_is_pending,
    assess_finish_mode,
    force_complete_current_rule,
    install_backpack_paint,
    level_sources,
    normalize_brush_options,
    place_current_rule_anchor,
    random_brush_choice_scenario,
    source_accepts_paint,
    tile_bonus_level,
    tile_aura_level,
    tile_final_level,
    random_state,
    recommend_affixes,
    recommend_brush_types,
    recommend_install_plan,
    recommend_placements,
    recommend_special_actions,
    sort_backpack_for_rule,
    state_from_observation,
)
from .learning import (
    LAST_PERSIST_TIMING,
    cached_learning_stats,
    export_learning_bundle,
    load_learning_stats,
    lowest_trajectory_score,
    merge_learning_bundle,
    record_correction,
    record_high_score_trajectory,
    record_observation,
    trajectory_count,
)
from .level_reconcile import ReconcileReport, reconcile as reconcile_levels
from .level_smoothing import LevelStabilizer
from .line_capacity import summarize_lines
from .models import BoardObservation, BrushOptionsObservation, PaintCategory, PaintDirection, PaintObject, Rect, RegionKind, SceneObservation, TileObservation, TileState
from .mouse_monitor import MouseClickMonitor
from .outcome_learning import decision_records_from_steps
from .pet_brain import PetBrainState, evaluate_pet_state
from .run_recorder import AutomaticRunRecorder, describe_record_status
from .tile_art import (
    VOIDED_SUBTYPE,
    compose_icon,
    family_name as tile_family_name,
    icon_stem as tile_icon_stem,
)
from . import layout
from . import paint_icon
from . import vision
from .vision import VisionTracker, analyze_and_save_debug
from .windows_ocr import warm_windows_ocr_async


STATE_LABELS = {
    TileState.BLANK: "空",
    TileState.BLOCKED: "阻",
    TileState.VOID: "虚",
    TileState.FADED: "失",
    TileState.PAINT: "颜",
    TileState.UNKNOWN: "?",
}

STATE_COLORS = {
    TileState.BLANK: "#eef3fb",
    TileState.BLOCKED: "#00a8d8",
    TileState.VOID: "#151b22",
    TileState.FADED: "#78808a",
    # 颜料格底色压深：格子里现在画的是官方品质框 + 图案，亮黄底会把框的辉光吃掉，
    # 看起来像「糊成一团」。深底才跟游戏里一致。
    TileState.PAINT: "#0d141b",
    TileState.UNKNOWN: "#b33030",
}

#: 「虚空上的等级光环」用的颜色，按 ``frame_level.HUE_WINDOWS`` 的色相配：
#: Lv3 青蓝(95~122) / Lv4 紫(132~162) / Lv5 橙(10~24) / Lv6 红(168~180 + 0~8)。
#:
#: 为什么要单独画：颜料落到虚空后失去自身内容、退化成笔触效果，画面上只剩底色
#: 和一圈等级光环（用户 2026-11-24 口径，实测一般 3-5 级）。这格没有颜料，
#: 原来只画一个「虚」字，玩家看不出这里还挂着笔触效果。
AURA_LEVEL_COLORS = {
    3: "#3aa6ff",
    4: "#b45cf0",
    5: "#ff8a1f",
    6: "#ff3b3b",
}
#: 等级读不出来时的兜底色（中性紫），照样要把「有光环」这件事画出来。
AURA_UNKNOWN_COLOR = "#8a7cff"

NOTICE_STYLES = {
    "info": ("#18324a", "#d7e3ee"),
    "success": ("#123d30", "#c8ffe8"),
    "warning": ("#4a3513", "#ffe8a3"),
    "error": ("#4a1f1f", "#ffd6d6"),
}

#: 记录状态灯的文字颜色，按 ``run_recorder.describe_record_status`` 给的等级取。
#: 灰=没开、红=抓不到画面、黄=有画面没开局、绿=正在写盘、蓝=本局刚收尾。
RECORD_BADGE_COLORS = {
    "off": "#8fa9bd",
    "error": "#ff9a9a",
    "waiting": "#ffd84d",
    "recording": "#5ef2a8",
    "finished": "#9ecbff",
}

BRUSH_STAGE1_ABSENCE_GRACE_FRAMES = 12
BRUSH_STAGE2_ABSENCE_GRACE_FRAMES = 2


def _parse_slot_index(option: str | None) -> int | None:
    """从模型给的 option 里抠出「第几格」。

    实现在 ``llm.binding.parse_slot_index``（那里和"落点是否合法"的判断放在一起）。
    这里只留一层薄封装，免得存量调用点改签名。
    """
    from .llm.binding import parse_slot_index

    return parse_slot_index(option)


class HuahuaApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        warm_windows_ocr_async()
        # 官方图标库（ORB 27 家族）冷启一次 1.3~3 秒。背包 15 格要认家族，
        # 这笔钱要是落在第一帧，用户拖好窗口的瞬间会看到一次「卡住」。
        paint_icon.warm_async()
        # 界面进程才允许自动标定棋盘布局：窗口被最大化/改尺寸后 layout.py 的
        # 固定比例会偏十几像素，实时路径整片报「没有棋盘」。自愈找到位置后落盘，
        # 下次不用再搜。测试进程不打开（几十帧不同尺寸的截图会把文件写乱）。
        vision.set_layout_autosave(True)
        # 「开局识别卡十秒」这类问题，日志里原来只有 auto_follow_restart（泵停了 12 秒），
        # 看不出是标定没命中、全搜索太慢还是 OCR 卡住。把识别侧慢路径接进同一条日志。
        vision.set_diagnostic_hook(
            lambda payload: record_observation(
                "vision_diagnostic", dict(payload), source="vision"
            )
        )
        self.title("Huahua Puzzle Assistant v0.1")
        # 工具栏拆成两行后多了约 36px。棋盘边长是 min(可用宽, 可用高-留白) 算出来的，
        # 高度一少格子就整圈变小——屏幕够高就把窗口同步加高，让棋盘尺寸跟拆行前一致。
        # 注意 ``winfo_screenheight`` 是**逻辑**像素：1080p + 125% 缩放只有 864，
        # 所以按「留 90px 给标题栏/任务栏」算，算出来比 720 还小就维持 720 不动。
        self.geometry(
            f"1180x{max(720, min(756, self.winfo_screenheight() - 90))}"
        )
        self.minsize(1080, 640)
        self.configure(bg="#101820")

        self.status_var = tk.StringVar(
            value=f"就绪。优先捕捉窗口：{DEFAULT_WINDOW_TITLE}"
        )
        self.notice_var = tk.StringVar(value="提示：关键风险和高价值记录会显示在这里。")
        self.notice_label: tk.Label | None = None
        self.perf_var = tk.StringVar(value="性能：等待识别")
        self.perf_label: tk.Label | None = None
        self.board_var = tk.StringVar(value="局面：尚未识别")
        self.game_log_var = tk.StringVar(value="日志：未接上")
        # 记录状态灯：说清楚「现在到底有没有在录」，不靠用户去翻 jsonl 有没有变长。
        self.record_var = tk.StringVar(value="记录 关")
        #: 状态灯最近一次判定的等级（off/error/waiting/recording/finished）。
        #: 桌宠在「推荐关掉」时用这一级决定气泡文字和配色。
        self.record_level: str = "off"
        self.record_label: tk.Label | None = None
        self._record_badge_after_id: str | None = None
        #: 最近一次采集报错（抓不到游戏窗口等）。成功出帧后清空。
        self._last_capture_error: str | None = None
        #: 最近一帧是否锁定了游戏窗口。``False`` = 退回截全屏，说明根本没找到游戏，
        #: 此时帧是有的、phase 也停在 waiting，光看 phase 会误报「待开局」。
        self._capture_locked_window: bool | None = None
        #: 最近一帧是不是因为「游戏不在前台」被暂停（``PAUSED_SOURCE``）。
        self._capture_foreground_paused = False
        #: 最近一帧是不是「连窗口都没有」（游戏没开/最小化）。与上面分开，是因为
        #: 界面要说清到底是"你切走了"还是"游戏没开"，不能都叫"暂停"。
        self._capture_window_missing = False
        #: 最近一帧的来源分级（``capture.capture_kind``）。落盘/外发闸门只看它。
        self._last_capture_kind: str = "unknown"
        self.debug_path = Path("runtime/debug.png")
        self.current_scene: SceneObservation | None = None
        self.current_state: PuzzleState | None = None
        self.recommendations: list[PlacementRecommendation] = []
        self.install_plan: InstallPlan | None = None
        self.selected_backpack_index: int | None = None
        self.board_cell_rects: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        #: 格子图（官方品质框 + 家族图案）的 Tk 图片缓存。
        #: 键里带像素尺寸：窗口一改大小就得重合成，不能复用。
        #: 这里必须自己拿着引用——Tk 只要收不到强引用就会把图丢掉，
        #: 画布上立刻变成一片空白（而且不报错，极难查）。
        self._tile_photo_cache: dict[tuple, ImageTk.PhotoImage] = {}
        self.backpack_slot_rects: dict[int, tuple[float, float, float, float]] = {}
        self.brush_stage1_rects: dict[BrushType, tuple[float, float, float, float]] = {}
        self.brush_affix_rects: dict[int, tuple[float, float, float, float]] = {}
        self.brush_scenario: BrushChoiceScenario | None = None
        self.selected_brush_type: BrushType | None = None
        self.selected_affix: AffixRule | None = None
        self._brush_panel_absent_frames = 0
        self._last_brush_observation: BrushOptionsObservation | None = None
        self._brush_stage1_vote_observations: list[BrushOptionsObservation] = []
        self._locked_brush_stage1: BrushType | None = None
        self._locked_brush_stage2: AffixRule | None = None
        self.affix_catalog = build_affix_catalog()
        # 宠物等级决定刷新次数：Lv1–2 为 0、Lv3–4 为 1、Lv5–6 为 2。
        # 是否"能用刷新"由等级推出，不再单独用一个复选框。
        self.pet_level_var = tk.IntVar(value=DEFAULT_PET_LEVEL)
        self.refresh_available_var = tk.BooleanVar(value=False)
        self.force_used_var = tk.IntVar(value=0)
        self.refresh_used_var = tk.IntVar(value=0)
        self.auto_follow_var = tk.BooleanVar(value=True)
        self.auto_follow_interval_var = tk.IntVar(value=100)
        # 大模型托管：勾上之后，每隔 min_interval 把最新一帧交给视觉大模型，
        # 由它直接给出「点哪里」。本地链路仍然并行跑，作为超时/失败时的降级。
        self.llm_var = tk.BooleanVar(value=bool(config_value("llm.enabled", False)))
        self.llm_advice: AdvisorAdvice | None = None
        #: 这条建议出生时那一帧的指纹 + 采集时刻（``llm.binding.AdviceBinding``）。
        #: 展示前用它判断"还属于当前画面吗"，顺手提供不随重绘延长的截止时刻。
        self.llm_advice_binding = None
        #: 最近一次拒绝的原因（过期/阶段变了/卡片没匹配上）。只用于状态与诊断。
        self.llm_reject_reason: str | None = None
        self.llm_busy = False
        self.llm_after_id: str | None = None
        self._llm_pending_frame = None
        #: 待发帧的采集时刻（monotonic）。建议有效期从这里算起，不从重绘时刻算。
        self._llm_pending_captured_at: float = 0.0
        #: 上一帧为什么没交给模型（来源不可信/发送前被拦）。只用于状态与诊断，
        #: 不参与任何判断——判断永远重算，不读这个字段。
        self._llm_block_reason: str | None = None
        #: 上一次因为来源不可信而提示过的来源分级，用来去重（每 100ms 一帧）。
        self._llm_last_blocked_kind: str | None = None
        # 工作线程只往队列里丢结果，由主线程 drain。tkinter 不是线程安全的，
        # 从工作线程直接调 after() 会让事件循环卡住——表现就是「点了取消勾选，
        # 界面半天不刷新」。
        self._llm_results: queue.Queue = queue.Queue()
        # 每次开/关托管都 +1。工作线程回来时对不上就丢弃，避免关闭后旧建议
        # 又被写回 llm_advice、继续在箭头上挂 6 秒。
        self._llm_generation = 0
        self.overlay_var = tk.BooleanVar(value=False)
        self.overlay_window: tk.Toplevel | None = None
        self.overlay_canvas: tk.Canvas | None = None
        self.overlay_after_id: str | None = None
        self.overlay_phase = False
        self.overlay_wndprocs: dict[int, tuple[int, object]] = {}
        self.last_capture_window: WindowInfo | None = None
        self.last_capture_size: tuple[int, int] | None = None
        self.last_fast_snapshot: FastVisionSnapshot | None = None
        self.previous_fast_snapshot: FastVisionSnapshot | None = None
        self.last_fast_events: tuple[FastVisionEvent, ...] = ()
        self.last_pet_state: PetBrainState | None = None
        self.tracker = VisionTracker()
        self.run_recorder = AutomaticRunRecorder(external_brush_choices=True)
        self.mouse_click_monitor = MouseClickMonitor()
        self.mouse_click_monitor.start()
        self.brush_evidence = BrushEvidenceService(self.mouse_click_monitor, store=EvidenceStore(
            max_bytes=int(config_value("brush_recording.max_image_mb", 128)) * 1024 * 1024,
            mode=str(config_value("brush_recording.mode", "validation")),
        ))
        self.backpack_accumulator = BackpackAccumulator()
        self.backpack_session_key: str | None = None
        #: 最近一帧是否真的在画面里看到了背包面板。None = 还没有任何一帧。
        #: 背包面板不常驻，看不到时界面显示的是累积器里的旧读数，
        #: 必须让用户能分清「这是这一帧拍的」还是「这是上次留下的」。
        self.backpack_frame_visible: bool | None = None
        self.last_backpack_paint_tiles: int | None = None
        self.manual_correction_applied = False
        self.real_trajectory_steps: list[dict] = []
        self.last_real_step_key: str | None = None
        self.last_auto_trajectory_key: str | None = None
        self.last_real_capture_step: dict | None = None
        self.real_initial_snapshot: dict | None = None
        self.last_compressed_summary_key: str | None = None
        self.last_capture_quality: dict | None = None
        self.last_rule_anomalies: list[dict] = []
        #: 最近一次「模型推算 vs 画面识别」的逐格对账结果（见 :mod:`level_reconcile`）。
        self.level_reconcile: ReconcileReport = ReconcileReport()
        #: 已经提醒过的对账差异集合。同一笔每帧都会重算，靠它做到「变了才提醒」。
        self.last_reconcile_notice_key = ""
        #: 棋盘等级/种类的时序稳定器。挡「点击地块后浮窗遮挡」造成的 1↔7 抖动，
        #: 并在本局结算后冻结（等级与种类不再变化）。
        self.level_stabilizer = LevelStabilizer()
        self.last_quick_step_key: str | None = None
        self.pending_tentative_step: dict | None = None
        self.pending_tentative_signature: str | None = None
        self.quick_capture_record_count = 0
        self.real_session_id = _new_session_id()
        self.real_session_started_at = time.time()
        self.real_session_step_index = 0
        self.last_settlement_key: str | None = None
        self.real_session_finished = False
        #: 本局是否已经出现过「满铺」（blank=0 且棋盘上有一整盘颜料）。
        #: 为什么不能只看 ``real_session_finished``：那个标志要玩家真去点「结束本局」
        #: 才会置位，而实际玩法是画满就点「入画」开下一张——标志永远停在 False，
        #: 于是 :func:`_looks_like_new_run` 从来没被问过，界面一直挂着上一局的规则。
        #: 满铺是**不可逆**的：同一张画纸满铺之后格子不会自己空出来，
        #: 所以「满铺过」配上「空位又回来」就是换画纸，不需要等结算。
        self.board_reached_full = False
        #: 上一局**收尾时**那一帧的棋盘状态串。只为了拦一件事：刚收完尾，
        #: 画面还停在旧棋盘上，这时再去问 :func:`_looks_like_new_run` 有可能
        #: 因为没有参照帧（``last_real_capture_step`` 已随新局清空）而直接判成新局，
        #: 于是凭空开出一局 0 步的假局、还要白跑一遍十几秒的收尾写盘。
        #: 实测 2026-11-22 10:04:21 就这么假开了一局（1000.68 分、0 步）。
        #: 这里故意**不随新局重置**：它描述的是「上一局留下的画面」。
        self.last_finished_board_states = ""
        #: 本局是否已经在满铺那一刻尝试过自动存档。只尝试**一次**：
        #: 存档失败（没到高分门槛）时不能每帧重试，否则一次失败会在记录里刷出几十条。
        self.board_full_autosave_attempted = False
        #: 棋盘/颜料/词缀编号字典。跨局累积——样本越多，同一个 ``B1234`` 覆盖的
        #: 局面就越多，记录里重复出现的那部分就只花一个编号。
        self.codebook = Codebook.load(CODEBOOK_PATH)
        self.codebook_saved_at = time.time()
        #: 日志里出现「入画」时置位，下一帧无条件开新局。
        #: 为什么不能只靠 ``real_session_finished``：用户「重绘」开新局时不走结算，
        #: 那个标志永远是 False，于是 ``previous_state`` 一直被继承，界面留着上一局
        #: 的调色目标/锚点/背包，看起来就是「换局了还显示旧棋盘」。
        self.pending_new_run = False
        #: 日志给出的换局信号，``"new_run"`` / ``"finished"``；被
        #: :meth:`_record_auto_snapshot` 取走后立刻清空（一次性）。
        #: 视觉认换局在「两局只隔 4 秒」时会失效：交界那几帧整盘被替换，被当成
        #: 入画画面误读丢掉，结果两局合并成一条记录（2026-11-22 21:56 实测）。
        self._recorder_run_boundary: str | None = None
        self.auto_follow_after_id: str | None = None
        self.auto_follow_busy = False
        self.auto_follow_overlap_paused = False
        self.auto_follow_pause_reported = False
        self.auto_follow_capture_probed = False
        #: 工具窗口「对截屏免疫」的复查定时器。桌宠/浮层/对话框都是运行中才建的，
        #: 逐个在创建点埋一行太容易漏；统一扫窗口树，顺带自愈驱动重置亲和度。
        self._capture_exclusion_after_id: str | None = None
        self._capture_exclusion_reported = False
        #: 采集自检结论（``capture_diagnostics`` 的返回值）。
        self._capture_selftest_report: dict | None = None
        #: 上一帧锁定的游戏窗口句柄。变了就说明重开过游戏，WGC 值得重新试一次。
        self._capture_window_hwnd = 0
        self.auto_follow_last_frame_at = time.monotonic()
        self.auto_follow_generation = 0
        self._capture_inflight = False
        self.game_log_watcher: GameLogWatcher | None = None
        self.game_log_retry_at = 0.0
        self.game_log_missing_reported = False
        self.game_log_tick_after_id: str | None = None
        self.game_log_event_count = 0
        self.game_log_last_event: GameLogEvent | None = None
        self.game_log_nudge_at = 0.0
        self.auto_follow_log_revision = 0
        self.auto_follow_last_event_at = 0.0
        self.auto_follow_burst_until = 0.0
        #: 当前是否处于「日志在写、却解析不出画中雪原事件」的状态。
        #: 进入就切兜底节拍，并且**只在翻转时**记一条观测，不每 tick 刷屏。
        self.auto_follow_log_silent = False
        #: 静默提示是否已经顶到界面上过。置位表示「这轮静默已经提醒过」。
        self.auto_follow_silence_notice_key = False
        #: 最近一次真正写进轨迹的时间（monotonic）。用来发现「一直在截，却一帧都没记」
        #: 这种空档——2026-11-22 就是靠人工数日志才发现，不该再靠人工。
        self.last_step_recorded_at = 0.0
        self.capture_gap_frames = 0
        self.capture_gap_reported = False
        self.last_game_log_events: list[GameLogEvent] = []
        #: 落笔后播出的升级动画（``Anim_PutongShengji`` / ``CaiShengji``）条数，
        #: 用来和「模型预计升几格」对账。窗口由 :data:`LEVEL_ANIMATION_WINDOW` 定。
        self.level_animation_window: dict | None = None
        self.level_animation_seq = 0
        self.auto_follow_pump: FramePump | None = None
        self.auto_follow_last_sequence = 0
        self.last_high_value_key: str | None = None
        self.last_finish_skip_reason: str | None = None
        self.last_finish_diagnostic: dict | None = None
        self.last_perf: dict = {}
        self.last_desktop_pet_perf: dict = {}
        self._learning_stats_cache = load_learning_stats()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # 让本工具的所有窗口对截屏不可见：WGC 抓不到窗口内容时，采集退回屏幕复制，
        # 这时工具窗口压在棋盘上就会进帧——这是「软件窗口挡棋盘」的真正原因。
        self._capture_exclusion_after_id = self.after(300, self._capture_exclusion_tick)
        # 采集自检：把「为什么抓不到窗口内容」写成一条可读结论落在 runtime/ 里。
        threading.Thread(target=self._run_capture_selftest, name="HuahuaCaptureSelftest", daemon=True).start()
        # 日志监听是常驻的：不点「自动跟随」也要知道日志接没接上、刚刚有没有落子。
        self._start_game_log_tick()
        # 记录状态灯同样常驻：它就是用来回答「现在到底有没有在录」，不能等用户去翻文件。
        self._record_badge_after_id = self.after(self.RECORD_BADGE_INTERVAL_MS, self._refresh_record_badge)
        # 首帧之后把窗口插到最前：游戏窗口会反复抢前台，Tk 自己的 lift 跨不了进程，
        # 用户看到的就是「任务栏有、点不出来」（2026-11-18 实机）。
        self.after(60, self._show_in_front)
        # 窗口状态的客观快照：黑屏时唯一的可靠判据。
        self.after(2000, lambda: self._dump_window_state("startup"))
        self.after(8000, lambda: self._dump_window_state("steady"))
        self.after(500, self.toggle_auto_follow)
        # 诊断：HUAHUA_SELF_SHOT=<目录> 时在启动 12 秒后撤掉截屏免疫自拍一组图。
        # 免疫生效时进程外只能抓到纯黑，只有进程内能回答「界面到底画出来没有」。
        self_screenshot_dir = os.environ.get("HUAHUA_SELF_SHOT")
        if self_screenshot_dir:
            self.after(12000, lambda: self._diagnose_self_shot(self_screenshot_dir))

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self, bg="#101820")
        toolbar.pack(fill=tk.X, padx=14, pady=(14, 8))

        # 一行放不下：整条工具栏的 reqwidth 实测 1893px，而 1180 宽的窗口只有
        # 1152px 可用。于是「纠错识别」及其右边 8 项（识别统计/导出数据/合并数据/
        # 强制完成/宠物等级/强制已用/刷新已用/局面）一直排在窗口外面，点不到——
        # 恰恰是录数据要用的那几个。这里按「用得多 / 用得少」拆成两行：
        #   上行 = 对局操作（几乎每局都在点）
        #   下行 = 数据与状态（偶尔点，但必须点得到）
        # 拆成 toolbar 内部的子行，而不是再往窗口里 pack 一个兄弟框：桌宠那一行是
        # desktop_pet 用 ``after=toolbar`` 插到工具栏后面的，多一个兄弟框就要跟它
        # 抢 pack 顺序（``after`` 是「插到某个控件后面」，插两次的先后很反直觉）。
        primary_row = tk.Frame(toolbar, bg="#101820")
        primary_row.pack(fill=tk.X)
        secondary_row = tk.Frame(toolbar, bg="#101820")
        secondary_row.pack(fill=tk.X, pady=(6, 0))

        capture_btn = tk.Button(primary_row, text="完整识别", command=self.capture_async, width=14)
        capture_btn.pack(side=tk.LEFT)

        quick_capture_btn = tk.Button(primary_row, text="重新快速获取", command=self.quick_capture_async, width=14)
        quick_capture_btn.pack(side=tk.LEFT, padx=(8, 0))

        overlay_check = tk.Checkbutton(
            primary_row,
            text="坐标提示",
            variable=self.overlay_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
            command=self.toggle_overlay,
        )
        overlay_check.pack(side=tk.LEFT, padx=(8, 0))

        calibration_btn = tk.Button(
            primary_row,
            text="定位预览",
            command=self.open_region_calibration,
            width=10,
        )
        calibration_btn.pack(side=tk.LEFT, padx=(6, 0))

        auto_follow_check = tk.Checkbutton(
            primary_row,
            text="自动跟随",
            variable=self.auto_follow_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
            command=self.toggle_auto_follow,
        )
        auto_follow_check.pack(side=tk.LEFT, padx=(8, 0))

        tk.Spinbox(
            primary_row,
            from_=100,
            to=5000,
            increment=50,
            width=5,
            textvariable=self.auto_follow_interval_var,
        ).pack(side=tk.LEFT, padx=(3, 0))

        llm_check = tk.Checkbutton(
            primary_row,
            text="大模型托管",
            variable=self.llm_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
            command=self.toggle_llm_hosting,
        )
        llm_check.pack(side=tk.LEFT, padx=(8, 0))

        new_run_btn = tk.Button(primary_row, text="新开一局", command=self.start_new_real_session, width=10)
        new_run_btn.pack(side=tk.LEFT, padx=(8, 0))

        finish_run_btn = tk.Button(primary_row, text="结束本局", command=self.finish_real_session, width=10)
        finish_run_btn.pack(side=tk.LEFT, padx=(8, 0))

        simulate_btn = tk.Button(primary_row, text="模拟模式", command=self.simulate_random, width=14)
        simulate_btn.pack(side=tk.LEFT, padx=(8, 0))

        manual_brush_btn = tk.Button(primary_row, text="手动笔触", command=self._open_manual_brush_dialog, width=14)
        manual_brush_btn.pack(side=tk.LEFT, padx=(8, 0))

        verify_btn = tk.Button(secondary_row, text="纠错识别", command=self._open_capture_verify_dialog, width=14)
        verify_btn.pack(side=tk.LEFT, padx=(8, 0))

        stats_btn = tk.Button(secondary_row, text="识别统计", command=self._open_correction_stats_dialog, width=15)
        stats_btn.pack(side=tk.LEFT, padx=(8, 0))

        export_btn = tk.Button(secondary_row, text="导出数据", command=self._export_learning_data, width=11)
        export_btn.pack(side=tk.LEFT, padx=(8, 0))

        merge_btn = tk.Button(secondary_row, text="合并数据", command=self._merge_learning_data, width=11)
        merge_btn.pack(side=tk.LEFT, padx=(8, 0))

        force_btn = tk.Button(secondary_row, text="强制完成", command=self.force_complete_current, width=10)
        force_btn.pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(secondary_row, text="宠物等级", fg="#d7e3ee", bg="#101820").pack(side=tk.LEFT, padx=(10, 3))
        pet_spin = tk.Spinbox(
            secondary_row,
            from_=1,
            to=6,
            width=3,
            textvariable=self.pet_level_var,
            command=self._on_pet_level_change,
        )
        pet_spin.pack(side=tk.LEFT)
        pet_spin.bind("<FocusOut>", lambda _event: self._on_pet_level_change())

        tk.Label(secondary_row, text="强制已用", fg="#d7e3ee", bg="#101820").pack(side=tk.LEFT, padx=(10, 3))
        tk.Spinbox(
            secondary_row,
            from_=0,
            to=3,
            width=3,
            textvariable=self.force_used_var,
            command=self._rerender_side_panel,
        ).pack(side=tk.LEFT)

        tk.Label(secondary_row, text="刷新已用", fg="#d7e3ee", bg="#101820").pack(side=tk.LEFT, padx=(10, 3))
        tk.Spinbox(
            secondary_row,
            from_=0,
            to=2,
            width=3,
            textvariable=self.refresh_used_var,
            command=self._rerender_side_panel,
        ).pack(side=tk.LEFT)

        # 局面贴到这一行最右端：录数据时一直在看「走了第几步」，靠右才好扫。
        self.board_label = tk.Label(secondary_row, textvariable=self.board_var, fg="#d7e3ee", bg="#101820")
        self.board_label.pack(side=tk.RIGHT, padx=(14, 0))

        # 日志状态单独贴到提示条右端（不是工具栏最后一项）：它每帧都在变，
        # 塞进工具栏会被按钮挤着走，位置一跳就看不见了。
        self.game_log_label = tk.Label(
            self,
            textvariable=self.game_log_var,
            anchor="e",
            fg="#8fa9bd",
            bg="#101820",
            font=("Microsoft YaHei UI", 9),
        )
        self.game_log_label.pack(side=tk.RIGHT, padx=(8, 14))

        self.notice_label = tk.Label(
            self,
            textvariable=self.notice_var,
            anchor="w",
            justify=tk.LEFT,
            padx=10,
            pady=7,
            font=("Microsoft YaHei UI", 10, "bold"),
            relief=tk.FLAT,
            wraplength=940,
        )
        self.notice_label.pack(fill=tk.X, padx=14, pady=(0, 6))
        self._set_notice("提示：完整识别建立地图；快速获取/自动跟随用于实战节奏。低置信度不会写入高分轨迹。", "info")

        self.perf_label = tk.Label(
            self,
            textvariable=self.perf_var,
            anchor="w",
            justify=tk.LEFT,
            padx=10,
            pady=5,
            fg="#d7e3ee",
            bg="#162232",
            font=("Microsoft YaHei UI", 9),
            wraplength=1120,
        )
        self.perf_label.pack(fill=tk.X, padx=14, pady=(0, 6))

        # body 的 pack 必须晚于 footer：pack 按调用顺序分配空间，body 带 expand=True
        # 会把剩余高度全吃掉，最后 pack 的底部行在窗口不够高时会被压成 0 高 ——
        # 1180x720 下状态行原本就是这么消失的（winfo_ismapped()==0）。
        body = tk.Frame(self, bg="#101820")

        self.canvas = tk.Canvas(body, bg="#0d141b", highlightthickness=0, width=560)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.region_text = tk.Text(
            body,
            width=28,
            bg="#0d141b",
            fg="#d7e3ee",
            insertbackground="#d7e3ee",
            relief=tk.FLAT,
            font=("Microsoft YaHei UI", 10),
            wrap=tk.WORD,
        )
        self.region_text.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        self.region_text.configure(state=tk.DISABLED)

        footer = tk.Frame(self, bg="#101820")
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=(0, 14))

        self.record_label = tk.Label(
            footer,
            textvariable=self.record_var,
            anchor="e",
            fg=RECORD_BADGE_COLORS["off"],
            bg="#101820",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self.record_label.pack(side=tk.RIGHT, padx=(12, 0))
        tk.Button(footer, text="笔触核对", command=self._open_brush_review, width=8).pack(side=tk.RIGHT)

        status = tk.Label(
            footer,
            textvariable=self.status_var,
            anchor="w",
            fg="#d7e3ee",
            bg="#101820",
        )
        status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

        self._draw_empty_grid()

    def _open_brush_review(self) -> None:
        from .brush_review import write_report
        try:
            path = self.brush_evidence.store.root / "review.html"
            write_report(self.brush_evidence.store, path)
            os.startfile(str(path.resolve()))
        except OSError as exc:
            self._set_notice(f"笔触核对页打开失败：{exc}", "warning")

    def _set_notice(self, message: str, level: str = "info") -> None:
        bg, fg = NOTICE_STYLES.get(level, NOTICE_STYLES["info"])
        self.notice_var.set(message)
        if self.notice_label is not None:
            self.notice_label.configure(bg=bg, fg=fg)

    def _update_perf_panel(
        self,
        *,
        mode: str | None = None,
        elapsed: float | None = None,
        capture_source: str | None = None,
        capture_perf: dict | None = None,
        recommendation_elapsed: float | None = None,
    ) -> None:
        perf = dict(self.last_perf)
        if mode is not None:
            perf["mode"] = mode
        if elapsed is not None:
            perf["elapsed"] = round(float(elapsed), 3)
        if capture_source is not None:
            perf["source"] = capture_source
        if capture_perf is not None:
            perf["capture_perf"] = dict(capture_perf)
        if recommendation_elapsed is not None:
            perf["recommendation"] = round(float(recommendation_elapsed), 3)
        quality = self.last_capture_quality
        if quality is not None:
            perf["quality_ok"] = bool(quality.get("ok"))
            perf["quality_score"] = quality.get("score")
            perf["quality_reason"] = quality.get("reason")
        pet_state = self.last_pet_state
        if pet_state is not None:
            perf["pet_state"] = pet_state.mode
            perf["pet_message"] = pet_state.message
        pet_perf = getattr(self, "last_desktop_pet_perf", None)
        if isinstance(pet_perf, dict):
            perf["pet_render"] = pet_perf.get("render")
        self.last_perf = perf
        self.perf_var.set(_format_perf_panel(perf))
        self._persist_perf(perf)

    #: ``runtime/perf.json`` 的落盘间隔。太快没意义（同一个读数），
    #: 太慢又抓不到「刚才那波卡顿」——5 秒够用。
    PERF_PERSIST_SECONDS = 5.0

    def _persist_perf(self, perf: dict) -> None:
        """把性能面板的读数写到 ``runtime/perf.json``。

        2026-11-18：用户报「CPU 占用很高」，但窗口压不出来、看不到右下角的性能行。
        这份文件是那行字的镜子——帧耗时、采集来源、质量分都在里面，
        排查时不用让用户念屏。只写变化，异步无阻塞。
        """
        now = time.monotonic()
        if now - getattr(self, "_perf_persisted_at", 0.0) < self.PERF_PERSIST_SECONDS:
            return
        self._perf_persisted_at = now
        payload = {"ts": time.time(), **perf}
        try:
            path = Path("runtime/perf.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp.replace(path)
        except (OSError, TypeError, ValueError):
            pass  # 只读目录/半截读数：性能日志丢了不影响识别

    #: 记录状态灯的刷新周期。500ms 足够跟上「开局/收尾」这种秒级变化，
    #: 又不会让 40ms 一轮的采集节奏多出负担。
    RECORD_BADGE_INTERVAL_MS = 500

    def _refresh_record_badge(self) -> None:
        """刷新右下角的记录状态灯。

        只读 ``run_recorder.status()`` 的快照 + 界面自己知道的采集状态，
        不碰任何录制逻辑：灯只是把「在不在录」这件事说出来，
        不允许它反过来影响录制。
        """
        self._record_badge_after_id = None
        try:
            if not self.winfo_exists():
                return
            auto = bool(self.auto_follow_var.get())
            frame_age = None
            if auto and self.auto_follow_last_frame_at:
                frame_age = max(0.0, time.monotonic() - self.auto_follow_last_frame_at)
            try:
                recorder_status = self.run_recorder.status()
            except Exception:  # 灯不能把主循环带崩
                recorder_status = None
            if self.auto_follow_overlap_paused:
                paused_reason = "工具挡住游戏"
            elif self._capture_window_missing:
                paused_reason = "等待游戏窗口"
            elif self._capture_foreground_paused:
                paused_reason = "游戏不在前台"
            else:
                paused_reason = None
            text, level = describe_record_status(
                recorder_status,
                auto_follow=auto,
                capture_error=self._last_capture_error,
                frame_age=frame_age,
                game_window=self._capture_locked_window,
                paused=paused_reason,
                run_archived=self.real_session_finished and not self.pending_new_run,
            )
            service = getattr(self, "brush_evidence", None)
            error = (service.last_error or service.store.last_error) if service is not None else None
            if error:
                text += " · " + {"image_budget_exhausted": "笔触图片已达上限",
                                  "id_only_requires_verified_accuracy": "笔触待准确率核验"}.get(error, "笔触记录需检查")
            self.record_var.set(text)
            self.record_level = level
            if self.record_label is not None:
                self.record_label.configure(fg=RECORD_BADGE_COLORS.get(level, RECORD_BADGE_COLORS["off"]))
            self._record_badge_after_id = self.after(self.RECORD_BADGE_INTERVAL_MS, self._refresh_record_badge)
        except tk.TclError:
            # 窗口正在销毁，静默收摊。
            self._record_badge_after_id = None

    def capture_async(self) -> None:
        if self._capture_inflight:
            self.status_var.set("上一次完整识别还没结束，请稍候…")
            return
        self._capture_inflight = True
        self.status_var.set("正在完整识别游戏窗口...")
        self._set_notice("正在完整识别：会短暂隐藏工具窗口，避免识别到工具自身。", "info")
        self.auto_follow_generation += 1
        self._stop_auto_follow_pump(wait=True)
        self.tracker.reset()
        self.withdraw()
        self.update_idletasks()
        self.after(CAPTURE_WINDOW_RESTORE_MS, self._capture_watchdog)
        thread = threading.Thread(target=self._capture_worker, daemon=True)
        thread.start()

    def _capture_watchdog(self) -> None:
        """完整识别把主窗口 withdraw 了，必须保证它一定回来。

        原来只有 ``_render_result`` / ``_show_error`` 会 deiconify；后台线程一旦
        卡在窗口抓屏（刚拖过尺寸 / 激活失败 / 抓屏重连）或者被回收，工具窗口就
        永久消失，玩家看到的正是「点完整识别卡住了」。这里只恢复窗口和说明情况，
        不杀线程：它晚到的话 ``_render_result`` 照常刷新结果。
        """
        if not self._capture_inflight:
            return
        self._restore_app_window()
        self.status_var.set("完整识别超时：工具窗口已恢复，请重试点一次。")
        self._set_notice("完整识别超过 12 秒未返回，窗口已恢复；若反复出现请重启工具。", "warning")

    def toggle_auto_follow(self) -> None:
        if self.auto_follow_var.get():
            self.auto_follow_generation += 1
            self.auto_follow_last_sequence = 0
            self.auto_follow_busy = False
            self.auto_follow_capture_probed = False
            self.auto_follow_overlap_paused = False
            self.auto_follow_pause_reported = False
            self.auto_follow_last_frame_at = time.monotonic()
            self._last_capture_error = None
            self._capture_locked_window = None
            self._capture_foreground_paused = False
            self.status_var.set("自动跟随已开启：正在接管游戏窗口画面。")
            self._set_notice("自动跟随已开启：直接读游戏窗口自己的画面，工具窗口盖在上面也不影响识别。", "success")
            self._schedule_auto_follow(delay_ms=50)
        else:
            self.auto_follow_generation += 1
            if self.auto_follow_after_id is not None:
                self.after_cancel(self.auto_follow_after_id)
                self.auto_follow_after_id = None
            self._stop_auto_follow_pump()
            self._stop_game_log_watcher()
            self.auto_follow_busy = False
            self.auto_follow_overlap_paused = False
            self.auto_follow_capture_probed = False
            self.status_var.set("自动跟随已关闭。")
            self._set_notice("自动跟随已关闭：可以手动点“完整识别”或“重新快速获取”。", "info")

    def toggle_llm_hosting(self) -> None:
        """大模型托管开关。

        勾选框本身就是 ``llm.enabled``，勾上就写回配置文件；缺 key 时当场说清楚，
        不要等到点了没反应。改完必须让 Advisor 重读配置——它是单例，缓存着旧的
        ``enabled=False``，不重读就会一直报「未开启」。
        """
        from .config import save_strategy_config
        from .llm import get_advisor
        from .llm.advisor import load_config

        want = bool(self.llm_var.get())
        save_strategy_config({"llm": {"enabled": want}})
        get_advisor().reload()
        # 开或关都换一代：正在进行的那次调用（可能要 10 秒以上）回来时对不上号，
        # 直接丢弃，不会把旧建议写回去。
        self._llm_generation += 1
        self._stop_llm_polling()

        if not want:
            self.llm_advice = None
            self.llm_advice_binding = None
            self._llm_pending_frame = None
            # 关键：把 busy 放掉。否则关闭那一刻正在飞的那次调用会把 busy 一直占着，
            # 下次再勾上时 _llm_tick 看到 busy=True 直接 return，托管再也起不来。
            # 飞出去的那次由 generation 作废，它回来时会被丢掉。
            self.llm_busy = False
            self._set_notice("大模型托管已关闭，回到本地识别链路。", "info")
            self.status_var.set("大模型托管已关闭。")
            return

        config = load_config()
        if not config.configured:
            self.llm_var.set(False)
            save_strategy_config({"llm": {"enabled": False}})
            get_advisor().reload()
            self._set_notice(
                "大模型托管需要先配置：runtime/strategy_config.json 的 llm 段填 api_key"
                "（云端）或把 provider 改成 ollama 指向本地模型。",
                "warning",
            )
            self.status_var.set("大模型托管未配置：缺少 api_key 或 base_url。")
            return

        self._set_notice(
            f"大模型托管已开启：{config.provider} / {config.model}，"
            f"每 {config.min_interval:.1f}s 问一次，超时会自动退回本地识别。",
            "success",
        )
        self.status_var.set(f"大模型托管已开启（{config.model}）。")
        self._schedule_llm_ask(delay_ms=200)

    def _stop_llm_polling(self) -> None:
        """取消待触发的定时器，并把队列里没处理的结果扔掉。"""
        if self.llm_after_id is not None:
            try:
                self.after_cancel(self.llm_after_id)
            except Exception:  # noqa: BLE001 - 窗口已销毁时 after_cancel 会抛
                pass
            self.llm_after_id = None
        while True:
            try:
                self._llm_results.get_nowait()
            except queue.Empty:
                break

    def _schedule_llm_ask(self, delay_ms: int = 0) -> None:
        if not self.llm_var.get():
            return
        if self.llm_after_id is not None:
            self.after_cancel(self.llm_after_id)
        self.llm_after_id = self.after(max(0, int(delay_ms)), self._llm_tick)

    def _llm_tick(self) -> None:
        self.llm_after_id = None
        self._llm_drain()
        if not self.llm_var.get():
            return
        if self.llm_busy:
            # 上一次还在飞，隔一会儿回来看队列
            self._schedule_llm_ask(delay_ms=120)
            return
        frame = getattr(self, "_llm_pending_frame", None)
        if frame is None:
            self._schedule_llm_ask(delay_ms=400)
            return
        # 发送前再验一次。入队时是窗口内容，不代表现在还是：窗口可能在这几百
        # 毫秒里关掉/最小化。判断依据是"最近一帧的来源"，两者不一致就说明中间
        # 掉过帧——这时候发出去的图已经不能代表当前局面了。
        if not _llm_may_send_source(self._last_capture_kind):
            self._llm_pending_frame = None
            self._llm_block_reason = f"send_blocked:{self._last_capture_kind}"
            self._schedule_llm_ask(delay_ms=400)
            return
        self._llm_pending_frame = None
        # 冻结"这张图属于哪一帧"的上下文：会话、窗口、阶段、候选卡、采集时刻。
        # 答复回来时拿它比对，对不上就丢掉——不能把旧图的理解套到新画面上。
        binding = self._llm_binding_for_pending_frame()
        self.llm_busy = True
        context = self._llm_context_text()
        thread = threading.Thread(
            target=self._llm_worker,
            args=(frame, context, self._llm_generation, binding),
            name="huahua-llm-ask",
            daemon=True,
        )
        thread.start()

    def _llm_current_signature(self):
        """当前画面的语义指纹。请求时冻结、答复时比对，用的都是这一个函数。"""
        from .llm.binding import scene_signature

        return scene_signature(
            self.current_scene,
            session=int(self.auto_follow_generation or 0),
            window=int(getattr(self, "_capture_window_hwnd", 0) or 0),
        )

    def _llm_binding_for_pending_frame(self):
        """把待发帧钉到它自己的采集时刻上。"""
        from .llm.binding import AdviceBinding

        captured_at = float(getattr(self, "_llm_pending_captured_at", 0.0) or 0.0)
        if captured_at <= 0.0:
            captured_at = time.monotonic()
        return AdviceBinding(
            signature=self._llm_current_signature(),
            captured_at=captured_at,
            generation=int(self._llm_generation),
        )

    def llm_advice_deadline(self) -> float | None:
        """这条建议的截止时刻（monotonic 时钟）。

        由**原始帧的采集时刻**决定：界面重绘多少次都不会延长它。缺 binding
        （历史调用、测试直塞）时返回 ``None``，由调用方按默认 TTL 处理。
        """
        binding = getattr(self, "llm_advice_binding", None)
        return binding.expires_at if binding is not None else None

    def _llm_drain(self) -> None:
        """主线程侧：把工作线程丢过来的结果取出来应用。"""
        while True:
            try:
                generation, advice, binding = self._llm_results.get_nowait()
            except queue.Empty:
                return
            if generation != self._llm_generation or not self.llm_var.get():
                # 这一代已经作废（用户关了托管或重开），丢掉
                continue
            self._apply_llm_advice(advice, binding)

    def _llm_context_text(self) -> str:
        """把本地链路已经知道的东西写成文字，减少模型瞎猜。"""
        from .llm.context import describe_scene

        top = self.recommendations[0] if getattr(self, "recommendations", None) else None
        return describe_scene(self.current_scene, self.current_state, top)

    def _llm_worker(self, frame, context: str, generation: int, binding=None) -> None:
        from .llm import get_advisor

        try:
            advice = get_advisor().advise(frame, context)
        except Exception as exc:  # noqa: BLE001
            advice = AdvisorAdvice(ok=False, error=f"{type(exc).__name__}: {exc}")
        # 只丢队列，不碰 tkinter。
        self._llm_results.put((generation, advice, binding))

    def _apply_llm_advice(self, advice: AdvisorAdvice, binding=None) -> None:
        self.llm_busy = False
        if not self.llm_var.get():
            # 关闭之后才回来的结果，一律不采纳
            return
        if advice.ok and binding is not None:
            reason = binding.why_not_current(
                now=time.monotonic(),
                current=self._llm_current_signature(),
                generation=int(self._llm_generation),
            )
            if reason:
                # 画面已经不是它看过的那一张了。**必须把旧建议清掉**，
                # 否则 arrow 会继续指着上一局的格子。
                self.llm_advice = None
                self.llm_advice_binding = None
                self.llm_reject_reason = reason
                self.status_var.set(f"大模型建议已作废（{reason}），回落本地识别。")
                self._schedule_llm_ask(delay_ms=200)
                return
        self.llm_advice = advice
        self.llm_advice_binding = binding
        self.llm_reject_reason = None
        if advice.ok:
            self.status_var.set(f"大模型：{advice.summary()}")
            self._set_notice(f"大模型建议：{advice.summary()}", "success")
        elif advice.error:
            self.status_var.set(f"大模型不可用，已退回本地识别：{advice.error}")
        self._schedule_llm_ask(delay_ms=200)

    def _llm_target_point(self) -> tuple[int, int] | None:
        """把大模型的建议换算成屏幕坐标。指不出来就返回 ``None``。

        **宁可不指，也不指错。** 三道闸：
        1. 建议还属于当前画面吗（会话/窗口/阶段/候选/截止时刻）；
        2. 这个落点现在还合法吗（格子还能放、背包槽还在、卡还属于这一阶段）；
        3. 只有唯一命中的卡片才给坐标——同名卡或陌生词一律不指。
        """
        advice = self.llm_advice
        if advice is None or not advice.ok:
            return None
        binding = getattr(self, "llm_advice_binding", None)
        if binding is not None:
            reason = binding.why_not_current(
                now=time.monotonic(),
                current=self._llm_current_signature(),
                generation=int(self._llm_generation),
            )
            if reason:
                self.llm_reject_reason = reason
                return None
        from .llm.binding import (
            match_option_index,
            parse_slot_index,
            stage_candidates,
            why_target_invalid,
        )

        invalid = why_target_invalid(advice, scene=self.current_scene, state=self.current_state)
        if invalid:
            self.llm_reject_reason = invalid
            return None
        if advice.action == ACTION_PLACE and advice.cell is not None:
            return self._screen_point_for_board_cell(advice.row, advice.col)
        scene = self.current_scene
        if scene is None:
            return None
        if advice.action in (ACTION_BRUSH1, ACTION_BRUSH2) and scene.brush_options is not None:
            panel = scene.brush_options.panel_rect
            # 这里以前读 `brush_options.affixes` / `.brush_types`——``BrushOptionsObservation``
            # 根本没有这两个字段（只有 stage1_options / stage2_options），一旦模型
            # 给出笔锋建议就抛 AttributeError。改成走同一份"当前阶段候选"来源。
            labels = stage_candidates(scene)
            index = match_option_index(advice.option, labels)
            if index is None:
                # 陌生词、同名卡、空列表：都不指。指错的箭头比没有箭头更糟。
                self.llm_reject_reason = "option_unmatched"
                return None
            y_ratio = 0.30 if index == 0 else 0.71
            cards = getattr(scene.brush_options, "card_rects", ())
            if index < len(cards):
                return self._screen_point_for_capture_rect(cards[index])
            return self._screen_point_for_capture_rect(panel, 0.55, y_ratio)
        if advice.action == ACTION_FINISH:
            region = scene.region(RegionKind.FINISH_DRAWING)
            if region is not None:
                return self._screen_point_for_capture_rect(region.rect)
        if advice.action == ACTION_BACKPACK:
            # 模型给的 option 是「第几格，从 1 开始」。原来这里没有分支，
            # 于是它一说 backpack 箭头就没了——明明背包格子坐标本地是知道的。
            paints = getattr(getattr(scene, "backpack", None), "paints", None) or ()
            index = parse_slot_index(advice.option)
            if index is not None and 1 <= index <= len(paints):
                return self._screen_point_for_capture_rect(paints[index - 1].slot_rect)
        return None

    def open_region_calibration(self) -> None:
        """Preview the very same automatic geometry used by recognition."""
        from .region_calibration_ui import AutomaticGeometryOverlay

        existing = getattr(self, "_calibration_overlay", None)
        if existing is not None:
            existing.close()
            self._calibration_overlay = None
            self._set_notice("自动定位预览已关闭，识别继续运行。", "info")
            return
        if not self.auto_follow_var.get():
            self.auto_follow_var.set(True)
            self.toggle_auto_follow()
        self._calibration_overlay = AutomaticGeometryOverlay(self)
        self._set_notice("自动定位预览已打开：棋盘和笔触卡片随游戏窗口自动更新。", "info")

    def toggle_overlay(self) -> None:
        self._destroy_overlay_window()
        if self.overlay_var.get():
            self._rerender_side_panel()
            self.status_var.set("坐标提示已开启：不覆盖游戏窗口，只在工具内显示推荐坐标。")
            self._set_notice("坐标提示已开启：只在工具内显示坐标，不再创建覆盖游戏的蒙版。", "info")
        else:
            self._rerender_side_panel()
            self.status_var.set("坐标提示已关闭。")
            self._set_notice("坐标提示已关闭。", "info")

    def _ensure_overlay_window(self) -> None:
        if self.overlay_window is not None and self.overlay_window.winfo_exists():
            return
        overlay = tk.Toplevel(self)
        overlay.title("Huahua Highlight Overlay")
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.42)
        overlay.configure(bg="#010203")
        try:
            overlay.attributes("-transparentcolor", "#010203")
        except tk.TclError:
            pass
        canvas = tk.Canvas(overlay, bg="#010203", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.overlay_window = overlay
        self.overlay_canvas = canvas
        self._make_overlay_clickthrough()
        self.after(80, self._make_overlay_clickthrough)

    def _destroy_overlay_window(self) -> None:
        if self.overlay_after_id is not None:
            self.after_cancel(self.overlay_after_id)
            self.overlay_after_id = None
        self._restore_overlay_window_procs()
        if self.overlay_window is not None and self.overlay_window.winfo_exists():
            self.overlay_window.destroy()
        self.overlay_window = None
        self.overlay_canvas = None

    def _make_overlay_clickthrough(self) -> None:
        if self.overlay_window is None:
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            GWL_EXSTYLE = -20
            GWLP_WNDPROC = -4
            WM_NCHITTEST = 0x0084
            HTTRANSPARENT = -1
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_FRAMECHANGED = 0x0020

            hwnds = [self.overlay_window.winfo_id()]
            if self.overlay_canvas is not None:
                hwnds.append(self.overlay_canvas.winfo_id())

            get_window_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            get_window_long.restype = ctypes.c_ssize_t
            set_window_long.restype = ctypes.c_ssize_t
            call_window_proc = user32.CallWindowProcW
            call_window_proc.restype = ctypes.c_ssize_t
            wndproc_type = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )

            for hwnd in hwnds:
                style = get_window_long(hwnd, GWL_EXSTYLE)
                set_window_long(
                    hwnd,
                    GWL_EXSTYLE,
                    style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                )
                if hwnd not in self.overlay_wndprocs:
                    original_proc = get_window_long(hwnd, GWLP_WNDPROC)

                    def _transparent_proc(window_hwnd, msg, wparam, lparam, *, _original_proc=original_proc):
                        if msg == WM_NCHITTEST:
                            return HTTRANSPARENT
                        return call_window_proc(_original_proc, window_hwnd, msg, wparam, lparam)

                    callback = wndproc_type(_transparent_proc)
                    set_window_long(hwnd, GWLP_WNDPROC, ctypes.cast(callback, ctypes.c_void_p).value)
                    self.overlay_wndprocs[int(hwnd)] = (int(original_proc), callback)
                user32.SetWindowPos(
                    hwnd,
                    0,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
                )
            try:
                self.overlay_window.attributes("-disabled", True)
            except tk.TclError:
                pass
        except Exception:
            return

    def _restore_overlay_window_procs(self) -> None:
        if not self.overlay_wndprocs:
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            GWLP_WNDPROC = -4
            set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            set_window_long.restype = ctypes.c_ssize_t
            for hwnd, (original_proc, _callback) in list(self.overlay_wndprocs.items()):
                try:
                    set_window_long(hwnd, GWLP_WNDPROC, original_proc)
                except Exception:
                    pass
        finally:
            self.overlay_wndprocs.clear()

    def _refresh_overlay(self) -> None:
        if not self.overlay_var.get():
            return
        # Full-screen transparent overlays can still consume input in some
        # games/Windows configurations. Keep this mode input-safe by rendering
        # hints inside the assistant UI instead of covering the game window.
        return

    def _overlay_tick(self) -> None:
        self.overlay_after_id = None
        if not self.overlay_var.get():
            return
        self.overlay_phase = not self.overlay_phase
        self._draw_overlay()
        self.overlay_after_id = self.after(520, self._overlay_tick)

    def _draw_overlay(self) -> None:
        if self.overlay_window is None or self.overlay_canvas is None:
            return
        window = self.last_capture_window
        scene = self.current_scene
        state = self.current_state
        if window is None or scene is None or state is None:
            self.overlay_window.withdraw()
            if self.overlay_var.get():
                self.status_var.set("高亮蒙版等待真实窗口坐标：请先完整识别游戏窗口。")
            return
        # 只在真的不可见时才 deiconify。原来每 520ms 无条件调一次，
        # 等于每半秒把蒙版重新 ShowWindow 一遍，游戏画面跟着闪。
        if not self.overlay_window.winfo_viewable():
            self.overlay_window.deiconify()
            self._make_overlay_clickthrough()
        geometry = f"{window.width}x{window.height}+{window.left}+{window.top}"
        if geometry != getattr(self, "_overlay_last_geometry", None):
            self.overlay_window.geometry(geometry)
            self._overlay_last_geometry = geometry
        canvas = self.overlay_canvas
        canvas.configure(width=window.width, height=window.height)
        canvas.delete("all")
        color = "#ffd84d" if self.overlay_phase else "#64e6ff"
        self._draw_overlay_board_hints(canvas, scene, color)
        self._draw_overlay_backpack_hints(canvas, scene, state, color)

    def _set_last_capture_context(
        self,
        window: WindowInfo | None,
        capture_size: tuple[int, int] | None,
    ) -> None:
        self.last_capture_window = window
        self.last_capture_size = capture_size if _valid_capture_size(capture_size) else None

    def _update_fast_snapshot(
        self,
        snapshot: FastVisionSnapshot | None,
        *,
        evaluate_pet: bool = True,
    ) -> None:
        if snapshot is None:
            self.previous_fast_snapshot = self.last_fast_snapshot
            self.last_fast_snapshot = None
            self.last_fast_events = ()
            self.last_pet_state = PetBrainState(mode="observing", message="等待识别", confidence=0.2)
            return
        previous = self.last_fast_snapshot
        self.previous_fast_snapshot = previous
        self.last_fast_snapshot = snapshot
        self.last_fast_events = detect_fast_events(previous, snapshot)
        if evaluate_pet:
            self.last_pet_state = evaluate_pet_state(self)

    def _screen_point_from_capture_point(self, x: float, y: float) -> tuple[int, int] | None:
        local = self._window_local_point_from_capture_point(x, y)
        if local is None:
            return None
        window = self.last_capture_window
        if window is None:
            return int(round(local[0])), int(round(local[1]))
        return int(round(window.left + local[0])), int(round(window.top + local[1]))

    def _screen_point_for_capture_rect(
        self,
        rect: Rect,
        x_ratio: float = 0.5,
        y_ratio: float = 0.5,
    ) -> tuple[int, int] | None:
        x = float(rect.x) + float(rect.w) * x_ratio
        y = float(rect.y) + float(rect.h) * y_ratio
        return self._screen_point_from_capture_point(x, y)

    def _window_local_point_from_capture_point(self, x: float, y: float) -> tuple[float, float] | None:
        window = self.last_capture_window
        capture_size = self.last_capture_size
        if window is not None and capture_size is not None and window.width > 0 and window.height > 0:
            capture_w, capture_h = capture_size
            return float(x) / capture_w * window.width, float(y) / capture_h * window.height
        return float(x), float(y)

    def _window_local_rect_from_capture_rect(self, rect: Rect, pad: float = 0.0) -> tuple[float, float, float, float]:
        x1 = float(rect.x) + pad
        y1 = float(rect.y) + pad
        x2 = float(rect.right) - pad
        y2 = float(rect.bottom) - pad
        p1 = self._window_local_point_from_capture_point(x1, y1) or (x1, y1)
        p2 = self._window_local_point_from_capture_point(x2, y2) or (x2, y2)
        return p1[0], p1[1], p2[0], p2[1]

    def _screen_rect_from_capture_rect(self, rect: Rect, pad: float = 0.0) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = self._window_local_rect_from_capture_rect(rect, pad=pad)
        window = self.last_capture_window
        if window is None:
            return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        return (
            int(round(window.left + x1)),
            int(round(window.top + y1)),
            int(round(window.left + x2)),
            int(round(window.top + y2)),
        )

    def _draw_overlay_board_hints(self, canvas: tk.Canvas, scene: SceneObservation, color: str) -> None:
        if not self.recommendations:
            return
        best = self.recommendations[0]
        for index, rec in enumerate(self.recommendations[:3], start=1):
            tile = scene.board.tile(rec.row, rec.col)
            width = 7 if rec == best else 4
            outline = color if rec == best else "#9ff5ff"
            x1, y1, x2, y2 = self._window_local_rect_from_capture_rect(tile.rect, pad=5)
            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=outline,
                width=width,
            )
            tx, ty = self._window_local_point_from_capture_point(tile.rect.x + 12, tile.rect.y + 14) or (
                tile.rect.x + 12,
                tile.rect.y + 14,
            )
            canvas.create_text(
                tx,
                ty,
                text=f"{index}",
                fill=outline,
                font=("Microsoft YaHei UI", 16, "bold"),
                anchor=tk.W,
            )

    def _draw_overlay_backpack_hints(self, canvas: tk.Canvas, scene: SceneObservation, state: PuzzleState, color: str) -> None:
        if scene.backpack is None or not state.backpack:
            return
        desired = Counter(step.paint for step in self.install_plan.steps) if self.install_plan else Counter()
        if not desired and self.recommendations and self.recommendations[0].paint is not None:
            desired[self.recommendations[0].paint] += 1
        if not desired:
            return
        for observed in scene.backpack.paints:
            paint = Paint(
                category=observed.category,
                level=observed.level,
                subtype=observed.subtype,
                is_treasure=observed.is_treasure,
            )
            if desired[paint] <= 0:
                continue
            desired[paint] -= 1
            rect = observed.slot_rect
            x1, y1, x2, y2 = self._window_local_rect_from_capture_rect(rect, pad=3)
            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=color,
                width=6,
            )
            tx, ty = self._window_local_point_from_capture_point((rect.x + rect.right) / 2, rect.y - 10) or (
                (rect.x + rect.right) / 2,
                rect.y - 10,
            )
            canvas.create_text(
                tx,
                ty,
                text="先点",
                fill=color,
                font=("Microsoft YaHei UI", 13, "bold"),
            )

    def _schedule_auto_follow(self, delay_ms: int | None = None) -> None:
        if not self.auto_follow_var.get():
            return
        if self.auto_follow_after_id is not None:
            self.after_cancel(self.auto_follow_after_id)
        interval = delay_ms if delay_ms is not None else 40
        self.auto_follow_after_id = self.after(interval, self._auto_follow_tick)

    def _start_auto_follow_pump(self) -> None:
        if self.auto_follow_pump is None:
            self.auto_follow_pump = FramePump(
                capture_func=default_auto_capture,
                prepare_capture_func=self.tracker.set_capture_context,
                geometry_probe=find_capture_window,
                analyze_func=self.tracker.analyze_quick,
                needs_full_refresh=lambda: False,
                full_analyze_func=self.tracker.analyze_realtime,
                maybe_save_frame=self._maybe_save_auto_frame,
                record_func=self._record_auto_snapshot,
                model_state_func=lambda: self.current_state,
                slow_frame_func=lambda payload: record_observation(
                    "slow_frame", dict(payload), source="auto_follow"
                ),
            )
        interval_ms = max(100, int(self.auto_follow_interval_var.get() or 100))
        self.auto_follow_pump.start(interval=interval_ms / 1000.0)
        self.brush_evidence.start()
        self.auto_follow_last_frame_at = time.monotonic()

    def _stop_auto_follow_pump(self, *, wait: bool = False) -> None:
        evidence = getattr(self, "brush_evidence", None)
        if evidence is not None:
            evidence.stop()
        if self.auto_follow_pump is not None:
            self.auto_follow_pump.stop(wait=wait)
        self.auto_follow_pump = None
        self.auto_follow_last_sequence = 0

    def _maybe_save_auto_frame(self, frame) -> float:
        if not bool(config_value("debug.save_fast_captures", False)):
            return 0.0
        write_started_at = time.perf_counter()
        write_image("runtime/capture_auto.png", frame)
        return time.perf_counter() - write_started_at

    def _record_auto_snapshot(self, snapshot: FastVisionSnapshot | None, capture) -> object:
        clicks = self.mouse_click_monitor.drain()
        local_clicks: list[tuple[float, float, float]] = []
        window = capture.window
        geometry = (window.hwnd, window.left, window.top, window.width, window.height,
                    capture.image.shape[:2]) if window is not None else None
        previous_geometry = getattr(self, "_recorder_click_geometry", geometry)
        self._recorder_click_geometry = geometry
        if geometry != previous_geometry:
            # Drained clicks may belong to coordinates before the resize/move.
            clicks = ()
        if window is not None and snapshot is not None and window.width > 0 and window.height > 0:
            capture_height, capture_width = capture.image.shape[:2]
            for click in clicks:
                x = (click.x - window.left) / window.width * capture_width
                y = (click.y - window.top) / window.height * capture_height
                local_clicks.append((click.ts, x, y))
        # 换局信号「用掉就清」：日志事件是一次性的，不能每帧都拿它去切局。
        boundary, self._recorder_run_boundary = self._recorder_run_boundary, None
        result = self.run_recorder.process(
            snapshot, clicks=tuple(local_clicks), run_boundary=boundary
        )
        service = getattr(self, "brush_evidence", None)
        if service is not None:
            service.context(result.session_id if result.phase in {"opening", "playing"} else None, snapshot)
            evidence = dict(result.evidence or {})
            evidence["brush_choices"] = service.evidence(result.session_id)
            result = replace(result, evidence=evidence)
        return result

    # --- 游戏日志驱动 -------------------------------------------------------

    def _sync_game_log_watcher(self) -> None:
        """确保日志监听挂上了。找不到日志就安静退回固定节拍。"""

        watcher = self.game_log_watcher
        if watcher is not None and watcher.running:
            return
        now = time.monotonic()
        if now < self.game_log_retry_at:
            return
        # 找不到就每 5 秒再试一次，别让 40ms 的 tick 反复去翻磁盘。
        self.game_log_retry_at = now + 5.0

        game_window = find_window(DEFAULT_WINDOW_TITLE)
        pids = (game_window.process_id,) if game_window is not None else ()
        configured = str(config_value("game_log.path", "") or "").strip() or None
        path = find_game_log(configured, pids)
        if path is None:
            if not self.game_log_missing_reported:
                self.game_log_missing_reported = True
                self._set_notice(
                    "没找到游戏日志，先按固定节拍识别（慢一些但能用）。日志一般在 "
                    "<游戏安装目录>\\UE_game\\TorchLight\\Saved\\Logs\\UE_game.log",
                    "warning",
                )
            return
        watcher = GameLogWatcher(path)
        watcher.start()
        self.game_log_watcher = watcher
        self.game_log_missing_reported = False
        self.auto_follow_log_revision = 0
        self._sync_game_log_status(watcher)
        self.status_var.set("已接上游戏日志：按画面变化触发识别。")
        record_observation(
            "game_log_attached",
            {"path": str(path)},
            source="game_log",
        )

    def _stop_game_log_watcher(self) -> None:
        watcher = self.game_log_watcher
        self.game_log_watcher = None
        self.auto_follow_burst_until = 0.0
        # 重开自动跟随时不该等上一个 5 秒冷却。
        self.game_log_retry_at = 0.0
        if watcher is not None:
            watcher.stop()
        self.game_log_var.set("日志：未接上")

    def _start_game_log_tick(self) -> None:
        """把日志监听挂成一条常驻循环，跟自动跟随开关无关。

        以前这条循环只在 ``_auto_follow_tick`` 里跑，于是「自动跟随没开」就等于
        「日志根本没在看」：用户放完颜料，界面和桌宠一点动静都没有，也看不出
        到底是没接上还是不该动。
        """

        if self.game_log_tick_after_id is not None:
            return
        self.game_log_tick_after_id = self.after(GAME_LOG_TICK_MS, self._game_log_tick)

    def _stop_game_log_tick(self) -> None:
        if self.game_log_tick_after_id is not None:
            try:
                self.after_cancel(self.game_log_tick_after_id)
            except tk.TclError:
                pass
            self.game_log_tick_after_id = None

    def _game_log_tick(self) -> None:
        self.game_log_tick_after_id = None
        try:
            self._sync_game_log_watcher()
            self._consume_game_log_events()
        finally:
            self.game_log_tick_after_id = self.after(GAME_LOG_TICK_MS, self._game_log_tick)

    def _sync_game_log_status(self, watcher: GameLogWatcher | None = None) -> None:
        watcher = watcher if watcher is not None else self.game_log_watcher
        if watcher is None or not watcher.running:
            self.game_log_var.set("日志：未接上")
            return
        if self.game_log_event_count <= 0:
            self.game_log_var.set("日志：已接上，等待落子")
            return
        last = self.game_log_last_event
        tail = ""
        if last is not None and last.game_time is not None:
            tail = f" · {last.game_time:%H:%M:%S}"
        self.game_log_var.set(f"日志事件 {self.game_log_event_count}{tail}")

    def _consume_game_log_events(self) -> None:
        """把日志里新出的事件换成「现在该截一帧」，并让它在界面上看得见。"""

        watcher = self.game_log_watcher
        pump = self.auto_follow_pump
        if watcher is None or not watcher.running:
            # 没有日志可用：保持用户设的固定节拍，不要因为这条路断了就变慢。
            if pump is not None:
                pump.set_interval(self._auto_follow_base_interval())
            self._sync_game_log_status(None)
            return

        events = watcher.drain()
        now = time.monotonic()
        if events:
            self.last_game_log_events = events
            self.auto_follow_last_event_at = now
            self.auto_follow_log_revision = watcher.revision
            self.game_log_event_count += len(events)
            self.game_log_last_event = events[-1]
            for event in events:
                record_observation(
                    "game_log_event",
                    {
                        "name": event.name,
                        "kind": event.kind,
                        "captures": event.captures,
                        "game_time": event.game_time.isoformat() if event.game_time else None,
                        "auto_follow": bool(self.auto_follow_var.get()),
                    },
                    source="game_log",
                )
            self._note_level_animation_events(events)
            # 「入画」= 一张新画纸开始了。实测两局各只有一次，且都紧跟在开局：
            #   15.37.17 Into → 15.37.18 TearUp → 15.37.19 Select → 15.37.20 Unfold
            # 用它开新局，比等「结算成功」可靠——重绘根本不经结算。
            if any(event.kind == "into" for event in events):
                self.pending_new_run = True
                record_observation("new_run_pending", {"trigger": "game_log_into"}, source="game_log")
            # 同一个信号也要递给 run_recorder：它只看画面，而换局那几帧整盘被替换，
            # 会被 `_frame_looks_replaced` 当成「入画画面误读」丢掉，于是两局合并成
            # 一条（2026-11-22 21:56→21:57 实测：一条 214 秒的记录装了两局）。
            # 日志是服务端驱动的，没有这个歧义。
            if any(event.kind == "into" for event in events):
                self._recorder_run_boundary = "new_run"
                service = getattr(self, "brush_evidence", None)
                if service is not None:
                    service.context(None, None)
            elif any(event.kind == "finish" for event in events):
                self._recorder_run_boundary = "finished"
                service = getattr(self, "brush_evidence", None)
                if service is not None:
                    service.context(None, None)
            self._sync_game_log_status(watcher)

        # 先定「静默与否」。它只看 watcher 的时间戳，跟这一批事件无关，
        # 所以放在前面：事件一恢复就能立刻把静默状态清掉（否则会一直挂着兜底节拍）。
        silent = self._note_log_silence(watcher, now)
        capturing = [event for event in events if event.captures]
        if silent:
            self._report_log_silence(watcher)
        elif self.auto_follow_silence_notice_key:
            self._report_log_resumed(watcher)

        if capturing and pump is not None:
            # 落子/开面板的动画还没走完就截会读到中间态，等一小会儿再叫醒采集线程。
            pump.set_interval(LOG_BURST_INTERVAL)
            self.auto_follow_burst_until = now + LOG_BURST_WINDOW
            self.after(LOG_SETTLE_MS, self._wake_auto_follow_pump)
        elif pump is not None and now > self.auto_follow_burst_until:
            # 静默时用更慢的兜底节拍：词汇对不上至少还能出帧，而不是整段沉默。
            pump.set_interval(
                self._auto_follow_fallback_interval() if silent else self._auto_follow_idle_interval()
            )

        if capturing and not self.auto_follow_var.get():
            self._nudge_auto_follow_after_event(capturing[-1], now)

    def _auto_follow_fallback_interval(self) -> float:
        """日志在写但认不出事件时用的兜底节拍。

        为什么需要它：捕获的唤醒**完全绑死在日志词汇**上。2026-11-22 实测第二局，
        用户大量安装颜料（放在初始颜料生效范围外，任务没完成），游戏只播通用按钮音
        ``ECommonBtnType``，一条 ``S8_*`` 都没有 → 日志驱动整段沉默，很久才出一帧。
        兜底节拍让「词汇对不上」最多慢一点，而不是整段丢。
        """

        try:
            value = float(config_value("game_log.fallback_interval", LOG_FALLBACK_INTERVAL))
        except (TypeError, ValueError):
            value = LOG_FALLBACK_INTERVAL
        return max(0.2, value)

    def _log_is_silent(self, watcher: GameLogWatcher) -> bool:
        """日志在持续写入，但已经很久没有可识别事件了。"""

        if not watcher.running or watcher.line_age > LOG_ALIVE_SECONDS:
            return False
        return watcher.silent_seconds >= LOG_SILENT_SECONDS

    def _note_log_silence(self, watcher: GameLogWatcher, now: float) -> bool:
        """静默状态翻转时记一笔，并说明「日志在写但认不出」这件事本身。"""

        silent = self._log_is_silent(watcher)
        if silent == self.auto_follow_log_silent:
            return silent
        self.auto_follow_log_silent = silent
        record_observation(
            "game_log_silent" if silent else "game_log_silent_cleared",
            {
                "silent_seconds": round(watcher.silent_seconds, 1),
                "line_age": round(watcher.line_age, 1),
                "lines_seen": watcher.lines_seen,
                "fallback_interval": self._auto_follow_fallback_interval() if silent else None,
                "note": (
                    "日志在写但解析不出画中雪原事件，已按兜底节拍采集"
                    if silent
                    else "日志事件恢复，回到日志驱动"
                ),
            },
            source="game_log",
        )
        return silent

    def _report_log_silence(self, watcher: GameLogWatcher) -> None:
        """把「日志在动、工具却认不出」顶到界面上，别让它静默降级。"""

        if not self._log_is_silent(watcher):
            return
        if self.auto_follow_silence_notice_key:
            return
        self.auto_follow_silence_notice_key = True
        seconds = int(max(0, watcher.silent_seconds))
        message = (
            f"日志在写入但已 {seconds} 秒没有可识别的画中雪原事件，"
            f"已切到 {self._auto_follow_fallback_interval():.1f}s 兜底节拍。"
            "若是把颜料放在初始颜料生效范围外，游戏只播通用按钮音，属于已知情况。"
        )
        self.status_var.set(message)
        self._set_notice(message, "warning")

    def _report_log_resumed(self, watcher: GameLogWatcher) -> None:
        self.auto_follow_silence_notice_key = False
        self.status_var.set("日志事件已恢复，回到日志驱动采集。")

    def _nudge_auto_follow_after_event(self, event: GameLogEvent, now: float) -> None:
        """自动跟随没开时，至少让用户知道「刚才那下我看见了」。"""

        if now - self.game_log_nudge_at < 3.0:
            return
        self.game_log_nudge_at = now
        self.status_var.set(f"检测到游戏事件：{event.name}（自动跟随未开启，画面不会自动重算）")
        self._set_notice(
            f"日志抓到「{event.name}」：画面已经变了，但自动跟随没开，所以不会自动重算。"
            "勾上工具栏的「自动跟随」就会跟着落子实时刷新。",
            "info",
        )

    def _wake_auto_follow_pump(self) -> None:
        pump = self.auto_follow_pump
        if pump is not None and self.auto_follow_var.get():
            pump.wake()

    def _auto_follow_base_interval(self) -> float:
        try:
            return max(0.1, float(self.auto_follow_interval_var.get() or 100) / 1000.0)
        except (tk.TclError, ValueError):
            return 0.1

    def _auto_follow_idle_interval(self) -> float:
        try:
            return max(0.2, float(config_value("game_log.idle_interval", LOG_IDLE_INTERVAL)))
        except (TypeError, ValueError):
            return LOG_IDLE_INTERVAL

    def _auto_follow_tick(self) -> None:
        self.auto_follow_after_id = None
        if not self.auto_follow_var.get():
            return
        if self.auto_follow_busy:
            self._schedule_auto_follow()
            return
        if self._app_overlaps_game_window():
            self._stop_auto_follow_pump()
            self.auto_follow_overlap_paused = True
            self._report_overlap_pause()
            self._schedule_auto_follow(delay_ms=1000)
            return
        self.auto_follow_overlap_paused = False
        self.auto_follow_pause_reported = False
        self._start_game_log_tick()
        if self._auto_follow_pump_stalled():
            self._restart_stalled_auto_follow_pump()
        if self.auto_follow_pump is None:
            self._start_auto_follow_pump()
        self._poll_auto_follow_pump(self.auto_follow_generation)
        self._report_capture_gap()

    def _report_capture_gap(self) -> None:
        """「采集线程在跑，却很久没往轨迹里写东西」时留一条证据。

        为什么需要：这种情况界面上一切正常——灯是绿的、日志是接上的、进程也不报错，
        但棋盘上的变化一帧都没进库。2026-11-22 那 64 秒就是这么过去的，最后只能
        人工去数日志时间戳才发现。把当时的节拍、日志状态、丢帧原因一起落盘，
        下次直接看这一条就能定位。
        """

        if self.auto_follow_pump is None:
            self.capture_gap_reported = False
            return
        now = time.monotonic()
        last = self.last_step_recorded_at or self.real_session_started_at or now
        gap = now - last
        if gap < CAPTURE_GAP_SECONDS:
            self.capture_gap_reported = False
            return
        if self.capture_gap_reported:
            return
        self.capture_gap_reported = True
        watcher = self.game_log_watcher
        record_observation(
            "capture_gap",
            {
                "session": self.real_session_id or None,
                "gap_seconds": round(gap, 1),
                "frames_since_step": self.capture_gap_frames,
                "pump_interval": round(self.auto_follow_pump.interval, 2),
                "log_running": bool(watcher.running) if watcher is not None else False,
                "log_line_age": round(watcher.line_age, 1) if watcher is not None else None,
                "log_idle_seconds": round(watcher.idle_seconds, 1) if watcher is not None else None,
                "log_silent": self.auto_follow_log_silent,
                "last_quick_step_key_matched": True,
                "note": "采集线程在跑但没有新步骤入库：多半是帧内容没变被去重，或日志驱动没唤醒",
            },
            source="auto_follow",
        )

    def _auto_follow_pump_stalled(self) -> bool:
        """采集线程还挂着，却一直不出帧。

        出帧的线程挂了不会报错：``pop_latest`` 永远返回 None，tick 每 80ms 空转，
        界面和桌宠就那么定住了。留一个超时兜底，比让用户以为工具死了强。
        """

        if self.auto_follow_pump is None:
            return False
        return time.monotonic() - self.auto_follow_last_frame_at > AUTO_FOLLOW_STALL_SECONDS

    def _restart_stalled_auto_follow_pump(self) -> None:
        self._stop_auto_follow_pump(wait=True)
        self.auto_follow_last_frame_at = time.monotonic()
        record_observation(
            "auto_follow_restart",
            {"note": "No frame for a while; restarting the capture pump."},
            source="auto_follow",
        )

    def _report_overlap_pause(self) -> None:
        """遮挡暂停时说清楚「为什么会被挡住」，并只在状态变化时记一次。

        这条分支以前只丢一句「把工具移开」，用户看到的就是桌宠一动不动。
        真正的原因几乎都是窗口内容抓取没接管、退回了屏幕复制——把那句话
        说出来，比让人反复挪窗口有用。
        """

        reason = wgc_status()
        degraded = reason != "wgc_ready"
        if reason == "wgc_blocked_low_integrity":
            # 实测结论：自动化环境拉起的进程是 LOW 完整性，WGC 对**任何**目标
            # 都回 E_ACCESSDENIED。这不是游戏窗口的问题，重启方式才是。
            message = (
                "自动跟随暂停：窗口内容抓取被系统拒绝——当前进程是 LOW 完整性"
                "（多半是被自动化/脚本拉起的）。请关掉工具后双击 启动.bat 重新启动。"
            )
        elif degraded:
            message = (
                "自动跟随暂停：窗口内容抓取没有接管（"
                f"{reason}），退回屏幕复制时工具窗口会挡进画面。"
                "把工具窗口移到游戏窗口外，或装好 windows-capture 后重启工具。"
            )
        else:
            message = "自动跟随暂停：工具窗口覆盖了游戏画面，请把工具移到游戏窗口外后会自动继续。"
        self.status_var.set(message)
        self._set_notice(message, "warning")
        if self.auto_follow_pause_reported:
            return
        self.auto_follow_pause_reported = True
        record_observation(
            "auto_follow_pause",
            {
                "wgc": reason,
                "probed": bool(self.auto_follow_capture_probed),
                "note": "Auto follow stopped because the tool window covers the game.",
            },
            source="auto_follow",
        )

    def _poll_auto_follow_pump(self, generation: int) -> None:
        if not self.auto_follow_var.get() or generation != self.auto_follow_generation:
            return
        pump = self.auto_follow_pump
        if pump is None:
            self._schedule_auto_follow(delay_ms=500)
            return
        result = pump.pop_latest(self.auto_follow_last_sequence)
        if result is None:
            self._schedule_auto_follow(delay_ms=80)
            return
        self.auto_follow_last_sequence = result.sequence
        self.auto_follow_last_frame_at = time.monotonic()
        # 到过一帧就记一次：配合 last_step_recorded_at 能看出「截了但没入库」。
        self.capture_gap_frames += 1
        # 换了游戏窗口（重开游戏、窗口重建）：把「WGC 不可用」的判定清掉重试一次。
        # 那条判定只是为了别每帧白抛异常，不该变成永久结论。
        captured_window = result.capture.window
        # The UI may consume a queued result after another resize or move.
        if result.scene is not None and captured_window is not None and not same_window_geometry(
            captured_window, find_capture_window()
        ):
            result = replace(result, scene=None, snapshot=None,
                             perf={**result.perf, "geometry_pending": True})
        captured_hwnd = int(getattr(captured_window, "hwnd", 0) or 0)
        if captured_hwnd and captured_hwnd != self._capture_window_hwnd:
            previous_hwnd, self._capture_window_hwnd = self._capture_window_hwnd, captured_hwnd
            if previous_hwnd:
                reset_wgc_state()
        # 灯要能分辨「抓到的是游戏」还是「抓到的是桌面」：后者 window 可能还在，
        # 但像素是合成后的桌面，不能拿"window 对象存在"当"抓到游戏"。
        source_kind = result.capture.kind
        self._last_capture_kind = source_kind
        self._capture_foreground_paused = source_kind == CAPTURE_KIND_PAUSED
        self._capture_window_missing = (
            source_kind == CAPTURE_KIND_PAUSED and result.capture.window is None
        )
        if not self._capture_foreground_paused:
            self._capture_locked_window = result.capture.window is not None
        if result.error:
            self._last_capture_error = result.error
            self._auto_follow_error(RuntimeError(result.error), generation)
            return
        # 第一帧回来了，说明「能不能直接抓窗口内容」已经有答案，遮挡判断从此按实际情况走。
        self.auto_follow_capture_probed = True
        self._last_capture_error = None
        # 大模型托管：**只把窗口自己渲染的像素交出去**。暂停、整屏兜底、桌面区域
        # 一律不入队，并且立刻清掉待发帧——否则"窗口消失前一瞬"留下的那张图
        # 会在下一次 _llm_tick 被悄悄发出去。
        if self.llm_var.get():
            if _llm_may_receive_frame(result.capture) and not result.perf.get("geometry_pending"):
                self._llm_pending_frame = result.capture.image
                self._llm_pending_captured_at = float(result.captured_at or 0.0)
                self._llm_block_reason = None
                self._llm_last_blocked_kind = None
                if self.llm_after_id is None and not self.llm_busy:
                    self._schedule_llm_ask(delay_ms=0)
            else:
                if self._llm_pending_frame is not None:
                    self._llm_block_reason = f"capture_source_untrusted:{source_kind}"
                self._llm_pending_frame = None
                # 只在"变成不可信"的那一次提示，不然每 100ms 弹一条。
                if source_kind != self._llm_last_blocked_kind:
                    self._llm_last_blocked_kind = source_kind
                    self._set_notice(
                        "大模型托管暂停：当前画面不是游戏窗口内容，只用本地识别。",
                        "info",
                    )
        capture_size = _capture_shape_size(result.frame_shape)
        self._render_auto_follow_result(
            result.scene,
            result.capture.source,
            result.elapsed,
            result.capture.window,
            capture_size,
            generation,
            result.perf,
            recording_evidence=result.recording_evidence,
        )

    def _auto_follow_error(self, exc: Exception, generation: int) -> None:
        if generation != self.auto_follow_generation or not self.auto_follow_var.get():
            return
        self.auto_follow_busy = False
        self._last_capture_error = str(exc)
        self.status_var.set(f"自动跟随失败：{exc}")
        self._set_notice(f"自动跟随失败：{exc}", "error")
        self._schedule_auto_follow(delay_ms=1500)

    def quick_capture_async(self) -> None:
        self.status_var.set("正在快速获取棋盘状态...")
        self._set_notice("正在快速获取：跳过 OCR，优先给出 1 秒级局面建议。", "info")
        self.auto_follow_generation += 1
        self._stop_auto_follow_pump(wait=True)
        self.withdraw()
        self.update_idletasks()
        thread = threading.Thread(target=self._quick_capture_worker, daemon=True)
        thread.start()

    def _capture_worker(self) -> None:
        try:
            time.sleep(0.04)
            started_at = time.perf_counter()
            capture = capture_target(
                DEFAULT_WINDOW_TITLE,
                fallback_to_screen=True,
                fallback_if_target_uncaptureable=False,
                activate=True,
                activation_delay=0.02,
            )
            frame = capture.image
            self.tracker.set_capture_context(capture)
            write_image("runtime/capture.png", frame)
            scene, debug_path = analyze_and_save_debug(
                frame,
                self.debug_path,
                tracker=self.tracker,
                save_crops=False,
                realtime=True,
            )
            if scene is None:
                # 实时快路径只按布局算框：刚拖过窗口、标定还没落盘时会整片判否。
                # 「完整识别」本来就该做一次冷启动全搜索，不能让玩家点了个叫
                # 完整识别的按钮却只拿到「未识别到棋盘」。冷启动成功也按窗口尺寸
                # 落盘标定，下一次自动跟随直接走快路径。
                self.tracker.reset()
                scene, debug_path = analyze_and_save_debug(
                    frame,
                    self.debug_path,
                    tracker=self.tracker,
                    save_crops=False,
                    realtime=False,
                )
                if scene is not None:
                    vision.remember_board_calibration(frame, scene)
            elapsed = time.perf_counter() - started_at
        except Exception as exc:
            self.after(0, lambda: self._show_error(exc))
            return

        capture_size = _capture_image_size(frame)
        self.after(0, lambda: self._render_result(scene, debug_path, capture.source, elapsed, capture.window, capture_size))

    def _quick_capture_worker(self) -> None:
        try:
            time.sleep(0.03)
            started_at = time.perf_counter()
            capture_started_at = time.perf_counter()
            capture = capture_target(
                DEFAULT_WINDOW_TITLE,
                fallback_to_screen=True,
                fallback_if_target_uncaptureable=False,
                activate=True,
                activation_delay=0.02,
            )
            capture_elapsed = time.perf_counter() - capture_started_at
            frame = capture.image
            self.tracker.set_capture_context(capture)
            write_elapsed = 0.0
            if bool(config_value("debug.save_fast_captures", False)):
                write_started_at = time.perf_counter()
                write_image("runtime/capture_quick.png", frame)
                write_elapsed = time.perf_counter() - write_started_at
            analysis_started_at = time.perf_counter()
            scene = self.tracker.analyze_quick(frame)
            analysis_elapsed = time.perf_counter() - analysis_started_at
            full_refresh_elapsed = 0.0
            if self._needs_stage2_brush_refresh():
                full_started_at = time.perf_counter()
                scene = self.tracker.analyze_realtime(frame)
                full_refresh_elapsed = time.perf_counter() - full_started_at
            elapsed = time.perf_counter() - started_at
            perf = _fast_capture_perf(capture_elapsed, analysis_elapsed, write_elapsed, full_refresh_elapsed)
        except Exception as exc:
            self.after(0, lambda: self._show_error(exc))
            return

        capture_size = _capture_image_size(frame)
        self.after(0, lambda: self._render_quick_result(scene, capture.source, elapsed, capture.window, capture_size, perf))

    def _needs_stage2_brush_refresh(self) -> bool:
        return (
            self.current_state is not None
            and current_rule_is_pending(self.current_state)
            and self.brush_scenario is not None
            and self.selected_brush_type is not None
            and self.selected_affix is None
        )

    def _show_error(self, exc: Exception) -> None:
        self._capture_inflight = False
        self._restore_app_window()
        self.status_var.set("识别失败。")
        messagebox.showerror("Huahua Assistant", str(exc))

    def _chosen_affix(self) -> AffixRule | None:
        """本局当前挂着的词缀：界面选过的优先，其次模型状态里的 ``active_affix``。"""
        if self.selected_affix is not None:
            return self.selected_affix
        if self.current_state is not None:
            return self.current_state.active_affix
        return None

    def _self_upgrade_from_choice(self) -> int:
        """这一笔额外吃到的「使自身等级提升 N 级」。

        只认**已记录的选择**：界面里选过的词缀（``selected_affix``）优先，其次模型状态里
        挂着的 ``active_affix``。模型不自己猜这个数——猜了就等于自己给自己发证据。
        """
        candidates = [self.selected_affix]
        if self.current_state is not None:
            candidates.append(self.current_state.active_affix)
        for affix in candidates:
            if affix is None:
                continue
            if getattr(affix, "effect", None) == AffixEffect.UPGRADE_SELF:
                return max(0, int(getattr(affix, "amount", 0) or 0))
        return 0

    def _stabilize_scene(self, scene: SceneObservation) -> SceneObservation:
        """棋盘等级/种类过一遍时序稳定（见 :mod:`level_smoothing`）。

        用户 2026-11-22：点击地块看加成时浮窗会挡住格子，等级读数在 1↔7、3↔6
        之间抖。抖动帧一旦进模型，会同时污染观测、对账和地块加成反推。
        """

        try:
            board = self.level_stabilizer.observe(scene.board)
        except Exception:  # 稳定器出问题不能连累采集，宁可这帧不平滑
            return scene
        if board is scene.board:
            return scene
        return replace(scene, board=board)

    def _surface_level_mismatch(self, report: ReconcileReport) -> None:
        """模型和画面不一致时，把结论顶到状态栏和提示条。

        以前只写进右侧「识别区域」里的「模型对账」小节，用户看不见——
        2026-11-22 原话：「日志标了对不上，不在 ui 上给我看？」。
        对账是判断「识别错还是规则错」的唯一线索，必须显眼。

        只在**差异集合变化**时提醒一次：同一笔每一帧都重算，逐帧弹会刷屏。
        """

        if not report.ran or not report.mismatches:
            self.last_reconcile_notice_key = ""
            return
        key = "|".join(
            f"{item.row},{item.col},{item.predicted},{item.observed}" for item in report.mismatches
        )
        if key == self.last_reconcile_notice_key:
            return
        self.last_reconcile_notice_key = key
        detail = "；".join(
            f"r{item.row + 1}c{item.col + 1} 模型{item.predicted}≠画面{item.observed}"
            for item in report.mismatches[:3]
        )
        more = f" 等{len(report.mismatches)}处" if len(report.mismatches) > 3 else ""
        self.status_var.set(f"对账不一致：{detail}{more}")
        self._set_notice(
            f"模型推算与画面不一致：{detail}{more}（逐格来源见右侧「模型对账」）",
            "warning",
        )

    def _update_level_reconcile(self, previous_state: PuzzleState | None) -> None:
        """落笔后用模型跑一遍「这一笔该变成几级」，和画面逐格比。

        这就是用户要的「代码推算和识别结果不一致时能不能标出来」：
        差异会落到具体格子（棋盘红框 + ``预N≠实M``），同时写一条观测记录。
        """
        try:
            report = reconcile_levels(
                previous_state,
                self.current_state,
                self_upgrade=self._self_upgrade_from_choice(),
                active_affix=self._chosen_affix(),
            )
        except Exception:
            # 对账只是旁路，绝不能因为它把整帧识别搞挂。
            self.level_reconcile = ReconcileReport(note="对账异常，已跳过")
            return
        self.level_reconcile = report
        self._open_level_animation_window(report)
        self._surface_level_mismatch(report)
        if report.ran and (report.mismatches or report.uncertain):
            record_observation(
                "level_reconcile",
                {
                    "session": self.real_session_id,
                    "step": self.real_session_step_index,
                    "installed": [list(cell) for cell in report.installed],
                    "self_upgrade": self._self_upgrade_from_choice(),
                    "affix": getattr(self._chosen_affix(), "canonical_affix_id", None),
                    # 来源表原样落盘：调色 / 笔触 / 自身各自的**范围与目标类别**。
                    # 复盘时「这格凭什么涨 2 级」直接看这里，不用再猜。
                    "sources": [
                        {
                            "key": source.key,
                            "label": source.label,
                            "target": source.target_text,
                            "scope_cells": len(source.scope),
                            "amount": source.amount,
                            "random_count": source.random_count,
                        }
                        for source in report.sources
                    ],
                    "mismatches": [
                        {
                            "r": item.row + 1,
                            "c": item.col + 1,
                            "predicted": item.predicted,
                            "upper": item.upper,
                            "observed": item.observed,
                            "kind": item.kind,
                            "breakdown": [
                                {"source": label, "amount": amount}
                                for label, amount in item.breakdown
                            ],
                        }
                        for item in report.mismatches
                    ],
                    # 「随机 N 个」这种由服务端摇的目标，模型只能给区间。
                    # 单独记一份，复盘时才知道「画面涨了但没错」的格子为什么没报警。
                    "uncertain": [
                        {"r": row + 1, "c": col + 1} for row, col in report.uncertain
                    ],
                    # 这一笔模型预计升几格（区间）。落盘是为了和随后的
                    # ``level_animation_check`` 里「动画播了几条」直接对。
                    "upgrade_cells": list(report.upgrade_cells),
                    "caveat": report.caveat,
                    "summary": report.summary(),
                },
                source="real_capture",
            )

    # ---- 「模型预计升几格」对「游戏播了几条升级动画」 ---------------------
    #
    # 实测：一条 ``Anim_PutongShengji`` = 一格升 1 级，``CaiShengji`` = 升到 7 级；
    # 一次落笔 1~4 条是**排队**播的（0 / 1.8s / 2.9s），所以不能在落笔当帧就判，
    # 要开一个窗口等它播完。窗口内的条数与 :attr:`ReconcileReport.upgrade_cells`
    # 对不上，就说明模型漏了一条等级来源（或这一格被读错），比逐格看红框更快定位。

    def _open_level_animation_window(self, report: ReconcileReport) -> None:
        if not report.ran:
            return
        low, high = report.upgrade_cells
        if high <= 0:
            return
        self.level_animation_seq += 1
        seq = self.level_animation_seq
        self.level_animation_window = {
            "seq": seq,
            "low": low,
            "high": high,
            "count": 0,
            "installed": [list(cell) for cell in report.installed],
        }
        self.after(int(LEVEL_ANIMATION_WINDOW * 1000), lambda: self._flush_level_animation(seq))

    def _note_level_animation_events(self, events: list[GameLogEvent]) -> None:
        window = self.level_animation_window
        if window is None:
            return
        added = sum(1 for event in events if event.kind in ("upgrade", "rainbow_upgrade"))
        if not added:
            return
        window["count"] += added
        current = self.level_reconcile
        if current.ran:
            self.level_reconcile = replace(current, animation_upgrades=window["count"])

    def _flush_level_animation(self, seq: int) -> None:
        window = self.level_animation_window
        if window is None or window.get("seq") != seq:
            return
        self.level_animation_window = None
        low, high, count = window["low"], window["high"], window["count"]
        matched = low <= count <= high
        current = self.level_reconcile
        if current.ran:
            self.level_reconcile = replace(current, animation_upgrades=count)
        record_observation(
            "level_animation_check",
            {
                "session": self.real_session_id,
                "step": self.real_session_step_index,
                "installed": window["installed"],
                "expected_min": low,
                "expected_max": high,
                "animations": count,
                "match": matched,
            },
            source="real_capture",
        )
        if matched:
            return
        self._set_notice(
            f"升级动画播了 {count} 条，模型预计 {low}~{high} 格 —— 对不上。"
            "要么漏了一条等级来源，要么有一格被读错；看右下「模型对账」的逐格差异。",
            "warning",
        )

    def _render_result(
        self,
        scene: SceneObservation | None,
        debug_path: Path | None,
        capture_source: str,
        elapsed: float,
        capture_window: WindowInfo | None,
        capture_size: tuple[int, int] | None,
    ) -> None:
        self._capture_inflight = False
        self._restore_app_window()
        self._set_last_capture_context(capture_window, capture_size)
        # 完整识别这条路径也要消费「入画」信号。否则用户点「完整刷新」时
        # 仍会走进下面的 previous_state 继承，把上一局的规则/锚点带过来。
        if self.pending_new_run:
            self.pending_new_run = False
            self.start_new_real_session(record_event=True)
        elif scene is not None:
            # 没有「入画」日志时，靠棋盘自己认新画纸（画满过 → 空位回来）。
            self._maybe_start_new_run_from_board(scene, trigger="board_changed_on_full_refresh")
        self.current_scene = scene
        if scene is None:
            self.current_state = None
            self.recommendations = []
            self.install_plan = None
            self._update_fast_snapshot(None)
            self._clear_brush_choice()
            self.last_capture_quality = None
            self._update_perf_panel(mode="完整识别", elapsed=elapsed, capture_source=capture_source)
            self.board_var.set("局面：未识别到棋盘")
            self.status_var.set(f"来源：{capture_source}。未找到棋盘。{elapsed:.2f}s。调试图：{debug_path}")
            self._draw_empty_grid()
            self._write_regions(None)
            return

        board = scene.board
        backpack = self._merge_detected_backpack(scene)
        self.backpack_frame_visible = bool(scene.backpack and scene.backpack.paints)
        self.manual_correction_applied = False
        previous_state = self.current_state
        scene = self._stabilize_scene(scene)
        self.current_state = state_from_observation(
            scene.board, scene.current_paint, backpack, previous=previous_state
        )
        self._update_level_reconcile(previous_state)
        quality = _capture_quality_gate(scene, backpack)
        self.last_capture_quality = quality
        self.last_rule_anomalies = (
            _detect_capture_rule_anomalies(self.last_real_capture_step, scene, self.current_state)
            if quality["ok"]
            else []
        )
        self._record_scene_observation(scene, backpack, quality, self.last_rule_anomalies)
        self.current_state = _state_for_visible_brush_panel(self.current_state, scene.brush_options)
        self._maybe_autosave_full_board(scene)
        self._clear_brush_choice()
        self._apply_detected_brush_options(scene.brush_options)
        recommendation_started_at = time.perf_counter()
        if recording_mode_enabled():
            # 记录模式：推荐一律不算（见 recording_mode_enabled 的注释）。
            self.recommendations = []
            self.install_plan = None
        else:
            self.recommendations = recommend_placements(
                self.current_state,
                include_lookahead=False,
                learning_stats=self._learning_stats_cache,
            )
            self.install_plan = recommend_install_plan(self.current_state)
        recommendation_elapsed = time.perf_counter() - recommendation_started_at
        self._update_fast_snapshot(snapshot_from_scene(scene, self.current_state))
        self._update_perf_panel(
            mode="完整识别",
            elapsed=elapsed,
            capture_source=capture_source,
            recommendation_elapsed=recommendation_elapsed,
        )
        high_value_notice = self._record_high_value_seen(
            scene,
            self.current_state,
            backpack,
            quality,
            mode="完整识别",
            recommendations=self.recommendations,
        )
        auto_saved = self._auto_record_real_trajectory(scene, backpack)
        rect = board.rect
        self.board_var.set(f"棋盘：x={rect.x} y={rect.y} size={rect.w} 置信={board.score:.2f}")
        status = f"来源：{capture_source}。{elapsed:.2f}s。调试图已保存：{debug_path}"
        if not quality["ok"]:
            status += f" 低置信度：{quality['reason']}。"
        if self.last_rule_anomalies:
            status += f" 规则异常 x{len(self.last_rule_anomalies)}。"
        if auto_saved:
            status += " 已自动保存高价值轨迹。"
        settlement_note = self._detect_settlement_candidate(scene, self.current_state) if quality["ok"] else None
        if settlement_note:
            status += f" {settlement_note}"
        self.status_var.set(status)
        if not quality["ok"]:
            self._set_notice(f"本次完整识别低置信度：{quality['reason']}。已记录观察，但不会进入高分轨迹。", "warning")
        elif high_value_notice:
            self._set_notice(high_value_notice, "success")
        elif self.last_rule_anomalies:
            self._set_notice(f"发现规则异常 x{len(self.last_rule_anomalies)}：右侧面板可查看细节。", "warning")
        else:
            self._set_notice(f"完整识别完成：质量 {quality['score']:.2f}，可用于学习记录。", "success")
        self._draw_scene(scene)
        self._write_regions(scene)
        self._refresh_overlay()

    def _render_quick_result(
        self,
        scene: SceneObservation | None,
        capture_source: str,
        elapsed: float,
        capture_window: WindowInfo | None,
        capture_size: tuple[int, int] | None,
        capture_perf: dict | None = None,
    ) -> None:
        self._restore_app_window()
        self._set_last_capture_context(capture_window, capture_size)
        self._render_fast_scene_result(scene, capture_source, elapsed, auto=False, capture_perf=capture_perf)

    def _render_auto_follow_result(
        self,
        scene: SceneObservation | None,
        capture_source: str,
        elapsed: float,
        capture_window: WindowInfo | None,
        capture_size: tuple[int, int] | None,
        generation: int,
        capture_perf: dict | None = None,
        recording_evidence: dict | None = None,
    ) -> None:
        if generation != self.auto_follow_generation or not self.auto_follow_var.get():
            self.auto_follow_busy = False
            return
        try:
            if capture_kind(capture_source) == CAPTURE_KIND_PAUSED:
                # 用户切到别的软件、或者游戏窗口没了。这一帧什么都不做：不清空状态、
                # 不弹「未识别」，保持上一帧的提示，等回来再说。
                return
            self._set_last_capture_context(capture_window, capture_size)
            self._render_fast_scene_result(scene, capture_source, elapsed, auto=True,
                                           capture_perf=capture_perf, recording_evidence=recording_evidence)
        finally:
            self.auto_follow_busy = False
            self._schedule_auto_follow()

    def _render_fast_scene_result(
        self,
        scene: SceneObservation | None,
        capture_source: str,
        elapsed: float,
        *,
        auto: bool,
        capture_perf: dict | None = None,
        recording_evidence: dict | None = None,
    ) -> None:
        # 日志说「入画」了：先无条件把旧局清干净，再谈识别。
        # 放在「没找到棋盘」的提前返回之前是故意的——选画纸、集雪这些界面本来就
        # 看不到棋盘，若等识别成功才清，旧棋盘会一直挂在界面上。
        if self.pending_new_run:
            self.pending_new_run = False
            self.start_new_real_session(record_event=True)

        if scene is None:
            prefix = "自动跟随" if auto else "快速获取"
            perf_text = _format_fast_capture_perf(capture_perf)
            self.last_capture_quality = None
            self._update_fast_snapshot(None)
            self._update_perf_panel(mode=prefix, elapsed=elapsed, capture_source=capture_source, capture_perf=capture_perf)
            if capture_perf and capture_perf.get("geometry_pending"):
                self.status_var.set("窗口正在变化，稳定后自动重新定位棋盘和笔触卡片。")
                return
            if self.real_session_finished or self.board_reached_full:
                # 结算/奖励/选画纸这类界面本来就看不到棋盘。本局已经收尾，
                # 这里就是「待开局」状态——不要报成「未找到棋盘」，那看起来像识别坏了。
                self.status_var.set(
                    f"{prefix}：本局已收尾，等待下一局入画。来源={capture_source}，耗时={elapsed:.2f}s。{perf_text}"
                )
            else:
                self.status_var.set(f"{prefix}：未找到棋盘。来源={capture_source}，耗时={elapsed:.2f}s。{perf_text}")
            return

        # 上一局已经「结束」有两条路：玩家真去点了「结束本局」（real_session_finished），
        # 或者这一局画满过（board_reached_full）。后者才是常态——画满就直接入画下一张，
        # 没人会去点结束本局，所以只看前者的话新画纸永远等不到重置。
        self._maybe_start_new_run_from_board(scene, trigger="board_changed_after_finish")

        previous_scene = self.current_scene
        # 稳定化要在写入 current_scene **之前**做：后面所有判断（落子识别、对账、
        # 观测落盘）都读 current_scene，晚一步就会用抖动值算一遍再丢。
        scene = self._stabilize_scene(scene)
        self.current_scene = scene
        previous_state = self.current_state
        previous_backpack_visible = bool(previous_scene and previous_scene.backpack and previous_scene.backpack.paints)
        backpack = self._merge_detected_backpack(scene)
        backpack_visible = bool(scene.backpack and scene.backpack.paints)
        self.backpack_frame_visible = backpack_visible
        backpack_just_appeared = backpack_visible and not previous_backpack_visible
        state = state_from_observation(
            scene.board, scene.current_paint, backpack, previous=previous_state
        )
        fast_current_changed = _fast_current_paint_changed(scene, previous_state)
        if previous_state is not None:
            state = _merge_fast_rule_state(
                state,
                previous_state,
                scene,
                backpack or previous_state.backpack,
                changed=fast_current_changed,
            )
        state = _state_for_visible_brush_panel(state, scene.brush_options)
        if backpack_just_appeared and previous_state is not None:
            state = self._confirm_pending_anchor_from_backpack_appearance(previous_state, state, scene)
            if current_rule_is_pending(previous_state) and not current_rule_is_pending(state):
                self._clear_brush_choice()
        self.current_state = state
        self._maybe_autosave_full_board(scene)
        self._update_level_reconcile(previous_state)
        self.manual_correction_applied = False
        self.last_capture_quality = None
        self.last_rule_anomalies = []
        self._merge_quick_brush_stage2(scene.brush_options)
        recommendation_started_at = time.perf_counter()
        if recording_mode_enabled():
            # 记录模式：推荐一律不算。这一对每帧要花几十到几百毫秒（含 lookahead
            # 之外的打分与安装计划求解），录数据时没有任何用处。
            self.recommendations = []
            self.install_plan = None
        else:
            self.recommendations = recommend_placements(
                self.current_state,
                include_lookahead=False,
                learning_stats=self._learning_stats_cache,
            )
            self.install_plan = recommend_install_plan(self.current_state)
        recommendation_elapsed = time.perf_counter() - recommendation_started_at
        quick_quality = _quick_capture_quality(scene, elapsed, capture_perf, capture_source=capture_source)
        self.last_capture_quality = quick_quality
        self._update_fast_snapshot(
            snapshot_from_scene(scene, self.current_state),
            evaluate_pet=False,
        )
        # 桌宠决策在记录模式下也跳过：evaluate_pet_state 会调 recommend_brush_types /
        # recommend_affixes，同样是白烧（用户 2026-11-22：记录模式不要任何推荐）。
        self.last_pet_state = None if recording_mode_enabled() else evaluate_pet_state(self)
        self._update_perf_panel(
            mode="自动跟随" if auto else "快速获取",
            elapsed=elapsed,
            capture_source=capture_source,
            capture_perf=capture_perf,
            recommendation_elapsed=recommendation_elapsed,
        )
        quick_saved = self._record_quick_capture_step(
            scene,
            self.current_state,
            elapsed,
            capture_perf=capture_perf,
            capture_source=capture_source,
            recording_evidence=recording_evidence,
        )
        high_value_notice = self._record_high_value_seen(
            scene,
            self.current_state,
            self.current_state.backpack,
            quick_quality,
            mode=("自动跟随" if auto else "快速获取"),
            recommendations=self.recommendations,
        )
        rect = scene.board.rect
        self.board_var.set(f"棋盘：x={rect.x} y={rect.y} size={rect.w} 置信={scene.board.score:.2f}")
        prefix = "自动跟随" if auto else "快速获取"
        status = (
            f"{prefix}：{elapsed:.2f}s。"
            "棋盘快路径，已跳过 OCR。"
        )
        perf_text = _format_fast_capture_perf(capture_perf)
        if perf_text:
            status += f" {perf_text}"
        if fast_current_changed:
            status += " 检测到新初始颜料图标变化，已暂停旧规则推荐，请点完整识别或完成笔触选择后快速获取。"
        if backpack_just_appeared and self.current_state is not None and not current_rule_is_pending(self.current_state):
            status += " 已检测到背包出现，确认落笔完成。"
        if quick_saved:
            status += f" 已记录快速样本 #{self.quick_capture_record_count}。"
            if self.quick_capture_record_count in {50, 100, 200, 500} or self.quick_capture_record_count % 1000 == 0:
                status += " 数据量达到分析节点，请查看识别统计。"
        elif self.pending_tentative_step is not None:
            status += " 检测到初始颜料疑似试放，等待下一次截图确认后再写入轨迹。"
        settlement_note = self._detect_settlement_candidate(scene, self.current_state) if quick_quality["ok"] else None
        if settlement_note:
            status += f" {settlement_note}"
            if auto and not self.real_session_finished:
                saved, score = self._finish_real_session_internal(reason="auto_settlement")
                if saved:
                    status += f" 自动结束本局，内部评分={score:.0f}，已保存。"
                else:
                    suffix = self.last_finish_skip_reason or "重复/未保存"
                    status += f" 自动结束本局，内部评分={score:.0f}，{suffix}。"
        self.status_var.set(status)
        if not quick_quality["ok"]:
            self._set_notice(f"{prefix}低置信度：棋盘={quick_quality['score']:.2f}，本次不写入真实轨迹。", "warning")
        elif high_value_notice:
            self._set_notice(high_value_notice, "success")
        elif settlement_note:
            self._set_notice(f"检测到可能结算：{settlement_note}", "warning")
        elif quick_saved:
            self._set_notice(f"{prefix}已记录真实样本 #{self.quick_capture_record_count}。", "success")
        elif self.pending_tentative_step is not None:
            self._set_notice("检测到初始颜料疑似试放：等待下一次截图确认，暂不写入轨迹。", "warning")
        self._draw_scene(scene)
        self._write_regions(scene)
        self._refresh_overlay()

    def _detect_settlement_candidate(self, scene: SceneObservation, state: PuzzleState | None) -> str | None:
        if state is None:
            return None
        finish_region = scene.region(RegionKind.FINISH_DRAWING)
        # 只认「完成绘制」按钮 / 结束横幅。右下角那个蓝色「入画」按钮在**作画界面常驻**
        # （实测每帧都检出，置信 0.92，位置 x=1263 y=779），它的意思是「可以进下一张画纸」，
        # 不代表本局已经画完。以前把它当成同类证据 → 新一局的第一帧就被判「可结算」，
        # 随即 run_session_finished，之后整局的落子都不进轨迹，状态灯还一直报「本局已收尾」。
        is_finish_button = (
            finish_region is not None
            and finish_region.confidence >= 0.58
            and not finish_region.is_enter_painting
        )
        if is_finish_button:
            key = _stable_payload_key(
                {
                    "session": self.real_session_id,
                    "finish_button": [
                        finish_region.rect.x,
                        finish_region.rect.y,
                        finish_region.rect.w,
                        finish_region.rect.h,
                    ],
                    "terminal": _best_available_terminal_summary(state, self.real_trajectory_steps),
                }
            )
            if key == self.last_settlement_key:
                return "已识别到“完成绘制”按钮，等待点击“结束本局”保存内部评分。"
            self.last_settlement_key = key
            record_observation(
                "settlement_candidate",
                {
                    "session": self.real_session_id,
                    "board": _compact_board_summary(scene),
                    "terminal": _best_available_terminal_summary(state, self.real_trajectory_steps),
                    "steps": len(self.real_trajectory_steps),
                    "reason": "finish_drawing_button",
                    "button_confidence": round(float(finish_region.confidence), 3),
                },
                source="real_capture",
            )
            return "识别到“完成绘制”按钮，已进入可结算状态；可点“结束本局”保存内部评分。"
        counts = Counter(tile.state for tile in state.tiles)
        blank_cells = counts.get(TileState.BLANK, 0)
        open_cells = blank_cells + counts.get(TileState.VOID, 0)
        if blank_cells > 0:
            return None
        key = _stable_payload_key({"session": self.real_session_id, "board": _compact_board_summary(scene)})
        reason = "no_blank_cells" if open_cells > 0 else "no_blank_or_void_cells"
        note = "无空白地块" if open_cells > 0 else "棋盘已满"
        if key == self.last_settlement_key:
            return f"{note}，等待点击“结束本局”保存内部评分。"
        self.last_settlement_key = key
        record_observation(
            "settlement_candidate",
            {
                "session": self.real_session_id,
                "board": _compact_board_summary(scene),
                "terminal": _trajectory_terminal_summary(state, state.backpack),
                "steps": len(self.real_trajectory_steps),
                "reason": reason,
            },
            source="real_capture",
        )
        return f"识别到{note}，可能已可完成绘制；请点“结束本局”保存内部评分。"

    def _merge_quick_brush_stage2(self, observation: BrushOptionsObservation | None) -> None:
        if observation is not None:
            self._brush_panel_absent_frames = 0
            if observation.stage2_options:
                self._brush_stage1_vote_observations.clear()
                self._locked_brush_stage1 = None
                self._apply_detected_brush_options(observation)
                return
            if (
                self.brush_scenario is not None
                and self.selected_brush_type is None
                and not self.brush_scenario.stage2_options
            ):
                return
            signature = tuple(_brush_types_from_values(observation.stage1_options))
            if not signature:
                return
            self._brush_stage1_vote_observations.append(observation)
            self._brush_stage1_vote_observations = self._brush_stage1_vote_observations[-5:]
            signatures = [
                tuple(_brush_types_from_values(item.stage1_options))
                for item in self._brush_stage1_vote_observations
            ]
            winner, votes = Counter(signatures).most_common(1)[0]
            if votes < 2:
                return
            stable_observation = next(
                item
                for item in reversed(self._brush_stage1_vote_observations)
                if tuple(_brush_types_from_values(item.stage1_options)) == winner
            )
            self._brush_stage1_vote_observations.clear()
            self._locked_brush_stage1 = None
            self._locked_brush_stage2 = None
            self._apply_detected_brush_options(stable_observation)
            return

        self._brush_panel_absent_frames += 1
        scenario = self.brush_scenario
        stage2_active = bool(
            scenario is not None
            and self.selected_brush_type is not None
            and scenario.stage2_options
        )
        absence_limit = (
            BRUSH_STAGE2_ABSENCE_GRACE_FRAMES
            if stage2_active
            else BRUSH_STAGE1_ABSENCE_GRACE_FRAMES
        )
        if self._brush_panel_absent_frames < absence_limit:
            return

        state = self.current_state
        if (
            not recording_mode_enabled()
            and state is not None
            and current_rule_is_pending(state)
            and scenario is not None
            and self.selected_brush_type is not None
            and scenario.stage2_options
            and self.selected_affix is None
        ):
            # 注意：这是**猜**——拿「推荐排序第一」当玩家选了哪条词缀。记录模式里
            # 一律不猜（用户 2026-11-22：记录模式不要任何推荐）。坐实的选择靠
            # run_recorder 的 affix_card_clicked / 窄化判定。
            recommendations = recommend_affixes(state, scenario.stage2_options)
            if recommendations:
                inferred_affix = recommendations[0].affix
                result = apply_affix_to_state(state, inferred_affix)
                if result.success:
                    self.current_state = result.state
                    self.selected_affix = inferred_affix

        self.brush_scenario = None
        self.selected_brush_type = None
        self._last_brush_observation = None
        self._brush_stage1_vote_observations.clear()
        self._locked_brush_stage1 = None
        self._locked_brush_stage2 = None
        self.brush_stage1_rects = {}
        self.brush_affix_rects = {}

    def _maybe_start_new_run_from_board(self, scene: SceneObservation, *, trigger: str) -> bool:
        """上一局已经满铺/结束，而这一帧的棋盘明显是新画纸 → 开新局。

        为什么要有这个函数：``_looks_like_new_run`` 以前只挂在
        ``real_session_finished`` 上，而那个标志要玩家真去点「结束本局」才会置位。
        实际玩法是**画满就点「入画」开下一张**，所以新画纸永远等不到重置，
        界面一直挂着上一局的规则/锚点，玩家看到的就是「换局了还显示旧棋盘」。

        ``board_reached_full`` 记的是「本局画满过」这一格事实（blank=0 且有一整盘颜料）。
        满铺是不可逆的，所以它配上「空位又回来」就是换画纸，不需要等结算。

        首帧必须要有上一帧参照才判：``_looks_like_new_run(..., None)`` 一律返回 True，
        没有参照就去问的话，满铺画面上会每帧重置一次，整局什么都记不下来。
        """
        if scene is None:
            return False
        board = _compact_board_summary(scene)
        finished_before = self.real_session_finished or (
            self.board_reached_full and self.last_real_capture_step is not None
        )
        # 先拦一道「画面还是上一局那块」：收完尾之后 ``last_real_capture_step``
        # 可能已经被清空，``_looks_like_new_run(..., None)`` 会无条件返回 True，
        # 于是同一块满铺棋盘被当成新画纸，凭空开出一局 0 步的假局。
        # 用上一局收尾时记下的状态串比一下，同款就直接不认。
        if finished_before and self.last_finished_board_states:
            if str(board.get("states") or "") == self.last_finished_board_states:
                return False
        if finished_before and _looks_like_new_run(scene, self.last_real_capture_step):
            self.start_new_real_session(record_event=True)
            record_observation(
                "new_run_detected",
                {"trigger": trigger, "board": board},
                source="real_capture",
            )
            return True
        if int(board.get("blank", 0)) == 0 and int(board.get("paint", 0)) >= 12:
            # 满铺：可放置格全填满，剩下的只有阻挡/虚空。整局不会再变回去。
            # 是否自动存档交给 :meth:`_maybe_autosave_full_board`——它必须在
            # ``self.current_state`` 换成这一帧之后才跑，否则评分用的是满铺前
            # 少一格、少一枚颜料的旧局面。
            self.board_reached_full = True
            # 记下这块满铺棋盘：下一帧如果还是它，就不能当新画纸。
            self.last_finished_board_states = str(board.get("states") or "")
        return False

    def _maybe_autosave_full_board(self, scene: SceneObservation) -> None:
        """本局第一次画满时自动存一局（只有高分才落盘）。

        为什么必须自动：实机玩法是**画满就点「入画」开下一张**，没人会去点
        「结束本局」，所以 ``_finish_real_session_internal`` 以前从来没被调用过，
        整局的轨迹全丢。门槛沿用 ``_high_score_gate``（没有 7 级颜料就不写盘），
        所以「满铺就存档」不会把普通局塞进高分库。

        只尝试一次：门槛不过时 ``_finish_real_session_internal`` 不会置
        ``real_session_finished``，若不记 ``board_full_autosave_attempted``，
        满铺之后的每一帧都会重跑一遍评分，记录里刷出一串重复的 skipped。
        """
        if not self.board_reached_full or self.board_full_autosave_attempted:
            return
        self.board_full_autosave_attempted = True
        if self.current_state is None or scene is None:
            return
        board = _compact_board_summary(scene)
        try:
            saved, score = self._finish_real_session_internal(reason="board_full_autosave")
        except Exception as exc:  # noqa: BLE001 - 存档不能把自动跟随带崩
            record_observation(
                "board_full_autosave",
                {"saved": False, "error": f"{type(exc).__name__}: {exc}", "board": board},
                source="real_capture",
            )
            return
        record_observation(
            "board_full_autosave",
            {
                "session": self.real_session_id,
                "saved": saved,
                "internal_score": score,
                "skip_reason": self.last_finish_skip_reason,
                "steps": len(self.real_trajectory_steps),
                "board": board,
            },
            source="real_capture",
        )
        if saved:
            self._set_notice(f"画满已自动存档：内部评分 {score:.0f}。", "success")
        elif self.last_finish_skip_reason:
            self._set_notice(f"画满但未存档：{self.last_finish_skip_reason}", "info")

    def start_new_real_session(self, record_event: bool = True) -> None:
        self.real_session_id = _new_session_id()
        self.real_session_started_at = time.time()
        self.real_session_step_index = 0
        self.real_trajectory_steps = []
        self.last_real_step_key = None
        self.last_quick_step_key = None
        self.last_auto_trajectory_key = None
        self.last_real_capture_step = None
        self.real_initial_snapshot = None
        self.last_compressed_summary_key = None
        self.last_high_value_key = None
        self.last_finish_skip_reason = None
        self.last_finish_diagnostic = None
        self.pending_tentative_step = None
        self.pending_tentative_signature = None
        self.quick_capture_record_count = 0
        self.last_settlement_key = None
        self.real_session_finished = False
        self.board_reached_full = False
        self.board_full_autosave_attempted = False
        self.pending_new_run = False
        # 所有新局入口（日志、手动、棋盘判定）都要清空格子身份并解除结算冻结。
        # 只在棋盘判定分支重置，会让日志开局后的怪物被旧局同位置的玩法覆盖。
        self.level_stabilizer.reset()
        self.current_scene = None
        self.current_state = None
        self.previous_fast_snapshot = None
        self.last_fast_snapshot = None
        self.last_fast_events = ()
        self.last_pet_state = None
        self.last_capture_quality = None
        self.last_perf = {}
        self._brush_panel_absent_frames = 0
        self._clear_brush_choice()
        self._update_perf_panel(mode="新局", elapsed=0.0)
        self.recommendations = []
        self.install_plan = None
        self.selected_backpack_index = None
        self.backpack_accumulator.reset()
        self.backpack_session_key = None
        self.backpack_frame_visible = None
        self.last_backpack_paint_tiles = None
        self._clear_brush_choice()
        self.tracker.reset()
        if record_event:
            record_observation(
                "run_session_started",
                {"session": self.real_session_id, "started_at": self.real_session_started_at},
                source="real_capture",
            )
        self.status_var.set(f"已开始新的一局：{self.real_session_id}")

    def finish_real_session(self) -> None:
        if not self.real_trajectory_steps:
            reason = "本局还没有真实步骤记录，请先完整识别或重新快速获取。"
            self._set_finish_diagnostic(saved=False, score=0.0, terminal={}, reliable_steps=[], finish_seen=False, reason_text=reason)
            self.status_var.set(reason)
            self._set_notice(_format_finish_diagnostic(self.last_finish_diagnostic), "warning")
            return
        if self.current_state is None:
            reason = "当前没有可评分的局面，请先识别一次终局画面。"
            reliable_steps = _trusted_trajectory_steps(self.real_trajectory_steps)
            self._set_finish_diagnostic(saved=False, score=0.0, terminal={}, reliable_steps=reliable_steps, finish_seen=False, reason_text=reason)
            self.status_var.set(reason)
            self._set_notice(_format_finish_diagnostic(self.last_finish_diagnostic), "warning")
            return
        if self.real_session_finished:
            reliable_steps = _trusted_trajectory_steps(self.real_trajectory_steps)
            finish_seen = self.current_scene is not None and _scene_has_finish_evidence(self.current_scene, self.current_state)
            terminal = _best_trusted_terminal_summary(
                self.current_state,
                self.real_trajectory_steps,
                current_quality=self.last_capture_quality,
            )
            score = _internal_run_score(terminal)
            reason = "本局已经自动/手动结束处理过，不重复保存；要记录下一局请点“新开一局”。"
            self.last_finish_skip_reason = reason
            self._set_finish_diagnostic(
                saved=False,
                score=score,
                terminal=terminal,
                reliable_steps=reliable_steps,
                finish_seen=finish_seen,
                reason_text=reason,
            )
            self.status_var.set(f"本局已结束 {self.real_session_id}：不重复保存。")
            self._set_notice(_format_finish_diagnostic(self.last_finish_diagnostic), "warning")
            return
        saved, score = self._finish_real_session_internal(reason="manual_finish")
        suffix = "已保存" if saved else (self.last_finish_skip_reason or "重复/未保存")
        self.status_var.set(
            f"本局结束 {self.real_session_id}：内部评分={score:.0f}，步骤={len(self.real_trajectory_steps)}，{suffix}。"
        )
        self._set_notice(
            _format_finish_diagnostic(self.last_finish_diagnostic),
            "success" if saved else "warning",
        )

    def _set_finish_diagnostic(
        self,
        *,
        saved: bool,
        score: float,
        terminal: dict,
        reliable_steps: list[dict],
        finish_seen: bool,
        reason_text: str,
        quality: dict | None = None,
    ) -> None:
        self.last_finish_diagnostic = {
            "saved": saved,
            "score": score,
            "reason": reason_text,
            "steps": len(self.real_trajectory_steps),
            "reliable_steps": len(reliable_steps),
            "finish_seen": finish_seen,
            "last_quality": quality if quality is not None else self.last_capture_quality,
            "terminal": terminal,
        }

    def _finish_real_session_internal(self, *, reason: str) -> tuple[bool, float]:
        self.last_finish_skip_reason = None
        if self.current_state is None:
            self.last_finish_skip_reason = "当前没有可评分局面"
            self._set_finish_diagnostic(
                saved=False,
                score=0.0,
                terminal={},
                reliable_steps=[],
                finish_seen=False,
                reason_text=self.last_finish_skip_reason,
            )
            return False, 0.0
        raw_finish_seen = self.current_scene is not None and _scene_has_finish_evidence(self.current_scene, self.current_state)
        current_trusted = _quality_is_trusted_for_trajectory(self.last_capture_quality)
        finish_seen = raw_finish_seen and current_trusted
        reliable_steps = _trusted_trajectory_steps(self.real_trajectory_steps)
        terminal = _best_trusted_terminal_summary(
            self.current_state,
            reliable_steps,
            current_quality=self.last_capture_quality,
        )
        score = _internal_run_score(terminal)
        score_breakdown = _internal_score_breakdown(terminal)
        terminal_has_high_value = _summary_has_high_value(terminal.get("known") or {})
        if self.last_capture_quality is not None and not current_trusted and not finish_seen and not terminal_has_high_value:
            self.last_finish_skip_reason = _finish_reliability_reason(
                terminal,
                reliable_steps,
                finish_seen=finish_seen,
                last_quality=self.last_capture_quality,
            )
            self._set_finish_diagnostic(
                saved=False,
                score=score,
                terminal=terminal,
                reliable_steps=reliable_steps,
                finish_seen=finish_seen,
                reason_text=self.last_finish_skip_reason,
                quality=self.last_capture_quality,
            )
            record_observation(
                "run_session_finish_skipped",
                {
                    "session": self.real_session_id,
                    "reason": reason,
                    "skip_reason": self.last_finish_skip_reason,
                    "quality": self.last_capture_quality,
                    "internal_score": score,
                    "score_breakdown": score_breakdown,
                    "steps": len(self.real_trajectory_steps),
                    "reliable_steps": len(reliable_steps),
                    "raw_finish_seen": raw_finish_seen,
                    "terminal": terminal,
                },
                source="real_capture",
            )
            return False, score
        reliable, reason_text = _terminal_reliability_for_finish(terminal, reliable_steps, finish_seen=finish_seen)
        if not reliable:
            self.last_finish_skip_reason = reason_text
            self._set_finish_diagnostic(
                saved=False,
                score=score,
                terminal=terminal,
                reliable_steps=reliable_steps,
                finish_seen=finish_seen,
                reason_text=self.last_finish_skip_reason,
            )
            record_observation(
                "run_session_finish_skipped",
                {
                    "session": self.real_session_id,
                    "reason": reason,
                    "skip_reason": self.last_finish_skip_reason,
                    "internal_score": score,
                    "score_breakdown": score_breakdown,
                    "steps": len(self.real_trajectory_steps),
                    "reliable_steps": len(reliable_steps),
                    "raw_finish_seen": raw_finish_seen,
                    "terminal": terminal,
                },
                source="real_capture",
            )
            return False, score
        high_score_gate = _high_score_gate(terminal)
        if not high_score_gate["passed"]:
            self.last_finish_skip_reason = high_score_gate["summary"]
            compressed_saved = self._record_compressed_run_summary(
                reason=reason,
                skip_reason=self.last_finish_skip_reason,
                terminal=terminal,
                high_score_gate=high_score_gate,
                score=score,
                score_breakdown=score_breakdown,
                reliable_steps=reliable_steps,
                raw_finish_seen=raw_finish_seen,
                finish_seen=finish_seen,
            )
            self._set_finish_diagnostic(
                saved=False,
                score=score,
                terminal=terminal,
                reliable_steps=reliable_steps,
                finish_seen=finish_seen,
                reason_text=self.last_finish_skip_reason,
            )
            record_observation(
                "run_session_finish_skipped",
                {
                    "session": self.real_session_id,
                    "reason": reason,
                    "skip_reason": self.last_finish_skip_reason,
                    "internal_score": score,
                    "score_breakdown": score_breakdown,
                    "steps": len(self.real_trajectory_steps),
                    "reliable_steps": len(reliable_steps),
                    "raw_finish_seen": raw_finish_seen,
                    "terminal": terminal,
                    "high_score_gate": high_score_gate,
                    "compressed_summary_saved": compressed_saved,
                },
                source="real_capture",
            )
            return False, score
        initial_snapshot = self._initial_snapshot_from_steps(reliable_steps)
        payload = {
            "version": 2,
            "kind": "internal_scored_run",
            "session": self.real_session_id,
            "reason": reason,
            "started_at": self.real_session_started_at,
            "finished_at": _trajectory_finished_at(self.real_session_started_at, reliable_steps),
            "internal_score": score,
            "score_model": "huahua-v2-treasure-equivalence",
            "score_breakdown": score_breakdown,
            "terminal": terminal,
            "high_score_gate": high_score_gate,
            "initial_snapshot": initial_snapshot,
            "finish_seen": finish_seen,
            "steps": reliable_steps[-240:],
        }
        payload = _with_initial_aliases(payload)
        payload = _with_state_action_result_metadata(payload)
        saved = record_high_score_trajectory(payload, source="real_capture")
        self._set_finish_diagnostic(
            saved=saved,
            score=score,
            terminal=terminal,
            reliable_steps=reliable_steps,
            finish_seen=finish_seen,
            reason_text=("结算轨迹已保存" if saved else "终局可信，但轨迹重复或已保存过，未重复写入"),
        )
        record_observation(
            "run_session_finished",
            _with_initial_aliases({
                "session": self.real_session_id,
                "reason": reason,
                "internal_score": score,
                "score_model": "huahua-v2-treasure-equivalence",
                "score_breakdown": score_breakdown,
                "steps": len(self.real_trajectory_steps),
                "saved": saved,
                "terminal": terminal,
                "high_score_gate": high_score_gate,
                "initial_snapshot": initial_snapshot,
                "raw_finish_seen": raw_finish_seen,
                "finish_seen": finish_seen,
                # 收尾各步耗时的快照：下次再「卡十几秒」时，一眼就能看出
                # 是读盘、规范化、写轨迹还是回写统计吃掉了时间。
                "persist_timing": dict(LAST_PERSIST_TIMING),
            }),
            source="real_capture",
        )
        # 记下收尾时的棋盘：下一帧如果还是这块，就不能当新画纸
        # （见 :meth:`_maybe_start_new_run_from_board` 里的同款拦截）。
        if self.current_scene is not None:
            self.last_finished_board_states = str(
                _compact_board_summary(self.current_scene).get("states") or ""
            )
        # 统计和轨迹都变了：让后台算完后把界面上的数字和推荐排序换掉。
        self._schedule_learning_stats_refresh()
        self.real_session_finished = True
        # 本局已经结算：颜料等级和种类从此不该再变（只有加成可能还在变）。
        # 用户 2026-11-22 要求「本局完成后颜料等级和种类不再变化」——结算后还让
        # 抖动读数改写终局盘面，会把已经入库的成绩和轨迹一起改坏。
        self.level_stabilizer.freeze()
        # 存档完成后立刻取下一张确认帧，不再等空闲采集间隔。
        self._wake_auto_follow_pump()
        return saved, score

    def _intern_board(self, step: dict) -> str:
        """把这一步的局面登记进字典，返回 ``B12`` 这样的编号。

        签名由「地块状态 + 每格颜料与加成 + 当前调色目标」决定，三者任一变化
        就是另一个局面。实测 926 步只有 283 种，所以后面 69% 的步都只花一个
        编号就能指回同一个棋盘。
        """

        board = step.get("board") or {}
        current = step.get("current") or {}
        signature = board_signature(
            board.get("states") or "",
            step.get("tiles") or [],
            current.get("code"),
        )
        code = self.codebook.intern_board(
            signature,
            {
                "states": board.get("states"),
                "tiles": step.get("tiles"),
                "current": current.get("code"),
                "score": board.get("score"),
            },
        )
        if self.codebook.dirty and time.time() - self.codebook_saved_at >= 30.0:
            # 每 30 秒落一次盘。不能每步都写——那会把 IO 拖进采集热路径。
            try:
                self.codebook.save(CODEBOOK_PATH)
            except OSError:
                pass
            self.codebook_saved_at = time.time()
        return code

    #: 背包快照内容没变时，最多每 60 秒刷一次时间戳（保活，不重复写盘）。
    BACKPACK_SNAPSHOT_REFRESH = 60.0

    def _persist_backpack_snapshot(self, state: PuzzleState) -> None:
        """把背包的两页读数写进 ``runtime/backpack_snapshot.json``。

        用户要求（2026-11-18）：背包按初始颜料的目标排序，**要记两页**——
        怪物初始颜料时记 3 排 15 格怪物排序的，玩法初始颜料出现后再记 3 排 15 格
        玩法排序的；两页合计最多 30 格，**允许重复**。每格带家族/类别/等级/宝箱/
        格子坐标，这局打完后复现时据此还原背包。只在读数变化时写盘。
        """
        accumulator = getattr(self, "backpack_accumulator", None)
        if accumulator is None:
            return
        pages = accumulator.pages()
        if not pages:
            return
        rule = state.current_rule
        target_object = getattr(rule, "target_object", None) if rule is not None else None
        now = time.time()
        payload = snapshot_payload(
            pages,
            session=self.real_session_id,
            target=getattr(target_object, "value", None),
            now=now,
        )
        key = _stable_payload_key(payload["pages"])
        if key == getattr(self, "_backpack_snapshot_key", None):
            if now - getattr(self, "_backpack_snapshot_at", 0.0) < self.BACKPACK_SNAPSHOT_REFRESH:
                return
        try:
            BACKPACK_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = BACKPACK_SNAPSHOT_PATH.with_name(BACKPACK_SNAPSHOT_PATH.name + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(BACKPACK_SNAPSHOT_PATH)
        except OSError:
            return
        self._backpack_snapshot_key = key
        self._backpack_snapshot_at = now

    def _backpack_pages_payload(self) -> list[dict]:
        """两页背包的紧凑载荷，写进观测/轨迹记录（复现时看得见当时有哪些颜料）。"""
        accumulator = getattr(self, "backpack_accumulator", None)
        if accumulator is None:
            return []
        pages: list[dict] = []
        for page_key, entries in accumulator.pages():
            pages.append(
                {
                    "target": page_key,
                    "slots": [
                        {
                            "i": index,
                            "category": entry.paint.category.value,
                            "level": int(entry.paint.level),
                            "subtype": entry.paint.subtype,
                            "treasure": bool(entry.paint.is_treasure),
                        }
                        for index, entry in enumerate(entries)
                    ],
                }
            )
        return pages

    def _record_quick_capture_step(
        self,
        scene: SceneObservation,
        state: PuzzleState,
        elapsed: float,
        *,
        capture_perf: dict | None = None,
        capture_source: str | None = None,
        recording_evidence: dict | None = None,
    ) -> bool:
        quality = _quick_capture_quality(scene, elapsed, capture_perf, capture_source=capture_source)
        step = _compact_real_capture_step(
            scene=scene,
            state=state,
            backpack=state.backpack,
            backpack_details=self._backpack_cache_details(),
            backpack_pages=self._backpack_pages_payload(),
            recommendations=self.recommendations,
            brush_scenario=self.brush_scenario,
            resources=self._resource_state(),
            quality=quality,
        )
        step["type"] = "quick_capture"
        step["recording_evidence"] = recording_evidence
        step["mode"] = "quick_capture"
        step["ocr"] = False
        step["board_id"] = self._intern_board(step)
        step["inferred_action"] = _infer_real_capture_action(self.last_real_capture_step, step)
        # 背包读数落盘（复现这局时要知道当时有哪些颜料）。与轨迹可信度无关：
        # 低置信度的整帧不进场，但背包那 15 格仍要留档。
        self._persist_backpack_snapshot(state)
        if not _quality_is_trusted_for_trajectory(quality):
            record_observation(
                "quick_capture_ignored",
                {
                    "session": self.real_session_id,
                    "quality": quality,
                    "board": step.get("board"),
                    "backpack": step.get("backpack"),
                    "current": step.get("current"),
                    "inferred_action": step.get("inferred_action"),
                    "note": "低置信度快速截图只保留观察，不进入真实轨迹/结算",
                },
                source="real_capture",
            )
            return False
        tentative_saved = self._handle_tentative_step(step, observation_kind="quick_capture_step")
        if tentative_saved is not None:
            if tentative_saved:
                self.quick_capture_record_count += 1
            return tentative_saved
        if (_is_noop_capture_update(step["inferred_action"])
                and not _recording_metadata_changed(self.last_real_capture_step, step)):
            return False

        step_key = _stable_payload_key(
            {
                "board": step.get("board"),
                "current": step.get("current"),
                "anchor": step.get("anchor"),
                "pending": step.get("pending"),
                "backpack": step.get("backpack"),
                "brush": step.get("brush"),
                "resources": step.get("resources"),
                "tiles": step.get("tiles"),
                "action": step.get("inferred_action"),
                "recording_evidence": step.get("recording_evidence"),
                "observed_tiles": step.get("observed_tiles"),
            }
        )
        if step_key == self.last_quick_step_key:
            return False
        step = self._stamp_real_step(step)
        self.last_quick_step_key = step_key
        self.last_real_capture_step = step
        self._ensure_initial_snapshot(step)
        self.real_trajectory_steps.append(step)
        self.real_trajectory_steps = self.real_trajectory_steps[-80:]
        self.quick_capture_record_count += 1
        # 「一直截却没有新步骤」的计时从这一刻重新开始（见 _report_capture_gap）。
        self.last_step_recorded_at = time.monotonic()
        self.capture_gap_frames = 0
        record_observation(
            "quick_capture_step",
            step,
            source="real_capture",
        )
        return True

    def _stamp_real_step(self, step: dict) -> dict:
        stamped = dict(step)
        stamped["session"] = self.real_session_id
        stamped["step_index"] = self.real_session_step_index
        stamped["t"] = round(time.time() - self.real_session_started_at, 3)
        # 帧标识：这一步到底对应哪一帧。
        # 现在不存图，但必须留下能定位的线索——将来要「步骤 ↔ 截图」对上时，
        # 靠 id 回查同一帧；靠 at 去录屏里按时间找；靠 source 知道是从哪条
        # 采集链路拿的（窗口内容 / 屏幕 / 暂停）。
        stamped["frame"] = {
            "id": f"{self.real_session_id}#{self.real_session_step_index}",
            "at": round(time.time(), 3),
            "source": (stamped.get("quality") or {}).get("capture_source") or "",
        }
        self.real_session_step_index += 1
        return stamped

    def _handle_tentative_step(self, step: dict, *, observation_kind: str) -> bool | None:
        if not _quality_is_trusted_for_trajectory(step.get("quality")):
            record_observation(
                "trajectory_step_ignored",
                {
                    "session": self.real_session_id,
                    "kind": "untrusted_tentative_or_confirm_step",
                    "quality": step.get("quality"),
                    "action": step.get("inferred_action"),
                    "note": "Untrusted capture kept as observation only; not added to trajectory.",
                },
                source="real_capture",
            )
            return False
        action = step.get("inferred_action") or {}
        if _is_tentative_place_current_action(action, step):
            signature = _tentative_place_signature(step)
            if _backpack_visible_in_step(step):
                pending_trusted = _quality_is_trusted_for_trajectory(
                    (self.pending_tentative_step or {}).get("quality")
                )
                confirmed_from_pending = (
                    signature
                    and signature == self.pending_tentative_signature
                    and self.pending_tentative_step is not None
                    and pending_trusted
                )
                confirmed = dict(self.pending_tentative_step) if confirmed_from_pending else dict(step)
                confirmed["tentative"] = False
                confirmed["confirmed_by_next_capture"] = confirmed_from_pending
                confirmed["confirmed_by_backpack"] = True
                confirmed["inferred_action"] = dict(confirmed.get("inferred_action") or {})
                confirmed["inferred_action"]["tentative"] = False
                confirmed["inferred_action"]["confirmed"] = True
                confirmed["inferred_action"]["confirmed_by_backpack"] = True
                stamped = self._stamp_real_step(confirmed)
                self.last_real_capture_step = stamped
                self._ensure_initial_snapshot(stamped)
                self.real_trajectory_steps.append(stamped)
                self.real_trajectory_steps = self.real_trajectory_steps[-80:]
                self.pending_tentative_step = None
                self.pending_tentative_signature = None
                record_observation(observation_kind, stamped, source="real_capture")
                return True

            tentative = dict(step)
            tentative["tentative"] = True
            tentative["session"] = self.real_session_id
            tentative["inferred_action"] = dict(action)
            tentative["inferred_action"]["tentative"] = True
            self.pending_tentative_step = tentative
            self.pending_tentative_signature = signature
            record_observation(
                "tentative_place_current_paint",
                {
                    "session": self.real_session_id,
                    "signature": signature,
                    "action": tentative.get("inferred_action"),
                    "note": "初始颜料疑似试放，等待下一次截图确认后再写入轨迹",
                },
                source="real_capture",
            )
            return False

        if self.pending_tentative_step is not None:
            record_observation(
                "tentative_place_cancelled",
                {
                    "session": self.real_session_id,
                    "signature": self.pending_tentative_signature,
                    "next_action": action,
                    "note": "初始颜料试放未保持，已丢弃，不写入轨迹",
                },
                source="real_capture",
            )
            self.pending_tentative_step = None
            self.pending_tentative_signature = None
        return None

    def _confirm_pending_anchor_from_backpack_appearance(
        self,
        previous_state: PuzzleState,
        current_state: PuzzleState,
        scene: SceneObservation,
    ) -> PuzzleState:
        if not current_rule_is_pending(previous_state) or previous_state.current_rule is None:
            return current_state
        anchor = _find_new_anchor_from_scene(scene, previous_state)
        if anchor is None:
            return current_state
        result = place_current_rule_anchor(previous_state, anchor[0], anchor[1])
        if not result.success:
            return current_state
        return replace(
            result.state,
            backpack=current_state.backpack,
            current_buff_id=result.state.current_buff_id,
        )

    def report_callback_exception(self, exc_type, exc_value, exc_tb) -> None:
        """Tk 回调里的异常默认只打到 stderr —— ``pythonw`` 下等于直接消失。

        实机 2026-11-18 的教训：双击启动后「任务栏什么都没有」，事后一句痕迹都没留。
        任何 ``after`` 回调里炸掉，至少要在 ``runtime/crash.log`` 里看得出是哪一行。
        """

        try:
            import traceback

            path = Path("runtime/crash.log")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} === [tk callback]\n")
                traceback.print_exception(exc_type, exc_value, exc_tb, file=handle)
                handle.write("\n")
        except OSError:
            pass

    def _restore_app_window(self) -> None:
        self._show_in_front()

    #: 窗口状态的客观快照落盘路径。
    WINDOW_STATE_PATH = Path("runtime/window_state.json")

    def _dump_window_state(self, tag: str) -> None:
        """把每个工具窗口的**系统层**状态写到 ``runtime/window_state.json``。

        实机 2026-11-18：用户报「黑屏、桌宠也不出现，类似卡死」，但
        `runtime/perf.json` 显示帧管线一直在跑（0.46s/帧、WGC 正常）。所以
        必须能把下面五种情况分开，否则只能靠猜：

        ① 窗口根本没建（进程里没有这个顶层窗口）
        ② 建了但被 DWM cloak（``cloaked=1``，屏幕上就是一块黑的）
        ③ 建了但在屏幕外 / 被最小化
        ④ 可见但内容没画（Tk 状态正常，得看截图）
        ⑤ 可见且正常，只是对截屏免疫（``wda=17``）——外部截图和远程串流
           看到的都是黑的，用户从串流里看就像是软件黑屏了
        """

        try:
            import ctypes
            import ctypes.wintypes as wintypes
        except Exception:  # 非 Windows 或裁剪过的运行时不值得崩
            return
        user32 = ctypes.windll.user32
        dwmapi = ctypes.windll.dwmapi

        def describe(hwnd: int, label: str) -> dict:
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            ex_style = user32.GetWindowLongW(hwnd, -20)
            cloaked = ctypes.c_int(0)
            hr = dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), 4)
            affinity = ctypes.c_uint(0)
            got_affinity = user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity))
            return {
                "label": label,
                "hwnd": hex(hwnd),
                "rect": [rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top],
                "visible": bool(user32.IsWindowVisible(hwnd)),
                "iconic": bool(user32.IsIconic(hwnd)),
                "cloaked": cloaked.value if hr == 0 else f"hr={hr:#x}",
                "topmost": bool(ex_style & 0x00000008),
                "layered": bool(ex_style & 0x00080000),
                "capture_exclusion": affinity.value if got_affinity else None,
            }

        windows: list[dict] = []
        try:
            windows.append(describe(user32.GetAncestor(self.winfo_id(), 2), "main"))
            windows.append(
                {
                    "label": "main-tk",
                    "state": str(self.state()),
                    "viewable": bool(self.winfo_viewable()),
                    "mapped": bool(self.winfo_ismapped()),
                    "geometry": self.winfo_geometry(),
                }
            )
            for child in self.winfo_children():
                if isinstance(child, tk.Toplevel):
                    try:
                        windows.append(
                            describe(
                                user32.GetAncestor(child.winfo_id(), 2),
                                f"toplevel:{child.title()}",
                            )
                        )
                    except tk.TclError:
                        continue
        except tk.TclError:
            return

        payload = {"tag": tag, "ts": time.time(), "windows": windows}
        try:
            path = self.WINDOW_STATE_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            temp.replace(path)
        except OSError:
            pass

    def _diagnose_self_shot(self, out_dir: str) -> None:
        """撤掉截屏免疫，把自己每个窗口拍一张存盘（只在设了环境变量时跑）。

        免疫（``wda=17``）生效后，外部**任何**抓取都只能拿到纯黑（``PrintWindow``
        实测 mean=0.0 / uniq=1），跨进程也清不掉这个设置。所以「界面黑屏」到底是
        ①内容没画出来 还是②只是被免疫排除掉，只有进程内自拍能回答。
        拍完立刻恢复免疫，不改变正常运行时的行为。
        """

        try:
            from .self_shot import capture_self_windows

            # 免疫定时器每 2 秒会重新设 17，自拍期间先摘掉，拍完再挂回去。
            if self._capture_exclusion_after_id is not None:
                try:
                    self.after_cancel(self._capture_exclusion_after_id)
                except tk.TclError:
                    pass
                self._capture_exclusion_after_id = None
            try:
                results = capture_self_windows(self, out_dir)
                payload: dict = {"ok": True, "windows": results}
            finally:
                try:
                    self._capture_exclusion_after_id = self.after(
                        self.CAPTURE_EXCLUSION_INTERVAL_MS, self._capture_exclusion_tick
                    )
                except tk.TclError:
                    self._capture_exclusion_after_id = None
        except Exception as error:  # 诊断自己失败也要留证据，绝不静默
            payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        try:
            report = Path(out_dir) / "report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass
        try:
            record_observation("self_shot", payload, source="diagnostics")
        except Exception:
            pass

    def _show_in_front(self) -> None:
        """把主窗口插到 Z 序最前（不只是本进程内 ``lift``）。

        实机 2026-11-18：「任务栏里有，但点不出来」——窗口 ``IsWindowVisible`` 为真、
        矩形也在屏幕内，可就是压不出来。原因不在 Tk：游戏（UE 无边框窗口）会反复
        把自己顶回前台，任务栏点击送来的激活消息被它吞掉，``deiconify()``/``lift()``
        只在**本进程**的窗口之间调序，动不了别的进程。

        做法是「短暂置顶」——``-topmost`` 400ms 足以穿过系统的激活/重绘，然后立刻
        还原，避免长期压住游戏画面（工具需要能看见棋盘）。
        """
        if not self.winfo_exists():
            return
        try:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.after(400, self._drop_topmost)
            self.focus_force()
        except tk.TclError:
            pass

    def _drop_topmost(self) -> None:
        try:
            if self.winfo_exists():
                self.attributes("-topmost", False)
        except tk.TclError:
            pass

    #: 工具窗口「对截屏免疫」的复查周期。2000ms 足够接住新建的桌宠/浮层窗口，
    #: 又完全不影响 100ms 一轮的采集节奏。
    CAPTURE_EXCLUSION_INTERVAL_MS = 2000

    def _capture_exclusion_tick(self) -> None:
        """周期性把所有工具窗口标成「对截屏不可见」，并只在状态变化时记一笔。

        为什么是轮询而不是在创建点逐个调用：桌宠窗口、坐标浮层、各种对话框
        都是运行中才出现的，而且用户拖动/重建时句柄会变。扫一遍窗口树最省事，
        也顺手能自愈「驱动或系统把亲和度重置了」这种情况。
        """

        self._capture_exclusion_after_id = None
        # 诊断开关：设置 HUAHUA_NO_CAPTURE_EXCLUSION=1 启动时完全不免疫。
        # 免疫生效后本窗口对**所有**截屏 API（含 PrintWindow / 远程桌面串流）都是
        # 黑或透明，外部就没法再看到界面内容判断「是窗口没画，还是被排除掉了」。
        # 2026-11-18 实机排查「黑屏」时需要这个对照，别删。
        if os.environ.get("HUAHUA_NO_CAPTURE_EXCLUSION") == "1":
            return
        # 远程会话（RDP/串流）里「对截屏免疫」等于把界面彻底抹掉：
        # 实测 wda=17 时 PrintWindow mean 0.0、屏幕复制拍到的是窗口背后的内容，
        # 远程用户看到的就是「有边框、无标题栏、不能拖动」的一块黑。
        # 用户 2026-11-18 确认自己就是远程连接 → 远程优先保证「能看见」。
        # 代价：工具窗口压在游戏上时可能被判定成遮挡，界面会提示切前台。
        # 强制开关：HUAHUA_CAPTURE_EXCLUSION=1 强制免疫，=0 强制关闭。
        force = os.environ.get("HUAHUA_CAPTURE_EXCLUSION")
        remote = remote_session_reason()
        if force != "1" and (force == "0" or remote is not None):
            if not self._capture_exclusion_reported:
                self._capture_exclusion_reported = True
                record_observation(
                    "capture_exclusion",
                    {
                        "enabled": False,
                        "reason": remote if force != "0" else "env_disabled",
                        "note": "远程会话下关闭截屏免疫，否则远程看到的是黑屏。",
                    },
                    source="capture",
                )
            return
        try:
            if not self.winfo_exists():
                return
            excluded = exclude_tool_windows_from_capture(self)
            if excluded and not self._capture_exclusion_reported:
                self._capture_exclusion_reported = True
                record_observation(
                    "capture_exclusion",
                    {
                        "windows": list(excluded),
                        "wgc": wgc_status(),
                        "note": "Tool windows are now invisible to screen capture.",
                    },
                    source="capture",
                )
        except tk.TclError:
            return
        except Exception:  # 免疫失败不该把界面带崩，退回原来的遮挡判断
            pass
        finally:
            try:
                self._capture_exclusion_after_id = self.after(
                    self.CAPTURE_EXCLUSION_INTERVAL_MS, self._capture_exclusion_tick
                )
            except tk.TclError:
                self._capture_exclusion_after_id = None

    def _run_capture_selftest(self) -> None:
        """把「窗口内容抓取到底为什么不可用」写成一条可读结论。

        实机踩过：``windows_capture`` 只抛一句没有 HRESULT 的字符串，看起来
        像「这台机器的 WGC 坏了」，实际是进程被降到了 LOW 完整性（自动化环境
        拉起时必然如此），系统对**任何**目标都回 ``E_ACCESSDENIED``。
        结论落盘到 ``runtime/capture_selftest.log``，界面只在不健康时提示一次。
        """

        try:
            targets: dict[str, int] = {}
            game_window = find_window(DEFAULT_WINDOW_TITLE)
            if game_window is not None:
                targets["game"] = game_window.hwnd
            report = capture_diagnostics(targets or None)
            text = format_capture_diagnostics(report)
            path = Path("runtime/capture_selftest.log")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{text}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
            self._capture_selftest_report = report
            verdict = str(report.get("verdict") or "")
            if verdict == "ready":
                return
            advice = str(report.get("advice") or "")
            message = (
                "采集自检：窗口内容抓取不可用，已改用「截屏免疫」方案"
                "（工具窗口对截屏不可见，压在棋盘上也不会挡画面）。"
                f"原因：{advice}"
            )
            self.after(0, lambda: self._set_notice(message, "warning"))
        except Exception:
            pass

    def _app_overlaps_game_window(self) -> bool:
        try:
            if not self.winfo_viewable():
                return False
            game_window = find_window(DEFAULT_WINDOW_TITLE)
            if game_window is None or not game_window.captureable:
                return False
            # 本工具的所有窗口已经对截屏免疫：屏幕复制根本拍不到它们，
            # 压在棋盘上也进不了帧。这条路和 WGC 无关，是 WGC 整体失效时
            # 唯一能真正解决「工具窗口挡棋盘」的办法。
            if windows_excluded_from_capture(self):
                return False
            self.update_idletasks()
            app_rect = (
                self.winfo_rootx(),
                self.winfo_rooty(),
                self.winfo_rootx() + self.winfo_width(),
                self.winfo_rooty() + self.winfo_height(),
            )
            game_rect = (game_window.left, game_window.top, game_window.right, game_window.bottom)
            if not _rects_intersect(app_rect, game_rect):
                return False
            # 窗口内容抓取读的是游戏自己渲染的表面，工具窗口盖上去不会进帧，
            # 所以只要它已经接管，遮挡就不再是需要暂停的理由。
            if window_content_capture_ready(game_window):
                return False
            # 会话还没建（游戏刚重开、句柄变了）但 WGC 本身可用：不能暂停。
            # 一暂停就停掉采集线程，而会话只能由采集线程建立——会把自己锁死，
            # 表现就是「自动跟随一直停在覆盖提示上，怎么移窗口都不动」。
            if window_content_capture_available():
                return False
            # 第一帧还没有建立会话：先放行一次，让采集线程去试。试成了就再也不会
            # 因为遮挡暂停；试不成下一轮就会照常暂停。
            if not self.auto_follow_capture_probed:
                return False
            return True
        except tk.TclError:
            return False

    def _auto_record_real_trajectory(self, scene: SceneObservation, backpack: tuple[Paint, ...]) -> bool:
        if self.current_state is None:
            return False
        # 背包留档优先于轨迹可信度判断：整帧不可信时轨迹不进，但「当时背包里
        # 有哪些颜料」这件事必须留下来，否则复现时无从对照。
        self._persist_backpack_snapshot(self.current_state)
        quality = self.last_capture_quality or _capture_quality_gate(scene, backpack)
        finish_seen = _scene_has_finish_evidence(scene, self.current_state)
        if not _quality_is_trusted_for_trajectory(quality):
            record_observation(
                "capture_step_ignored",
                {
                    "session": self.real_session_id,
                    "quality": quality,
                    "finish_seen": finish_seen,
                    "reason": "untrusted_capture",
                    "board": _compact_board_summary(scene),
                    "note": "Untrusted full/auto capture was not appended to the real trajectory.",
                },
                source="real_capture",
            )
            return False
        if any(item.get("severity") == "block" for item in self.last_rule_anomalies):
            return False
        step = _compact_real_capture_step(
            scene=scene,
            state=self.current_state,
            backpack=backpack,
            backpack_details=self._backpack_cache_details(),
            backpack_pages=self._backpack_pages_payload(),
            recommendations=self.recommendations,
            brush_scenario=self.brush_scenario,
            resources=self._resource_state(),
            quality=quality,
        )
        step["board_id"] = self._intern_board(step)
        step["inferred_action"] = _infer_real_capture_action(self.last_real_capture_step, step)
        tentative_saved = self._handle_tentative_step(step, observation_kind="capture_step")
        if tentative_saved is not None:
            return tentative_saved
        step = self._stamp_real_step(step)
        self.last_real_capture_step = step
        step_key = _stable_payload_key(step)
        if step_key != self.last_real_step_key:
            self._ensure_initial_snapshot(step)
            self.real_trajectory_steps.append(step)
            self.real_trajectory_steps = self.real_trajectory_steps[-60:]
            self.last_real_step_key = step_key

        reliable_steps = _trusted_trajectory_steps(self.real_trajectory_steps)
        terminal = (
            _best_trusted_terminal_summary(self.current_state, reliable_steps, current_quality=quality)
            if finish_seen
            else _trajectory_terminal_summary(self.current_state, backpack)
        )
        reliable, _reason_text = _terminal_reliability_for_finish(terminal, reliable_steps, finish_seen=finish_seen)
        if not reliable:
            return False
        high_score_gate = _high_score_gate(terminal)
        should_save, reason = high_score_gate["passed"], high_score_gate["primary_reason"]
        if not should_save:
            return False
        score = _internal_run_score(terminal)
        score_breakdown = _internal_score_breakdown(terminal)
        initial_snapshot = self._initial_snapshot_from_steps(reliable_steps)
        payload = {
            "version": 2,
            "kind": "auto_capture_trajectory",
            "session": self.real_session_id,
            "capture_session": self.backpack_session_key,
            "reason": reason,
            "started_at": self.real_session_started_at,
            "finished_at": _trajectory_finished_at(self.real_session_started_at, reliable_steps),
            "internal_score": score,
            "score_model": "huahua-v2-treasure-equivalence",
            "score_breakdown": score_breakdown,
            "terminal": terminal,
            "high_score_gate": high_score_gate,
            "initial_snapshot": initial_snapshot,
            "finish_seen": finish_seen,
            "steps": reliable_steps[-40:],
        }
        payload = _with_initial_aliases(payload)
        payload = _with_state_action_result_metadata(payload)
        payload_key = _stable_payload_key(payload)
        if payload_key == self.last_auto_trajectory_key:
            return False
        saved = record_high_score_trajectory(payload, source="real_capture")
        if saved:
            self.last_auto_trajectory_key = payload_key
            # 统计在后台算，算完把界面上的数字换掉（以前只在启动时算过一次）。
            self._schedule_learning_stats_refresh()
        return saved

    def _ensure_initial_snapshot(self, step: dict) -> None:
        if self.real_initial_snapshot is not None:
            return
        if not _quality_is_trusted_for_trajectory(step.get("quality")):
            return
        self.real_initial_snapshot = _initial_snapshot_from_step(step)

    def _initial_snapshot_from_steps(self, reliable_steps: list[dict]) -> dict | None:
        if self.real_initial_snapshot is not None:
            return self.real_initial_snapshot
        if reliable_steps:
            return _initial_snapshot_from_step(reliable_steps[0])
        return None

    def _record_compressed_run_summary(
        self,
        *,
        reason: str,
        skip_reason: str,
        terminal: dict,
        high_score_gate: dict,
        score: float,
        score_breakdown: dict,
        reliable_steps: list[dict],
        raw_finish_seen: bool,
        finish_seen: bool,
    ) -> bool:
        initial_snapshot = self._initial_snapshot_from_steps(reliable_steps)
        if not initial_snapshot:
            return False
        summary_kind = _compressed_summary_kind(terminal, high_score_gate)
        sample_metadata = _compressed_summary_sample_metadata(summary_kind)
        payload = {
            "version": 1,
            "kind": summary_kind,
            "weight": sample_metadata["training_weight"],
            "training_weight": sample_metadata["training_weight"],
            "sample_class": sample_metadata["sample_class"],
            "sample_role": sample_metadata["sample_role"],
            "candidate_sample": sample_metadata["candidate_sample"],
            "candidate_metadata": sample_metadata,
            "session": self.real_session_id,
            "reason": reason,
            "skip_reason": skip_reason,
            "started_at": self.real_session_started_at,
            "finished_at": _trajectory_finished_at(self.real_session_started_at, reliable_steps),
            "internal_score": score,
            "score_model": "huahua-v2-treasure-equivalence",
            "score_breakdown": score_breakdown,
            "terminal": terminal,
            "high_score_gate": high_score_gate,
            "initial_snapshot": initial_snapshot,
            "decision_records": decision_records_from_steps(
                reliable_steps[-40:],
                session=self.real_session_id,
            ),
            "steps": len(self.real_trajectory_steps),
            "reliable_steps": len(reliable_steps),
            "raw_finish_seen": raw_finish_seen,
            "finish_seen": finish_seen,
        }
        payload = _with_initial_aliases(payload)
        payload_key = _stable_payload_key(
            {
                "session": self.real_session_id,
                "kind": summary_kind,
                "initial": initial_snapshot.get("signature"),
                "terminal": terminal.get("known"),
                "score": round(float(score or 0.0), 3),
            }
        )
        if payload_key == self.last_compressed_summary_key:
            return False
        self.last_compressed_summary_key = payload_key
        record_observation("compressed_run_summary", payload, source="real_capture")
        return True

    def _export_learning_data(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Learning Data",
            defaultextension=".json",
            filetypes=[("Huahua learning bundle", "*.json"), ("All files", "*.*")],
            initialfile="huahua_learning_bundle.json",
        )
        if not path:
            return
        try:
            result = export_learning_bundle(path)
        except OSError as exc:
            messagebox.showerror("Export Data", f"导出失败：{exc}")
            return
        self.status_var.set(
            f"已导出学习数据：观察{result['observations']}条，轨迹{result['trajectories']}条，纠错{result['corrections']}条 -> {result['path']}"
        )

    def _merge_learning_data(self) -> None:
        path = filedialog.askopenfilename(
            title="Merge Learning Data",
            filetypes=[("Huahua learning bundle", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            result = merge_learning_bundle(path)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Merge Data", f"合并失败：{exc}")
            return
        self.status_var.set(
            f"已合并学习数据：新增观察{result['added_observations']}条，新增轨迹{result['added_trajectories']}条，新增纠错{result['added_corrections']}条"
        )
        self._rerender_side_panel()

    def _draw_empty_grid(self) -> None:
        self.canvas.delete("all")
        self._draw_grid(None)

    def _draw_scene(self, scene: SceneObservation) -> None:
        self.canvas.delete("all")
        self._draw_grid(scene.board, self.current_state)

    def simulate_random(self) -> None:
        self.current_scene = None
        self.backpack_accumulator.reset()
        self.backpack_session_key = None
        self.backpack_frame_visible = None
        self.last_backpack_paint_tiles = None
        self.manual_correction_applied = False
        self.current_state = random_state()
        self.selected_backpack_index = None
        self.recommendations = recommend_placements(self.current_state)
        self.install_plan = recommend_install_plan(self.current_state)
        self._clear_brush_choice()
        self.board_var.set("模式：模拟拼图")
        self.status_var.set("模拟模式：本地随机生成棋盘和初始颜料，不写入真实学习记录。")
        self.canvas.delete("all")
        self._draw_grid(None, self.current_state)
        self._write_simulation()

    def _merge_detected_backpack(self, scene: SceneObservation) -> tuple[Paint, ...]:
        # 会话 = **一局**。以前把「调色目标（类别/等级/方向/作用对象）」拼进键里，
        # 目标一变就 reset 整个累计器——而目标从怪物换成玩法，恰恰是要**另记一页**
        # 的时候（用户 2026-11-18：两页各 15 格、合计最多 30 格）。所以键只留局号，
        # 目标作为「页」的标识交给 page_key。
        session_key = self.real_session_id
        paint_tiles = sum(1 for tile in scene.board.tiles if tile.state is TileState.PAINT)
        consumed = self.last_backpack_paint_tiles is not None and paint_tiles > self.last_backpack_paint_tiles
        self.last_backpack_paint_tiles = paint_tiles
        if session_key != self.backpack_session_key:
            self.backpack_accumulator.reset(session_key)
            self.backpack_session_key = session_key
        if scene.backpack is None or not scene.backpack.paints:
            # 背包面板这一帧没露出来：保持上一次的结果，别把它当成"背包空了"
            return self.backpack_accumulator.paints()
        return self.backpack_accumulator.update(
            scene.backpack.paints,
            session_key,
            consumed=consumed,
            # 目标读不出来时给 None（沿用当前页），不能造 ``unknown`` 页——
            # 那一天会在快照里多出 15 格空读数，见 _scene_backpack_page_key。
            page_key=_scene_backpack_page_key(scene) or None,
        )

    def _backpack_cache_details(self) -> list[dict]:
        details: list[dict] = []
        index = 0
        for entry in self.backpack_accumulator.entries():
            observed = entry.observed
            if observed is None:
                continue
            for _ in range(max(1, entry.count)):
                detail = _compact_observed_backpack_paint(observed, index=index)
                detail["i"] = index
                details.append(detail)
                index += 1
        return details

    def _record_scene_observation(
        self,
        scene: SceneObservation,
        backpack: tuple[Paint, ...],
        quality: dict | None = None,
        anomalies: list[dict] | None = None,
    ) -> None:
        quality = quality or _capture_quality_gate(scene, backpack)
        record_observation(
            "capture_quality",
            quality,
            source="real_capture",
        )
        if scene.current_paint is not None:
            paint = scene.current_paint
            observed_rule = (
                CurrentPaintRule(
                    paint=Paint(
                        category=paint.category,
                        level=paint.level,
                        subtype=paint.subtype,
                        is_treasure=paint.is_treasure,
                    ),
                    direction=paint.direction,
                    target_object=paint.target_object,
                    match_required=paint.match_required,
                    upgrade_amount=paint.upgrade_amount,
                    match_current=paint.match_current or 0,
                )
                if paint.level is not None
                and paint.match_required is not None
                and paint.upgrade_amount is not None
                else None
            )
            record_observation(
                "current_paint",
                {
                    "paint_code": (
                        compact_paint_with_code(
                            Paint(
                                category=paint.category,
                                level=paint.level,
                                subtype=paint.subtype,
                                is_treasure=paint.is_treasure,
                            )
                        ).get("code")
                        if paint.level is not None
                        else None
                    ),
                    "rule_code": compact_rule_with_code(observed_rule).get("code") if observed_rule is not None else None,
                    "category": paint.category.value,
                    "level": paint.level,
                    "direction": paint.direction.value,
                    "target": paint.target_object.value,
                    "match_current": paint.match_current,
                    "match_required": paint.match_required,
                    "upgrade_amount": paint.upgrade_amount,
                    "raw_text_hint": paint.raw_text_hint,
                },
                source="real_capture",
            )
            self._record_current_paint_rule_warnings(paint)
        for anomaly in anomalies or []:
            record_observation(
                "rule_anomaly",
                anomaly,
                source="real_capture",
            )
        if scene.brush_options is not None:
            brush = scene.brush_options
            record_observation(
                "brush_options",
                {
                    "stage1_options": brush.stage1_options,
                    "stage1_codes": [code for item in brush.stage1_options if (code := brush_code(item))],
                    "chosen_stage1": brush.chosen_stage1,
                    "chosen_stage1_code": brush_code(brush.chosen_stage1),
                    "stage2_options": brush.stage2_options,
                    "stage2_codes": [code for item in brush.stage2_options if (code := affix_code(item))],
                    "stage2_ids": [identity for item in brush.stage2_options if (identity := canonical_affix_id(item))],
                    "confidence": brush.confidence,
                    "stage1_scores": brush.stage1_scores,
                    "stage2_scores": brush.stage2_scores,
                },
                source="real_capture",
            )
            for name in brush.stage2_options:
                if (
                    "进度要求减少1" in name
                    or "进度要求减少2" in name
                    or "所需数量-2" in name
                    or ("调色目标" in name and "减少" in name)
                ):
                    record_observation(
                        "rule_conflict",
                        {
                            "kind": "match_required_delta",
                            "affix_name": name,
                            "affix_code": affix_code(name),
                            "note": "已记录转锋减进度 -1/-2 实战文案，按识别词条 delta 保存",
                        },
                        source="real_capture",
                    )
        if scene.backpack is not None:
            record_observation(
                "backpack_capture",
                {
                    "visible": len(scene.backpack.paints),
                    "cumulative": len(backpack),
                    "visible_paints": [
                        {
                            "category": observed.category.value,
                            "level": observed.level,
                            "subtype": observed.subtype,
                            "is_treasure": observed.is_treasure,
                            "category_confidence": round(observed.category_confidence, 3),
                            "level_confidence": round(observed.level_confidence, 3),
                        }
                        for observed in scene.backpack.paints
                    ],
                    "slots": [
                        {
                            "slot": [observed.slot_rect.x, observed.slot_rect.y, observed.slot_rect.w, observed.slot_rect.h],
                            "category": observed.category.value,
                            "category_confidence": round(observed.category_confidence, 3),
                            "level": observed.level,
                            "level_confidence": round(observed.level_confidence, 3),
                            "subtype": observed.subtype,
                            "is_treasure": observed.is_treasure,
                            "fingerprint": observed.fingerprint,
                            "merge_key": paint_signature(paint_from_observation(observed)),
                        }
                        for observed in scene.backpack.paints
                    ],
                    "paints": [
                        {
                            "code": compact_paint_with_code(paint).get("code"),
                            "category": paint.category.value,
                            "level": paint.level,
                            "subtype": paint.subtype,
                            "is_treasure": paint.is_treasure,
                        }
                        for paint in backpack
                    ],
                },
                source="real_capture",
            )

    def _record_high_value_seen(
        self,
        scene: SceneObservation,
        state: PuzzleState | None,
        backpack: tuple[Paint, ...],
        quality: dict | None,
        *,
        mode: str,
        recommendations: list[PlacementRecommendation] | None = None,
    ) -> str | None:
        quality = quality or _capture_quality_gate(scene, backpack)
        if not quality.get("ok"):
            return None
        board_paints = tuple(tile.paint for tile in state.tiles if tile.paint is not None) if state is not None else ()
        board_summary = _paint_collection_summary(board_paints)
        backpack_summary = _paint_collection_summary(backpack)
        combined_summary = _paint_collection_summary(tuple(paint for paint in (*board_paints, *backpack) if paint is not None))
        if not _summary_has_high_value(combined_summary):
            return None
        payload = {
            "session": self.real_session_id,
            "mode": mode,
            "quality": quality,
            "board": _compact_board_summary(scene),
            "board_known": board_summary,
            "backpack": backpack_summary,
            "known": combined_summary,
            "current": _compact_rule(state.current_rule) if state is not None else None,
            "score": _quality_proxy(combined_summary),
            "proxy_score": {
                "value": _quality_proxy(combined_summary),
                "model": "huahua-v2-treasure-equivalence",
                "breakdown": _internal_score_breakdown({"known": combined_summary}),
            },
            "score_breakdown": _internal_score_breakdown({"known": combined_summary}),
            "recommendations": [
                _compact_recommendation(rec, rank=rank)
                for rank, rec in enumerate((recommendations or [])[:36], start=1)
            ],
        }
        event_key = _stable_payload_key(
            {
                "session": self.real_session_id,
                "mode": mode,
                "board_states": payload["board"].get("states"),
                "known": combined_summary,
                "current": payload.get("current"),
            }
        )
        if event_key != self.last_high_value_key:
            self.last_high_value_key = event_key
            record_observation("high_value_seen", payload, source="real_capture")
        return f"已记录高价值识别：{_format_high_value_summary(combined_summary)}。"

    def _record_current_paint_rule_warnings(self, paint) -> None:
        if paint.level is None:
            return
        expected_match, expected_upgrade = _empirical_current_paint_defaults(paint.level)
        if paint.match_required is not None and paint.match_required != expected_match:
            record_observation(
                "rule_mismatch",
                {
                    "kind": "current_match_required",
                    "level": paint.level,
                    "observed": paint.match_required,
                    "expected": expected_match,
                    "raw_text_hint": paint.raw_text_hint,
                },
                source="real_capture",
            )
        if paint.upgrade_amount is not None and not _empirical_upgrade_is_allowed(paint.level, paint.upgrade_amount):
            record_observation(
                "rule_mismatch",
                {
                    "kind": "current_upgrade_amount",
                    "level": paint.level,
                    "observed": paint.upgrade_amount,
                    "expected": expected_upgrade,
                    "allowed": _empirical_allowed_upgrades(paint.level),
                    "raw_text_hint": paint.raw_text_hint,
                },
                source="real_capture",
            )

    def force_complete_current(self) -> None:
        if self.current_state is None:
            self.status_var.set("当前没有可强制完成的模拟棋局")
            return
        force_used = _clamp_int_var(self.force_used_var, 0, 3)
        if force_used >= 3:
            self.status_var.set("强制完成次数已用完")
            return
        result = force_complete_current_rule(self.current_state, force_used)
        self.current_state = result.state
        if result.success:
            self.force_used_var.set(force_used + 1)
            self.selected_backpack_index = None
            self._begin_pending_brush_choice()
        self.status_var.set(result.message)
        self._refresh_simulation_state_views()

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        self.canvas.delete("all")
        if self.current_scene is None:
            self._draw_grid(None, self.current_state)
            return
        self._draw_grid(self.current_scene.board, self.current_state)

    def _rerender_side_panel(self) -> None:
        if self.current_scene is not None:
            self._write_regions(self.current_scene)
        elif self.current_state is not None:
            self._write_simulation()

    def _refresh_simulation_state_views(self) -> None:
        if self.current_state is None:
            return
        self.recommendations = recommend_placements(self.current_state)
        self.install_plan = recommend_install_plan(self.current_state)
        self.canvas.delete("all")
        self._draw_grid(self.current_scene.board if self.current_scene else None, self.current_state)
        self._rerender_side_panel()
        self._refresh_overlay()

    def _clear_brush_choice(self) -> None:
        self.brush_scenario = None
        self.selected_brush_type = None
        self.selected_affix = None
        self._last_brush_observation = None
        self._brush_stage1_vote_observations.clear()
        self._locked_brush_stage1 = None
        self._locked_brush_stage2 = None
        self.brush_stage1_rects = {}
        self.brush_affix_rects = {}

    def _apply_detected_brush_options(self, observation: BrushOptionsObservation | None) -> None:
        if observation is None:
            return
        self._last_brush_observation = observation
        stage1 = _brush_types_from_values(observation.stage1_options)
        stage2 = tuple(
            affix
            for name in observation.stage2_options
            if (affix := resolve_affix(name)) is not None
        )
        chosen = _brush_type_from_value(observation.chosen_stage1)
        if chosen is None and stage2:
            chosen = stage2[0].brush_type
        if not stage1 and chosen is not None:
            stage1 = (chosen,)
        if not stage1 and not stage2:
            return
        if chosen is None:
            chosen = stage1[0]
        if chosen not in stage1:
            stage1 = normalize_brush_options((*stage1, chosen))

        self.brush_scenario = BrushChoiceScenario(
            stage1_options=stage1,
            chosen_stage1=chosen,
            stage2_options=stage2,
        )
        self.selected_brush_type = chosen if stage2 else None
        self.selected_affix = None
        self.selected_backpack_index = None

    def _begin_pending_brush_choice(self) -> None:
        if self.current_state is None or not current_rule_is_pending(self.current_state):
            self._clear_brush_choice()
            return
        stage1 = normalize_brush_options(tuple(random.sample(list(BrushType), 2)))
        self.brush_scenario = BrushChoiceScenario(
            stage1_options=stage1,
            chosen_stage1=stage1[0],
            stage2_options=(),
        )
        self.selected_brush_type = None
        self.selected_affix = None
        self.selected_backpack_index = None

    def _pending_brush_choice_required(self) -> bool:
        return (
            self.current_state is not None
            and current_rule_is_pending(self.current_state)
            and self.brush_scenario is not None
            and self.selected_affix is None
        )

    def _choose_brush_stage1(self, brush_type: BrushType) -> None:
        if self.current_state is None or self.brush_scenario is None:
            return
        if self.selected_affix is not None:
            self.status_var.set("词缀已经选择，当前初始颜料放置前不能重新选择大类")
            return
        if brush_type not in self.brush_scenario.stage1_options:
            return
        options = [affix for affix in self.affix_catalog if affix.brush_type == brush_type]
        stage2 = tuple(random.sample(options, min(2, len(options)))) if options else ()
        self.brush_scenario = BrushChoiceScenario(
            stage1_options=self.brush_scenario.stage1_options,
            chosen_stage1=brush_type,
            stage2_options=stage2,
        )
        self.selected_brush_type = brush_type
        self.selected_affix = None
        self.status_var.set(f"已选择{_brush_label(brush_type)}，请从右侧2个词缀中再选1个")
        self._refresh_simulation_state_views()

    def _choose_brush_affix(self, affix_index: int) -> None:
        if self.current_state is None or self.brush_scenario is None:
            return
        if self.selected_affix is not None:
            self.status_var.set("词缀已经选择，请直接在棋盘上放置新的初始颜料")
            return
        if not 0 <= affix_index < len(self.brush_scenario.stage2_options):
            return
        affix = self.brush_scenario.stage2_options[affix_index]
        result = apply_affix_to_state(self.current_state, affix)
        self.current_state = result.state
        if result.success:
            self.selected_affix = affix
        self.status_var.set(result.message)
        self._refresh_simulation_state_views()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.current_state is None:
            return
        point = (float(event.x), float(event.y))
        for brush_type, rect in self.brush_stage1_rects.items():
            if _point_in_rect(point, rect):
                self._choose_brush_stage1(brush_type)
                return
        for affix_index, rect in self.brush_affix_rects.items():
            if _point_in_rect(point, rect):
                self._choose_brush_affix(affix_index)
                return
        for index, rect in self.backpack_slot_rects.items():
            if _point_in_rect(point, rect):
                if current_rule_is_pending(self.current_state):
                    self.status_var.set("请先在棋盘空白地块放置新的初始颜料")
                    return
                self.selected_backpack_index = index
                paint = self.current_state.backpack[index]
                self.status_var.set(f"已选中背包第{index + 1}格：{_paint_display_label(paint)}")
                self.canvas.delete("all")
                self._draw_grid(self.current_scene.board if self.current_scene else None, self.current_state)
                self._refresh_overlay()
                return
        if current_rule_is_pending(self.current_state):
            for (row, col), rect in self.board_cell_rects.items():
                if not _point_in_rect(point, rect):
                    continue
                if self._pending_brush_choice_required():
                    self.status_var.set("请先在右侧完成转锋/堆砌/扩散二阶段选择，再放置新的初始颜料")
                    return
                result = place_current_rule_anchor(self.current_state, row, col)
                self.current_state = result.state
                self.selected_backpack_index = None
                if result.success:
                    self._clear_brush_choice()
                self.status_var.set(result.message)
                self._refresh_simulation_state_views()
                return
        if self.selected_backpack_index is None:
            return
        for (row, col), rect in self.board_cell_rects.items():
            if not _point_in_rect(point, rect):
                continue
            result = install_backpack_paint(self.current_state, self.selected_backpack_index, row, col)
            self.current_state = result.state
            if result.success:
                self.selected_backpack_index = None
                if result.completed_match:
                    self._begin_pending_brush_choice()
            elif self.selected_backpack_index is not None and self.selected_backpack_index >= len(self.current_state.backpack):
                self.selected_backpack_index = None
            self.status_var.set(result.message)
            self._refresh_simulation_state_views()
            return

    def _open_capture_verify_dialog(self) -> None:
        if self.current_scene is None:
            self.status_var.set("请先完整识别真实游戏画面；模拟模式不参与学习纠错")
            return

        scene = self.current_scene
        window = tk.Toplevel(self)
        window.title("纠错真实识别")
        window.configure(bg="#101820")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()

        content = tk.Frame(window, bg="#101820")
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(
            content,
            text="确认/纠错本次真实识别",
            fg="#ffffff",
            bg="#101820",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        risk_hint = _risk_hint_text(self._learning_stats_cache)
        tk.Label(
            content,
            text=risk_hint,
            fg="#f2c14e" if "高风险" in risk_hint else "#95abc1",
            bg="#101820",
            font=("Microsoft YaHei UI", 9),
            wraplength=430,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 8))

        paint = scene.current_paint
        brush = scene.brush_options
        backpack_count = self.backpack_accumulator.total

        category_var = tk.StringVar(value=paint.category.value if paint else PaintCategory.UNKNOWN.value)
        level_var = tk.StringVar(value="" if paint is None or paint.level is None else str(paint.level))
        direction_var = tk.StringVar(value=paint.direction.value if paint else PaintDirection.UNKNOWN.value)
        target_var = tk.StringVar(value=paint.target_object.value if paint else PaintObject.UNKNOWN.value)
        match_required_var = tk.StringVar(value="" if paint is None or paint.match_required is None else str(paint.match_required))
        upgrade_var = tk.StringVar(value="" if paint is None or paint.upgrade_amount is None else str(paint.upgrade_amount))
        stage1_var = tk.StringVar(value=", ".join(brush.stage1_options) if brush else "")
        stage2_var = tk.StringVar(value="; ".join(brush.stage2_options) if brush else "")
        backpack_var = tk.StringVar(value=str(backpack_count))
        note_var = tk.StringVar()
        apply_var = tk.BooleanVar(value=True)

        fields = [
            ("类型", category_var, [item.value for item in PaintCategory]),
            ("等级", level_var, None),
            ("方向", direction_var, [item.value for item in PaintDirection]),
            ("目标", target_var, [item.value for item in PaintObject]),
            ("需求数", match_required_var, None),
            ("升级数", upgrade_var, None),
            ("一阶段", stage1_var, None),
            ("二阶段词缀", stage2_var, None),
            ("背包累计数", backpack_var, None),
            ("备注", note_var, None),
        ]

        for row, (label, variable, options) in enumerate(fields, start=2):
            tk.Label(content, text=label, fg="#95abc1", bg="#101820").grid(row=row, column=0, sticky="w", pady=3)
            if options:
                widget = tk.OptionMenu(content, variable, *options)
                widget.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=22)
            else:
                widget = tk.Entry(content, textvariable=variable, bg="#172331", fg="#d7e3ee", insertbackground="#d7e3ee", width=30)
            widget.grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)

        tk.Checkbutton(
            content,
            text="保存后应用到当前局面（重算金框/推荐；背包数量只记录不造假）",
            variable=apply_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
        ).grid(row=len(fields) + 2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        original = {
            "category": paint.category.value if paint else None,
            "level": paint.level if paint else None,
            "direction": paint.direction.value if paint else None,
            "target": paint.target_object.value if paint else None,
            "match_required": paint.match_required if paint else None,
            "upgrade_amount": paint.upgrade_amount if paint else None,
            "stage1_options": list(brush.stage1_options) if brush else [],
            "stage2_options": list(brush.stage2_options) if brush else [],
            "backpack_count": backpack_count,
        }

        def save() -> None:
            corrected = {
                "category": category_var.get(),
                "level": _optional_int(level_var.get()),
                "direction": direction_var.get(),
                "target": target_var.get(),
                "match_required": _optional_int(match_required_var.get()),
                "upgrade_amount": _optional_int(upgrade_var.get()),
                "stage1_options": _split_manual_list(stage1_var.get(), separator=","),
                "stage2_options": _split_manual_list(stage2_var.get(), separator=";"),
                "backpack_count": _optional_int(backpack_var.get()),
                "note": note_var.get().strip(),
            }
            changed = [
                key
                for key, value in corrected.items()
                if key != "note" and value != original.get(key)
            ]
            applied = self._apply_capture_correction(corrected) if apply_var.get() else False
            record_correction(
                {
                    "original": original,
                    "corrected": corrected,
                    "changed_fields": changed,
                    "applied_requested": apply_var.get(),
                    "applied_to_current_state": applied,
                    "board_rect": scene.board.rect,
                    "board_score": scene.board.score,
                },
                source="real_capture",
            )
            suffix = "，已应用到当前局面" if applied else "，仅记录未应用"
            self.status_var.set(f"已保存真实识别纠错：{len(changed)} 个字段变化{suffix}")
            self._refresh_after_correction()
            # 纠错也会进统计（后台算），算完把界面数字换掉。
            self._schedule_learning_stats_refresh()
            window.destroy()

        button_row = len(fields) + 4
        tk.Button(content, text="保存纠错", command=save, width=18).grid(row=button_row, column=1, sticky="e", pady=(12, 0))
        tk.Button(content, text="取消", command=window.destroy, width=12).grid(row=button_row, column=2, sticky="w", padx=(8, 0), pady=(12, 0))

    def _apply_capture_correction(self, corrected: dict) -> bool:
        if self.current_state is None:
            return False
        rule = self.current_state.current_rule
        if rule is None:
            return False

        category = _paint_category_from_value(corrected.get("category"), rule.paint.category)
        direction = _paint_direction_from_value(corrected.get("direction"), rule.direction)
        target = _paint_object_from_value(corrected.get("target"), rule.target_object)
        level = corrected.get("level") if corrected.get("level") is not None else rule.paint.level
        match_required = corrected.get("match_required") if corrected.get("match_required") is not None else rule.match_required
        upgrade_amount = corrected.get("upgrade_amount") if corrected.get("upgrade_amount") is not None else rule.upgrade_amount
        paint = replace(rule.paint, category=category, level=max(1, min(7, int(level))))
        next_rule = CurrentPaintRule(
            paint=paint,
            direction=direction,
            target_object=target,
            match_required=max(1, int(match_required)),
            upgrade_amount=max(1, int(upgrade_amount)),
            match_current=rule.match_current,
            extra_directions=rule.extra_directions,
        )
        if self.current_state.current_anchor is not None:
            match_current = min(
                _safe_current_match_count(self.current_state, next_rule),
                next_rule.match_required,
            )
            next_rule = replace(next_rule, match_current=match_current)
        self.current_state = replace(
            self.current_state,
            current_rule=next_rule,
            backpack=sort_backpack_for_rule(self.current_state.backpack, next_rule),
        )
        self._apply_corrected_brush_options(corrected)
        self.manual_correction_applied = True
        return True

    def _apply_corrected_brush_options(self, corrected: dict) -> None:
        stage1 = normalize_brush_options(
            [
                brush_type
                for value in corrected.get("stage1_options", [])
                if (brush_type := _brush_type_from_value(value)) is not None
            ]
        )
        stage2_names = corrected.get("stage2_options", [])
        stage2 = tuple(
            affix
            for name in stage2_names
            if (affix := resolve_affix(name)) is not None
        )
        if not stage1 and not stage2:
            return
        chosen = stage2[0].brush_type if stage2 else stage1[0]
        if chosen not in stage1:
            stage1 = normalize_brush_options((*stage1, chosen))
        self.brush_scenario = BrushChoiceScenario(
            stage1_options=stage1,
            chosen_stage1=chosen,
            stage2_options=stage2,
        )
        self.selected_brush_type = chosen if stage2 else None
        self.selected_affix = None

    def _sync_learning_stats(self) -> bool:
        """把后台刚算好的统计换进来，返回「换到了新的一份」。

        绝不在这里现算：算一次要 1.8 秒（几乎全在 ``build_outcome_risk_report``），
        主线程等不起。后台线程算完会回填 ``learning`` 的内存缓存，这里只是取指针。
        """
        fresh = cached_learning_stats()
        if fresh is None or fresh is self._learning_stats_cache:
            return False
        self._learning_stats_cache = fresh
        return True

    def _schedule_learning_stats_refresh(self, attempts: int = 6, delay_ms: int = 700) -> None:
        """收尾后隔一会儿再看统计算好没有。

        为什么要等：``_write_stats`` 现在是后台线程，一局收尾后它还要跑 1.8 秒左右。
        立刻去看必然是旧的，所以按小步重试几次；拿到就停，拿不到就算了——
        刷新界面上的数字不值得为它卡住任何东西。
        """
        try:
            self.after(delay_ms, lambda: self._learning_stats_refresh_tick(attempts, delay_ms))
        except Exception:  # noqa: BLE001 - 窗口正在销毁时 after 会抛
            pass

    def _learning_stats_refresh_tick(self, attempts: int, delay_ms: int) -> None:
        if self._sync_learning_stats():
            self._apply_learning_stats_to_ui()
            return
        if attempts > 1:
            try:
                self.after(delay_ms, lambda: self._learning_stats_refresh_tick(attempts - 1, delay_ms))
            except Exception:  # noqa: BLE001
                pass

    def _apply_learning_stats_to_ui(self) -> None:
        """统计换了之后把界面上依赖它的东西重画一遍。

        以前这里没人做，于是 ``_learning_stats_cache`` 只在启动时算过一次：
        打完一局不重启，界面上的统计数字、风险提示、推荐排序全都停在开机那一刻。
        """
        try:
            self._write_learning_stats()
            self._write_current_terrain_bucket_hint()
        except Exception:  # noqa: BLE001 - 刷数字不能影响主流程
            pass
        # 统计回调也要遵守记录模式；已结束的对局无需再搜索推荐。
        if self.current_state is not None and not self.real_session_finished and not recording_mode_enabled():
            try:
                self._refresh_after_correction()
            except Exception:  # noqa: BLE001
                pass

    def _refresh_after_correction(self) -> None:
        if self.current_state is None:
            return
        if self.current_scene is not None:
            self.last_capture_quality = _capture_quality_gate(self.current_scene, self.current_state.backpack)
            self.last_rule_anomalies = (
                _detect_capture_rule_anomalies(self.last_real_capture_step, self.current_scene, self.current_state)
                if self.last_capture_quality["ok"]
                else []
            )
        if recording_mode_enabled() or self.real_session_finished:
            self.recommendations = []
            self.install_plan = None
        else:
            self.recommendations = recommend_placements(
                self.current_state,
                learning_stats=self._learning_stats_cache if self.current_scene is not None else None,
            )
            self.install_plan = recommend_install_plan(self.current_state)
        self.selected_backpack_index = None
        self.canvas.delete("all")
        self._draw_grid(self.current_scene.board if self.current_scene else None, self.current_state)
        if self.current_scene is not None:
            self._write_regions(self.current_scene)
        else:
            self._write_simulation()
        self._refresh_overlay()

    def _open_correction_stats_dialog(self) -> None:
        stats = self._learning_stats_cache
        window = tk.Toplevel(self)
        window.title("Correction Stats")
        window.configure(bg="#101820")
        window.geometry("560x460")
        window.transient(self)

        content = tk.Frame(window, bg="#101820")
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        tk.Label(
            content,
            text="真实纠错统计",
            fg="#ffffff",
            bg="#101820",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        text = tk.Text(
            content,
            bg="#0d141b",
            fg="#d7e3ee",
            insertbackground="#d7e3ee",
            relief=tk.FLAT,
            font=("Microsoft YaHei UI", 10),
            wrap=tk.WORD,
        )
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, _format_correction_stats(stats))
        text.configure(state=tk.DISABLED)

    def _open_manual_brush_dialog(self) -> None:
        window = tk.Toplevel(self)
        window.title("手动笔触选项")
        window.configure(bg="#101820")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()

        stage1_default = self.brush_scenario.stage1_options if self.brush_scenario else (BrushType.TRANSFORM, BrushType.STACK)
        chosen_default = self.brush_scenario.chosen_stage1 if self.brush_scenario else stage1_default[0]
        stage_vars = {
            brush_type: tk.BooleanVar(value=brush_type in stage1_default)
            for brush_type in BrushType
        }
        chosen_var = tk.StringVar(value=chosen_default.value)
        affix1_var = tk.StringVar()
        affix2_var = tk.StringVar()

        content = tk.Frame(window, bg="#101820")
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(content, text="一阶段：出现的两个大类", fg="#d7e3ee", bg="#101820", font=("Microsoft YaHei UI", 10, "bold")).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 6),
        )
        for col, brush_type in enumerate(BrushType):
            tk.Checkbutton(
                content,
                text=_brush_label(brush_type),
                variable=stage_vars[brush_type],
                fg="#d7e3ee",
                bg="#101820",
                activeforeground="#ffffff",
                activebackground="#101820",
                selectcolor="#0d141b",
            ).grid(row=1, column=col, sticky="w", padx=(0, 12))

        tk.Label(content, text="二阶段：已选择的大类", fg="#d7e3ee", bg="#101820").grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 4))
        chosen_menu = tk.OptionMenu(content, chosen_var, *[brush.value for brush in BrushType])
        chosen_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=18)
        chosen_menu.grid(row=3, column=0, columnspan=3, sticky="w")

        tk.Label(content, text="二阶段词缀 A", fg="#d7e3ee", bg="#101820").grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 4))
        affix1_menu = tk.OptionMenu(content, affix1_var, "")
        affix1_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=42)
        affix1_menu.grid(row=5, column=0, columnspan=3, sticky="ew")

        tk.Label(content, text="二阶段词缀 B", fg="#d7e3ee", bg="#101820").grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 4))
        affix2_menu = tk.OptionMenu(content, affix2_var, "")
        affix2_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=42)
        affix2_menu.grid(row=7, column=0, columnspan=3, sticky="ew")

        affix_display: dict[str, AffixRule] = {}

        def refresh_affix_menus(*_args: object) -> None:
            brush_type = BrushType(chosen_var.get())
            affixes = [affix for affix in self.affix_catalog if affix.brush_type == brush_type]
            affix_display.clear()
            labels = []
            for affix in affixes:
                code = affix_code(affix) or "?"
                label = f"{code}. {affix.name}"
                affix_display[label] = affix
                labels.append(label)
            if not labels:
                labels = ["-"]
            _replace_option_menu(affix1_menu, affix1_var, labels)
            _replace_option_menu(affix2_menu, affix2_var, labels)
            if affix1_var.get() not in labels:
                affix1_var.set(labels[0])
            if affix2_var.get() not in labels:
                affix2_var.set(labels[min(1, len(labels) - 1)])

        chosen_var.trace_add("write", refresh_affix_menus)
        refresh_affix_menus()

        buttons = tk.Frame(content, bg="#101820")
        buttons.grid(row=8, column=0, columnspan=3, sticky="e", pady=(16, 0))

        def apply_manual() -> None:
            selected = normalize_brush_options([brush for brush in BrushType if stage_vars[brush].get()])
            if len(selected) != 2:
                messagebox.showwarning("Huahua Assistant", "一阶段必须正好选择两个笔触大类。")
                return
            chosen = BrushType(chosen_var.get())
            if chosen not in selected:
                messagebox.showwarning("Huahua Assistant", "二阶段大类必须来自一阶段出现的两个大类。")
                return
            affix_a = affix_display.get(affix1_var.get())
            affix_b = affix_display.get(affix2_var.get())
            if affix_a is None or affix_b is None:
                messagebox.showwarning("Huahua Assistant", "请选择两个二阶段词缀。")
                return
            self.brush_scenario = BrushChoiceScenario(stage1_options=selected, chosen_stage1=chosen, stage2_options=(affix_a, affix_b))
            self._rerender_side_panel()
            window.destroy()

        def clear_manual() -> None:
            self.brush_scenario = None
            self._rerender_side_panel()
            window.destroy()

        tk.Button(buttons, text="应用", command=apply_manual, width=10).pack(side=tk.RIGHT)
        tk.Button(buttons, text="清空", command=clear_manual, width=10).pack(side=tk.RIGHT, padx=(0, 8))
        tk.Button(buttons, text="取消", command=window.destroy, width=10).pack(side=tk.RIGHT, padx=(0, 8))

    def _tile_photo(self, paint: Paint, size: int, *, voided: bool = False) -> ImageTk.PhotoImage | None:
        """一格图（品质框 + 家族图案）的 Tk 图片；画不出时返回 ``None``。

        缓存键包含像素边长：窗口尺寸一变就得按新尺寸重合成，复用旧图会糊。
        """
        category = paint.category.value if paint.category is not None else None
        level = int(paint.level) if paint.level else None
        # 宝箱常常只有 is_treasure 没有子类型：目录里它就叫「宝箱」，补上才能取到图标。
        subtype = paint.subtype or ("宝箱" if paint.is_treasure else None)
        key = (category, subtype, level, int(size), bool(voided))
        cached = self._tile_photo_cache.get(key)
        if cached is not None:
            return cached
        image = compose_icon(category, subtype, level, size=int(size), voided=voided)
        if image is None:
            return None
        try:
            photo = ImageTk.PhotoImage(image)
        except tk.TclError:
            return None
        if len(self._tile_photo_cache) > 256:
            # 换局/改窗口会攒出大量尺寸变体，别让它无限长。
            self._tile_photo_cache.clear()
        self._tile_photo_cache[key] = photo
        return photo

    def _draw_chip(
        self,
        x: float,
        y: float,
        text: str,
        *,
        fill: str,
        fg: str,
        anchor: str = tk.SE,
        font_size: int = 9,
    ) -> None:
        """带底色的角标：先写字再按 bbox 补一块底、压到字下面。

        Tk 画布没有「圆角矩形」，也不支持给文字加描边；深色棋盘上直接把
        白字压在彩色框上会看不清，所以必须垫一块实心底。
        """
        item = self.canvas.create_text(
            x,
            y,
            text=text,
            fill=fg,
            anchor=anchor,
            font=("Microsoft YaHei UI", font_size, "bold"),
        )
        box = self.canvas.bbox(item)
        if box is None:
            return
        rect = self.canvas.create_rectangle(
            box[0] - 3,
            box[1] - 1,
            box[2] + 3,
            box[3] + 1,
            fill=fill,
            outline="",
        )
        self.canvas.tag_lower(rect, item)

    def _draw_grid(self, board: BoardObservation | None, model_state: PuzzleState | None = None) -> None:
        self.board_cell_rects = {}
        self.backpack_slot_rects = {}
        self.brush_stage1_rects = {}
        self.brush_affix_rects = {}
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        rule_panel_w = 260 if model_state and model_state.current_rule and width >= 760 else 0
        rule_panel_gap = 18 if rule_panel_w else 0
        backpack_width = max(320, width - 48)
        _, backpack_rows, _, backpack_slot_h, backpack_gap = _backpack_layout(
            len(model_state.backpack) if model_state else 0,
            backpack_width,
        )
        summary_reserve = 30 if model_state and model_state.current_rule else 0
        backpack_reserve = 0
        if backpack_rows:
            backpack_reserve = 52 + backpack_rows * backpack_slot_h + max(0, backpack_rows - 1) * backpack_gap
        board_width = max(300, width - 48 - rule_panel_w - rule_panel_gap)
        size = min(board_width, height - backpack_reserve - summary_reserve - 48)
        size = max(260, min(size, 520))
        used_width = size + rule_panel_w + rule_panel_gap
        left = max(24, (width - used_width) // 2)
        top = 42 + summary_reserve
        cell = size / 6
        rec_by_cell = {(rec.row, rec.col): rec for rec in self.recommendations}
        plan_by_cell = {(step.row, step.col): index for index, step in enumerate(self.install_plan.steps, start=1)} if self.install_plan else {}
        best_cell = (self.recommendations[0].row, self.recommendations[0].col) if self.recommendations else None
        scope_cells = set(current_scope_cells(model_state, include_anchor=True)) if model_state else set()
        anchor_cell = model_state.current_anchor if model_state else None
        # 「模型推算 vs 画面识别」对不上的格子：棋盘上要能一眼看到是哪几格。
        mismatch_by_cell = {(item.row, item.col): item for item in self.level_reconcile.mismatches}

        if model_state and model_state.current_rule:
            self._draw_rule_summary(model_state, width / 2, 18)
            if rule_panel_w:
                self._draw_current_rule_card(
                    model_state,
                    left + size + rule_panel_gap,
                    top,
                    rule_panel_w,
                    min(max(size, 360), height - top - 24),
                )

        for col in range(6):
            x = left + (col + 0.5) * cell
            self.canvas.create_text(
                x,
                top - 15,
                text=f"c{col + 1}",
                fill="#9fb4c8",
                font=("Microsoft YaHei UI", 9, "bold"),
            )
        for row in range(6):
            y = top + (row + 0.5) * cell
            self.canvas.create_text(
                left - 14,
                y,
                text=f"r{row + 1}",
                fill="#9fb4c8",
                font=("Microsoft YaHei UI", 9, "bold"),
                anchor=tk.E,
            )
            for col in range(6):
                tile = board.tile(row, col) if board else None
                model_tile = model_state.tile(row, col) if model_state else None
                state = tile.state if tile else (model_tile.state if model_tile else TileState.UNKNOWN)
                x1 = left + col * cell
                y1 = top + row * cell
                x2 = left + (col + 1) * cell
                y2 = top + (row + 1) * cell
                self.board_cell_rects[(row, col)] = (x1, y1, x2, y2)
                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=STATE_COLORS[state],
                    outline="#314152",
                    width=2,
                )
                if (row, col) in scope_cells:
                    self.canvas.create_rectangle(
                        x1 + 2,
                        y1 + 2,
                        x2 - 2,
                        y2 - 2,
                        outline="#f2c14e",
                        width=5 if (row, col) == anchor_cell else 3,
                    )
                if (row, col) in plan_by_cell:
                    self.canvas.create_oval(
                        x1 + 9,
                        y1 + 9,
                        x1 + 32,
                        y1 + 32,
                        fill="#f2c14e",
                        outline="#101820",
                        width=2,
                    )
                    self.canvas.create_text(
                        x1 + 20,
                        y1 + 20,
                        text=str(plan_by_cell[(row, col)]),
                        fill="#101820",
                        font=("Microsoft YaHei UI", 10, "bold"),
                    )
                if (row, col) in rec_by_cell:
                    rec = rec_by_cell[(row, col)]
                    outline = "#ffd84d" if (row, col) == best_cell else "#64e6ff"
                    self.canvas.create_rectangle(
                        x1 + 4,
                        y1 + 4,
                        x2 - 4,
                        y2 - 4,
                        outline=outline,
                        width=5 if (row, col) == best_cell else 3,
                    )
                    self.canvas.create_text(
                        x2 - 10,
                        y1 + 14,
                        text=f"r{row + 1}c{col + 1}\n{int(rec.score)}",
                        fill=outline,
                        font=("Microsoft YaHei UI", 8, "bold"),
                        anchor=tk.E,
                        justify=tk.RIGHT,
                    )
                if model_state and current_rule_is_pending(model_state) and (row, col) in rec_by_cell:
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        y2 - 13,
                        text="放初始",
                        fill="#101820" if state not in {TileState.VOID, TileState.UNKNOWN} else "#f2c14e",
                        font=("Microsoft YaHei UI", 8, "bold"),
                    )
                # 界面显示「这一帧在棋盘上真正看到了什么」：模型层会把锚点格的
                # 颜料替换成面板上的当前颜料（那是推断，不是画面），直接画模型值
                # 会让用户看到「图案 A、等级 5」而游戏里明明是另一张图，核对就永远对不上。
                observed_paint = _paint_from_tile_observation_for_display(tile)
                model_paint = model_tile.paint if model_tile else None
                display_paint = observed_paint or model_paint
                stale_paint = bool(
                    board is not None
                    and model_paint is not None
                    and observed_paint is None
                    and state is not TileState.BLANK
                    and state is not TileState.BLOCKED
                    and state is not TileState.VOID
                    and state is not TileState.UNKNOWN
                )
                paint_drawn = False
                if display_paint is not None and state == TileState.PAINT:
                    subtype = display_paint.subtype
                    # 虚空形态：图案被抹掉，只剩品质框。框色此时是**初始颜料**的等级，
                    # 不是这格自己的等级——所以角标仍按读到的 level 写，并在图例里说明。
                    voided = subtype == VOIDED_SUBTYPE
                    photo = self._tile_photo(display_paint, int(cell * 0.94), voided=voided)
                    if photo is not None:
                        self.canvas.create_image((x1 + x2) / 2, (y1 + y2) / 2, image=photo)
                        paint_drawn = True
                        if not voided and tile_icon_stem(subtype) is None:
                            # 有框没图案：家族没认出来。用「?」说清楚，别让它看着像虚空。
                            self.canvas.create_text(
                                (x1 + x2) / 2,
                                (y1 + y2) / 2,
                                text="?",
                                fill="#8a4b00",
                                font=("Microsoft YaHei UI", max(11, int(cell * 0.2)), "bold"),
                            )
                    else:
                        # 连框都画不出来（类别/等级都缺）：退回文字，不能留一格空的。
                        self.canvas.create_text(
                            (x1 + x2) / 2,
                            (y1 + y2) / 2,
                            text=_tile_display_label(state, display_paint, tile),
                            fill="#d7e3ee",
                            font=("Microsoft YaHei UI", 11, "bold"),
                            justify=tk.CENTER,
                        )
                    if stale_paint:
                        # 这一帧没在这格看到颜料，图来自上一帧/模型推断。
                        # 用**橙色虚框 + 左上角问号**标出来：以前打半透明网点，
                        # 结果把图案和等级都糊暗了，反而更难核对。
                        self.canvas.create_rectangle(
                            x1 + 2,
                            y1 + 2,
                            x2 - 2,
                            y2 - 2,
                            outline="#8a4b00",
                            width=3,
                            dash=(5, 3),
                        )
                        self._draw_chip(x1 + 6, y1 + 5, "?", fill="#8a4b00", fg="#ffffff", anchor=tk.NW)
                    # 等级角标：屏幕上的数字（= 识别到的等级），有场地加成时补「→最终」。
                    # 「→最终」里的加成只来自模型（地块加成 + 同种光环），画面上看不见，
                    # 所以必须显式标出来，否则用户没法判断这个数是游戏给的还是软件算的。
                    base_level = int(display_paint.level or 0)
                    bonus_level = (
                        tile_bonus_level(model_tile) + tile_aura_level(model_tile) if model_tile is not None else 0
                    )
                    final_level = min(MAX_LEVEL, base_level + bonus_level) if base_level else 0
                    if not paint_drawn:
                        # 连框都画不出来，已经用文字写了「未识 L4」，再补一个角标就是重复。
                        pass
                    elif voided:
                        # 虚空格的框色是**本局初始颜料**的等级，不是这格自己的。
                        # 写「初N」而不是「LvN」，免得被当成这格的价值。
                        self._draw_chip(x2 - 4, y2 - 4, f"初{base_level}", fill="#3a2a4a", fg="#f6e7ff", font_size=8)
                    elif final_level and base_level and final_level != base_level:
                        self._draw_chip(
                            x2 - 4,
                            y2 - 4,
                            f"Lv{base_level}→{final_level}",
                            fill="#f2c14e",
                            fg="#101820",
                            font_size=8,
                        )
                    elif base_level:
                        self._draw_chip(x2 - 4, y2 - 4, f"Lv{base_level}", fill="#101820", fg="#ffffff")
                else:
                    label = _tile_display_label(state, display_paint, tile)
                    if stale_paint:
                        label = f"?{label}"
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=label,
                        fill="#ffffff" if state in {TileState.VOID, TileState.UNKNOWN} else "#101820",
                        font=("Microsoft YaHei UI", 12, "bold"),
                        justify=tk.CENTER,
                    )
                    # 颜料落到虚空后只剩底色 + 等级光环：这格没有颜料，光一个「虚」字
                    # 看不出它还挂着笔触效果（用户 2026-11-24：「只有虚字，无其他任何
                    # 东西，不行」）。先把环画出来，再在右下补等级。
                    #
                    # 观测和模型两边都取：某一帧漏读时另一边的值还在，不至于闪烁。
                    retained_aura = max(
                        float(getattr(tile, "retained_aura", 0.0) or 0.0),
                        float(getattr(model_tile, "retained_aura", 0.0) or 0.0),
                    )
                    if retained_aura > 0:
                        retained_level = getattr(tile, "retained_aura_level", None)
                        if retained_level is None:
                            retained_level = getattr(model_tile, "retained_aura_level", None)
                        aura_color = AURA_LEVEL_COLORS.get(retained_level, AURA_UNKNOWN_COLOR)
                        inset = max(6, int(cell * 0.10))
                        self.canvas.create_oval(
                            x1 + inset,
                            y1 + inset,
                            x2 - inset,
                            y2 - inset,
                            outline=aura_color,
                            width=max(3, int(cell * 0.05)),
                        )
                        self._draw_chip(
                            x2 - 4,
                            y2 - 4,
                            f"环L{retained_level}" if retained_level else "环",
                            fill=aura_color,
                            fg="#101820",
                            font_size=8,
                        )
                if paint_drawn and (row, col) in scope_cells:
                    # 图标是彩色的，作用范围那圈金色细线会被盖住：重描一次。
                    self.canvas.create_rectangle(
                        x1 + 2,
                        y1 + 2,
                        x2 - 2,
                        y2 - 2,
                        outline="#f2c14e",
                        width=5 if (row, col) == anchor_cell else 3,
                    )
                if (row, col) == anchor_cell:
                    # 模型认定的「初始颜料」格。它对不对直接决定作用范围，
                    # 必须在棋盘上单独标出来，而不是只体现在旁边那行文字里。
                    self.canvas.create_text(
                        x1 + 8,
                        y2 - 10,
                        text="初",
                        fill="#b3121f",
                        font=("Microsoft YaHei UI", 9, "bold"),
                        anchor=tk.W,
                    )
                mismatch = mismatch_by_cell.get((row, col))
                if mismatch is not None:
                    # 模型推算与画面识别对不上：红虚框 + 「预N≠实M」。
                    # 这是唯一一种「软件自己知道可能算错了」的格子，必须显眼。
                    self.canvas.create_rectangle(
                        x1 + 1,
                        y1 + 1,
                        x2 - 1,
                        y2 - 1,
                        outline="#ff4d4f",
                        width=3,
                        dash=(6, 3),
                    )
                    self._draw_chip(
                        x1 + 5,
                        y1 + 4,
                        f"预{mismatch.predicted}≠实{mismatch.observed}",
                        fill="#b3121f",
                        fg="#ffffff",
                        anchor=tk.NW,
                        font_size=8,
                    )

        self._draw_backpack_bar(model_state, 24, top + size + 18, width - 48)

    def _draw_current_rule_card(
        self,
        model_state: PuzzleState,
        left: float,
        top: float,
        width: float,
        height: float,
    ) -> None:
        rule = model_state.current_rule
        if rule is None:
            return
        pending = current_rule_is_pending(model_state)
        title = "待放置初始颜料" if pending else "当前初始颜料"
        outline = "#ff5d5d" if pending else "#f2c14e"
        fill = "#111b24"
        self.canvas.create_rectangle(
            left,
            top,
            left + width,
            top + height,
            fill=fill,
            outline=outline,
            width=4 if pending else 3,
        )
        self.canvas.create_text(
            left + 12,
            top + 14,
            text=title,
            fill="#ffffff",
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor=tk.W,
        )
        if self.manual_correction_applied:
            self.canvas.create_text(
                left + width - 12,
                top + 14,
                text="人工修正",
                fill="#f2c14e",
                font=("Microsoft YaHei UI", 9, "bold"),
                anchor=tk.E,
            )

        icon_x1 = left + 16
        icon_y1 = top + 42
        icon_x2 = left + 112
        icon_y2 = icon_y1 + 86
        icon_fill = "#ffd84d" if rule.paint.category.value == "gameplay" else "#b983ff"
        self.canvas.create_rectangle(icon_x1, icon_y1, icon_x2, icon_y2, fill=icon_fill, outline="#101820", width=3)
        self.canvas.create_text(
            (icon_x1 + icon_x2) / 2,
            icon_y1 + 30,
            text=_paint_short_label(rule.paint),
            fill="#101820",
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        self.canvas.create_text(
            (icon_x1 + icon_x2) / 2,
            icon_y1 + 62,
            text=f"Lv{rule.paint.level}",
            fill="#101820",
            font=("Microsoft YaHei UI", 17, "bold"),
        )

        lines = [
            f"方向: {_direction_display_label(rule.direction)}",
            f"目标: {_target_display_label(rule.target_object)}",
            f"匹配: {rule.match_current}/{rule.match_required}",
            f"效果: 提升{rule.upgrade_amount}级",
        ]
        y = icon_y1 + 8
        for line in lines:
            self.canvas.create_text(
                icon_x2 + 12,
                y,
                text=line,
                fill="#d7e3ee",
                font=("Microsoft YaHei UI", 9),
                anchor=tk.W,
            )
            y += 21

        body_top = icon_y2 + 18
        if pending and self.brush_scenario is not None:
            body_top = self._draw_brush_choice_panel(left + 12, body_top, width - 24)

        hint = self._current_rule_hint(pending)
        self.canvas.create_rectangle(
            left + 12,
            top + height - 44,
            left + width - 12,
            top + height - 14,
            fill="#203044",
            outline="#314152",
            width=1,
        )
        self.canvas.create_text(
            left + width / 2,
            top + height - 29,
            text=hint,
            fill="#f5f7fa",
            font=("Microsoft YaHei UI", 9, "bold"),
        )

    def _draw_brush_choice_panel(self, left: float, top: float, width: float) -> float:
        scenario = self.brush_scenario
        if scenario is None:
            return top

        self.canvas.create_text(
            left,
            top,
            text="二阶段选择",
            fill="#ffffff",
            font=("Microsoft YaHei UI", 10, "bold"),
            anchor=tk.W,
        )
        y = top + 24
        self.canvas.create_text(
            left,
            y,
            text="1. 选择大类",
            fill="#95abc1",
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
        )
        y += 16
        button_gap = 8
        button_w = (width - button_gap) / 2
        button_h = 56
        learning_stats = self._learning_stats_cache if self.current_scene is not None else None
        stage1_recs = (
            recommend_brush_types(self.current_state, scenario.stage1_options, learning_stats=learning_stats)
            if self.current_state
            else []
        )
        score_by_type = {rec.brush_type: rec.score for rec in stage1_recs}
        reason_by_type = {rec.brush_type: rec.reason for rec in stage1_recs}
        best_type = stage1_recs[0].brush_type if stage1_recs else None
        for index, brush_type in enumerate(scenario.stage1_options):
            x1 = left + index * (button_w + button_gap)
            x2 = x1 + button_w
            selected = brush_type == self.selected_brush_type
            self.brush_stage1_rects[brush_type] = (x1, y, x2, y + button_h)
            rec_mark = "推荐 " if brush_type == best_type else ""
            label = (
                f"{rec_mark}{brush_code(brush_type)} {_brush_label(brush_type)}\n"
                f"评分 {score_by_type.get(brush_type, 0):.0f}\n"
                f"{reason_by_type.get(brush_type, '按当前局面估算')}"
            )
            self._draw_canvas_button(
                x1,
                y,
                x2,
                y + button_h,
                label,
                selected=selected,
                disabled=self.selected_affix is not None,
                font_size=8,
            )

        y += button_h + 14
        if self.selected_brush_type is None:
            self.canvas.create_text(
                left,
                y,
                text="先点上方一个大类，才会出现2个词缀",
                fill="#f2c14e",
                font=("Microsoft YaHei UI", 9),
                anchor=tk.W,
            )
            return y + 26

        self.canvas.create_text(
            left,
            y,
            text="2. 选择词缀",
            fill="#95abc1",
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
        )
        y += 18
        affix_recs = (
            recommend_affixes(self.current_state, scenario.stage2_options, learning_stats=learning_stats)
            if self.current_state
            else []
        )
        score_by_affix = {rec.affix: rec.score for rec in affix_recs}
        reason_by_affix = {rec.affix: rec.reason for rec in affix_recs}
        for index, affix in enumerate(scenario.stage2_options):
            selected = affix == self.selected_affix
            x1 = left
            x2 = left + width
            y2 = y + 52
            self.brush_affix_rects[index] = (x1, y, x2, y2)
            score_text = f"评分 {score_by_affix.get(affix, 0):.0f}"
            label = f"{affix_code(affix) or '?'} {affix.name}\n{score_text} | {reason_by_affix.get(affix, '按当前局面估算收益')}"
            self._draw_canvas_button(
                x1,
                y,
                x2,
                y2,
                label,
                selected=selected,
                disabled=self.selected_affix is not None and not selected,
                font_size=8,
            )
            y += 60

        return y

    def _draw_canvas_button(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        text: str,
        *,
        selected: bool = False,
        disabled: bool = False,
        font_size: int = 9,
    ) -> None:
        fill = "#f2c14e" if selected else "#223248" if not disabled else "#18222e"
        outline = "#ffffff" if selected else "#3e536b"
        text_fill = "#101820" if selected else "#d7e3ee" if not disabled else "#738497"
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=2)
        self.canvas.create_text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            text=text,
            fill=text_fill,
            font=("Microsoft YaHei UI", font_size, "bold"),
            justify=tk.CENTER,
            width=max(20, int(x2 - x1 - 12)),
        )

    def _current_rule_hint(self, pending: bool) -> str:
        if not pending:
            return "金框范围内安装背包颜料"
        if self.brush_scenario is None:
            return "点击左侧推荐格放置初始颜料"
        if self.selected_brush_type is None:
            return "先选一阶段大类，再点重新快速获取"
        if self.selected_affix is None:
            return "二阶段词条出现后选1条"
        return "已选词条，点击左侧推荐格放置初始颜料"

    def _draw_rule_summary(self, model_state: PuzzleState, center_x: float, top: float) -> None:
        rule = model_state.current_rule
        if rule is None:
            return
        if current_rule_is_pending(model_state):
            summary = (
                f"待放置初始颜料 Lv{rule.paint.level} {_paint_short_label(rule.paint)} | "
                f"目标: {_direction_display_label(rule.direction)} {_target_display_label(rule.target_object)} "
                f"(0/{rule.match_required}) | 请选择空白地块"
            )
        else:
            summary = (
                f"初始颜料 Lv{rule.paint.level} {_paint_short_label(rule.paint)} | "
                f"目标: {_direction_display_label(rule.direction)} {_target_display_label(rule.target_object)} "
                f"({rule.match_current}/{rule.match_required}) | "
                f"效果: 提升{rule.upgrade_amount}级"
            )
        self.canvas.create_text(
            center_x,
            top,
            text=summary,
            fill="#f5f7fa",
            font=("Microsoft YaHei UI", 10, "bold"),
            anchor=tk.N,
        )

    def _draw_backpack_bar(self, model_state: PuzzleState | None, left: float, top: float, width: float) -> None:
        if model_state is None or not model_state.backpack:
            return
        per_row, rows, slot_w, slot_h, gap = _backpack_layout(len(model_state.backpack), width)
        row_width = per_row * slot_w + max(0, per_row - 1) * gap
        start_x = left + max(0, (width - row_width) / 2)
        stale = self.backpack_frame_visible is False
        title = f"背包 {len(model_state.backpack)} 格"
        title += " · 沿用上次读数（本帧没看到背包面板）" if stale else " · 本帧实拍"
        self.canvas.create_text(
            left,
            top,
            text=title,
            fill="#f2c14e" if stale else "#d7e3ee",
            font=("Microsoft YaHei UI", 10, "bold"),
            anchor=tk.W,
        )
        if model_state.current_rule is not None:
            action_hint = (
                "新的初始颜料待放置：先点棋盘空白格"
                if current_rule_is_pending(model_state)
                else f"已按 {_target_display_label(model_state.current_rule.target_object)} 目标优先排序，直接按安装计划目标地块放置"
            )
            self.canvas.create_text(
                left,
                top + 18,
                text=action_hint,
                fill="#95abc1",
                font=("Microsoft YaHei UI", 9),
                anchor=tk.W,
            )
        planned = Counter(step.paint for step in self.install_plan.steps) if self.install_plan else Counter()
        icon_size = max(18, slot_h - 24)
        for index, paint in enumerate(model_state.backpack):
            row = index // per_row
            col = index % per_row
            x1 = start_x + col * (slot_w + gap)
            y1 = top + 38 + row * (slot_h + gap)
            x2 = x1 + slot_w
            y2 = y1 + slot_h
            self.backpack_slot_rects[index] = (x1, y1, x2, y2)
            planned_here = planned[paint] > 0
            if planned_here:
                planned[paint] -= 1
            fill = "#243349" if planned_here else "#101a25"
            selected_here = self.selected_backpack_index == index
            outline = "#64e6ff" if selected_here else "#f2c14e" if planned_here else "#314152"
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=4 if selected_here else 3 if planned_here else 2)
            # 背包格也画成游戏里那样：品质框 + 图案。文字只剩序号和等级，
            # 因为「这格是什么」看图案最快，写两行字反而挤掉了图。
            photo = self._tile_photo(paint, icon_size)
            if photo is not None:
                self.canvas.create_image((x1 + x2) / 2, y1 + 4 + icon_size / 2, image=photo)
            else:
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    y1 + 6 + icon_size / 2,
                    text=_backpack_slot_label(paint),
                    fill="#d7e3ee",
                    font=("Microsoft YaHei UI", 8, "bold"),
                )
            self.canvas.create_text(
                x1 + 4,
                y2 - 3,
                text=f"#{index + 1}",
                fill="#8fa6ba" if not planned_here else "#101820",
                font=("Microsoft YaHei UI", 7, "bold"),
                anchor=tk.SW,
            )
            self._draw_chip(x2 - 3, y2 - 2, f"Lv{paint.level}", fill="#101820" if not planned_here else "#f2c14e", fg="#ffffff" if not planned_here else "#101820", font_size=7)

    def _write_regions(self, scene: SceneObservation | None) -> None:
        self.region_text.configure(state=tk.NORMAL)
        self.region_text.delete("1.0", tk.END)
        if scene is None:
            self.region_text.insert(tk.END, "识别区域\n\n")
            self._write_perf_debug()
            self.region_text.insert(tk.END, "无")
            self.region_text.configure(state=tk.DISABLED)
            return

        region_names = {
            RegionKind.CURRENT_PAINT: "当前初始颜料",
            RegionKind.BACKPACK: "背包",
            RegionKind.BRUSH_OPTIONS: "笔触选项",
            RegionKind.BRUSH_CARD_1: "笔触卡片 1",
            RegionKind.BRUSH_CARD_2: "笔触卡片 2",
            RegionKind.CONFIRM_BUTTON: "确认按钮",
            RegionKind.START_DRAWING: "开始绘制",
            RegionKind.FINISH_DRAWING: "完成绘制",
            RegionKind.FORCE_COMPLETE: "强制完成",
            RegionKind.REFRESH_AFFIX: "刷新词缀",
        }
        self.region_text.insert(tk.END, "识别区域\n\n")
        self._write_perf_debug()
        if self.manual_correction_applied:
            self.region_text.insert(tk.END, "已应用人工纠错\n当前局面已使用纠错值重算；上方识别区域仍保留原始截图识别值。\n\n")
        self.region_text.insert(
            tk.END,
            f"棋盘\nx={scene.board.rect.x} y={scene.board.rect.y} "
            f"w={scene.board.rect.w} h={scene.board.rect.h}\n"
            f"置信={scene.board.score:.2f}\n\n",
        )
        self._write_coordinate_debug(scene)
        for region in scene.regions:
            name = region_names.get(region.kind, region.kind.value)
            text = (
                f"{name}\n"
                f"x={region.rect.x} y={region.rect.y} "
                f"w={region.rect.w} h={region.rect.h}\n"
                f"置信={region.confidence:.2f}\n\n"
            )
            self.region_text.insert(tk.END, text)
        if scene.current_paint is not None:
            paint = scene.current_paint
            level = paint.level if paint.level is not None else "?"
            self.region_text.insert(tk.END, "当前初始颜料\n")
            self.region_text.insert(
                tk.END,
                f"类型={paint.category.value} ({paint.category_confidence:.2f})\n"
                f"小类={paint.subtype}\n"
                f"宝箱={paint.is_treasure}\n"
                f"等级={level} ({paint.level_confidence:.2f})\n"
                f"目标={paint.target_object.value}\n"
                f"方向={paint.direction.value}\n"
                f"升级={paint.upgrade_amount}\n"
                f"匹配={paint.match_current}/{paint.match_required}\n"
                f"提示={paint.raw_text_hint}\n"
                f"关键词={_format_keyword_scores(paint.keyword_scores)}\n"
                "截图=runtime/current_paint_*.png\n\n",
            )
        if scene.brush_options is not None:
            brush = scene.brush_options
            self.region_text.insert(tk.END, "笔触选项识别\n")
            self.region_text.insert(
                tk.END,
                f"一阶段={', '.join(brush.stage1_options) or '-'}\n"
                f"已选={brush.chosen_stage1 or '-'}\n"
                f"二阶段={'; '.join(brush.stage2_options) or '-'}\n"
                f"置信={brush.confidence:.2f}\n"
                f"一阶段评分={_format_keyword_scores(brush.stage1_scores)}\n"
                f"二阶段评分={_format_keyword_scores(brush.stage2_scores)}\n"
                "截图=runtime/brush_options_panel.png\n\n",
            )
        if scene.backpack is not None:
            truncated = "（满屏，可能还有 1 格在屏外，落一枚后再识别）" if self.backpack_accumulator.may_be_truncated else ""
            self.region_text.insert(
                tk.END,
                f"背包识别\n"
                f"可见={len(scene.backpack.paints)} 累计={self.backpack_accumulator.total}{truncated}\n"
                f"置信={scene.backpack.confidence:.2f}\n"
                "支持手动滚动：你在游戏里滚动背包后，再点完整识别/快速获取\n\n",
            )
        if self.last_capture_quality is not None:
            quality = self.last_capture_quality
            status = "通过" if quality.get("ok") else "暂缓"
            self.region_text.insert(
                tk.END,
                f"识别质量：{status} 评分={quality.get('score', 0.0):.2f}\n"
                f"{quality.get('reason', '-')}\n\n",
            )
        if self.last_rule_anomalies:
            self.region_text.insert(tk.END, "规则异常提醒\n")
            for anomaly in self.last_rule_anomalies[:4]:
                self.region_text.insert(tk.END, f"- {_format_rule_anomaly(anomaly)}\n")
            blocked = any(item.get("severity") == "block" for item in self.last_rule_anomalies)
            note = "warn/info 仅记录提醒；block 会阻止本次进入高分轨迹。"
            if blocked:
                note = "存在 block 级规则异常，本次截图不会进入高分轨迹；建议复查识别或补充规则。"
            self.region_text.insert(tk.END, f"{note}\n\n")
        self._write_learning_stats()
        self._write_current_terrain_bucket_hint()
        self._write_backpack()
        self._write_level_ledger()
        self._write_install_plan()
        self._write_recommendations()
        self._write_brush_recommendations()
        self._write_special_actions()
        self.region_text.configure(state=tk.DISABLED)

    def _write_perf_debug(self) -> None:
        self.region_text.insert(tk.END, "性能面板\n")
        for line in _format_perf_details(self.last_perf):
            self.region_text.insert(tk.END, f"{line}\n")
        self.region_text.insert(tk.END, "\n")

    def _write_coordinate_debug(self, scene: SceneObservation) -> None:
        window = self.last_capture_window
        capture_size = self.last_capture_size
        board_rect = scene.board.rect
        board_screen_rect = self._screen_rect_from_capture_rect(board_rect)
        board_center = self._screen_point_for_capture_rect(board_rect)

        if window is None:
            window_text = "无窗口信息"
        else:
            window_text = f"({window.left},{window.top}) {window.width}x{window.height}"
        if capture_size is None:
            capture_text = "未知"
            scale_text = "未知"
        else:
            capture_w, capture_h = capture_size
            capture_text = f"{capture_w}x{capture_h}"
            if window is not None and capture_w > 0 and capture_h > 0:
                scale_text = f"x={window.width / capture_w:.4f} y={window.height / capture_h:.4f}"
            else:
                scale_text = "无窗口，使用截图坐标"

        self.region_text.insert(
            tk.END,
            "坐标调试\n"
            f"窗口={window_text}\n"
            f"截图={capture_text}\n"
            f"缩放={scale_text}\n"
            f"棋盘截图=x{board_rect.x} y{board_rect.y} w{board_rect.w} h{board_rect.h}\n",
        )
        if board_screen_rect is not None:
            x1, y1, x2, y2 = board_screen_rect
            self.region_text.insert(tk.END, f"棋盘屏幕=({x1},{y1})-({x2},{y2})\n")
        if board_center is not None:
            self.region_text.insert(tk.END, f"棋盘中心=({board_center[0]},{board_center[1]})\n")
        for index, rec in enumerate(self.recommendations[:2], start=1):
            point = self._screen_point_for_board_cell(rec.row, rec.col)
            if point is None:
                continue
            self.region_text.insert(
                tk.END,
                f"推荐{index}=r{rec.row + 1}c{rec.col + 1} -> 屏幕({point[0]},{point[1]})\n",
            )
        if scene.backpack is not None and scene.backpack.paints:
            slot_point = self._screen_point_for_backpack_slot(0)
            if slot_point is not None:
                self.region_text.insert(tk.END, f"背包#1=屏幕({slot_point[0]},{slot_point[1]})\n")
        if scene.brush_options is not None:
            brush_point = self._screen_point_for_capture_rect(scene.brush_options.panel_rect)
            if brush_point is not None:
                self.region_text.insert(tk.END, f"词条面板中心=屏幕({brush_point[0]},{brush_point[1]})\n")
        self.region_text.insert(tk.END, "\n")

    def _write_simulation(self) -> None:
        self.region_text.configure(state=tk.NORMAL)
        self.region_text.delete("1.0", tk.END)
        self.region_text.insert(tk.END, "模拟棋局\n\n")
        if self.current_state and self.current_state.current_rule:
            rule = self.current_state.current_rule
            treasure = "是" if rule.paint.is_treasure else "否"
            pending = current_rule_is_pending(self.current_state)
            anchor_text = "待选择" if pending or self.current_state.current_anchor is None else f"({self.current_state.current_anchor[0] + 1}, {self.current_state.current_anchor[1] + 1})"
            scope_count = 0 if pending else len(current_scope_cells(self.current_state, include_anchor=True))
            note = "请选择一个空白地块放置新的初始颜料" if pending else f"每安装1次都会立即结算，不用等{rule.match_required}/{rule.match_required}"
            self.region_text.insert(
                tk.END,
                f"当前初始颜料\n"
                f"类型={_paint_display_label(rule.paint)}\n"
                f"宝箱={treasure}\n"
                f"方向={_direction_display_label(rule.direction)}\n"
                f"目标={_target_display_label(rule.target_object)}\n"
                f"效果=范围内目标颜料提升{rule.upgrade_amount}级\n"
                f"进度={rule.match_current}/{rule.match_required}\n"
                f"锚点={anchor_text}\n"
                f"金框覆盖={scope_count}格\n"
                f"说明={note}\n\n",
            )
        self._write_backpack()
        self._write_level_ledger()
        self._write_current_terrain_bucket_hint()
        self._write_install_plan()
        self._write_recommendations()
        self._write_brush_recommendations()
        self._write_special_actions()
        self.region_text.configure(state=tk.DISABLED)

    def _write_recommendations(self) -> None:
        finish = assess_finish_mode(self.current_state) if self.current_state is not None else None
        if finish is not None and not finish.reachable:
            # 目标已经凑不满：这时候再列落点等于误导。玩家要决定的是
            # 「要不要强制调色」和「什么时候点完成绘制」，直接给结论。
            self.region_text.insert(tk.END, f"⚠ 收尾建议\n{finish.headline}\n")
            for line in finish.actions:
                self.region_text.insert(tk.END, f"  · {line.replace('**', '')}\n")
            self.region_text.insert(tk.END, "\n")
            return
        if not self.recommendations:
            reason = "背包为空 / 真实模式尚未识别背包" if self.current_state and not self.current_state.backpack else "暂无"
            self.region_text.insert(tk.END, f"单步预览\n{reason}\n\n")
            return
        if finish is not None:
            self.region_text.insert(tk.END, f"{finish.headline}\n")
        pending = self.current_state is not None and current_rule_is_pending(self.current_state)
        self.region_text.insert(tk.END, "初始颜料放置推荐\n" if pending else "单步预览（安装即生效）\n")
        for index, rec in enumerate(self.recommendations, start=1):
            mark = "最佳 " if index == 1 else ""
            action = "放初始" if pending else "安装"
            self.region_text.insert(
                tk.END,
                f"{mark}{index}. {action} r{rec.row + 1} c{rec.col + 1} "
                f"评分={rec.score:.1f} {rec.reason}\n",
            )
            for line in _recommendation_explanation_lines(rec):
                self.region_text.insert(tk.END, f"   {line}\n")
            if self.overlay_var.get():
                screen = self._screen_point_for_board_cell(rec.row, rec.col)
                if screen is not None:
                    self.region_text.insert(tk.END, f"   坐标：棋盘({rec.row + 1},{rec.col + 1}) 屏幕({screen[0]},{screen[1]})\n")
        self.region_text.insert(tk.END, "\n")

    def _screen_point_for_board_cell(self, row: int, col: int) -> tuple[int, int] | None:
        scene = self.current_scene
        if scene is None:
            return None
        tile = scene.board.tile(row, col)
        return self._screen_point_for_capture_rect(tile.rect)

    def _visible_backpack_index_for_paint(self, paint: Paint | None) -> int | None:
        scene = self.current_scene
        if paint is None or scene is None or scene.backpack is None:
            return None
        for index, observed in enumerate(scene.backpack.paints):
            observed_paint = Paint(
                category=observed.category,
                level=observed.level,
                subtype=observed.subtype,
                is_treasure=observed.is_treasure,
            )
            if observed_paint == paint:
                return index
        return None

    def _screen_point_for_backpack_slot(self, index: int) -> tuple[int, int] | None:
        scene = self.current_scene
        if scene is None or scene.backpack is None or not 0 <= index < len(scene.backpack.paints):
            return None
        rect = scene.backpack.paints[index].slot_rect
        return self._screen_point_for_capture_rect(rect)

    def _write_reconcile_lines(self) -> None:
        """写「模型推算 vs 画面识别」的对账结论。

        这一段回答的是用户最容易困惑的那个问题：**软件算的和游戏显示的不一样，到底哪边错**。
        所以先把这一笔的**来源表**摆出来（调色 / 笔触 / 各自的范围与目标类别），
        再写逐格差异——差异行后面还带上该格的来源构成，一眼能看出差在哪个来源上。
        没有对账时也要写一行，否则用户会以为功能没接上。
        """
        report = self.level_reconcile
        self.region_text.insert(tk.END, "模型对账\n")
        if not report.ran:
            self.region_text.insert(tk.END, f"{report.note or '暂无对账'}\n\n")
            return
        source_lines = report.source_lines()
        if source_lines:
            self.region_text.insert(tk.END, "来源（分开算，各自带范围与目标）\n")
            for line in source_lines:
                self.region_text.insert(tk.END, f"  {line}\n")
        if report.caveat:
            # 推演时做过妥协要写出来，否则「模型和画面一致」会被误当成完全确定。
            self.region_text.insert(tk.END, f"推演口径：{report.caveat}\n")
        low, high = report.upgrade_cells
        if high > 0:
            span = f"{low}" if low == high else f"{low}~{high}"
            if report.animation_upgrades is None:
                self.region_text.insert(tk.END, f"这一笔预计升{span}格（等升级动画播完再核）\n")
            else:
                verdict = "吻合" if report.animation_matches else "对不上"
                self.region_text.insert(
                    tk.END,
                    f"这一笔预计升{span}格 / 动画播了{report.animation_upgrades}条（{verdict}）\n",
                )
        if not report.mismatches:
            self.region_text.insert(
                tk.END, f"已推演 {len(report.levels)} 格，模型与画面一致\n\n"
            )
            return
        self.region_text.insert(
            tk.END, f"已推演 {len(report.levels)} 格，{len(report.mismatches)} 格对不上：\n"
        )
        for item in report.mismatches[:8]:
            detail = f"（{item.breakdown_text}）" if item.breakdown_text else ""
            self.region_text.insert(
                tk.END,
                f"r{item.row + 1}c{item.col + 1}{'（新落笔）' if item.kind == 'new' else ''} "
                f"模型{item.predicted} ≠ 画面{item.observed}（差{item.delta:+d}）{detail}\n",
            )
        if len(report.mismatches) > 8:
            self.region_text.insert(tk.END, f"…… 另有 {len(report.mismatches) - 8} 格\n")
        self.region_text.insert(tk.END, "\n")

    def _write_current_level_sources(self) -> None:
        """这一笔的等级来源：调色 / 笔触 / 自身，各自范围与目标类别。

        用户 2026-11-15 第 3 点：来源必须分开算。这里把「哪个来源、什么目标类别、
        覆盖几格、加几级」摊开，再对范围内每个已有颜料格给一行
        「现在 Lv4：调色1 + 笔触1 → 落笔后 Lv6」，对不上时一眼看出差在哪条来源上。
        """
        state = self.current_state
        if state is None or state.current_rule is None:
            return
        affix = self.selected_affix if self.selected_affix is not None else state.active_affix
        sources = level_sources(
            state, affix=affix, self_upgrade=self._self_upgrade_from_choice()
        )
        if not sources:
            return
        self.region_text.insert(tk.END, "这一笔的等级来源\n")
        for source in sources:
            self.region_text.insert(tk.END, f"  {source.describe()}\n")
        cells: list[tuple[int, int]] = []
        for source in sources:
            for cell in source.scope:
                if cell not in cells:
                    cells.append(cell)
        printed = 0
        for row, col in sorted(cells):
            tile = state.tile(row, col)
            if tile.paint is None:
                continue
            parts = [
                (source.label, source.amount)
                for source in sources
                if source.covers(row, col)
                and source_accepts_paint(source, tile.paint, state.current_rule)
            ]
            now = tile_final_level(tile)
            after = min(MAX_LEVEL, now + sum(amount for _label, amount in parts))
            detail = " + ".join(f"{label}{amount}" for label, amount in parts) or "无加成"
            self.region_text.insert(
                tk.END,
                f"  r{row + 1}c{col + 1} {_paint_display_label(tile.paint)} "
                f"Lv{now}：{detail} → Lv{after}\n",
            )
            printed += 1
            if printed >= 10:
                self.region_text.insert(tk.END, "  ……\n")
                break
        self.region_text.insert(tk.END, "\n")

    def _write_level_ledger(self) -> None:
        """把「识别等级」和「软件加成」分开写清楚。

        棋盘上的 ``LvN`` 是**画面直接读到**的数字，游戏里也是这个数；
        ``LvN→M`` 里的 ``M`` 是本地模型叠了「地块等级加成 + 同种颜料光环」之后
        的**生效等级**，游戏界面上不会写出来。复原/复盘时必须能分开看，
        否则「这格到底是游戏给的 4 级，还是我们自己算的 4 级」永远说不清。
        """
        state = self.current_state
        if state is None:
            return
        # 坐标要从下标算：模型层的 ``Tile`` **不带 row/col**（它只是 tiles 元组里的一个元素，
        # 位置就是坐标）。以前这里写 ``tile.row``，只要场上有带加成的格子就会 AttributeError，
        # 而全场加成都是 0 时函数提前 return，问题一直没暴露。
        boosted = [
            (index // BOARD_SIZE, index % BOARD_SIZE, tile)
            for index, tile in enumerate(state.tiles)
            if tile.paint is not None and (tile_bonus_level(tile) + tile_aura_level(tile)) > 0
        ]
        self.region_text.insert(tk.END, "等级口径\n")
        self.region_text.insert(
            tk.END,
            "格子上的 LvN = 画面识别到的等级（游戏里显示的就是这个数）\n"
            "LvN→M = 本地模型再叠加地块加成/同种光环后的生效等级（游戏界面不显示）\n",
        )
        # 先列加成的来源（逐格），再写对账结论——顺序反过来会让「模型对账」那几行
        # 夹在「等级口径」的明细中间，读起来像是同一个列表。
        if not boosted:
            self.region_text.insert(tk.END, "当前没有带加成的地块\n")
        for row, col, tile in boosted:
            base = int(tile.paint.level)
            bonus = tile_bonus_level(tile)
            aura = tile_aura_level(tile)
            total = min(MAX_LEVEL, base + bonus + aura)
            parts = []
            if bonus:
                parts.append(f"地块+{bonus}")
            if aura:
                parts.append(f"光环+{aura}")
            self.region_text.insert(
                tk.END,
                f"r{row + 1}c{col + 1} Lv{base} + {' + '.join(parts)} → {total}\n",
            )
        self.region_text.insert(tk.END, "\n")
        self._write_current_level_sources()
        self._write_reconcile_lines()

    def _write_backpack(self) -> None:
        self.region_text.insert(tk.END, "背包（已按当前目标优先排序；滚动背包后再次完整识别可累计）\n")
        if not self.current_state or not self.current_state.backpack:
            self.region_text.insert(tk.END, "空 / 等待识别\n\n")
            return
        for index, paint in enumerate(self.current_state.backpack, start=1):
            self.region_text.insert(tk.END, f"{index}. {_paint_display_label(paint)}\n")
        self.region_text.insert(tk.END, "\n")

    def _write_current_terrain_bucket_hint(self) -> None:
        if self.current_state is None:
            return
        stats = self._learning_stats_cache
        hint = _current_terrain_bucket_hint(self.current_state, stats)
        if not hint:
            return
        self.region_text.insert(tk.END, "当前地图分桶\n")
        self.region_text.insert(tk.END, f"{hint}\n\n")

    def _write_learning_stats(self) -> None:
        stats = self._learning_stats_cache
        if not stats:
            return
        conflicts = stats.get("event_counts", {}).get("rule_conflict", 0)
        mismatches = sum((stats.get("rule_mismatches") or {}).values())
        anomalies = stats.get("rule_anomalies") or {}
        anomaly_severity = stats.get("rule_anomaly_severity") or {}
        affix_seen = stats.get("affix_seen") or {}
        affix_code_seen = stats.get("affix_code_seen") or {}
        correction_fields = stats.get("correction_fields") or {}
        trajectories = stats.get("trajectories", 0)
        best_score = stats.get("trajectory_best_score")
        quick_steps = stats.get("quick_capture_steps", 0)
        ignored_quick = stats.get("quick_capture_ignored", 0)
        ignored_trajectory = stats.get("trajectory_step_ignored", 0)
        quick_avg = stats.get("quick_capture_avg_elapsed")
        finished_runs = stats.get("finished_runs", 0)
        finished_skipped = stats.get("finished_skipped", 0)
        finished_best = stats.get("finished_best_score")
        high_value_seen = stats.get("high_value_seen", 0)
        high_value_counts = stats.get("high_value_counts") or {}
        compressed_count = stats.get("compressed_run_summaries", 0)
        compressed_kinds = stats.get("compressed_summary_kinds") or {}
        compressed_avg = stats.get("compressed_summary_avg_score")
        compressed_weighted_avg = stats.get("compressed_summary_weighted_avg_score")
        compressed_best = stats.get("compressed_summary_best_score")
        expert_stage = stats.get("expert_sample_stage") if isinstance(stats.get("expert_sample_stage"), dict) else {}
        terrain_prior = stats.get("terrain_prior") if isinstance(stats.get("terrain_prior"), dict) else {}
        inferred_actions = stats.get("inferred_actions") or {}
        inferred_codes = stats.get("inferred_action_codes") or {}
        refresh_events = stats.get("refresh_events") or {}
        prior_samples = stats.get("policy_prior_samples") or {}
        imitation_policy = stats.get("imitation_policy") or {}
        recommendation_shorts = stats.get("recommendation_shorts") or {}
        recommendation_tags = stats.get("recommendation_tags") or {}
        high_value_recommendation_tags = stats.get("high_value_recommendation_tags") or {}
        gate_primary = stats.get("high_score_gate_primary") or {}
        gate_reasons = stats.get("high_score_gate_reasons") or {}
        appearance_rates = stats.get("affix_appearance_rates") or {}
        stage1_rate_text = _format_distribution_summary(appearance_rates.get("stage1"), limit=3)
        stage2_rate_text = _format_distribution_summary(appearance_rates.get("stage2_codes"), limit=3)
        if stage2_rate_text == "-":
            stage2_rate_text = _format_distribution_summary(appearance_rates.get("stage2"), limit=3)
        unknown_text = _format_unknown_affix_summary(stats, limit=2)
        top_affixes = _top_affix_display_items(affix_seen, affix_code_seen, limit=3)
        top_corrections = sorted(correction_fields.items(), key=lambda item: item[1], reverse=True)[:3]
        top_actions = sorted(inferred_actions.items(), key=lambda item: item[1], reverse=True)[:3]
        top_codes = sorted(inferred_codes.items(), key=lambda item: item[1], reverse=True)[:5]
        top_rec_shorts = sorted(recommendation_shorts.items(), key=lambda item: item[1], reverse=True)[:3]
        top_rec_tags = sorted(recommendation_tags.items(), key=lambda item: item[1], reverse=True)[:4]
        top_high_value_tags = sorted(high_value_recommendation_tags.items(), key=lambda item: item[1], reverse=True)[:4]
        affix_text = ", ".join(f"{name}x{count}" for name, count in top_affixes) if top_affixes else "-"
        correction_text = ", ".join(f"{_field_display_name(name)}x{count}" for name, count in top_corrections) if top_corrections else "-"
        action_text = ", ".join(f"{name}x{count}" for name, count in top_actions) if top_actions else "-"
        code_text = ", ".join(f"{name}x{count}" for name, count in top_codes) if top_codes else "-"
        rec_short_text = ", ".join(f"{name}x{count}" for name, count in top_rec_shorts) if top_rec_shorts else "-"
        rec_tag_text = ", ".join(f"{name}x{count}" for name, count in top_rec_tags) if top_rec_tags else "-"
        high_value_tag_text = ", ".join(f"{name}x{count}" for name, count in top_high_value_tags) if top_high_value_tags else "-"
        imitation_text = _format_imitation_policy_summary(imitation_policy)
        gate_text = _format_count_summary(gate_primary, limit=3)
        gate_reason_text = _format_count_summary(gate_reasons, limit=4)
        best_score_text = f"{best_score:.0f}" if isinstance(best_score, (int, float)) else "-"
        quick_avg_text = f"{quick_avg:.2f}s" if isinstance(quick_avg, (int, float)) else "-"
        finished_best_text = f"{finished_best:.0f}" if isinstance(finished_best, (int, float)) else "-"
        compressed_kind_text = _format_count_summary(compressed_kinds, limit=3)
        compressed_avg_text = f"{compressed_avg:.0f}" if isinstance(compressed_avg, (int, float)) else "-"
        compressed_weighted_text = f"{compressed_weighted_avg:.0f}" if isinstance(compressed_weighted_avg, (int, float)) else "-"
        compressed_best_text = f"{compressed_best:.0f}" if isinstance(compressed_best, (int, float)) else "-"
        compressed_terrain_text = _format_terrain_avg(stats.get("compressed_initial_terrain"))
        expert_stage_text = expert_stage.get("label") or "样本不足"
        terrain_prior_text = _format_terrain_prior(terrain_prior)
        self.region_text.insert(
            tk.END,
            "学习记录\n"
            f"事件={stats.get('total_events', 0)} 纠错={stats.get('corrections', 0)} 词条冲突={conflicts} 规则异常={mismatches}\n"
            f"异常校验={anomalies} 严重度={anomaly_severity}\n"
            f"快速获取={quick_steps} 平均={quick_avg_text} 忽略低质={ignored_quick} 信任拦截={ignored_trajectory} 已结束局数={finished_runs} 跳过结算={finished_skipped} 最佳内部评分={finished_best_text}\n"
            f"高价值识别={high_value_seen} 明细={high_value_counts}\n"
            f"{_risk_hint_text(stats)}\n"
            f"高频词条={affix_text}\n"
            f"高分门槛={gate_text} 原因={gate_reason_text}\n"
            f"词条概率：一阶段={stage1_rate_text} 二阶段={stage2_rate_text}\n"
            f"未知/新词条={unknown_text}\n"
            f"高频纠错={correction_text}\n"
            f"高价值轨迹={trajectories} 最佳={best_score_text}\n"
            f"专家样本={expert_stage_text}\n"
            f"压缩摘要={compressed_count} 类型={compressed_kind_text} 均分={compressed_avg_text} 加权={compressed_weighted_text} 最佳={compressed_best_text}\n"
            f"摘要初始地形={compressed_terrain_text}\n"
            f"地形先验={terrain_prior_text}\n"
            f"推断动作={action_text}\n"
            f"动作短码={code_text} 强制={stats.get('forced_events', 0)} 刷新={refresh_events} 升级事件={stats.get('upgrade_events', 0)}\n"
            f"推荐语义={rec_short_text} 标签={rec_tag_text} 高价值标签={high_value_tag_text}\n"
            f"经验样本：落点={prior_samples.get('placements', 0)} 大类={prior_samples.get('brushes', 0)} 词条={prior_samples.get('affixes', 0)}\n"
            f"宝箱占比={stats.get('treasure_ratio', 0.0):.3f}\n"
            "文件=runtime/observations.jsonl, runtime/corrected_observations.jsonl, runtime/high_score_trajectories.jsonl, runtime/stats.json\n\n",
        )

        self.region_text.insert(tk.END, f"模仿分析：{imitation_text}\n\n")

    def _write_install_plan(self) -> None:
        self.region_text.insert(tk.END, "安装计划（逐步模拟，每步立即结算）\n")
        if self.install_plan is None:
            self.region_text.insert(tk.END, "等待当前初始颜料和背包识别\n\n")
            return
        self.region_text.insert(tk.END, f"{'可完成' if self.install_plan.completes_match else '未完成'}: {self.install_plan.reason}\n")
        for index, step in enumerate(self.install_plan.steps, start=1):
            self.region_text.insert(
                tk.END,
                f"{index}. r{step.row + 1} c{step.col + 1} <- {_paint_display_label(step.paint)}\n",
            )
        self.region_text.insert(tk.END, "\n")

    def _write_brush_recommendations(self) -> None:
        if self.current_state is None:
            return
        if current_rule_is_pending(self.current_state) and self.brush_scenario is not None:
            self.region_text.insert(tk.END, "笔触选择流程\n")
            if self.selected_brush_type is None:
                self.region_text.insert(tk.END, "1. 先在右侧选择转锋/堆砌/扩散大类\n2. 点“重新快速获取”读取二阶段词条\n\n")
                return
            if self.selected_affix is None:
                self.region_text.insert(tk.END, f"已选大类：{_brush_label(self.selected_brush_type)}；请点“重新快速获取”后选择二阶段词条\n\n")
                return
        stage1 = (
            self.brush_scenario.stage1_options
            if self.brush_scenario
            else (BrushType.TRANSFORM, BrushType.STACK, BrushType.SPREAD)
        )
        learning_stats = self._learning_stats_cache if self.current_scene is not None else None
        self.region_text.insert(tk.END, "笔触一阶段\n")
        for rec in recommend_brush_types(self.current_state, stage1, learning_stats=learning_stats):
            self.region_text.insert(
                tk.END,
                f"- {brush_code(rec.brush_type)} {_brush_label(rec.brush_type)} 评分={rec.score:.1f} {rec.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

        if not self.brush_scenario:
            self.region_text.insert(tk.END, "笔触二阶段\n等待真实选项识别\n\n")
            return
        self.region_text.insert(tk.END, f"笔触二阶段（{_brush_label(self.brush_scenario.chosen_stage1)}）\n")
        for rec in recommend_affixes(
            self.current_state,
            self.brush_scenario.stage2_options,
            learning_stats=learning_stats,
        ):
            self.region_text.insert(
                tk.END,
                f"- {affix_code(rec.affix) or '?'} {rec.affix.name} 评分={rec.score:.1f} {rec.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

    def _write_special_actions(self) -> None:
        if self.current_state is None:
            return
        resources = self._resource_state()
        learning_stats = self._learning_stats_cache if self.current_scene is not None else None
        actions = recommend_special_actions(
            self.current_state,
            resources,
            self.brush_scenario,
            learning_stats=learning_stats,
        )
        self.region_text.insert(tk.END, "特殊动作\n")
        self.region_text.insert(
            tk.END,
            f"资源：强制={resources.force_remaining}/{MAX_FORCE_TIMES} "
            f"刷新={resources.refresh_remaining}/{PET_REFRESH_BY_LEVEL.get(self.pet_level, 0)} "
            f"宠物Lv{self.pet_level}刷新={'有' if resources.refresh_available else '无'}\n",
        )
        if not actions:
            refresh_state = "可用" if resources.refresh_available else "未启用"
            self.region_text.insert(tk.END, f"- 刷新={refresh_state}；等待可操作局面\n\n")
            return
        for action in actions:
            mark = "建议用" if action.recommended else "暂缓"
            self.region_text.insert(
                tk.END,
                f"- {mark} {action.action.value} 评分={action.score:.1f} {action.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

    @property
    def pet_level(self) -> int:
        """当前宠物等级（1–6），非法值按 1 处理。"""
        try:
            level = int(self.pet_level_var.get())
        except (tk.TclError, ValueError):
            return DEFAULT_PET_LEVEL
        return max(1, min(6, level))

    def _on_pet_level_change(self) -> None:
        """改宠物等级后，把刷新剩余钳到新上限并重绘侧栏。"""
        total = PET_REFRESH_BY_LEVEL.get(self.pet_level, 0)
        self.refresh_available_var.set(total > 0)
        try:
            used = int(self.refresh_used_var.get())
        except (tk.TclError, ValueError):
            used = 0
        self.refresh_used_var.set(max(0, min(total, used)))
        self._rerender_side_panel()

    def _resource_state(self) -> ResourceState:
        force_used = _clamp_int_var(self.force_used_var, 0, MAX_FORCE_TIMES)
        refresh_available = self.refresh_available_var.get()
        # 刷新次数按宠物等级给：Lv1–2 为 0、Lv3–4 为 1、Lv5–6 为 2。
        # 且只能刷新**第二阶段词缀**，第一阶段画笔类型不可刷新。
        refresh_total = PET_REFRESH_BY_LEVEL.get(self.pet_level, 0)
        refresh_used = _clamp_int_var(self.refresh_used_var, 0, refresh_total)
        return ResourceState(
            force_remaining=max(0, MAX_FORCE_TIMES - force_used),
            force_used=force_used,
            refresh_available=refresh_available and refresh_total > 0,
            refresh_remaining=max(0, refresh_total - refresh_used) if refresh_available else 0,
        )

    def _on_close(self) -> None:
        self._destroy_overlay_window()
        if self.auto_follow_after_id is not None:
            self.after_cancel(self.auto_follow_after_id)
            self.auto_follow_after_id = None
        self._stop_auto_follow_pump()
        self._stop_game_log_tick()
        self._stop_game_log_watcher()
        if self._record_badge_after_id is not None:
            try:
                self.after_cancel(self._record_badge_after_id)
            except tk.TclError:
                pass
            self._record_badge_after_id = None
        if self._capture_exclusion_after_id is not None:
            try:
                self.after_cancel(self._capture_exclusion_after_id)
            except tk.TclError:
                pass
            self._capture_exclusion_after_id = None
        # 编号字典是跨局累积的，关窗口前必须落盘，否则这一局新见的局面全丢。
        if self.codebook.dirty:
            try:
                self.codebook.save(CODEBOOK_PATH)
            except OSError:
                pass
        # 窗口内容抓取的会话持有 Direct3D 设备，不释放会在下次启动前一直占着显存。
        close_capture_backends()
        self.mouse_click_monitor.close()
        self.brush_evidence.close()
        self.run_recorder.close()
        self.destroy()


def main() -> None:
    app = HuahuaApp()
    app.mainloop()


def _format_keyword_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "-"
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:4]
    return ", ".join(f"{key}:{value:.2f}" for key, value in top)


def _risk_hint_text(stats: dict) -> str:
    risky = stats.get("risky_fields") or []
    corrections = stats.get("corrections", 0)
    if risky:
        labels = "、".join(_field_display_name(field) for field in risky[:4])
        return f"高风险字段：{labels}。本次建议重点复查这些项。"
    if corrections < 3:
        return "纠错样本不足：先积累几次真实纠错，工具再判断高风险字段。"
    return "暂无高风险字段：按当前纠错样本看识别较稳定。"


def _format_correction_stats(stats: dict) -> str:
    corrections = stats.get("corrections", 0)
    quick_steps = stats.get("quick_capture_steps", 0)
    ignored_quick = stats.get("quick_capture_ignored", 0)
    ignored_trajectory = stats.get("trajectory_step_ignored", 0)
    quick_avg = stats.get("quick_capture_avg_elapsed")
    quick_avg_text = f"{quick_avg:.2f}s" if isinstance(quick_avg, (int, float)) else "-"
    finished_best = stats.get("finished_best_score")
    finished_best_text = f"{finished_best:.0f}" if isinstance(finished_best, (int, float)) else "-"
    lines = [
        f"真实纠错样本：{corrections}",
        _risk_hint_text(stats),
        f"快速获取样本：{quick_steps} 平均耗时={quick_avg_text} 低质忽略={ignored_quick} 信任拦截={ignored_trajectory}",
        f"已结束局数：{stats.get('finished_runs', 0)} 跳过结算={stats.get('finished_skipped', 0)} 最佳内部评分={finished_best_text}",
        f"高价值识别：{stats.get('high_value_seen', 0)} 明细={stats.get('high_value_counts', {})}",
        f"普通/失败局压缩摘要：{stats.get('compressed_run_summaries', 0)} 条 类型={_format_count_summary(stats.get('compressed_summary_kinds') or {}, limit=6)}",
        "",
        "字段准确率",
    ]
    field_accuracy = stats.get("field_accuracy") or {}
    if not field_accuracy:
        lines.append("暂无纠错样本。请真实完整识别后用纠错识别保存几次。")
    else:
        ranked = sorted(
            field_accuracy.items(),
            key=lambda item: (item[1].get("error_rate", 0.0), item[1].get("errors", 0)),
            reverse=True,
        )
        for field, data in ranked:
            samples = data.get("samples", 0)
            errors = data.get("errors", 0)
            accuracy = data.get("accuracy", 0.0)
            error_rate = data.get("error_rate", 0.0)
            risk = " 高风险" if field in (stats.get("risky_fields") or []) else ""
            lines.append(
                f"- {_field_display_name(field)}: 样本{samples} 错{errors} "
                f"准确率{accuracy:.0%} 错误率{error_rate:.0%}{risk}"
            )
    lines.extend(_format_learning_stats_detail_lines(stats))
    lines.extend(_format_terrain_bucket_analysis_lines(stats))
    lines.extend(_format_imitation_policy_detail_lines(stats.get("imitation_policy") or {}))
    lines.extend(
        [
            "",
            f"词缀冲突记录：{stats.get('event_counts', {}).get('rule_conflict', 0)}",
            f"经验规则不一致：{sum((stats.get('rule_mismatches') or {}).values())}",
            f"宝箱识别占比：{stats.get('treasure_ratio', 0.0):.1%}",
            f"真实高分轨迹：{stats.get('trajectories', 0)} 条",
            f"高分轨迹初始快照：{stats.get('trajectory_initial_snapshots', 0)} 条",
            f"快速获取动作：{stats.get('quick_capture_actions', {})}",
            "",
            "说明：这里只统计真实 Capture 后保存的纠错；模拟模式不会进入统计。",
        ]
    )
    return "\n".join(lines)


def _format_learning_stats_detail_lines(stats: dict) -> list[str]:
    appearance_rates = stats.get("affix_appearance_rates") or {}
    stage1 = _format_distribution_summary(appearance_rates.get("stage1"), limit=6)
    stage1_codes = _format_distribution_summary(appearance_rates.get("stage1_codes"), limit=6)
    stage2 = _format_distribution_summary(appearance_rates.get("stage2"), limit=8)
    stage2_codes = _format_distribution_summary(appearance_rates.get("stage2_codes"), limit=8)
    unknown_stage1 = _format_distribution_summary(appearance_rates.get("unknown_stage1"), limit=5)
    unknown_stage2 = _format_distribution_summary(appearance_rates.get("unknown_stage2"), limit=5)
    compressed_avg = stats.get("compressed_summary_avg_score")
    compressed_weighted = stats.get("compressed_summary_weighted_avg_score")
    compressed_best = stats.get("compressed_summary_best_score")
    compressed_avg_text = f"{compressed_avg:.0f}" if isinstance(compressed_avg, (int, float)) else "-"
    compressed_weighted_text = f"{compressed_weighted:.0f}" if isinstance(compressed_weighted, (int, float)) else "-"
    compressed_best_text = f"{compressed_best:.0f}" if isinstance(compressed_best, (int, float)) else "-"
    expert_stage = stats.get("expert_sample_stage") if isinstance(stats.get("expert_sample_stage"), dict) else {}
    terrain_prior = stats.get("terrain_prior") if isinstance(stats.get("terrain_prior"), dict) else {}
    return [
        "",
        "高分轨迹门槛",
        f"- 主门槛：{_format_count_summary(stats.get('high_score_gate_primary') or {}, limit=8)}",
        f"- 命中原因：{_format_count_summary(stats.get('high_score_gate_reasons') or {}, limit=10)}",
        f"- 专家阶段：{expert_stage.get('label') or '样本不足'} Top20门槛={expert_stage.get('top20_threshold') or '-'}",
        f"- 初始快照：{stats.get('trajectory_initial_snapshots', 0)} 条 地形均值={_format_terrain_avg(stats.get('trajectory_initial_terrain'))}",
        f"- 地形先验：{_format_terrain_prior(terrain_prior)}",
        f"- 地形分桶：{_format_terrain_buckets(terrain_prior, limit=5)}",
        "",
        "普通/失败局压缩摘要",
        f"- 样本：{stats.get('compressed_run_summaries', 0)} 条 类型={_format_count_summary(stats.get('compressed_summary_kinds') or {}, limit=8)}",
        f"- 分数：均分={compressed_avg_text} 加权均分={compressed_weighted_text} 最佳={compressed_best_text}",
        f"- 初始地形：{_format_terrain_avg(stats.get('compressed_initial_terrain'))}",
        f"- 初始可用格：{_format_distribution_summary(stats.get('compressed_initial_open_cells'), limit=8)}",
        f"- 终局统计：{stats.get('compressed_terminal_counts') or {}}",
        "",
        "词条出现概率（真实识别）",
        f"- 一阶段名称：{stage1}",
        f"- 一阶段编码：{stage1_codes}",
        f"- 二阶段名称：{stage2}",
        f"- 二阶段编码：{stage2_codes}",
        "",
        "未知/新词条候选",
        f"- 未知一阶段：{unknown_stage1}",
        f"- 未知二阶段：{unknown_stage2}",
        f"- 候选明细：{_format_unknown_affix_summary(stats, limit=8)}",
    ]


def _format_count_summary(values: dict, *, limit: int) -> str:
    items = []
    for key, count in values.items():
        try:
            parsed = int(count)
        except (TypeError, ValueError):
            continue
        items.append((str(key), parsed))
    if not items:
        return "-"
    items.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(f"{key}x{count}" for key, count in items[:limit])


def _format_distribution_summary(distribution, *, limit: int) -> str:
    if not isinstance(distribution, dict):
        return "-"
    items = distribution.get("items") or []
    if not isinstance(items, list) or not items:
        return "-"
    parts: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "-"))
        count = item.get("count", 0)
        pct = item.get("pct")
        pct_text = f"{float(pct):.1f}%" if isinstance(pct, (int, float)) else "-"
        parts.append(f"{key}x{count}({pct_text})")
    return ", ".join(parts) if parts else "-"


def _format_terrain_avg(summary) -> str:
    if not isinstance(summary, dict):
        return "-"
    samples = int(summary.get("samples", 0) or 0)
    avg = summary.get("avg") if isinstance(summary.get("avg"), dict) else {}
    if samples <= 0 or not avg:
        return "-"
    labels = (
        ("blank", "空白"),
        ("paint", "颜料"),
        ("blocked", "阻挡"),
        ("void", "虚空"),
        ("faded", "失色"),
        ("open_cells", "可用"),
    )
    parts: list[str] = []
    for key, label in labels:
        value = avg.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{label}{float(value):.1f}")
    return f"样本{samples} " + " ".join(parts) if parts else f"样本{samples}"


def _format_terrain_prior(prior) -> str:
    if not isinstance(prior, dict) or not prior:
        return "-"
    samples = prior.get("samples") if isinstance(prior.get("samples"), dict) else {}
    ready = "可用" if prior.get("ready") is True else "样本不足"
    deltas = prior.get("delta_high_minus_compressed") if isinstance(prior.get("delta_high_minus_compressed"), dict) else {}
    open_delta = deltas.get("open_cells")
    blocked_delta = deltas.get("blocked")
    open_text = f"开放差{float(open_delta):+.1f}" if isinstance(open_delta, (int, float)) else "开放差-"
    blocked_text = f"阻挡差{float(blocked_delta):+.1f}" if isinstance(blocked_delta, (int, float)) else "阻挡差-"
    return f"{ready} 高分{samples.get('high', 0)} 压缩{samples.get('compressed', 0)} {open_text} {blocked_text}"


def _format_terrain_buckets(prior, *, limit: int) -> str:
    if not isinstance(prior, dict):
        return "-"
    buckets = prior.get("terrain_buckets") if isinstance(prior.get("terrain_buckets"), dict) else {}
    items = buckets.get("items") if isinstance(buckets.get("items"), list) else []
    if not items:
        return "-"
    parts: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "-"))
        total = int(item.get("total", 0) or 0)
        high = int(item.get("high", 0) or 0)
        pct = item.get("success_pct")
        pct_text = f"{float(pct):.1f}%" if isinstance(pct, (int, float)) else "-"
        parts.append(f"{key} 高{high}/{total}({pct_text})")
    return "；".join(parts) if parts else "-"


def _current_terrain_bucket_hint(state: PuzzleState, stats: dict) -> str:
    key = _terrain_bucket_key_from_state(state)
    label = _terrain_bucket_label(key)
    prior = stats.get("terrain_prior") if isinstance(stats.get("terrain_prior"), dict) else {}
    buckets = prior.get("terrain_buckets") if isinstance(prior.get("terrain_buckets"), dict) else {}
    items = buckets.get("items") if isinstance(buckets.get("items"), list) else []
    item = next((entry for entry in items if isinstance(entry, dict) and entry.get("key") == key), None)
    if item is None:
        return f"{label}：暂无历史样本，先按规则推荐，继续收集真实数据。"
    high = int(item.get("high", 0) or 0)
    total = int(item.get("total", 0) or 0)
    pct = item.get("success_pct")
    pct_text = f"{float(pct):.1f}%" if isinstance(pct, (int, float)) else "-"
    min_samples = 8
    if total < min_samples:
        advice = "样本不足，先按规则推荐，不建议因为这个桶额外消耗刷新/强制。"
    else:
        success_rate = float(item.get("success_rate", 0.0) or 0.0)
        if success_rate >= 0.58:
            advice = "历史表现偏好，可以正常投入资源；仍优先保护宝箱和7级。"
        elif success_rate <= 0.32:
            advice = "历史表现偏差，谨慎使用刷新/强制，除非当前已有宝箱或明确冲7机会。"
        else:
            advice = "历史表现中性，按当前推荐评分执行。"
    return f"{label}：历史高分 {high}/{total}（{pct_text}）。{advice}"


def _terrain_bucket_key_from_state(state: PuzzleState) -> str:
    blank = sum(1 for tile in state.tiles if tile.state == TileState.BLANK)
    paint = sum(1 for tile in state.tiles if tile.state == TileState.PAINT)
    blocked = sum(1 for tile in state.tiles if tile.state == TileState.BLOCKED)
    void = sum(1 for tile in state.tiles if tile.state == TileState.VOID)
    center_open = sum(
        1
        for row, col in ((2, 2), (2, 3), (3, 2), (3, 3))
        if state.tile(row, col).state in {TileState.BLANK, TileState.PAINT}
    )
    return "|".join(
        (
            f"open:{_terrain_bucket_range(blank + paint, (16, 22))}",
            f"block:{_terrain_bucket_range(blocked, (4, 8))}",
            f"void:{_terrain_bucket_range(void, (8, 13))}",
            f"center:{_terrain_bucket_range(center_open, (1, 3))}",
        )
    )


def _terrain_bucket_range(value: int, cuts: tuple[int, int]) -> str:
    low, high = cuts
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def _format_terrain_bucket_analysis_lines(stats: dict) -> list[str]:
    prior = stats.get("terrain_prior") if isinstance(stats.get("terrain_prior"), dict) else {}
    buckets = prior.get("terrain_buckets") if isinstance(prior.get("terrain_buckets"), dict) else {}
    items = buckets.get("items") if isinstance(buckets.get("items"), list) else []
    lines = ["", "地形分桶分析"]
    if not items:
        lines.append("- 暂无地形桶样本；继续积累真实高分局和普通/失败局摘要。")
        return lines

    enough: list[dict] = []
    sparse: list[dict] = []
    min_samples = 8
    for item in items:
        if not isinstance(item, dict):
            continue
        total = int(item.get("total", 0) or 0)
        if total >= min_samples:
            enough.append(item)
        else:
            sparse.append(item)

    invest = [item for item in enough if float(item.get("success_rate", 0.0) or 0.0) >= 0.58]
    caution = [item for item in enough if float(item.get("success_rate", 0.0) or 0.0) <= 0.32]
    neutral = [item for item in enough if item not in invest and item not in caution]
    lines.append(f"- 可分析桶：{len(enough)} 个；样本不足桶：{len(sparse)} 个。")
    lines.append(f"- 值得投入：{_format_terrain_bucket_items(invest, limit=5, advice='可正常投入资源')}")
    lines.append(f"- 谨慎投入：{_format_terrain_bucket_items(caution, limit=5, advice='少用刷新/强制')}")
    lines.append(f"- 中性观察：{_format_terrain_bucket_items(neutral, limit=4, advice='继续观察')}")
    lines.append(f"- 样本不足：{_format_terrain_bucket_items(sparse, limit=5, advice='继续收集')}")
    return lines


def _format_terrain_bucket_items(items: list[dict], *, limit: int, advice: str) -> str:
    if not items:
        return "-"
    parts: list[str] = []
    for item in items[:limit]:
        key = _terrain_bucket_label(str(item.get("key", "-")))
        high = int(item.get("high", 0) or 0)
        total = int(item.get("total", 0) or 0)
        pct = item.get("success_pct")
        gap = item.get("score_gap")
        pct_text = f"{float(pct):.1f}%" if isinstance(pct, (int, float)) else "-"
        gap_text = f" 分差{float(gap):+.0f}" if isinstance(gap, (int, float)) else ""
        parts.append(f"{key} 高{high}/{total}({pct_text}){gap_text} {advice}")
    return "；".join(parts)


def _terrain_bucket_label(key: str) -> str:
    labels = {
        "open": "开放",
        "block": "阻挡",
        "void": "虚空",
        "center": "中心",
        "low": "低",
        "mid": "中",
        "high": "高",
    }
    parts: list[str] = []
    for segment in key.split("|"):
        if ":" not in segment:
            continue
        name, value = segment.split(":", 1)
        parts.append(f"{labels.get(name, name)}{labels.get(value, value)}")
    return "/".join(parts) if parts else key


def _format_unknown_affix_summary(stats: dict, *, limit: int) -> str:
    candidates = stats.get("new_affix_candidates") or []
    if not isinstance(candidates, list) or not candidates:
        total = int(stats.get("unknown_affix_seen_total", 0) or 0)
        return f"暂无（累计{total}）"
    parts: list[str] = []
    for item in candidates[:limit]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "-"))
        count = int(item.get("count", 0) or 0)
        pct = item.get("pct")
        avg_score = item.get("avg_score")
        pct_text = f"{float(pct):.1f}%" if isinstance(pct, (int, float)) else "-"
        score_text = f"均分{float(avg_score):.2f}" if isinstance(avg_score, (int, float)) else "均分-"
        parts.append(f"{name}x{count}({pct_text},{score_text})")
    return ", ".join(parts) if parts else "-"


def _format_imitation_policy_summary(policy: dict) -> str:
    if not isinstance(policy, dict) or not policy:
        return "暂无高分样本"
    samples = policy.get("samples") if isinstance(policy.get("samples"), dict) else {}
    ready = policy.get("ready") if isinstance(policy.get("ready"), dict) else {}
    hits = policy.get("recommendation_hits") if isinstance(policy.get("recommendation_hits"), dict) else {}
    hit_pct = hits.get("hit_pct")
    hit_text = f"{float(hit_pct):.1f}%" if isinstance(hit_pct, (int, float)) else "-"
    ready_names = [str(name) for name, ok in ready.items() if ok]
    ready_text = ",".join(ready_names) if ready_names else "样本未达标"
    return (
        f"轨迹{samples.get('trajectories', 0)} 动作{samples.get('actions', 0)} "
        f"落点{samples.get('placements', 0)} 词条{samples.get('affixes', 0)} "
        f"推荐命中{hit_text} 可用={ready_text}"
    )


def _format_imitation_policy_detail_lines(policy: dict) -> list[str]:
    if not isinstance(policy, dict) or not policy:
        return [
            "",
            "模仿分析",
            "- 暂无可信高分轨迹样本",
        ]
    samples = policy.get("samples") if isinstance(policy.get("samples"), dict) else {}
    thresholds = policy.get("thresholds") if isinstance(policy.get("thresholds"), dict) else {}
    ready = policy.get("ready") if isinstance(policy.get("ready"), dict) else {}
    hits = policy.get("recommendation_hits") if isinstance(policy.get("recommendation_hits"), dict) else {}
    return [
        "",
        "模仿分析（可信高分轨迹）",
        f"- 样本：{samples}",
        f"- 阈值：{thresholds}",
        f"- 就绪：{ready}",
        f"- 推荐命中：决策{hits.get('decisions', 0)} 命中{hits.get('hits', 0)} 命中率{float(hits.get('hit_pct', 0.0) or 0.0):.1f}% Top1={float(hits.get('top1_pct', 0.0) or 0.0):.1f}% Top3={float(hits.get('top3_pct', 0.0) or 0.0):.1f}%",
        f"- 高频动作：{_format_distribution_summary(policy.get('top_actions'), limit=8)}",
        f"- 动作收益：{_format_action_delta_summary(policy.get('action_score_delta'), limit=8)}",
        f"- 高频落点：{_format_distribution_summary(policy.get('top_placement_cells'), limit=10)}",
        f"- 地形区域：{_format_distribution_summary(policy.get('top_placement_zones'), limit=6)}",
        f"- 高频笔触：{_format_distribution_summary(policy.get('top_brush_codes'), limit=6)}",
        f"- 高频词条：{_format_distribution_summary(policy.get('top_affix_codes'), limit=8)}",
        f"- 命中推荐语义：{_format_distribution_summary(policy.get('recommendation_shorts'), limit=6)}",
        f"- 命中推荐标签：{_format_distribution_summary(policy.get('recommendation_tags'), limit=8)}",
    ]


def _format_action_delta_summary(distribution, *, limit: int) -> str:
    if not isinstance(distribution, dict):
        return "-"
    items = distribution.get("items") or []
    if not isinstance(items, list) or not items:
        return "-"
    parts: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "-"))
        count = int(item.get("count", 0) or 0)
        avg_delta = item.get("avg_score_delta")
        delta_text = f"{float(avg_delta):+.1f}" if isinstance(avg_delta, (int, float)) else "-"
        parts.append(f"{key}x{count}({delta_text})")
    return ", ".join(parts) if parts else "-"


def _top_affix_display_items(affix_seen: dict, affix_code_seen: dict, *, limit: int) -> list[tuple[str, int]]:
    if affix_code_seen:
        return sorted(
            ((affix_display_from_code(str(code)) or str(code), int(count)) for code, count in affix_code_seen.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    items: list[tuple[str, int]] = []
    for name, count in affix_seen.items():
        code = affix_code(str(name))
        label = f"{code} {name}" if code else str(name)
        try:
            parsed_count = int(count)
        except (TypeError, ValueError):
            parsed_count = 0
        items.append((label, parsed_count))
    return sorted(items, key=lambda item: item[1], reverse=True)[:limit]


def _field_display_name(field: str) -> str:
    return {
        "category": "类型",
        "level": "等级",
        "direction": "方向",
        "target": "目标",
        "match_required": "需求数",
        "upgrade_amount": "升级数",
        "stage1_options": "一阶段",
        "stage2_options": "二阶段词缀",
        "backpack_count": "背包数量",
    }.get(field, field)


def _paint_category_from_value(value, fallback):
    try:
        return PaintCategory(value)
    except (TypeError, ValueError):
        return fallback


def _paint_direction_from_value(value, fallback):
    try:
        return PaintDirection(value)
    except (TypeError, ValueError):
        return fallback


def _paint_object_from_value(value, fallback):
    try:
        return PaintObject(value)
    except (TypeError, ValueError):
        return fallback


def _safe_current_match_count(state: PuzzleState, rule: CurrentPaintRule) -> int:
    if state.current_anchor is None:
        return 0
    row, col = state.current_anchor
    cells = _scope_cells_for_display(row, col, rule.direction)
    for extra in rule.extra_directions:
        cells.extend(_scope_cells_for_display(row, col, extra))
    seen = set()
    count = 0
    for r, c in cells:
        if (r, c) == state.current_anchor or (r, c) in seen:
            continue
        seen.add((r, c))
        tile = state.tile(r, c)
        if tile.paint is None:
            continue
        if _paint_matches_target_for_display(tile.paint, rule.target_object):
            count += 1
    return count


def _scope_cells_for_display(row: int, col: int, direction: PaintDirection) -> list[tuple[int, int]]:
    if direction == PaintDirection.HORIZONTAL:
        return [(row, c) for c in range(6)]
    if direction == PaintDirection.VERTICAL:
        return [(r, col) for r in range(6)]
    if direction == PaintDirection.DIAGONAL:
        cells = {(row, col)}
        for delta in range(-6, 7):
            for r, c in ((row + delta, col + delta), (row + delta, col - delta)):
                if 0 <= r < 6 and 0 <= c < 6:
                    cells.add((r, c))
        return list(cells)
    if direction == PaintDirection.ADJACENT:
        return [(r, c) for r, c in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)) if 0 <= r < 6 and 0 <= c < 6]
    if direction == PaintDirection.GLOBAL:
        return [(r, c) for r in range(6) for c in range(6)]
    return [(row, col)]


def _paint_matches_target_for_display(paint: Paint, target: PaintObject) -> bool:
    if target == PaintObject.MONSTER:
        return paint.category == PaintCategory.MONSTER
    if target == PaintObject.GAMEPLAY:
        return paint.category == PaintCategory.GAMEPLAY
    return True


def _optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _split_manual_list(value: str, *, separator: str) -> list[str]:
    normalized = value.replace("，", ",").replace("；", ";")
    if separator == ",":
        parts = normalized.split(",")
    else:
        parts = normalized.split(";")
    return [part.strip() for part in parts if part.strip()]


def _compact_real_capture_step(
    *,
    scene: SceneObservation,
    state: PuzzleState,
    backpack: tuple[Paint, ...],
    backpack_details: list[dict] | None = None,
    backpack_pages: list[dict] | None = None,
    recommendations: list[PlacementRecommendation],
    brush_scenario: BrushChoiceScenario | None,
    resources: ResourceState,
    quality: dict | None = None,
) -> dict:
    rule = state.current_rule
    brush = scene.brush_options
    # A UI scenario may default to the first candidate. It is not a selection.
    stage1 = list(brush.stage1_options) if brush else []
    chosen = brush.chosen_stage1 if brush else None
    stage2 = list(brush.stage2_options) if brush else []
    brush_payload = {
        "stage1": stage1,
        "chosen": chosen,
        "stage2": stage2,
        "chosen_source": "panel_recognition" if chosen else None,
    }
    brush_payload.update(brush_payload_codes(stage1, chosen, stage2))
    candidate_payloads = [
        _compact_recommendation(rec, rank=rank)
        for rank, rec in enumerate(recommendations[:36], start=1)
    ]
    # 面板「调色效果」与选卡得到的笔触词缀分开保存，不互相替代。
    hint = getattr(getattr(scene, "current_paint", None), "raw_text_hint", None) or ""
    effect_text = getattr(scene.current_paint, "effect_text", None)
    if not effect_text and "effect=" in hint:
        effect_text = hint.split("effect=", 1)[1].strip()
    current_payload = dict(_compact_rule(rule) or {})
    current_payload["effect"] = effect_text
    current_payload["brush_effect"] = getattr(scene.current_paint, "brush_effect_text", None)
    board_payload = _compact_board_summary(scene)
    tile_payloads = _compact_tile_paints(state)
    return {
        "type": "capture",
        "board": board_payload,
        "current": current_payload,
        "anchor": list(state.current_anchor) if state.current_anchor is not None else None,
        "pending": current_rule_is_pending(state),
        "backpack": _paint_collection_summary(backpack),
        "backpack_visible": _compact_visible_backpack(scene),
        "backpack_cumulative": _compact_cumulative_backpack(backpack, backpack_details),
        # 按排序目标分的两页（怪物页 + 玩法页，各 ≤15 格，允许重复）。
        # 复现这局时用它知道当时背包里到底有哪些颜料。
        "backpack_pages": backpack_pages or [],
        "brush": brush_payload,
        "active_affix": _compact_affix(state.active_affix),
        "resources": {
            "force_used": resources.force_used,
            "force_remaining": resources.force_remaining,
            "refresh_available": resources.refresh_available,
            "refresh_remaining": resources.refresh_remaining,
        },
        "tiles": tile_payloads,
        # Keep displayed levels separate from model base levels and buffs.
        "observed_tiles": [
            {"state": _tile_state_code(t.state), "c": t.paint_category.value,
             "s": t.paint_subtype, "l": t.paint_level, "t": int(t.paint_is_treasure),
             "confidence": round(t.confidence, 3),
             "level_confidence": round(t.paint_level_confidence, 3)}
            for t in getattr(getattr(scene, "board", None), "tiles", ())
        ],
        "line_capacity": summarize_lines(board_payload.get("states", ""), tile_payloads),
        "quality": quality or {},
        "top": candidate_payloads[:3],
        "candidates": candidate_payloads,
    }


def _initial_snapshot_from_step(step: dict) -> dict:
    board = step.get("board") if isinstance(step.get("board"), dict) else {}
    tiles = step.get("tiles") if isinstance(step.get("tiles"), list) else []
    current = step.get("current") if isinstance(step.get("current"), dict) else None
    quality = step.get("quality") if isinstance(step.get("quality"), dict) else {}
    snapshot = {
        "schema": "huahua-initial-snapshot-v1",
        "step_index": step.get("step_index"),
        "t": step.get("t"),
        "board": _learning_board_snapshot(step),
        "tiles": tiles,
        "line_capacity": summarize_lines(board.get("states", ""), tiles),
        "board_known": _paint_collection_summary_from_compact(tiles),
        "current": current,
        "anchor": step.get("anchor"),
        "pending": bool(step.get("pending")),
        "brush": step.get("brush"),
        "active_affix": step.get("active_affix"),
        "recording_evidence": step.get("recording_evidence"),
        "observed_tiles": step.get("observed_tiles"),
        "backpack": step.get("backpack") or {},
        "resources": step.get("resources") or {},
        "quality": quality,
        "signature": _stable_payload_key(
            {
                "board": board.get("states"),
                "tiles": tiles,
                "current": current,
                "anchor": step.get("anchor"),
                "pending": bool(step.get("pending")),
            }
        ),
    }
    return snapshot


def _compact_visible_backpack(scene: SceneObservation) -> list[dict]:
    if scene.backpack is None:
        return []
    return [
        _compact_observed_backpack_paint(observed, index=index)
        for index, observed in enumerate(scene.backpack.paints)
    ]


def _compact_observed_backpack_paint(observed, *, index: int) -> dict:
    paint = Paint(
        category=observed.category,
        level=observed.level,
        subtype=observed.subtype,
        is_treasure=observed.is_treasure,
    )
    confidence = observed.category_confidence * 0.55 + observed.level_confidence * 0.45
    payload = {
        "i": index,
        **_compact_paint(paint),
        "slot": [observed.slot_rect.x, observed.slot_rect.y, observed.slot_rect.w, observed.slot_rect.h],
        "fingerprint": observed.fingerprint,
        "confidence": round(float(confidence), 3),
        "category_confidence": round(float(observed.category_confidence), 3),
        "level_confidence": round(float(observed.level_confidence), 3),
    }
    return payload


def _compact_cumulative_backpack(backpack: tuple[Paint, ...], backpack_details: list[dict] | None = None) -> list[dict]:
    available = [dict(item) for item in (backpack_details or []) if isinstance(item, dict)]
    used: set[int] = set()
    result: list[dict] = []
    for index, paint in enumerate(backpack):
        compact = _compact_paint(paint)
        code = compact.get("code")
        detail_index = next(
            (
                candidate_index
                for candidate_index, detail in enumerate(available)
                if candidate_index not in used and detail.get("code") == code
            ),
            None,
        )
        if detail_index is None:
            result.append({"i": index, **compact})
            continue
        used.add(detail_index)
        detail = dict(available[detail_index])
        detail.update(compact)
        detail["i"] = index
        result.append(detail)
    return result


def _with_initial_aliases(payload: dict) -> dict:
    initial_snapshot = payload.get("initial_snapshot")
    if not isinstance(initial_snapshot, dict):
        return payload
    enriched = dict(payload)
    board = initial_snapshot.get("board")
    enriched["initial_board"] = board if isinstance(board, dict) else {}
    enriched["initial_anchor"] = initial_snapshot.get("anchor")
    return enriched


def _compact_board_summary(scene: SceneObservation) -> dict:
    state_counts = Counter(tile.state.value for tile in scene.board.tiles)
    states = "".join(_tile_state_code(tile.state) for tile in scene.board.tiles)
    return {
        "states": states,
        "blank": state_counts.get(TileState.BLANK.value, 0),
        "blocked": state_counts.get(TileState.BLOCKED.value, 0),
        "void": state_counts.get(TileState.VOID.value, 0),
        "faded": state_counts.get(TileState.FADED.value, 0),
        "paint": state_counts.get(TileState.PAINT.value, 0),
        "score": round(scene.board.score, 3),
    }


def _compact_tile_paints(state: PuzzleState) -> list[dict | None]:
    """每格的颜料 + 颜料加成 + 光环强度。

    ``b`` / ``au`` 必须落记录，两者归属不同（2026-11-23 口径）：

    * ``b`` 是**这枚颜料自己的**加成（``Paint.bonus_level``），颜料被炸毁/清掉就
      跟着没了。不记的话回放出来的棋盘等级是错的，而等级正是判断「这步该不该压」
      的核心依据。
    * ``au`` 是**锚点地块上的**笔触光环强度（``Tile.aura_amount``），颜料没了它还在，
      恢复后换一枚颜料就改挂到新颜料上。

    ``al``（从光环里吃到的派生等级）刻意不记——它由 ``_recompute_auras`` 从
    全场 ``au`` 重算，存了反而容易和真实状态打架。

    ``id`` 是 :func:`paint_id` 的短码（``sr1`` = 鼠人 lv1），和 ``code`` 并存：
    ``code`` 是老的「类别+等级」写法，下游一堆代码在用它，不能直接替换。
    """

    result: list[dict | None] = []
    for tile in state.tiles:
        if tile.paint is None:
            # 没有颜料，但可能留着「等级光环」——颜料落到虚空后失去自身内容、
            # 退化成笔触效果，只剩底色 + 一圈光环（用户 2026-11-24 实测 3-5 级）。
            # 以前这里直接记账成 None，那枚颜料的笔触效果就从记录里彻底消失了，
            # 复盘时看不出这格还有东西。用 ``ra``/``ral`` 单独标出来。
            aura_retained = float(getattr(tile, "retained_aura", 0.0) or 0.0)
            if aura_retained > 0:
                entry: dict = {"ra": round(aura_retained, 3)}
                aura_retained_level = getattr(tile, "retained_aura_level", None)
                if aura_retained_level is not None:
                    entry["ral"] = int(aura_retained_level)
                result.append(entry)
            else:
                result.append(None)
            continue
        payload = _compact_paint(tile.paint)
        payload["id"] = paint_id(
            tile.paint.category.value,
            tile.paint.level,
            tile.paint.subtype,
            is_treasure=tile.paint.is_treasure,
        )
        bonus = tile_bonus_level(tile)
        if bonus:
            payload["b"] = bonus
        aura = max(0, int(getattr(tile, "aura_amount", 0) or 0))
        if aura:
            payload["au"] = aura
        result.append(payload)
    return result


def _tile_state_code(state: TileState) -> str:
    return {
        TileState.BLANK: "B",
        TileState.BLOCKED: "K",
        TileState.VOID: "V",
        TileState.FADED: "F",
        TileState.PAINT: "P",
        TileState.UNKNOWN: "?",
    }[state]


def _compact_rule(rule: CurrentPaintRule | None) -> dict | None:
    if rule is None:
        return None
    return compact_rule_with_code(rule)


def _compact_affix(affix: AffixRule | None) -> dict | None:
    if affix is None:
        return None
    payload = {
        "name": affix.name,
        "code": affix_code(affix),
        "canonical_affix_id": canonical_affix_id(affix),
        "brush": affix.brush_type.value,
        "brush_code": brush_code(affix.brush_type),
        "effect": affix.effect.value,
        "persistent": bool(affix.persistent_until_match),
    }
    if affix.direction is not None:
        payload["direction"] = affix.direction.value
    if affix.target_object is not None:
        payload["target"] = affix.target_object.value
    if affix.delta:
        payload["delta"] = affix.delta
    if affix.amount:
        payload["amount"] = affix.amount
    if affix.count is not None:
        payload["count"] = affix.count
    if affix.nearest:
        payload["nearest"] = True
    return payload


def _compact_recommendation(rec: PlacementRecommendation, *, rank: int | None = None) -> dict:
    paint_payload = _compact_paint(rec.paint) if rec.paint is not None else None
    paint_id = paint_payload.get("code") if paint_payload is not None else "none"
    payload = {
        "r": rec.row,
        "c": rec.col,
        "paint": paint_payload,
        "score": round(rec.score, 1),
        "complete": rec.completes_match,
        "candidate_id": f"placement:{rec.row}:{rec.col}:{paint_id}",
        "action_id": f"board.place:{rec.row}:{rec.col}",
        "legal": True,
    }
    if rank is not None:
        payload["rank"] = int(rank)
    explanation = _compact_recommendation_explanation(rec)
    if explanation:
        payload["explanation"] = explanation
    payload["policy_score"] = {
        "value": round(rec.score, 1),
        "components": dict(explanation.get("layers") or {}),
    }
    return payload


def _compact_recommendation_explanation(rec: PlacementRecommendation) -> dict:
    explanation = getattr(rec, "explanation", None)
    if explanation is None:
        return {}
    payload: dict = {}
    if explanation.short:
        payload["short"] = explanation.short
    if explanation.tags:
        payload["tags"] = list(explanation.tags)
    layers = {}
    for name in ("immediate", "value", "terrain", "risk", "resource", "expected"):
        value = float(getattr(explanation, name, 0.0) or 0.0)
        if abs(value) >= 0.5:
            layers[name] = round(value, 1)
    if layers:
        payload["layers"] = layers
    return payload


def _recommendation_explanation_lines(rec: PlacementRecommendation) -> list[str]:
    explanation = getattr(rec, "explanation", None)
    if explanation is None:
        return []

    lines: list[str] = []
    if explanation.short:
        lines.append(f"解释：{explanation.short}")

    parts: list[str] = []
    for label, attr in (
        ("即时", "immediate"),
        ("价值", "value"),
        ("地形", "terrain"),
        ("风险", "risk"),
        ("资源", "resource"),
        ("期望", "expected"),
    ):
        value = float(getattr(explanation, attr, 0.0) or 0.0)
        if abs(value) >= 0.5:
            parts.append(f"{label}{value:+.0f}")
    if parts:
        lines.append("分层：" + " ".join(parts))

    tags = tuple(getattr(explanation, "tags", ()) or ())
    if tags:
        labels = [_recommendation_tag_label(tag) for tag in tags[:6]]
        lines.append("标签：" + "、".join(labels))
    return lines


def _recommendation_tag_label(tag: str) -> str:
    return {
        "anchor": "初始落点",
        "can_complete": "可完成",
        "complete_match": "完成匹配",
        "treasure": "宝箱",
        "placed_to_7": "本格到7",
        "treasure_to_7": "宝箱到7",
        "gameplay_to_7": "玩法到7",
        "monster_to_7": "怪物到7",
        "high_value_gain": "高价值",
        "overwrite_risk": "覆盖风险",
        "terrain_gain": "地形收益",
        "terrain_loss": "地形损失",
        "lookahead_gain": "后续更好",
        "lookahead_risk": "后续风险",
        "learned_prior": "历史偏好",
        "special_tile": "特殊地块",
        "scope_value_gain": "范围收益",
    }.get(tag, tag)


def _infer_real_capture_action(previous: dict | None, current: dict) -> dict:
    if previous is None:
        return {"type": "session_start", "confidence": 0.5}

    previous_board = previous.get("board") or {}
    current_board = current.get("board") or {}
    previous_current = previous.get("current")
    current_rule = current.get("current")
    previous_brush = previous.get("brush") or {}
    current_brush = current.get("brush") or {}
    previous_resources = previous.get("resources") or {}
    current_resources = current.get("resources") or {}
    previous_backpack = previous.get("backpack") or {}
    current_backpack = current.get("backpack") or {}

    board_delta = _state_count_delta(previous_board, current_board)
    changed_cells = _changed_capture_cells(previous, current)
    current_changed = previous_current != current_rule
    brush_changed = previous_brush != current_brush
    backpack_delta = int(current_backpack.get("total", 0)) - int(previous_backpack.get("total", 0))
    force_delta = int(previous_resources.get("force_remaining", 0)) - int(current_resources.get("force_remaining", 0))
    refresh_delta = int(previous_resources.get("refresh_remaining", 0)) - int(current_resources.get("refresh_remaining", 0))

    if force_delta > 0 or board_delta.get("faded", 0) > 0:
        return _action_payload(
            "force_complete",
            confidence=0.86,
            code="F",
            force_used_delta=force_delta,
            board_delta=board_delta,
            changed_cells=changed_cells,
            detail=_action_detail("force_complete", changed_cells, previous, current),
            summary=_action_summary("force_complete", changed_cells, previous, current),
        )
    if refresh_delta > 0:
        stage = "stage2" if previous_brush.get("chosen") == current_brush.get("chosen") else "stage1"
        return _action_payload(
            "refresh_affix",
            confidence=0.82,
            code="R2" if stage == "stage2" else "R1",
            stage=stage,
            refresh_used_delta=refresh_delta,
            detail=_action_detail("refresh_affix", changed_cells, previous, current),
            summary=_action_summary("refresh_affix", changed_cells, previous, current),
        )
    if current_changed and current_rule is not None:
        return _action_payload(
            "new_current_paint",
            confidence=0.78,
            code="N",
            previous=previous_current,
            current=current_rule,
            changed_cells=changed_cells,
            detail=_action_detail("new_current_paint", changed_cells, previous, current),
            summary=_action_summary("new_current_paint", changed_cells, previous, current),
        )
    if brush_changed and current_brush.get("stage2"):
        return _action_payload(
            "brush_options_changed",
            confidence=0.70,
            code="B",
            brush=current_brush,
            detail=_action_detail("brush_options_changed", changed_cells, previous, current),
            summary=_action_summary("brush_options_changed", changed_cells, previous, current),
        )
    if board_delta.get("paint", 0) > 0 or backpack_delta < 0:
        action_type = "place_current_paint" if previous.get("pending") else "install_paint"
        return _action_payload(
            action_type,
            confidence=0.76 if action_type == "place_current_paint" else 0.74,
            code="C" if action_type == "place_current_paint" else "P",
            backpack_delta=backpack_delta,
            board_delta=board_delta,
            changed_cells=changed_cells,
            detail=_action_detail(action_type, changed_cells, previous, current),
            summary=_action_summary(action_type, changed_cells, previous, current),
        )
    if board_delta.get("blank", 0) > 0 or board_delta.get("blocked", 0) < 0 or board_delta.get("void", 0) < 0:
        return _action_payload(
            "terrain_changed",
            confidence=0.68,
            code="T",
            board_delta=board_delta,
            changed_cells=changed_cells,
            detail=_action_detail("terrain_changed", changed_cells, previous, current),
            summary=_action_summary("terrain_changed", changed_cells, previous, current),
        )
    return _action_payload(
        "capture_update",
        confidence=0.45,
        code="U",
        backpack_delta=backpack_delta,
        board_delta=board_delta,
        changed_cells=changed_cells,
        detail=_action_detail("capture_update", changed_cells, previous, current),
        summary=_action_summary("capture_update", changed_cells, previous, current),
    )


def _action_payload(action_type: str, *, confidence: float, code: str, **fields) -> dict:
    payload = {
        "type": action_type,
        "code": code,
        "confidence": confidence,
    }
    payload.update(fields)
    return payload


def _action_detail(action_type: str, changed_cells: list[dict], previous: dict, current: dict) -> dict:
    summary = _action_summary(action_type, changed_cells, previous, current)
    current_rule = current.get("current") if isinstance(current.get("current"), dict) else None
    previous_rule = previous.get("current") if isinstance(previous.get("current"), dict) else None
    detail = {
        "code": _action_code_from_type(action_type),
        "placed": summary.get("placed_cells", []),
        "upgraded": summary.get("upgraded_cells", []),
        "faded": summary.get("faded_cells", []),
        "terrain": summary.get("terrain_cells", []),
        "match": summary.get("current_match"),
        "current_code": current_rule.get("code") if current_rule else None,
        "previous_code": previous_rule.get("code") if previous_rule else None,
    }
    if action_type in {"install_paint", "place_current_paint"}:
        detail["placed_paints"] = summary.get("placed_paints", [])
    if action_type == "refresh_affix":
        previous_brush = previous.get("brush") or {}
        current_brush = current.get("brush") or {}
        detail["from_stage2"] = previous_brush.get("stage2_codes") or previous_brush.get("stage2") or []
        detail["to_stage2"] = current_brush.get("stage2_codes") or current_brush.get("stage2") or []
    return detail


def _action_code_from_type(action_type: str) -> str:
    return {
        "session_start": "S",
        "force_complete": "F",
        "refresh_affix": "R",
        "new_current_paint": "N",
        "brush_options_changed": "B",
        "place_current_paint": "C",
        "install_paint": "P",
        "terrain_changed": "T",
        "capture_update": "U",
    }.get(action_type, "?")


def _action_summary(action_type: str, changed_cells: list[dict], previous: dict, current: dict) -> dict:
    placed = [cell for cell in changed_cells if cell.get("to_state") == "P" and cell.get("from_state") != "P"]
    upgraded = [
        cell
        for cell in changed_cells
        if isinstance(cell.get("from_paint"), dict)
        and isinstance(cell.get("to_paint"), dict)
        and (cell["to_paint"].get("l") or 0) > (cell["from_paint"].get("l") or 0)
    ]
    faded = [cell for cell in changed_cells if cell.get("to_state") == "F"]
    terrain = [cell for cell in changed_cells if cell.get("from_state") != cell.get("to_state") and cell.get("to_state") != "P"]
    return {
        "action": action_type,
        "placed_cells": _cell_refs(placed),
        "upgraded_cells": _cell_refs(upgraded),
        "faded_cells": _cell_refs(faded),
        "terrain_cells": _cell_refs(terrain),
        "placed_paints": [cell.get("to_paint") for cell in placed if cell.get("to_paint") is not None],
        "upgrade_count": len(upgraded),
        "changed_count": len(changed_cells),
        "current_match": ((current.get("current") or {}).get("match") if isinstance(current.get("current"), dict) else None),
    }


def _cell_refs(cells: list[dict]) -> list[list[int]]:
    refs: list[list[int]] = []
    for cell in cells[:12]:
        row = cell.get("r")
        col = cell.get("c")
        if isinstance(row, int) and isinstance(col, int):
            refs.append([row, col])
    return refs


def _state_count_delta(previous_board: dict, current_board: dict) -> dict:
    keys = ("blank", "blocked", "void", "faded", "paint")
    return {
        key: int(current_board.get(key, 0)) - int(previous_board.get(key, 0))
        for key in keys
    }


def _recording_metadata_changed(previous: dict | None, current: dict) -> bool:
    previous = previous or {}
    return any(previous.get(key) != current.get(key) for key in
               ("current", "brush", "active_affix", "recording_evidence"))


def _is_noop_capture_update(inferred_action: dict) -> bool:
    if inferred_action.get("type") != "capture_update":
        return False
    if inferred_action.get("changed_cells"):
        return False
    summary = inferred_action.get("summary") or {}
    return int(summary.get("changed_count", 0) or 0) == 0


def _is_tentative_place_current_action(action: dict, step: dict) -> bool:
    if action.get("type") != "place_current_paint":
        return False
    if step.get("ocr") is not False:
        return False
    summary = action.get("summary") or {}
    placed_cells = summary.get("placed_cells") or []
    if len(placed_cells) != 1:
        return False
    if (step.get("quality") or {}).get("mode") != "quick_capture":
        return False
    return True


def _backpack_visible_in_step(step: dict) -> bool:
    backpack = step.get("backpack") or {}
    return int(backpack.get("total", 0) or 0) > 0


def _find_new_anchor_from_scene(scene: SceneObservation, previous_state: PuzzleState) -> tuple[int, int] | None:
    candidates: list[tuple[float, int, int]] = []
    recommendations = recommend_placements(previous_state, limit=36, include_lookahead=False)
    recommended_cells = {(rec.row, rec.col): rec.score for rec in recommendations}
    for tile in scene.board.tiles:
        row, col = tile.row, tile.col
        before = previous_state.tile(row, col)
        if tile.state != TileState.PAINT or before.state == TileState.PAINT:
            continue
        score = float(tile.confidence)
        if (row, col) in recommended_cells:
            score += 2.0 + recommended_cells[(row, col)] / 1000.0
        if before.state == TileState.BLANK:
            score += 0.6
        candidates.append((score, row, col))
    if not candidates:
        return None
    _score, row, col = max(candidates, key=lambda item: item[0])
    return row, col


def _tentative_place_signature(step: dict) -> str | None:
    action = step.get("inferred_action") or {}
    summary = action.get("summary") or {}
    placed_cells = summary.get("placed_cells") or []
    if not placed_cells:
        return None
    return _stable_payload_key(
        {
            "session": step.get("session"),
            "current": step.get("current"),
            "anchor": step.get("anchor"),
            "placed": placed_cells[:1],
        }
    )


def _changed_capture_cells(previous: dict, current: dict) -> list[dict]:
    previous_states = str((previous.get("board") or {}).get("states") or "")
    current_states = str((current.get("board") or {}).get("states") or "")
    previous_tiles = previous.get("tiles") or []
    current_tiles = current.get("tiles") or []
    changed: list[dict] = []
    for index in range(36):
        prev_state = previous_states[index] if index < len(previous_states) else "?"
        curr_state = current_states[index] if index < len(current_states) else "?"
        prev_paint = previous_tiles[index] if index < len(previous_tiles) else None
        curr_paint = current_tiles[index] if index < len(current_tiles) else None
        if prev_state == curr_state and prev_paint == curr_paint:
            continue
        changed.append(
            {
                "r": index // 6,
                "c": index % 6,
                "from_state": prev_state,
                "to_state": curr_state,
                "from_paint": prev_paint,
                "to_paint": curr_paint,
            }
        )
    return changed[:12]


def _with_state_action_result_metadata(payload: dict) -> dict:
    enriched = dict(payload)
    steps = [dict(step) for step in enriched.get("steps") or [] if isinstance(step, dict)]
    terminal = enriched.get("terminal") if isinstance(enriched.get("terminal"), dict) else {}
    score = enriched.get("internal_score")
    if not isinstance(score, (int, float)):
        score = _internal_run_score(terminal) if terminal else 0.0
    score_breakdown = enriched.get("score_breakdown")
    if not isinstance(score_breakdown, dict):
        score_breakdown = _internal_score_breakdown(terminal) if terminal else {}

    steps = _attach_transition_metadata(steps, terminal=terminal, final_score=float(score or 0.0))
    enriched["steps"] = steps
    enriched["transition_schema"] = "huahua-state-action-result-v1"
    enriched["trajectory_summary"] = _trajectory_learning_summary(
        steps,
        terminal=terminal,
        score=float(score or 0.0),
        score_breakdown=score_breakdown,
    )
    return enriched


def _trajectory_finished_at(started_at: float, steps: list[dict]) -> float:
    if steps:
        last_t = steps[-1].get("t")
        if isinstance(last_t, (int, float)):
            return round(float(started_at) + float(last_t), 3)
    return round(float(started_at), 3)


def _attach_transition_metadata(steps: list[dict], *, terminal: dict, final_score: float) -> list[dict]:
    enriched: list[dict] = []
    previous: dict | None = None
    for index, step in enumerate(steps):
        item = dict(step)
        item["transition_index"] = index
        if previous is not None:
            action = item.get("inferred_action") if isinstance(item.get("inferred_action"), dict) else {}
            item["decision_context"] = _decision_context_from_steps(previous, item)
            item["outcome"] = _transition_outcome(previous, item, terminal=terminal, final_score=final_score, action=action)
            item["chosen_recommendation"] = _matched_previous_recommendation(previous, action)
        else:
            item.setdefault("inferred_action", {"type": "session_start", "code": "S", "confidence": 0.5})
            item["decision_context"] = _decision_context_from_steps(None, item)
            item["outcome"] = _session_start_outcome(item, terminal=terminal, final_score=final_score)
            item["chosen_recommendation"] = None
        enriched.append(item)
        previous = item
    return enriched


def _decision_context_from_steps(previous: dict | None, current: dict) -> dict:
    source = previous or current
    current_rule = source.get("current") if isinstance(source.get("current"), dict) else None
    brush = source.get("brush") if isinstance(source.get("brush"), dict) else {}
    action = current.get("inferred_action") if isinstance(current.get("inferred_action"), dict) else {}
    return {
        "source_step_index": source.get("step_index"),
        "t": source.get("t"),
        "board": _learning_board_snapshot(source),
        "current_code": current_rule.get("code") if current_rule else None,
        "current": current_rule,
        "anchor": source.get("anchor"),
        "pending": bool(source.get("pending")),
        "active_affix": source.get("active_affix"),
        "brush": {
            "stage1_codes": brush.get("stage1_codes") or [],
            "chosen_code": brush.get("chosen_code"),
            "stage2_codes": brush.get("stage2_codes") or [],
            "chosen": brush.get("chosen"),
        },
        "resources": source.get("resources") or {},
        "backpack": source.get("backpack") or {},
        "top": (source.get("top") or [])[:3],
        "action_type": action.get("type"),
        "action_code": action.get("code"),
    }


def _transition_outcome(
    previous: dict,
    current: dict,
    *,
    terminal: dict,
    final_score: float,
    action: dict,
) -> dict:
    before_terminal = _terminal_summary_from_step(previous)
    after_terminal = _terminal_summary_from_step(current)
    before_score = _internal_run_score(before_terminal)
    after_score = _internal_run_score(after_terminal)
    summary = action.get("summary") if isinstance(action.get("summary"), dict) else {}
    return {
        "before": _learning_terminal_snapshot(before_terminal),
        "after": _learning_terminal_snapshot(after_terminal),
        "score_delta": round(after_score - before_score, 3),
        "score_to_final": round(final_score - after_score, 3),
        "changed_cells": (action.get("changed_cells") or [])[:12],
        "summary": summary,
        "terminal": _learning_terminal_snapshot(terminal),
    }


def _session_start_outcome(step: dict, *, terminal: dict, final_score: float) -> dict:
    current_terminal = _terminal_summary_from_step(step)
    current_score = _internal_run_score(current_terminal)
    return {
        "before": None,
        "after": _learning_terminal_snapshot(current_terminal),
        "score_delta": 0.0,
        "score_to_final": round(final_score - current_score, 3),
        "changed_cells": [],
        "summary": {"action": "session_start"},
        "terminal": _learning_terminal_snapshot(terminal),
    }


def _matched_previous_recommendation(previous: dict, action: dict) -> dict | None:
    cells = _placement_cells_from_action(action)
    if not cells:
        return None
    top = [rec for rec in (previous.get("top") or []) if isinstance(rec, dict)]
    if not top:
        return {"matched": False, "cells": cells[:3]}
    for cell in cells:
        row = cell.get("r")
        col = cell.get("c")
        for rank, rec in enumerate(top, start=1):
            if rec.get("r") == row and rec.get("c") == col:
                return {
                    "matched": True,
                    "rank": rank,
                    "cell": [row, col],
                    "score": rec.get("score"),
                    "complete": rec.get("complete"),
                    "recommendation": rec,
                }
    return {"matched": False, "cells": cells[:3], "top_cells": [[rec.get("r"), rec.get("c")] for rec in top[:3]]}


def _placement_cells_from_action(action: dict) -> list[dict]:
    summary = action.get("summary") if isinstance(action.get("summary"), dict) else {}
    refs = summary.get("placed_cells") or []
    cells: list[dict] = []
    for ref in refs:
        if isinstance(ref, (list, tuple)) and len(ref) >= 2 and isinstance(ref[0], int) and isinstance(ref[1], int):
            cells.append({"r": ref[0], "c": ref[1]})
    if cells:
        return cells[:3]
    changed = [cell for cell in action.get("changed_cells") or [] if isinstance(cell, dict)]
    for cell in changed:
        if cell.get("to_state") == "P" and (cell.get("from_state") != "P" or cell.get("from_paint") is None):
            row = cell.get("r")
            col = cell.get("c")
            if isinstance(row, int) and isinstance(col, int):
                cells.append({"r": row, "c": col})
    return cells[:3]


def _learning_board_snapshot(step: dict) -> dict:
    board = step.get("board") if isinstance(step.get("board"), dict) else {}
    return {
        "states": board.get("states"),
        "blank": int(board.get("blank", 0) or 0),
        "blocked": int(board.get("blocked", 0) or 0),
        "void": int(board.get("void", 0) or 0),
        "faded": int(board.get("faded", 0) or 0),
        "paint": int(board.get("paint", 0) or 0),
        "score": board.get("score"),
    }


def _learning_terminal_snapshot(terminal: dict) -> dict:
    known = terminal.get("known") if isinstance(terminal.get("known"), dict) else {}
    return {
        "score": _internal_run_score(terminal) if terminal else 0.0,
        "known": {
            "total": int(known.get("total", 0) or 0),
            "lv7": int(known.get("lv7", 0) or 0),
            "gameplay7": int(known.get("gameplay7", 0) or 0),
            "monster7": int(known.get("monster7", 0) or 0),
            "treasure5": int(known.get("treasure5", 0) or 0),
            "treasure6": int(known.get("treasure6", 0) or 0),
            "treasure7": int(known.get("treasure7", 0) or 0),
            "max_treasure": int(known.get("max_treasure", 0) or 0),
            "levels": dict(known.get("levels") or {}),
        },
    }


def _trajectory_learning_summary(steps: list[dict], *, terminal: dict, score: float, score_breakdown: dict) -> dict:
    action_counts = Counter()
    recommendation_hits = Counter()
    brush_codes = Counter()
    affix_codes = Counter()
    reliable_steps = 0
    for step in steps:
        quality = step.get("quality") if isinstance(step.get("quality"), dict) else {}
        if quality.get("ok") is True:
            reliable_steps += 1
        inferred = step.get("inferred_action") if isinstance(step.get("inferred_action"), dict) else {}
        action_counts[str(inferred.get("type", "unknown"))] += 1
        chosen_rec = step.get("chosen_recommendation") if isinstance(step.get("chosen_recommendation"), dict) else {}
        if chosen_rec.get("matched") is True:
            recommendation_hits["top_hit"] += 1
            if chosen_rec.get("rank") is not None:
                recommendation_hits[f"rank_{chosen_rec.get('rank')}"] += 1
        elif chosen_rec:
            recommendation_hits["miss"] += 1
        brush = step.get("brush") if isinstance(step.get("brush"), dict) else {}
        if brush.get("chosen_code"):
            brush_codes[str(brush.get("chosen_code"))] += 1
        for code in brush.get("stage2_codes") or []:
            affix_codes[str(code)] += 1
        active_affix = step.get("active_affix") if isinstance(step.get("active_affix"), dict) else {}
        if active_affix.get("code"):
            affix_codes[str(active_affix.get("code"))] += 1
    return {
        "step_count": len(steps),
        "trusted_step_count": reliable_steps,
        "action_counts": dict(action_counts),
        "recommendation_hits": dict(recommendation_hits),
        "brush_codes": dict(brush_codes),
        "affix_codes": dict(affix_codes),
        "terminal": _learning_terminal_snapshot(terminal),
        "score": round(float(score or 0.0), 3),
        "score_breakdown": score_breakdown,
    }


def _trajectory_terminal_summary(state: PuzzleState, backpack: tuple[Paint, ...]) -> dict:
    board_paints = tuple(tile.paint for tile in state.tiles if tile.paint is not None)
    known_paints = tuple(paint for paint in (*board_paints, *backpack) if paint is not None)
    if not known_paints and state.current_rule is not None:
        known_paints = (state.current_rule.paint,)
    known = _paint_collection_summary(known_paints)
    return {
        "known": known,
        "board_known": _paint_collection_summary(board_paints),
        "backpack": _paint_collection_summary(backpack),
        "quality": _quality_proxy(known),
    }


def _quality_is_trusted_for_trajectory(quality: dict | None) -> bool:
    return isinstance(quality, dict) and quality.get("ok") is True


def _trusted_trajectory_steps(steps: list[dict]) -> list[dict]:
    return [step for step in steps if _quality_is_trusted_for_trajectory(step.get("quality"))]


def _empty_terminal_summary() -> dict:
    empty = {
        "total": 0,
        "levels": {},
        "lv7": 0,
        "gameplay7": 0,
        "monster7": 0,
        "treasure": 0,
        "treasure5": 0,
        "treasure6": 0,
        "treasure7": 0,
        "max_treasure": 0,
    }
    return {
        "known": dict(empty),
        "board_known": dict(empty),
        "backpack": dict(empty),
        "quality": 0.0,
    }


def _best_trusted_terminal_summary(
    state: PuzzleState,
    steps: list[dict],
    *,
    current_quality: dict | None = None,
) -> dict:
    reliable_steps = _trusted_trajectory_steps(steps)
    if _quality_is_trusted_for_trajectory(current_quality):
        best = _trajectory_terminal_summary(state, state.backpack)
        best_score = _internal_run_score(best)
    elif reliable_steps:
        best = _terminal_summary_from_step(reliable_steps[0])
        best_score = _internal_run_score(best)
    else:
        return _empty_terminal_summary()

    for step in reliable_steps:
        step_terminal = _terminal_summary_from_step(step)
        step_score = _internal_run_score(step_terminal)
        if step_score > best_score:
            best = step_terminal
            best_score = step_score
    return best


def _best_available_terminal_summary(state: PuzzleState, steps: list[dict]) -> dict:
    terminal = _trajectory_terminal_summary(state, state.backpack)
    best = terminal
    best_score = _internal_run_score(terminal)
    for step in steps:
        quality = step.get("quality") or {}
        if quality.get("ok") is not True:
            continue
        step_terminal = _terminal_summary_from_step(step)
        step_score = _internal_run_score(step_terminal)
        if step_score > best_score:
            best = step_terminal
            best_score = step_score
    return best


def _terminal_summary_from_step(step: dict) -> dict:
    backpack = step.get("backpack") or {}
    board_known = _paint_collection_summary_from_compact(step.get("tiles") or [])
    known = _merge_paint_summaries(board_known, backpack)
    return {
        "known": known,
        "board_known": board_known,
        "backpack": backpack,
        "quality": _quality_proxy(known),
    }


def _paint_collection_summary_from_compact(items: list) -> dict:
    paints = [item for item in items if isinstance(item, dict)]
    total = len(paints)
    levels = Counter(str(item.get("l")) for item in paints if item.get("l") is not None)
    treasure_levels = [int(item.get("l") or 0) for item in paints if item.get("t")]
    gameplay7 = sum(1 for item in paints if item.get("c") == PaintCategory.GAMEPLAY.value and not item.get("t") and int(item.get("l") or 0) >= 7)
    monster7 = sum(1 for item in paints if item.get("c") == PaintCategory.MONSTER.value and int(item.get("l") or 0) >= 7)
    return {
        "total": total,
        "levels": dict(levels),
        "lv7": sum(1 for item in paints if int(item.get("l") or 0) >= 7),
        "gameplay7": gameplay7,
        "monster7": monster7,
        "treasure": sum(1 for item in paints if item.get("t")),
        "treasure5": sum(1 for level in treasure_levels if level >= 5),
        "treasure6": sum(1 for level in treasure_levels if level >= 6),
        "treasure7": sum(1 for level in treasure_levels if level >= 7),
        "max_treasure": max(treasure_levels) if treasure_levels else 0,
    }


def _merge_paint_summaries(*summaries: dict) -> dict:
    levels = Counter()
    merged = {
        "total": 0,
        "lv7": 0,
        "gameplay7": 0,
        "monster7": 0,
        "treasure": 0,
        "treasure5": 0,
        "treasure6": 0,
        "treasure7": 0,
        "max_treasure": 0,
    }
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        merged["total"] += int(summary.get("total", 0) or 0)
        merged["lv7"] += int(summary.get("lv7", 0) or 0)
        merged["gameplay7"] += int(summary.get("gameplay7", 0) or 0)
        merged["monster7"] += int(summary.get("monster7", 0) or 0)
        merged["treasure"] += int(summary.get("treasure", 0) or 0)
        merged["treasure5"] += int(summary.get("treasure5", 0) or 0)
        merged["treasure6"] += int(summary.get("treasure6", 0) or 0)
        merged["treasure7"] += int(summary.get("treasure7", 0) or 0)
        merged["max_treasure"] = max(merged["max_treasure"], int(summary.get("max_treasure", 0) or 0))
        for level, count in (summary.get("levels") or {}).items():
            levels[str(level)] += int(count or 0)
    merged["levels"] = dict(levels)
    return merged


def _internal_run_score(terminal: dict) -> float:
    return round(_internal_score_breakdown(terminal)["total"], 3)


def _compressed_summary_kind(terminal: dict, high_score_gate: dict) -> str:
    if isinstance(high_score_gate, dict) and high_score_gate.get("passed") is True:
        return "candidate_high_score"
    known = terminal.get("known") if isinstance(terminal.get("known"), dict) else {}
    if _summary_has_high_value(known):
        return "candidate_low_weight"
    if _summary_has_level6_cluster(known):
        return "candidate_level6_cluster"
    return "low_value_terminal"


def _compressed_summary_weight(kind: str) -> float:
    return {
        "candidate_high_score": 0.75,
        "candidate_low_weight": 0.45,
        "candidate_level6_cluster": 0.30,
        "low_value_terminal": 0.15,
    }.get(kind, 0.1)


def _compressed_summary_sample_metadata(kind: str) -> dict:
    weight = _compressed_summary_weight(kind)
    metadata = {
        "schema": "huahua-candidate-sample-v1",
        "source": "compressed_run_summary",
        "summary_kind": kind,
        "sample_class": "terrain_baseline",
        "sample_role": "low_value_terminal",
        "candidate_sample": False,
        "training_weight": weight,
        "expert_eligible": False,
        "use_for": ["terrain_prior", "negative_baseline"],
        "exclude_from": ["expert_trajectory", "high_score_imitation"],
    }
    overrides = {
        "candidate_high_score": {
            "sample_class": "candidate",
            "sample_role": "high_score_candidate",
            "candidate_sample": True,
            "use_for": ["candidate_review", "terrain_prior"],
        },
        "candidate_low_weight": {
            "sample_class": "candidate",
            "sample_role": "high_value_terminal_low_weight",
            "candidate_sample": True,
            "use_for": ["candidate_review", "low_weight_training", "terrain_prior"],
        },
        "candidate_level6_cluster": {
            "sample_class": "candidate",
            "sample_role": "level6_cluster_low_weight",
            "candidate_sample": True,
            "use_for": ["candidate_review", "low_weight_training", "terrain_prior"],
        },
    }
    metadata.update(overrides.get(kind, {}))
    return metadata


def _summary_has_level6_cluster(summary: dict) -> bool:
    levels = summary.get("levels") if isinstance(summary.get("levels"), dict) else {}
    level6_or_higher = 0
    for level_text, count in levels.items():
        try:
            level = int(level_text)
            amount = int(count or 0)
        except (TypeError, ValueError):
            continue
        if 6 <= level < 7:
            level6_or_higher += amount
    treasure6_plus = int(summary.get("treasure6", 0) or 0)
    lv7 = int(summary.get("lv7", 0) or 0)
    return treasure6_plus <= 0 and lv7 <= 0 and level6_or_higher >= 2


def _internal_score_breakdown(terminal: dict) -> dict:
    known = terminal.get("known") or {}
    main = _quality_proxy(known)

    # Keep the main score on the explicit treasure equivalence scale:
    # Lv7 box=5000, Lv6 box=3000, Lv5 box=1000, gameplay Lv7=1000,
    # monster Lv7≈714. Low-level pieces are only tie-breakers.
    treasure7 = int(known.get("treasure7", 0) or 0)
    treasure6_only = max(0, int(known.get("treasure6", 0) or 0) - treasure7)
    treasure5_only = max(0, int(known.get("treasure5", 0) or 0) - int(known.get("treasure6", 0) or 0))
    treasure_tie = treasure7 * 10.0 + treasure6_only * 6.0 + treasure5_only * 3.0
    lv7_tie = max(0, int(known.get("lv7", 0) or 0) - treasure7) * 0.05
    low_level_tie = 0.0

    levels = known.get("levels") or {}
    for level_text, count in levels.items():
        try:
            level = int(level_text)
        except (TypeError, ValueError):
            continue
        if level <= 4:
            low_level_tie += count * level * 0.01
        elif level == 5:
            low_level_tie += count * 0.08
        elif level == 6:
            low_level_tie += count * 0.16
    total = main + treasure_tie + lv7_tie + low_level_tie
    return {
        "main": round(main, 3),
        "treasure_tie": round(treasure_tie, 3),
        "lv7_tie": round(lv7_tie, 3),
        "low_level_tie": round(low_level_tie, 3),
        "total": round(total, 3),
        "model": "huahua-v2-treasure-equivalence",
    }


def _looks_like_new_run(scene: SceneObservation, previous_step: dict | None) -> bool:
    """这一帧是不是「换了一张画纸」——上一局收尾之后又重开了一张。

    实机取证（2026-11-15，两局对照）先把「换纸」这一刻认清楚了，别再靠想象：

    * **终局/入画界面**：棋盘还是画满的样子——19 枚颜料 + 17 个不可放置格，
      ``blank=0``，右侧进度 ``0/11``，右下角是蓝色「入画」。这一帧**不是**新画纸，
      它只是上一局的棋盘还留在画布上（实测与终局帧的颜料位置逐格相同）。
    * **按下「入画」真正发牌**：实测 22:40 开局是 `blank=12 / paint=7 / void=13 /
      blocked=4`。也就是说新画纸的硬特征是——**空位从 0 猛增，颜料同时明显变少**。

    判据（任一成立即认新局，调用方只在「本局满铺过」或「已结束」时才问）：

    1. **空位回升 + 颜料减少**（主判据）：满铺是整局的终态，格子不会自己空出来。
       ``blank`` 增 ≥2 且 ``paint`` 减 ≥2 → 只能是重新发了牌。这条不依赖绝对数量，
       画纸开局自带 7 枚还是 15 枚颜料都成立。
    2. **满铺之后又出现空位**：上一帧连 VOID 都算满（``open=0``）时直接认。
    3. **棋盘布局重排**（VOID 位置变了 ≥3 格）：同一张画纸的不可放置空洞整局不动，
       实测一整局 13 个位置一个没变；换纸后重排。
    4. **开局盘**：空位多、颜料少（``paint<=8``，老判据，仅作兜底）。

    为什么不能只靠第 4 条（2026-11-14 实机纠正）：S8 画纸开局自带一整盘 1~2 级颜料，
    有的局开局 paint 就有 12 枚，``paint<=8`` 直接不成立。于是收尾之后一直挂着上一局的
    规则与推荐，状态灯停在「本局已收尾」，而棋盘早就是新的了。
    """
    if previous_step is None:
        return True
    board = _compact_board_summary(scene)
    previous_board = previous_step.get("board") or {}
    states = str(board.get("states") or "")
    previous_states = str(previous_board.get("states") or "")
    if states == previous_states:
        return False
    blank_now = int(board.get("blank", 0))
    paint_now = int(board.get("paint", 0))
    open_cells = blank_now + int(board.get("void", 0))
    if open_cells <= 0:
        return False
    blank_before = int(previous_board.get("blank", 0))
    paint_before = int(previous_board.get("paint", 0))
    # 主判据：空位回来了，而且颜料少了 —— 同一张画纸满铺之后不会发生这种事。
    if blank_now - blank_before >= 2 and paint_before - paint_now >= 2:
        return True
    previous_open = blank_before + int(previous_board.get("void", 0))
    if previous_open <= 0:
        return True
    if len(states) == len(previous_states):
        void_changed = sum(
            1
            for now, before in zip(states, previous_states)
            if (now == _tile_state_code(TileState.VOID)) != (before == _tile_state_code(TileState.VOID))
        )
        if void_changed >= 3:
            return True
    blocked_count = int(board.get("blocked", 0))
    return open_cells >= 8 and paint_now <= 8 and blocked_count <= 18


def _paint_collection_summary(paints: tuple[Paint, ...] | list[Paint]) -> dict:
    total = len(paints)
    level_counts = Counter(str(paint.level) for paint in paints)
    treasure_levels = [paint.level for paint in paints if paint.is_treasure]
    gameplay7 = sum(1 for paint in paints if paint.category == PaintCategory.GAMEPLAY and not paint.is_treasure and paint.level >= 7)
    monster7 = sum(1 for paint in paints if paint.category == PaintCategory.MONSTER and paint.level >= 7)
    treasure5 = sum(1 for level in treasure_levels if level >= 5)
    treasure6 = sum(1 for level in treasure_levels if level >= 6)
    treasure7 = sum(1 for level in treasure_levels if level >= 7)
    return {
        "total": total,
        "levels": dict(level_counts),
        "lv7": sum(1 for paint in paints if paint.level >= 7),
        "gameplay7": gameplay7,
        "monster7": monster7,
        "treasure": sum(1 for paint in paints if paint.is_treasure),
        "treasure5": treasure5,
        "treasure6": treasure6,
        "treasure7": treasure7,
        "max_treasure": max(treasure_levels) if treasure_levels else 0,
    }


def _capture_quality_gate(scene: SceneObservation, backpack: tuple[Paint, ...]) -> dict:
    min_board = float(config_value("quality.min_board", 0.48))
    min_current = float(config_value("quality.min_current", 0.45))
    min_backpack = float(config_value("quality.min_backpack", 0.35))
    min_brush = float(config_value("quality.min_brush", 0.35))
    min_overall = float(config_value("quality.min_overall", 0.58))
    max_unknown_tiles = int(config_value("quality.max_unknown_tiles", 1))
    board_score = float(scene.board.score)
    start_region = scene.region(RegionKind.START_DRAWING)
    start_scene = start_region is not None and float(start_region.confidence) >= 0.90
    current_score = 0.90 if start_scene else _current_paint_quality(scene.current_paint)
    # 笔锋面板一出来就把背包挡掉了，这是游戏设定，不是识别失败。
    # 不豁免的话整帧会被判「识别不稳」，阶段卡在 unstable，提示也就不会出箭头。
    brush_visible = scene.brush_options is not None
    if start_scene:
        backpack_score = 0.90
    elif scene.backpack is not None:
        backpack_score = float(scene.backpack.confidence)
    elif backpack:
        backpack_score = 0.72
    elif brush_visible:
        backpack_score = 0.72
    else:
        backpack_score = 0.30
    brush_score = float(scene.brush_options.confidence) if scene.brush_options is not None else 0.72
    tile_scores = [tile.confidence for tile in scene.board.tiles if tile.state != TileState.UNKNOWN]
    tile_score = sum(tile_scores) / len(tile_scores) if tile_scores else 0.0
    unknown_tiles = sum(1 for tile in scene.board.tiles if tile.state == TileState.UNKNOWN)
    score = (
        board_score * 0.34
        + current_score * 0.26
        + backpack_score * 0.20
        + brush_score * 0.10
        + tile_score * 0.10
    )
    reasons: list[str] = []
    if board_score < min_board:
        reasons.append(f"board={board_score:.2f}")
    if current_score < min_current:
        reasons.append(f"current={current_score:.2f}")
    if backpack_score < min_backpack:
        reasons.append(f"backpack={backpack_score:.2f}")
    if brush_score < min_brush:
        reasons.append(f"brush={brush_score:.2f}")
    if unknown_tiles > max_unknown_tiles:
        reasons.append(f"unknown_tiles={unknown_tiles}")
    if score < min_overall:
        reasons.append(f"overall={score:.2f}")
    ok = not reasons
    return {
        "ok": ok,
        "score": round(score, 3),
        "board": round(board_score, 3),
        "current": round(current_score, 3),
        "backpack": round(backpack_score, 3),
        "brush": round(brush_score, 3),
        "tile": round(tile_score, 3),
        "unknown_tiles": unknown_tiles,
        "reason": "ok" if ok else ", ".join(reasons),
    }


def _quick_capture_quality(
    scene: SceneObservation,
    elapsed: float,
    capture_perf: dict | None = None,
    *,
    capture_source: str | None = None,
) -> dict:
    board_score = round(float(scene.board.score), 3)
    unknown_tiles = sum(1 for tile in scene.board.tiles if tile.state == TileState.UNKNOWN)
    min_board = float(config_value("quality.quick_min_board", 0.66))
    max_unknown_tiles = int(config_value("quality.quick_max_unknown_tiles", 1))
    reasons: list[str] = []
    source = capture_source or "unknown"
    start_region = scene.region(RegionKind.START_DRAWING)
    strong_start_scene = (
        start_region is not None
        and float(start_region.confidence) >= 0.90
        and board_score >= 0.68
    )
    strong_active_scene = (
        scene.current_paint is not None
        and scene.backpack is not None
        and len(scene.backpack.paints) >= 5
        and board_score >= 0.72
    )
    if _capture_source_can_include_overlays(source) and not (strong_start_scene or strong_active_scene):
        reasons.append("capture_source_composited")
    if board_score < min_board:
        reasons.append(f"board={board_score:.2f}")
    if unknown_tiles > max_unknown_tiles:
        reasons.append(f"unknown_tiles={unknown_tiles}")
    reason = "quick_board_reuse"
    if not reasons and unknown_tiles:
        reason = f"quick_board_reuse_with_{unknown_tiles}_unknown"
    payload = {
        "ok": not reasons,
        "score": board_score,
        "reason": reason if not reasons else ", ".join(reasons),
        "mode": "quick_capture",
        "ocr": False,
        "elapsed": round(elapsed, 3),
        "unknown_tiles": unknown_tiles,
        "perf": capture_perf or {},
        "capture_source": source,
        # 本帧用的标定是哪一组、窗口到底多大。2026-11-23 那次「昨天稳定、
        # 今天全错但自评分没掉」就是缺这一项才查不出来：棋盘定位是按窗口
        # 尺寸分别标定的，没有匹配尺寸时会**静默**回落到内置默认比例，
        # 整盘偏移可每格依然读得「很自信」（unknown_tiles 仍是 0）。
        "layout": getattr(scene, "layout", None) or layout.last_layout_match(),
    }
    global _FAMILY_STATUS_TICK
    _FAMILY_STATUS_TICK += 1
    if _FAMILY_STATUS_TICK % _FAMILY_STATUS_EVERY == 0:
        # 背包「认不出家族」有两种完全不同的原因：后台线程死了 / 单纯没认出来。
        # 界面上都是空框，靠这一行才能在日志里分开。
        payload["family_worker"] = vision.family_worker_status()
    return payload


#: 「认家族」线程状态多久往观测里写一次（帧）。
_FAMILY_STATUS_EVERY = 20
_FAMILY_STATUS_TICK = 0


def _capture_source_can_include_overlays(source: str) -> bool:
    """这一帧里会不会混进别的窗口（整屏、桌面区域都算）。"""
    return capture_kind(source) in (CAPTURE_KIND_SCREEN, CAPTURE_KIND_WINDOW_REGION)


def _llm_may_receive_frame(capture) -> bool:
    """这一帧能不能进「待发给模型」的队列。

    只认窗口自己渲染的内容（``window_content``）。**不能只看 ``capture.window``
    是否存在**——窗口存在但各后端都失败时仍会回退成整屏，那条路径的
    ``CaptureResult`` 同样带着 ``window`` 对象。
    """
    return bool(capture is not None and getattr(capture, "trusted_for_automation", False))


def _llm_may_send_source(source_kind: str) -> bool:
    """发送前用最近一帧的来源分级再验一次。"""
    return str(source_kind or "") in TRUSTED_CAPTURE_KINDS


def _current_paint_quality(paint) -> float:
    if paint is None:
        return 0.0
    parts = [
        float(paint.category_confidence),
        float(paint.level_confidence),
        max(paint.keyword_scores.values(), default=0.0),
    ]
    if paint.match_required is None:
        parts.append(0.35)
    else:
        parts.append(0.75)
    if paint.upgrade_amount is None:
        parts.append(0.35)
    else:
        parts.append(0.75)
    return max(0.0, min(1.0, sum(parts) / len(parts)))


def _detect_capture_rule_anomalies(
    previous: dict | None,
    scene: SceneObservation,
    state: PuzzleState,
) -> list[dict]:
    anomalies: list[dict] = []
    current = scene.current_paint
    if current is not None and current.level is not None:
        expected_match, expected_upgrade = _empirical_current_paint_defaults(current.level)
        if current.match_required is not None and current.match_required != expected_match:
            anomalies.append(
                {
                    "kind": "current_match_required",
                    "severity": "warn",
                    "level": current.level,
                    "observed": current.match_required,
                    "expected": expected_match,
                    "hint": current.raw_text_hint,
                }
            )
        if current.upgrade_amount is not None and not _empirical_upgrade_is_allowed(current.level, current.upgrade_amount):
            anomalies.append(
                {
                    "kind": "current_upgrade_amount",
                    "severity": "warn",
                    "level": current.level,
                    "observed": current.upgrade_amount,
                    "expected": expected_upgrade,
                    "allowed": _empirical_allowed_upgrades(current.level),
                    "hint": current.raw_text_hint,
                }
            )
    if previous is not None:
        current_step = {
            "board": _compact_board_summary(scene),
            "tiles": _compact_tile_paints(state),
            "current": _compact_rule(state.current_rule),
        }
        changed_cells = _changed_capture_cells(previous, current_step)
        unexpected_levels = _unexpected_level_changes(previous, current_step, changed_cells)
        if unexpected_levels:
            anomalies.append(
                {
                    "kind": "unexpected_level_change",
                    "severity": "block",
                    "cells": unexpected_levels[:8],
                    "note": "相邻截图出现无法用当前规则直接解释的等级变化，可能是漏识别词缀、强制完成后状态变化或视觉误判",
                }
            )
        unexpected_transforms = _unexpected_paint_transforms(previous, current_step, changed_cells)
        if unexpected_transforms:
            anomalies.append(
                {
                    "kind": "unexpected_paint_transform",
                    "severity": "warn",
                    "cells": unexpected_transforms[:8],
                    "note": "颜料类型/宝箱属性发生变化；若非转锋/堆砌变化词条，可能是识别错误或漏规则",
                }
            )
        terrain_changes = _unexpected_terrain_changes(previous, current_step, changed_cells)
        if terrain_changes:
            anomalies.append(
                {
                    "kind": "unexpected_terrain_change",
                    "severity": "warn",
                    "cells": terrain_changes[:8],
                    "note": "地形发生变化；若非扩散/强制完成，应复查识别或补充词条规则",
                }
            )
        faded_cells = _forced_faded_cells(previous, current_step, changed_cells)
        if faded_cells:
            anomalies.append(
                {
                    "kind": "force_faded_seen",
                    "severity": "info",
                    "cells": faded_cells[:8],
                    "note": "检测到颜料变失色，通常来自强制完成破坏",
                }
            )
    return anomalies


def _unexpected_level_changes(previous: dict, current: dict, changed_cells: list[dict]) -> list[dict]:
    previous_rule = previous.get("current") or {}
    current_rule = current.get("current") or {}
    if previous_rule != current_rule:
        return []
    rule_paint = (current_rule.get("paint") or {})
    upgrade = int(current_rule.get("upgrade") or 0)
    if upgrade <= 0:
        return []
    target = current_rule.get("target")
    unexpected: list[dict] = []
    for cell in changed_cells:
        before = cell.get("from_paint")
        after = cell.get("to_paint")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        before_level = before.get("l")
        after_level = after.get("l")
        if not isinstance(before_level, int) or not isinstance(after_level, int):
            continue
        if after_level <= before_level:
            continue
        if _compact_paint_matches_target(after, target):
            expected_max = min(7, before_level + upgrade)
            if after_level <= expected_max:
                continue
        if before == rule_paint or after == rule_paint:
            continue
        unexpected.append(
            {
                "r": cell.get("r"),
                "c": cell.get("c"),
                "from": before,
                "to": after,
                "upgrade": upgrade,
                "target": target,
            }
        )
    return unexpected


def _unexpected_paint_transforms(previous: dict, current: dict, changed_cells: list[dict]) -> list[dict]:
    previous_rule = previous.get("current") or {}
    current_rule = current.get("current") or {}
    if previous_rule != current_rule:
        return []
    unexpected: list[dict] = []
    for cell in changed_cells:
        before = cell.get("from_paint")
        after = cell.get("to_paint")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        before_kind = (before.get("c"), before.get("t"), before.get("s"))
        after_kind = (after.get("c"), after.get("t"), after.get("s"))
        if before_kind == after_kind:
            continue
        # A pure level change is handled by _unexpected_level_changes.
        if before.get("code") != after.get("code") or before_kind != after_kind:
            unexpected.append(
                {
                    "r": cell.get("r"),
                    "c": cell.get("c"),
                    "from": before,
                    "to": after,
                }
            )
    return unexpected


def _unexpected_terrain_changes(previous: dict, current: dict, changed_cells: list[dict]) -> list[dict]:
    previous_rule = previous.get("current") or {}
    current_rule = current.get("current") or {}
    if previous_rule != current_rule:
        return []
    unexpected: list[dict] = []
    for cell in changed_cells:
        from_state = cell.get("from_state")
        to_state = cell.get("to_state")
        if from_state == to_state:
            continue
        if to_state == "P" or from_state == "P":
            continue
        if to_state == "F":
            continue
        if from_state == "?" or to_state == "?":
            continue
        unexpected.append(
            {
                "r": cell.get("r"),
                "c": cell.get("c"),
                "from_state": from_state,
                "to_state": to_state,
            }
        )
    return unexpected


def _forced_faded_cells(previous: dict, current: dict, changed_cells: list[dict]) -> list[dict]:
    result: list[dict] = []
    for cell in changed_cells:
        if cell.get("to_state") != "F":
            continue
        result.append(
            {
                "r": cell.get("r"),
                "c": cell.get("c"),
                "from_state": cell.get("from_state"),
                "from_paint": cell.get("from_paint"),
            }
        )
    return result


def _compact_paint_matches_target(paint: dict, target: str | None) -> bool:
    if target == PaintObject.MONSTER.value:
        return paint.get("c") == PaintCategory.MONSTER.value
    if target == PaintObject.GAMEPLAY.value:
        return paint.get("c") == PaintCategory.GAMEPLAY.value
    if target == PaintObject.ALL.value:
        return True
    return False


def _format_rule_anomaly(anomaly: dict) -> str:
    kind = anomaly.get("kind")
    severity = anomaly.get("severity", "warn")
    if kind in {"current_match_required", "current_upgrade_amount"}:
        label = "进度要求" if kind == "current_match_required" else "升级数"
        return (
            f"[{severity}] {label}异常：Lv{anomaly.get('level')} 识别={anomaly.get('observed')} "
            f"经验={anomaly.get('expected')}"
        )
    if kind == "unexpected_level_change":
        return f"[{severity}] 等级变化异常：{len(anomaly.get('cells') or [])}格变化暂不能用本地规则解释"
    if kind == "unexpected_paint_transform":
        return f"[{severity}] 颜料变化异常：{len(anomaly.get('cells') or [])}格类型/宝箱属性变化"
    if kind == "unexpected_terrain_change":
        return f"[{severity}] 地形变化提醒：{len(anomaly.get('cells') or [])}格地形变化"
    if kind == "force_faded_seen":
        return f"[{severity}] 强制破坏痕迹：{len(anomaly.get('cells') or [])}格变失色"
    return str(anomaly)


def _compact_paint(paint: Paint) -> dict:
    return compact_paint_with_code(paint)


def _quality_proxy(summary: dict) -> float:
    return (
        summary.get("treasure7", 0) * float(config_value("values.treasure7", 5000.0))
        + max(0, summary.get("treasure6", 0) - summary.get("treasure7", 0)) * float(config_value("values.treasure6", 3000.0))
        + max(0, summary.get("treasure5", 0) - summary.get("treasure6", 0)) * float(config_value("values.treasure5", 1000.0))
        + summary.get("gameplay7", 0) * float(config_value("values.gameplay7", 1000.0))
        + summary.get("monster7", 0) * float(config_value("values.monster7", 714.0))
    )


def _should_save_auto_trajectory(terminal: dict) -> tuple[bool, str]:
    gate = _high_score_gate(terminal)
    return bool(gate["passed"]), str(gate["primary_reason"])


def _high_score_gate(terminal: dict) -> dict:
    known = terminal.get("known") or {}
    score = _internal_run_score(terminal)
    treasure7 = int(known.get("treasure7", 0) or 0)
    treasure6_plus = int(known.get("treasure6", 0) or 0)
    treasure5_plus = int(known.get("treasure5", 0) or 0)
    lv7 = int(known.get("lv7", 0) or 0)
    gameplay7 = int(known.get("gameplay7", 0) or 0)
    monster7 = int(known.get("monster7", 0) or 0)
    thresholds = {
        "min_internal_score": float(config_value("autosave.min_internal_score", 3000.0)),
        "min_treasure7": int(config_value("autosave.min_treasure7", 1)),
        "min_treasure6": int(config_value("autosave.min_high_score_treasure6", config_value("autosave.min_treasure6", 1))),
        "min_lv7": int(config_value("autosave.min_high_score_lv7", 1)),
        "min_many_lv7": int(config_value("autosave.min_lv7", 3)),
        "min_gameplay7_mixed": int(config_value("autosave.min_gameplay7_mixed", 2)),
        "min_monster7_mixed": int(config_value("autosave.min_monster7_mixed", 1)),
        "min_treasure5_with_lv7": int(config_value("autosave.min_treasure5_with_lv7", 1)),
        "min_lv7_with_treasure5": int(config_value("autosave.min_lv7_with_treasure5", 2)),
    }
    reasons: list[str] = []
    #: 入库口径（用户 2026-11-15 二次确认，替代旧的「只看 7 级」）：
    #:
    #: 1. **库没满** → 全部入库。一晚上只攒几十局，先要样本量；
    #: 2. 库满了 → 至少 1 个 7 级颜料的局直接收；
    #: 3. 库满了且没有 7 级 → 只有内部评分**高于库内最低分**才写入，
    #:    写盘后 :func:`learning._prune_trajectories` 会按分数把最低那局挤出去。
    #:
    #: 以前「内部评分 ≥3000」「6 级宝箱」也能单独过线，于是 ``lv7=0`` 的普通局
    #: 照样进库（现存 14 条里就有 3 条）；现在这些条件只作为诊断信息。
    #: ``autosave.require_level7=false`` 表示库满后不再给 7 级局开直通，一律按分数比。
    require_level7 = bool(config_value("autosave.require_level7", True))
    limit = max(0, int(config_value("autosave.max_sessions", 500)))
    library_count = trajectory_count()
    library_full = limit > 0 and library_count >= limit
    lowest_score: float | None = None
    if not library_full:
        reasons.append("library_not_full")
    elif require_level7 and lv7 >= thresholds["min_lv7"]:
        reasons.append("level7_paint")
    else:
        lowest_score = lowest_trajectory_score()
        if lowest_score is None or score > lowest_score:
            reasons.append("beats_lowest_score")

    if "library_not_full" in reasons:
        summary = f"高分库未满（{library_count}/{limit}），本局直接入库"
    elif "level7_paint" in reasons:
        summary = "高分轨迹门槛通过（至少 1 个 7 级颜料）"
    elif "beats_lowest_score" in reasons:
        summary = f"评分 {score:.0f} 高于库内最低分 {lowest_score:.0f}，替换入库"
    else:
        summary = (
            f"未入库：库里已有 {library_count} 局，本局评分 {score:.0f} "
            f"不高于最低分 {lowest_score if lowest_score is not None else 0:.0f}，且没有 7 级颜料"
        )
    primary = reasons[0] if reasons else "below_high_score_gate"
    return {
        "passed": bool(reasons),
        "primary_reason": primary,
        "reasons": reasons,
        "rule": "library_then_score",
        "summary": summary,
        "score": score,
        "thresholds": thresholds,
        "library": {
            "count": library_count,
            "limit": limit,
            "full": library_full,
            "require_level7": require_level7,
            "lowest_score": lowest_score,
        },
        "counts": {
            "treasure7_plus": treasure7,
            "treasure6_plus": treasure6_plus,
            "treasure5_plus": treasure5_plus,
            "lv7": lv7,
            "gameplay7": gameplay7,
            "monster7": monster7,
        },
    }


def _summary_has_high_value(summary: dict) -> bool:
    return (
        int(summary.get("treasure5", 0) or 0) > 0
        or int(summary.get("gameplay7", 0) or 0) > 0
        or int(summary.get("monster7", 0) or 0) > 0
        or int(summary.get("lv7", 0) or 0) >= 2
    )


def _format_high_value_summary(summary: dict) -> str:
    parts: list[str] = []
    treasure7 = int(summary.get("treasure7", 0) or 0)
    treasure6 = int(summary.get("treasure6", 0) or 0)
    treasure5 = int(summary.get("treasure5", 0) or 0)
    gameplay7 = int(summary.get("gameplay7", 0) or 0)
    monster7 = int(summary.get("monster7", 0) or 0)
    if treasure7:
        parts.append(f"7级宝箱x{treasure7}")
    if treasure6:
        parts.append(f"6级以上宝箱x{treasure6}")
    elif treasure5:
        parts.append(f"5级以上宝箱x{treasure5}")
    if gameplay7:
        parts.append(f"7级玩法x{gameplay7}")
    if monster7:
        parts.append(f"7级怪物x{monster7}")
    if not parts:
        parts.append(f"7级颜料x{int(summary.get('lv7', 0) or 0)}")
    return "，".join(parts)


def _format_finish_diagnostic(diagnostic: dict | None) -> str:
    if not diagnostic:
        return "本局结束：暂无诊断信息。"
    terminal = diagnostic.get("terminal") or {}
    known = terminal.get("known") or {}
    board_known = terminal.get("board_known") or {}
    backpack = terminal.get("backpack") or {}
    quality = diagnostic.get("last_quality") or {}
    saved = bool(diagnostic.get("saved"))
    score = float(diagnostic.get("score", 0.0) or 0.0)
    reason = diagnostic.get("reason") or "-"
    finish_text = "有" if diagnostic.get("finish_seen") else "无"
    quality_text = _format_finish_quality(quality)
    high_value = _format_high_value_summary(known) if _summary_has_high_value(known) else "无高价值终局"
    board_total = int(board_known.get("total", 0) or 0)
    backpack_total = int(backpack.get("total", 0) or 0)
    result = "已保存结算轨迹" if saved else "未保存结算轨迹"
    return (
        f"结束诊断：{result}；原因：{reason}；内部评分={score:.0f}；"
        f"可靠步骤={int(diagnostic.get('reliable_steps', 0) or 0)}/{int(diagnostic.get('steps', 0) or 0)}；"
        f"结算证据={finish_text}；最后识别={quality_text}；"
        f"终局价值={high_value}；棋盘已识别颜料={board_total}，背包={backpack_total}。"
    )


def _format_finish_quality(quality: dict) -> str:
    if not quality:
        return "无"
    ok_text = "合格" if quality.get("ok") else "低置信"
    score = quality.get("score")
    score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "-"
    reason = quality.get("reason") or "-"
    return f"{ok_text}({score_text}, {reason})"


def _capture_image_size(image) -> tuple[int, int] | None:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _capture_shape_size(shape: tuple[int, ...] | None) -> tuple[int, int] | None:
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _valid_capture_size(size: tuple[int, int] | None) -> bool:
    if size is None:
        return False
    width, height = size
    return width > 0 and height > 0


def _fast_capture_perf(
    capture_elapsed: float,
    analysis_elapsed: float,
    write_elapsed: float = 0.0,
    full_refresh_elapsed: float = 0.0,
) -> dict:
    return {
        "capture": round(capture_elapsed, 3),
        "analysis": round(analysis_elapsed, 3),
        "write": round(write_elapsed, 3),
        "full_refresh": round(full_refresh_elapsed, 3),
    }


def _format_fast_capture_perf(perf: dict | None) -> str:
    if not perf:
        return ""
    parts = [
        f"截图={float(perf.get('capture', 0.0) or 0.0):.2f}s",
        f"分析={float(perf.get('analysis', 0.0) or 0.0):.2f}s",
    ]
    write_elapsed = float(perf.get("write", 0.0) or 0.0)
    full_refresh = float(perf.get("full_refresh", 0.0) or 0.0)
    record_elapsed = float(perf.get("record", 0.0) or 0.0)
    parts.append(f"record={record_elapsed:.3f}s")
    if perf.get("under_200ms") is False:
        parts.append("budget>0.2s")
    if write_elapsed > 0.01:
        parts.append(f"写图={write_elapsed:.2f}s")
    if full_refresh > 0.01:
        parts.append(f"完整补读={full_refresh:.2f}s")
    return "耗时拆分：" + "，".join(parts) + "。"


def _format_perf_panel(perf: dict) -> str:
    if not perf:
        return "性能：等待识别"
    mode = str(perf.get("mode") or "-")
    elapsed = _format_seconds(perf.get("elapsed"))
    capture_perf = perf.get("capture_perf") if isinstance(perf.get("capture_perf"), dict) else {}
    capture = _format_seconds(capture_perf.get("capture"))
    analysis = _format_seconds(capture_perf.get("analysis"))
    recommendation = _format_seconds(perf.get("recommendation"))
    record_elapsed = _format_seconds(capture_perf.get("record"))
    pet = _format_seconds(perf.get("pet_render"))
    quality = "通过" if perf.get("quality_ok") is True else "暂缓" if perf.get("quality_ok") is False else "-"
    pet_message = str(perf.get("pet_message") or "-")
    source = _short_source(perf.get("source"))
    return (
        f"性能：{mode} 总={elapsed} 截图={capture} 分析={analysis} "
        f"推荐={recommendation} 记录={record_elapsed} 桌宠={pet} 质量={quality} 桌宠状态={pet_message} 来源={source}"
    )


def _format_perf_details(perf: dict) -> list[str]:
    if not perf:
        return ["等待识别"]
    capture_perf = perf.get("capture_perf") if isinstance(perf.get("capture_perf"), dict) else {}
    lines = [
        f"模式={perf.get('mode', '-')}",
        f"总耗时={_format_seconds(perf.get('elapsed'))}",
        f"截图={_format_seconds(capture_perf.get('capture'))} 分析={_format_seconds(capture_perf.get('analysis'))}",
        f"写图={_format_seconds(capture_perf.get('write'))} 完整补读={_format_seconds(capture_perf.get('full_refresh'))}",
        f"自动记录={_format_seconds(capture_perf.get('record'))} 记录阶段={capture_perf.get('record_phase', '-')}",
        f"推荐={_format_seconds(perf.get('recommendation'))} 桌宠渲染={_format_seconds(perf.get('pet_render'))}",
        f"质量={'通过' if perf.get('quality_ok') is True else '暂缓' if perf.get('quality_ok') is False else '-'} 评分={_format_score(perf.get('quality_score'))}",
        f"原因={perf.get('quality_reason', '-')}",
        f"桌宠={perf.get('pet_state', '-')} / {perf.get('pet_message', '-')}",
        f"来源={perf.get('source', '-')}",
    ]
    if capture_perf.get("pump_interval") is not None:
        lines.insert(4, f"帧泵间隔={_format_seconds(capture_perf.get('pump_interval'))}")
    return lines


def _format_seconds(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}s"
    return "-"


def _format_score(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "-"


def _short_source(source) -> str:
    text = str(source or "-")
    if text.startswith("window-content"):
        return "窗口内容"
    if text.startswith("window-foreground-screen"):
        return "前台窗口"
    if text.startswith("window-screen"):
        return "桌面区域"
    if text.startswith("screen:fallback"):
        return "桌面兜底"
    if text.startswith("paused:game-window-missing"):
        return "等待游戏窗口"
    if text.startswith("paused:"):
        return "暂停采集"
    return text[:24]


def _scene_has_finish_evidence(scene: SceneObservation | None, state: PuzzleState | None) -> bool:
    if scene is None:
        return False
    finish_region = scene.region(RegionKind.FINISH_DRAWING)
    # 「入画」按钮在作画界面常驻，不是「本局已画完」的证据（见 _detect_settlement_candidate）。
    if (
        finish_region is not None
        and finish_region.confidence >= 0.58
        and not finish_region.is_enter_painting
    ):
        return True
    if state is None:
        return False
    counts = Counter(tile.state for tile in state.tiles)
    # User-observed rule: completion can be available once all placeable blank
    # cells are gone. Void cells may remain because they are not placeable unless
    # a special spread affix converts/allows them.
    return counts.get(TileState.BLANK, 0) == 0


def _finish_reliability_reason(
    terminal: dict,
    steps: list[dict],
    *,
    finish_seen: bool,
    last_quality: dict | None = None,
) -> str:
    if last_quality is not None and not last_quality.get("ok") and not finish_seen:
        return f"最后一次识别低置信度（{last_quality.get('reason', '-') }），已跳过高分轨迹保存"
    reliable, reason = _terminal_reliability_for_finish(terminal, steps, finish_seen=finish_seen)
    return reason if not reliable else "可保存"


def _terminal_reliability_for_finish(terminal: dict, steps: list[dict], *, finish_seen: bool = False) -> tuple[bool, str]:
    if not steps:
        return False, "没有真实步骤记录，请先完整识别或重新快速获取一次"
    reliable_steps = [step for step in steps if (step.get("quality") or {}).get("ok") is True]
    if not reliable_steps:
        return False, "没有通过质量门槛的真实步骤，已跳过高分轨迹保存"
    latest = reliable_steps[-1]
    latest_quality = latest.get("quality") or {}
    if not finish_seen and float(latest_quality.get("score", 0.0) or 0.0) < float(config_value("quality.finish_min_board", 0.66)):
        return False, f"最后可靠步骤棋盘分数过低（{float(latest_quality.get('score', 0.0) or 0.0):.2f}），已跳过高分轨迹保存"
    known = terminal.get("known") or {}
    board_known = terminal.get("board_known") or {}
    total_known = int(known.get("total", 0) or 0)
    board_total = int(board_known.get("total", 0) or 0)
    if total_known <= 0:
        return False, "终局没有识别到任何颜料/背包内容，已跳过高分轨迹保存"
    if _summary_has_high_value(known):
        return True, "高价值终局可保存"
    # Low-value terminal runs are still useful only when the board paint
    # recognition has meaningful content. Avoid saving empty/garbled endings.
    min_board_paints = int(config_value("quality.finish_min_board_paints", 8))
    if finish_seen and board_total < min_board_paints:
        return False, f"已识别结算，但棋盘颜料等级识别少于{min_board_paints}个；为避免假成功，已跳过高分轨迹保存"
    if finish_seen and total_known > 0:
        return True, "有完成绘制/无空白地块证据，且棋盘颜料识别足够，可保存"
    if board_total >= min_board_paints:
        return True, "终局棋盘颜料数量足够，可保存"
    return False, f"终局价值较低且棋盘颜料识别少于{min_board_paints}个，已跳过高分轨迹保存"


def _terminal_is_reliable_for_finish(terminal: dict, steps: list[dict], *, finish_seen: bool = False) -> bool:
    reliable, _reason = _terminal_reliability_for_finish(terminal, steps, finish_seen=finish_seen)
    return reliable


def _stable_payload_key(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_session_id() -> str:
    return time.strftime("run-%Y%m%d-%H%M%S")


def _empirical_current_paint_defaults(level: int) -> tuple[int, int]:
    if level >= 5:
        return 4, 2
    if level == 4:
        return 4, 1
    return 3, 1


def _empirical_allowed_upgrades(level: int) -> list[int]:
    if level >= 5:
        return [2]
    if level == 4:
        return [1, 2]
    return [1]


def _empirical_upgrade_is_allowed(level: int, upgrade_amount: int) -> bool:
    return upgrade_amount in _empirical_allowed_upgrades(level)


def _tile_display_label(
    state: TileState,
    paint: Paint | None,
    observation: TileObservation | None = None,
) -> str:
    """棋盘格中央的文字。

    原来只输出 ``M2`` / ``G3`` 这种类别缩写，用户没法把界面和游戏画面逐格对照
    ——「M2」到底是矮人还是鼠人得靠猜。这里优先用视觉层这一帧直接读到的子类型，
    读不到才退回类别缩写。
    """
    if paint is None:
        return STATE_LABELS[state]
    subtype = (observation.paint_subtype if observation is not None else None) or paint.subtype
    level = int(paint.level) if paint.level else 0
    if subtype and subtype != "宝箱":
        return f"{subtype} L{level}"
    if paint.is_treasure:
        return f"宝箱 L{level}"
    if paint.category == PaintCategory.MONSTER or paint.category.value == "monster":
        return f"怪 L{level}"
    if paint.category == PaintCategory.GAMEPLAY or paint.category.value == "gameplay":
        return f"玩 L{level}"
    # 类别也认不出来：以前这里返回状态名「颜」，等于什么都没说。
    return f"未识 L{level}" if level else STATE_LABELS[state]


def _tile_observation_is_fresh(observation: TileObservation | None) -> bool:
    """这一帧是否真的在棋盘上看到了这格颜料。

    看不到颜料却仍有 ``paint``，说明该值来自上一帧/模型推断；界面要标出来，
    否则用户会以为是自己刚放的那枚读错了。
    """
    if observation is None:
        return False
    if observation.state is not TileState.PAINT:
        return False
    return observation.paint_category is not PaintCategory.UNKNOWN


def _paint_from_tile_observation_for_display(observation: TileObservation | None) -> Paint | None:
    """把视觉层这一帧读到的颜料转成可绘制的 ``Paint``；没读到就是 None。"""
    if not _tile_observation_is_fresh(observation):
        return None
    assert observation is not None
    level = observation.paint_level
    return Paint(
        category=observation.paint_category,
        level=int(level) if level else 1,
        subtype=observation.paint_subtype,
        is_treasure=bool(observation.paint_is_treasure),
    )


def _paint_display_label(paint: Paint) -> str:
    if paint.is_treasure:
        return f"宝箱 Lv{paint.level}"
    category = "怪物" if paint.category.value == "monster" else "玩法" if paint.category.value == "gameplay" else "未知"
    subtype = f"-{paint.subtype}" if paint.subtype else ""
    return f"{category}{subtype} Lv{paint.level}"


def _paint_short_label(paint: Paint) -> str:
    if paint.is_treasure:
        return "宝箱"
    if paint.category.value == "monster":
        return "怪物"
    if paint.category.value == "gameplay":
        return "玩法"
    return "未知"


def _backpack_slot_label(paint: Paint) -> str:
    """背包格上那一行字。

    背包面板里每枚颜料都带名字（宝箱 / 矮人 / 鼠人 / 黑潮 …），原来只显示
    「怪物」「玩法」，用户没法核对到底读了什么。名字太长时截到 4 个字。
    """
    if paint.is_treasure:
        return "宝箱"
    subtype = paint.subtype
    if subtype:
        return subtype if len(subtype) <= 4 else subtype[:4]
    return _paint_short_label(paint)


def _brush_label(brush_type: BrushType) -> str:
    return {
        BrushType.TRANSFORM: "转锋",
        BrushType.STACK: "堆砌",
        BrushType.SPREAD: "扩散",
    }[brush_type]


def _brush_type_from_value(value: str | None) -> BrushType | None:
    if value is None:
        return None
    try:
        return BrushType(value)
    except ValueError:
        return None


def _brush_types_from_values(values: tuple[str, ...]) -> tuple[BrushType, ...]:
    parsed = [_brush_type_from_value(value) for value in values]
    return normalize_brush_options([brush_type for brush_type in parsed if brush_type is not None])


def _scene_backpack_page_key(scene: SceneObservation) -> str:
    """背包这一帧属于哪一页：按**当前初始颜料的目标**分页。

    怪物初始颜料 → 背包按怪物排序 → ``monster`` 页；
    玩法初始颜料 → 按玩法排序 → ``gameplay`` 页。两页各 15 格、合计最多 30 格
    （同一枚颜料两页都在是正常的，用户 2026-11-18 明确要求「可能有重复」）。

    **目标还没读出来时必须返回空串，不能造一个 ``unknown`` 页**（2026-11-18 实机
    抓到的 bug）：选色面板刚弹出、``current_paint`` 还认不出时，以前会开一个新页
    ``unknown`` 并往里写满 15 格空读数，快照于是变成 `monster 15 + unknown 15`，
    「这局有哪些颜料」被一整页 None 污染。返回空串 → 累计器沿用当前页。
    """
    # 实现放在 fast_vision：实时帧（frame_pump）也要用同一份口径，两份实现必然漂移。
    return backpack_page_of(scene)


def _anchor_tile_holds_rule_paint(
    scene: SceneObservation, rule: CurrentPaintRule, anchor: tuple[int, int]
) -> bool:
    """锚点那一格现在还是不是「同一类颜料」。

    这是判断「初始颜料有没有换」唯一靠得住的一条物理证据：颜料就在棋盘上，
    而面板上的等级/方向/作用对象都会被词缀合法改掉。类别识别不出（unknown）
    时按「还算在」处理——宁可认为锚点还在，也不要在识别抖动时把锚点丢掉。
    """
    try:
        tile = scene.board.tile(anchor[0], anchor[1])
    except (IndexError, AttributeError):
        return False
    if tile is None or tile.state != TileState.PAINT:
        return False
    if tile.paint_category == PaintCategory.UNKNOWN:
        return True
    return tile.paint_category == rule.paint.category


def _same_initial_paint(rule: CurrentPaintRule, previous: CurrentPaintRule) -> bool:
    """两条规则是不是**同一枚初始颜料**。

    只比类别与「是不是宝箱」，**不比等级**：
    「使自身等级提升2级」会把面板等级从 5 抬到 7（用户 2026-11-14 实机就是这么变的），
    「随机改变自身等级」(官方 4005) 更是每档都可能跳。拿等级当身份，等于每升一次级
    就把自己认成「换了颜料」，于是锚点被清空、词缀被丢弃。
    """
    return (
        rule.paint.category == previous.paint.category
        and bool(rule.paint.is_treasure) == bool(previous.paint.is_treasure)
    )


def _fast_current_paint_changed(scene: SceneObservation, previous_state: PuzzleState | None) -> bool:
    """这一帧是不是**换了新的初始颜料**（旧锚点必须作废）。

    只比类别/等级是不够的：新一局抽到「纵向·所有·2/4」而上一局是「纵向·所有·3/4」，
    类别和等级可能完全一样，旧判定会说「没变」，于是界面一直挂着上一局的规则
    （用户看到的目标/匹配数和面板上写的不是一回事）。
    所以方向、目标、匹配需求、升级量都比一遍——但只在**这一帧确实读到了**
    该字段时才比，读不到（unknown/None）就当作没证据，免得识别抖动被当成规则变化。

    ## 为什么先看棋盘上的锚点（2026-11-14 实机事故）

    旧实现只要面板上有任何一项字段变了就返回 True，调用方随即
    ``current_anchor=None`` + ``current_rule_pending=True``。实机后果是：
    伊吉已经落在 r5c6（面板 2/4），但作用对象在「所有 / 玩法」之间来回翻
    （栅格模板把「所有」读成「玩法」），每一翻都被当成「换了颜料」，
    于是推荐变成「先选一个空白地块放入初始颜料」——**让玩家把第二枚锚点往空白格怼**。
    锚点还在原位、还是同一类颜料时，任何字段差异都不可能是「换了颜料」，
    所以这里先过这道闸。
    """
    current = scene.current_paint
    if current is None or previous_state is None or previous_state.current_rule is None:
        return False
    previous = previous_state.current_rule
    anchor = previous_state.current_anchor
    if anchor is not None and _anchor_tile_holds_rule_paint(scene, previous, anchor):
        return False
    changed: list[bool] = []
    if current.category != PaintCategory.UNKNOWN and current.category_confidence >= 0.62:
        changed.append(current.category != previous.paint.category)
    if current.level is not None and current.level_confidence >= 0.62:
        changed.append(abs(int(current.level) - int(previous.paint.level)) >= 2)
    if current.is_treasure != previous.paint.is_treasure and current.category_confidence >= 0.70:
        changed.append(True)
    if current.direction != PaintDirection.UNKNOWN:
        changed.append(current.direction != previous.direction)
    if current.target_object != PaintObject.UNKNOWN:
        changed.append(current.target_object != previous.target_object)
    if current.match_required is not None:
        changed.append(int(current.match_required) != int(previous.match_required))
    if current.upgrade_amount is not None:
        changed.append(int(current.upgrade_amount) != int(previous.upgrade_amount))
    return any(changed)


def _merge_fast_rule_state(
    state: PuzzleState,
    previous_state: PuzzleState,
    scene: SceneObservation,
    backpack: tuple[Paint, ...],
    *,
    changed: bool | None = None,
) -> PuzzleState:
    """把本帧识别结果与上一帧的派生状态合起来（快路径的核心决策）。

    抽成纯函数是为了可测：这段以前内联在 ``_render_fast_scene_result`` 里，
    「目标抖一下就把锚点清掉」这类事故只能靠实机复现，单测碰不到。
    """
    if changed is None:
        changed = _fast_current_paint_changed(scene, previous_state)
    if changed:
        # 真的是新的初始颜料：上一枚的锚点/词缀全部作废。
        return replace(
            state,
            active_affix=None,
            backpack=sort_backpack_for_rule(backpack, state.current_rule),
            current_anchor=None,
            current_buff_id=previous_state.current_buff_id + 1,
            current_rule_pending=state.current_rule is not None,
        )
    if previous_state.current_rule is None:
        return state
    # 同一枚初始颜料 → 沿用上一帧的派生状态（词缀、锚点、进度）。
    # 规则本身仍然用本帧的：以前这里无条件写回 previous 的规则，
    # 结果只要「类别+等级」看起来没变，界面就永远挂着上一局的规则，
    # 用户看到的目标/匹配数和面板上写的对不上。
    #
    # 「同一枚」只比类别/宝箱，不比等级（见 :func:`_same_initial_paint`）：
    # 「使自身等级提升2级」会让面板等级 5→7，拿等级当身份就会把同一枚颜料
    # 认成新的，锚点和词缀全丢。
    rule = state.current_rule or previous_state.current_rule
    same_rule = _same_initial_paint(rule, previous_state.current_rule)
    # 锚点优先沿用上一帧的（避免识别抖动让作用范围来回跳），但旧锚点那一格
    # 已经没有同类颜料时必须让位给本帧的判定。
    latched_anchor = previous_state.current_anchor
    if latched_anchor is not None and not _anchor_tile_holds_rule_paint(
        scene, previous_state.current_rule, latched_anchor
    ):
        latched_anchor = None
    return replace(
        state,
        current_rule=rule,
        active_affix=previous_state.active_affix if same_rule else None,
        backpack=sort_backpack_for_rule(backpack, rule),
        current_anchor=(
            (latched_anchor if latched_anchor is not None else state.current_anchor)
            if same_rule
            else state.current_anchor
        ),
        current_buff_id=previous_state.current_buff_id,
        # 「待落初始颜料」是**粘性假**：一旦棋盘上看到锚点（或进度 >0），
        # 这条规则剩余时间里就不要再翻回「还没落」。旧写法无条件沿用上一帧，
        # 会把开头那几帧的 True 一直挂着，落子之后提示还是「先放初始颜料」。
        current_rule_pending=(
            previous_state.current_rule_pending and state.current_rule_pending
            if same_rule
            else state.current_rule_pending
        ),
        observed_scope_cells=state.observed_scope_cells or previous_state.observed_scope_cells,
    )


def recording_mode_enabled() -> bool:
    """记录模式：只录数据，不算任何推荐。

    用户 2026-11-22 明确要求：「现在是记录模式，不是推荐模式，任何推荐的都需要
    关掉！不要浪费不必要的时间」。

    管的是**主采集路径的计算**（``recommend_placements`` / ``recommend_install_plan``
    / ``recommend_affixes``）——每帧几十到几百毫秒，纯属白烧。
    和 ``pet.recommendations`` 不是一回事：那个只管桌宠浮层画不画标记。
    """

    return bool(config_value("recording_mode", False))


def _state_for_visible_brush_panel(
    state: PuzzleState,
    observation: BrushOptionsObservation | None,
) -> PuzzleState:
    """笔触面板可见时的状态修正：**只说明正在挑词缀**，不说明初始颜料没落。

    2026-11-14 实机：伊吉已经落在 r5c6（面板进度 2/4），笔触面板仍开着。
    旧实现顺手把 ``current_anchor`` 清空、``current_rule_pending`` 置真，
    于是推荐变成「先选一个空白地块放入初始颜料」——玩家会被推着往空白格落锚点，
    而这一笔本来就该落在调色范围里。锚点是否已经落子只有棋盘说了算
    （:func:`game_model.state_from_observation` 自己会判），面板在不在不构成证据。
    这里只清 ``active_affix``：面板开着说明正在挑新词缀，上一条不再生效。
    """
    if observation is None or state.current_rule is None:
        return state
    return replace(state, active_affix=None)


def _direction_display_label(direction) -> str:
    return {
        "horizontal": "横向",
        "vertical": "纵向",
        "diagonal": "对角斜向",
        "adjacent": "相邻",
        "global": "全图",
        "unknown": "未知方向",
    }.get(direction.value, direction.value)


def _target_display_label(target) -> str:
    return {
        "monster": "怪物",
        "gameplay": "玩法",
        "all": "所有",
        "blocked_tile": "阻挡地块",
        "blank_tile": "空白地块",
        "void_tile": "虚空地块",
        "unknown": "颜料",
    }.get(target.value, target.value)


def _backpack_layout(count: int, width: float) -> tuple[int, int, int, int, int]:
    if count <= 0:
        return 1, 0, 0, 0, 0
    gap = 6
    # 槽位改成能放下「品质框 + 图案」的方脸（原来 38 高只够写两行字）。
    # 高度固定 58、宽度一律 58：图标是方的，宽度不一会让同一张图在不同格
    # 被缩放成不同大小，看起来像「有的格子图案偏小」。
    per_row = min(count, max(6, min(10, int(width // 62))))
    slot_w = max(52, min(58, int((width - max(0, per_row - 1) * gap) // per_row)))
    slot_h = 58
    rows = math.ceil(count / per_row)
    return per_row, rows, slot_w, slot_h, gap


def _replace_option_menu(menu_widget: tk.OptionMenu, variable: tk.StringVar, labels: list[str]) -> None:
    menu = menu_widget["menu"]
    menu.delete(0, "end")
    for label in labels:
        menu.add_command(label=label, command=tk._setit(variable, label))


def _point_in_rect(point: tuple[float, float], rect: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _rects_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    left1, top1, right1, bottom1 = first
    left2, top2, right2, bottom2 = second
    return left1 < right2 and right1 > left2 and top1 < bottom2 and bottom1 > top2


def _clamp_int_var(value: tk.IntVar, minimum: int, maximum: int) -> int:
    try:
        raw = int(value.get())
    except (tk.TclError, ValueError):
        raw = minimum
    clamped = max(minimum, min(maximum, raw))
    if clamped != raw:
        value.set(clamped)
    return clamped
