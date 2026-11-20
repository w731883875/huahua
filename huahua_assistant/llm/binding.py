"""把「一条模型建议」钉死在「一帧画面」上。

模型一次调用要 2~5 秒。这段时间里玩家可能已经换了局、点了笔锋、把格子占了。
如果只在答复回来时看一眼 ``llm_var`` 开关，就会把**上一局的建议**画到这一局的
箭头上——玩家看到箭头指着刚放好颜料的那一格，或者指着已经不存在的下卡。

这个模块只做三件纯计算的事，不碰 tkinter、不碰线程、不读配置：

1. :func:`scene_signature` —— 把"这是哪一局的哪个阶段、候选卡是哪几张"压成一个
   可比较的值，请求时冻结、答复时比对；
2. :func:`match_option_index` —— 卡片词 → 候选序号。先按官方规范词缀 ID 认，
   认不到再认别名；**零个或多个匹配都返回 ``None``**（宁可不指，也不指错卡）；
3. :func:`why_target_invalid` —— 展示前再验一次落点：格子还能放吗、背包槽还
   在吗、这张卡还属于当前阶段吗。

时间一律由调用方传入（``now``），模块内部不读时钟——测试才能用假时钟锁住
"重绘不续期"这件事。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..codes import affix_aliases, canonical_affix_id, normalize_affix_text
from ..game_model import TileState
from ..models import RegionKind, SceneObservation
from .types import (
    ACTION_BACKPACK,
    ACTION_BRUSH1,
    ACTION_BRUSH2,
    ACTION_FINISH,
    ACTION_PLACE,
    ACTION_UNKNOWN,
    AdvisorAdvice,
)


#: 一条建议从采集那一帧算起最多挂多久。模型自己就要 2~5 秒，给到 6 秒，
#: 但**起点是原始帧的采集时刻**，不是界面重绘时刻——否则箭头能被无限续命。
ADVICE_TTL_SECONDS = 6.0

STAGE_NONE = "none"
STAGE_IDLE = "idle"
STAGE_PLACING = "placing"
STAGE_BRUSH1 = "brush_stage1"
STAGE_BRUSH2 = "brush_stage2"


def _option_label(item) -> str:
    """候选卡的名字。枚举就取 ``value``，字符串原样用。"""
    return str(getattr(item, "value", item) or "")


def stage_of(scene: SceneObservation | None) -> str:
    """当前处在哪一步。判断顺序按界面实际出现的顺序来，别按"哪个字段不为空"。"""
    if scene is None:
        return STAGE_NONE
    options = getattr(scene, "brush_options", None)
    if options is not None:
        if tuple(getattr(options, "stage2_options", ()) or ()):
            return STAGE_BRUSH2
        if tuple(getattr(options, "stage1_options", ()) or ()):
            return STAGE_BRUSH1
    if getattr(scene, "current_paint", None) is not None:
        return STAGE_PLACING
    return STAGE_IDLE


def stage_candidates(scene: SceneObservation | None) -> tuple[str, ...]:
    """当前阶段对应的候选卡名单（按界面顺序）。"""
    stage = stage_of(scene)
    options = getattr(scene, "brush_options", None)
    if options is None:
        return ()
    if stage == STAGE_BRUSH2:
        return tuple(_option_label(item) for item in (options.stage2_options or ()))
    if stage == STAGE_BRUSH1:
        return tuple(_option_label(item) for item in (options.stage1_options or ()))
    return ()


@dataclass(frozen=True)
class SceneSignature:
    """一帧画面的语义指纹。字段少而稳定，多一点都会让"没变"被判成"变了"。"""

    stage: str
    candidates: tuple[str, ...]
    #: 上一阶段已经选定的那一张。二阶段的合法性依赖它，换卡就要重算。
    chosen_stage1: str = ""
    #: 局号/跟随代次。重新开自动跟随就换号。
    session: int = 0
    #: 游戏窗口句柄。换窗口（重开游戏）就必须丢弃旧建议。
    window: int = 0

    @property
    def key(self) -> tuple:
        return (self.stage, self.candidates, self.chosen_stage1, self.session, self.window)


def scene_signature(
    scene: SceneObservation | None,
    *,
    session: int = 0,
    window: int = 0,
) -> SceneSignature:
    options = getattr(scene, "brush_options", None)
    chosen = str(getattr(options, "chosen_stage1", "") or "") if options is not None else ""
    return SceneSignature(
        stage=stage_of(scene),
        candidates=stage_candidates(scene),
        chosen_stage1=chosen,
        session=int(session or 0),
        window=int(window or 0),
    )


def match_option_index(option: str | None, labels: tuple[str, ...] | list[str]) -> int | None:
    """把模型说的卡片词落到候选名单里的位置。

    只认**唯一**命中：候选里有两张同名卡时返回 ``None``——这种局面下"指哪张"
    本身就是错的，宁可退回等待提示。零命中同样是 ``None``，绝不默认第 0 张。
    """
    text = str(option or "").strip()
    names = [str(label or "") for label in labels]
    if not text or not names:
        return None

    # ① 官方规范词缀 ID。别名、旧编号、``brush:alias`` 形式都会在这里归一。
    target_id = canonical_affix_id(text)
    if target_id:
        hits = [
            index
            for index, label in enumerate(names)
            if canonical_affix_id(label) == target_id
        ]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return None

    # ② 原始文本：模型常常把卡片名一字不差抄回来，这条最稳。
    key = normalize_affix_text(text)
    if not key:
        return None
    direct = [index for index, label in enumerate(names) if normalize_affix_text(label) == key]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        return None

    # ③ 别名：卡片自己登记的别名表。同样要求唯一命中。
    alias_hits: list[int] = []
    for index, label in enumerate(names):
        for alias in affix_aliases(label):
            if normalize_affix_text(alias) == key:
                alias_hits.append(index)
                break
    return alias_hits[0] if len(alias_hits) == 1 else None


def parse_slot_index(option: str | None) -> int | None:
    """从模型给的 option 里抠出「第几格」。只认阿拉伯数字，认不出返回 ``None``。"""
    if not option:
        return None
    match = re.search(r"\d+", str(option))
    if match is None:
        return None
    try:
        return int(match.group())
    except ValueError:
        return None


def why_target_invalid(
    advice: AdvisorAdvice,
    *,
    scene: SceneObservation | None,
    state,
) -> str | None:
    """展示前验一次落点。返回 ``None`` 表示可以指，否则是拒绝原因。

    ``state`` 是规则推演出的盘面（``game_model.PuzzleState``），可以是 ``None``：
    没有模型状态时只能验"格子存在"，验不了"还能不能放"。
    """
    if advice is None or not advice.ok:
        return "advice_not_ok"
    action = advice.action
    if action == ACTION_UNKNOWN:
        return "action_unknown"
    if scene is None:
        return "no_scene"

    if action == ACTION_PLACE:
        cell = advice.cell
        if cell is None:
            return "no_cell"
        row, col = cell
        board = getattr(scene, "board", None)
        if board is None or not _board_has(board, row, col):
            return "cell_out_of_board"
        if state is not None and not _state_tile_blank(state, row, col):
            # 模型说要放的格子已经不是空白了：它看的是旧画面。
            return "cell_not_placeable"
        return None

    if action in (ACTION_BRUSH1, ACTION_BRUSH2):
        expected = STAGE_BRUSH1 if action == ACTION_BRUSH1 else STAGE_BRUSH2
        if stage_of(scene) != expected:
            return "stage_mismatch"
        labels = stage_candidates(scene)
        if not labels:
            return "no_candidates"
        if match_option_index(advice.option, labels) is None:
            return "option_unmatched"
        return None

    if action == ACTION_BACKPACK:
        paints = tuple(getattr(getattr(scene, "backpack", None), "paints", ()) or ())
        if not paints:
            return "backpack_hidden"
        index = parse_slot_index(advice.option)
        if index is None or not (1 <= index <= len(paints)):
            return "slot_not_visible"
        return None

    if action == ACTION_FINISH:
        if scene.region(RegionKind.FINISH_DRAWING) is None:
            return "finish_region_missing"
        return None

    return "action_unsupported"


def _board_has(board, row: int, col: int) -> bool:
    try:
        board.tile(int(row), int(col))
    except Exception:  # noqa: BLE001 - 越界/缺格一律当"不在棋盘上"
        return False
    return True


def _state_tile_blank(state, row: int, col: int) -> bool:
    try:
        tile = state.tile(int(row), int(col))
    except Exception:  # noqa: BLE001
        return False
    return getattr(tile, "state", None) == TileState.BLANK


@dataclass(frozen=True)
class AdviceBinding:
    """一条建议 + 它出生时那一帧的指纹 + 采集时刻。"""

    signature: SceneSignature
    captured_at: float
    generation: int = 0

    @property
    def expires_at(self) -> float:
        return float(self.captured_at) + ADVICE_TTL_SECONDS

    def why_not_current(
        self,
        *,
        now: float,
        current: SceneSignature | None,
        generation: int,
    ) -> str | None:
        """这条建议还能不能用。返回 ``None`` 表示可用，否则是作废原因。"""
        if int(generation) != int(self.generation):
            return "generation_changed"
        if float(now) > self.expires_at:
            return "expired"
        if current is None:
            return "no_scene"
        mine, theirs = self.signature, current
        if mine.session != theirs.session:
            return "session_changed"
        if mine.window != theirs.window:
            return "window_changed"
        if mine.stage != theirs.stage:
            return "stage_changed"
        if mine.candidates != theirs.candidates:
            return "candidates_changed"
        if mine.chosen_stage1 != theirs.chosen_stage1:
            return "stage1_changed"
        return None
