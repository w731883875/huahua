"""雪原练习模式 + 高分复盘内核（与 UI 解耦，可被 CLI / 桌宠 / 测试共用）。

设计边界（按用户明确要求收敛）：

* **只练"放置颜料"** —— 不刷副本、不算技艺（天赋）、不涉及掉落结算；
* **棋盘 6×6**，与现有 ``game_model.random_state`` 一致；
* **刷新次数按宠物等级**：Lv1–2 为 0、Lv3–4 为 1、Lv5–6 为 2
  （``PET_REFRESH_BY_LEVEL``）；且**只能刷新第二阶段的词缀**，
  第一阶段（画笔类型）不可刷新；
* 随机源全部可注入，同一 seed 必须能完整复现（这是"能够还原"的前提）。

三条对外能力：

1. :func:`start_practice` —— 开一局练习，棋盘/规则/背包全部按官方口径生成；
2. :class:`PracticeSession` —— 逐步落笔，记录每一步的快照，可用于回放；
3. :func:`explain_high_score` —— 给一局终局打分并**说明为什么高分**，同时
   挑出对得分的技巧点（见 :data:`TACTIC_LIBRARY`）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from . import game_model as gm
from . import s8_official_data
from .engine.value import DEFAULT_WEIGHTS, EvalWeights, board_value, level7_counts


# ---------------------------------------------------------------------------
# 技巧库：每条都是「可判定的局面特征 → 一句人话」
# ---------------------------------------------------------------------------
#
# 这里的判据全部能由棋盘状态直接算出，不含主观描述。
# 用户点名的那条（宝箱 + 地块连接少 + 开局放一枚就强制调色）就是
# ``TACTIC_FORCE_ON_ISOLATED_TREASURE``。

TACTIC_FORCE_ON_ISOLATED_TREASURE = 'force_on_isolated_treasure'
TACTIC_STACK_BONUS_THEN_LOW_PAINT = 'stack_bonus_then_low_paint'
TACTIC_ANCHOR_CENTER = 'anchor_center'
TACTIC_TREASURE_HIGH_LEVEL = 'treasure_high_level'
TACTIC_AVOID_OVERFILL = 'avoid_overfill'
TACTIC_KEEP_BONUS_TILE = 'keep_bonus_tile'

#: 练习存档格式版本。改动 ``snapshot()`` 的字段含义时必须升版本，
#: 否则老存档会被静默按新格式解释，比直接报错更糟。
PRACTICE_SNAPSHOT_SCHEMA = 'huahua-practice-snapshot-v1'

#: 技巧编号 → (标题, 触发条件说明, 操作建议)
TACTIC_LIBRARY: dict[str, tuple[str, str, str]] = {
    TACTIC_FORCE_ON_ISOLATED_TREASURE: (
        '开局直接强制调色：先落 1 枚背包颜料，再强制调色',
        '官方规定必须先放入颜料才能强制调色（棋盘上此时共 2 枚：1 枚初始颜料 + 1 枚背包颜料）。'
        '当初始颜料的相邻可放格 ≤ 2、或作用范围内剩余空位凑不满连接数，且背包里带宝箱时成立。',
        '这种局面靠逐个匹配吃不到分，不如只落 1 枚就把当前规则强制结算，'
        '用一次强制调色的代价换一张干净的棋盘去接后续规则；宝箱留在背包里等下一轮更值分的规则。',
    ),
    TACTIC_STACK_BONUS_THEN_LOW_PAINT: (
        '先用低价值颜料把加成叠起来，再补高级颜料',
        '存在地块加成 ≥ 2 的空/失色地块，且有可用的低等级颜料。',
        '加成挂在地块上，炸毁颜料不会带走加成；先在加成格上垫一枚低级颜料，'
        '后面同格再补一枚就能直接跳到高等级。',
    ),
    TACTIC_ANCHOR_CENTER: (
        '初始颜料尽量放中腹',
        '初始颜料锚点位于棋盘外圈（行或列属于 0 / 5）。',
        '锚点的光环按作用方向辐射，放边上会白丢一半覆盖格；'
        '中腹锚点能让同一条规则多升一次级。',
    ),
    TACTIC_TREASURE_HIGH_LEVEL: (
        '宝箱优先堆到 7 级',
        '终局 7 级宝箱数 ≥ 1。',
        '宝箱的等级价值曲线比玩法/怪物陡得多，同样一次升级，'
        '加在宝箱上收益最高。',
    ),
    TACTIC_AVOID_OVERFILL: (
        '别把棋盘塞满低价值颜料',
        '终局存在 ≥ 3 枚 Lv1~Lv2 颜料，且空白地块 ≤ 3。',
        '低级颜料占格会挡住后续高价值落点，宁可留空。',
    ),
    TACTIC_KEEP_BONUS_TILE: (
        '轰掉颜料时保住地块加成',
        '强制调色后仍有地块加成留存。',
        '强制调色只炸颜料、不炸加成，这是"炸完再放一枚直接满级"的来源，'
        '所以炸的位置要挑加成最高的那几格之外的地方。',
    ),
}


@dataclass(frozen=True)
class TacticFinding:
    """一条被触发的技巧。"""

    key: str
    title: str
    evidence: str
    advice: str
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            'key': self.key,
            'title': self.title,
            'evidence': self.evidence,
            'advice': self.advice,
            'weight': self.weight,
        }


# ---------------------------------------------------------------------------
# 练习对局
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 局面的可序列化表示（"能够还原"的基础设施）
# ---------------------------------------------------------------------------
#
# 只存重建 ``PuzzleState`` 必需的最小集合。地形、地块加成、光环强度都要进，
# 因为它们是**棋盘自带**的信息，不能靠重放算出来（重放只负责改变它们）。

_TILE_STATE_CODE = {
    gm.TileState.BLANK: 0,
    gm.TileState.PAINT: 1,
    gm.TileState.FADED: 2,
    gm.TileState.BLOCKED: 3,
    gm.TileState.VOID: 4,
    gm.TileState.UNKNOWN: 5,
}
_TILE_STATE_FROM_CODE = {code: state for state, code in _TILE_STATE_CODE.items()}
_CATEGORY_CODE = {'gameplay': 0, 'monster': 1, 'treasure': 2}
_CATEGORY_FROM_CODE = {code: name for name, code in _CATEGORY_CODE.items()}


def paint_signature(paint: gm.Paint) -> str:
    """颜料的紧凑签名，形如 ``treasure/7/宝箱`` 或 ``gameplay/3/赛季``。

    用来在回放后校验"落下去的是不是同一枚颜料"，不能拿来重建对象
    （``Paint`` 没有额外字段，但签名故意不含可推导信息，避免两处口径漂移）。
    """
    if paint.is_treasure:
        category = 'treasure'
    elif paint.category is gm.PaintCategory.GAMEPLAY:
        category = 'gameplay'
    else:
        category = 'monster'
    return f'{category}/{int(paint.level)}/{paint.subtype or ""}'


def _paint_to_dict(paint: gm.Paint) -> dict[str, Any]:
    data: dict[str, Any] = {
        'category': ('treasure' if paint.is_treasure else paint.category.value),
        'level': int(paint.level),
        'subtype': paint.subtype,
        'is_treasure': bool(paint.is_treasure),
    }
    # 等级加成跟着颜料走，只有非 0 才写，老存档读进来默认 0。
    if paint.bonus_level:
        data['bonus'] = int(paint.bonus_level)
    # 加成的来源账（只有界面说明用，不参与算术）。
    if paint.bonus_parts:
        data['bonus_parts'] = [[name, int(value)] for name, value in paint.bonus_parts]
    return data


def _paint_from_dict(data: dict[str, Any]) -> gm.Paint:
    raw = str(data.get('category') or 'gameplay')
    if raw == 'treasure' or data.get('is_treasure'):
        category = gm.PaintCategory.GAMEPLAY
    else:
        category = gm.PaintCategory(raw)
    return gm.Paint(
        category=category,
        level=int(data.get('level') or 1),
        subtype=data.get('subtype'),
        is_treasure=bool(raw == 'treasure' or data.get('is_treasure')),
        bonus_level=int(data.get('bonus') or 0),
        bonus_parts=tuple(
            (str(name), int(value))
            for name, value in (data.get('bonus_parts') or ())
            if int(value)
        ),
    )


def _tile_to_dict(tile: gm.Tile) -> dict[str, Any]:
    data: dict[str, Any] = {
        's': _TILE_STATE_CODE.get(tile.state, 5),
        'p': _paint_to_dict(tile.paint) if tile.paint is not None else None,
        'a': tile.aura_amount,
        'buf': tile.buff_applied_id,
        'aff': tile.affix_applied_id,
    }
    # 被炸毁初始颜料留在这一格的效果包，只有非空才写。
    if tile.retained_effects:
        data['re'] = [
            [effect.direction.value, effect.target_object.value, int(effect.upgrade_amount)]
            for effect in tile.retained_effects
        ]
    return data


def _tile_from_dict(data: dict[str, Any]) -> gm.Tile:
    paint_raw = data.get('p')
    return gm.Tile(
        state=_TILE_STATE_FROM_CODE.get(int(data.get('s', 0)), gm.TileState.UNKNOWN),
        paint=_paint_from_dict(paint_raw) if paint_raw else None,
        buff_applied_id=data.get('buf'),
        affix_applied_id=data.get('aff'),
        aura_amount=int(data.get('a') or 0),
        retained_effects=tuple(
            gm.PaintEffect(
                gm.PaintDirection(direction),
                gm.PaintObject(target),
                int(amount),
            )
            for direction, target, amount in (data.get('re') or ())
        ),
    )


def _rule_to_dict(rule: gm.CurrentPaintRule | None) -> dict[str, Any] | None:
    if rule is None:
        return None
    data: dict[str, Any] = {
        'paint': _paint_to_dict(rule.paint),
        'target_object': rule.target_object.value,
        'direction': rule.direction.value,
        'match_required': int(rule.match_required),
        'match_current': int(rule.match_current),
        'upgrade_amount': int(rule.upgrade_amount),
        'extra_directions': [item.value for item in rule.extra_directions],
    }
    # 融合初始颜料有 2 条以上调色效果，只有非单条才写（单条能从上面几个字段推出来）。
    if len(rule.effects) > 1:
        data['effects'] = [
            [effect.direction.value, effect.target_object.value, int(effect.upgrade_amount)]
            for effect in rule.effects
        ]
    return data


def _rule_from_dict(data: dict[str, Any] | None) -> gm.CurrentPaintRule | None:
    if not data:
        return None
    raw_effects = data.get('effects') or ()
    return gm.CurrentPaintRule(
        paint=_paint_from_dict(data['paint']),
        target_object=gm.PaintObject(data['target_object']),
        direction=gm.PaintDirection(data['direction']),
        match_required=int(data.get('match_required') or 0),
        match_current=int(data.get('match_current') or 0),
        upgrade_amount=int(data.get('upgrade_amount') or 1),
        extra_directions=tuple(
            gm.PaintDirection(item) for item in data.get('extra_directions') or ()
        ),
        # 空元组时 __post_init__ 会从上面的标量字段补出单条效果，行为不变。
        effects=tuple(
            gm.PaintEffect(
                gm.PaintDirection(direction),
                gm.PaintObject(target),
                int(amount),
            )
            for direction, target, amount in raw_effects
        ),
    )


def _affix_to_dict(affix: gm.AffixRule | None) -> dict[str, Any] | None:
    if affix is None:
        return None
    return {
        'brush_type': affix.brush_type.value,
        'name': affix.name,
        'effect': affix.effect.value,
        'direction': affix.direction.value if affix.direction is not None else None,
        'target_object': (
            affix.target_object.value if affix.target_object is not None else None
        ),
        'delta': int(affix.delta),
        'amount': int(affix.amount),
        'count': affix.count,
    }


def _affix_from_dict(data: dict[str, Any] | None) -> gm.AffixRule | None:
    if not data:
        return None
    return gm.AffixRule(
        brush_type=gm.BrushType(data['brush_type']),
        name=str(data.get('name') or ''),
        effect=gm.AffixEffect(data['effect']),
        direction=(
            gm.PaintDirection(data['direction']) if data.get('direction') else None
        ),
        target_object=(
            gm.PaintObject(data['target_object']) if data.get('target_object') else None
        ),
        delta=int(data.get('delta') or 0),
        amount=int(data.get('amount') or 0),
        count=data.get('count'),
    )


def serialize_state(state: gm.PuzzleState) -> dict[str, Any]:
    """把棋盘状态压成纯 JSON 结构，供落盘 / 跨进程还原。

    故意不存 ``aura_level``：它是 ``_recompute_auras`` 的派生缓存，存下来反而
    容易和地形不一致。还原后由调用方重新算。
    """
    data: dict[str, Any] = {
        'tiles': [_tile_to_dict(tile) for tile in state.tiles],
        'current_rule': _rule_to_dict(state.current_rule),
        'active_affix': _affix_to_dict(state.active_affix),
        'backpack': [_paint_to_dict(paint) for paint in state.backpack],
        'current_anchor': list(state.current_anchor) if state.current_anchor else None,
        'current_buff_id': int(state.current_buff_id),
        'current_rule_pending': bool(state.current_rule_pending),
        'observed_scope_cells': [list(cell) for cell in state.observed_scope_cells],
    }
    # 融合初始颜料的第二、三条笔触效果，只有非空才写。
    if state.extra_affixes:
        data['extra_affixes'] = [
            _affix_to_dict(affix) for affix in state.extra_affixes
        ]
    return data


def deserialize_state(data: dict[str, Any]) -> gm.PuzzleState:
    """:func:`serialize_state` 的逆操作。"""
    anchor = data.get('current_anchor')
    return gm.PuzzleState(
        tiles=tuple(_tile_from_dict(item) for item in data.get('tiles') or ()),
        current_rule=_rule_from_dict(data.get('current_rule')),
        active_affix=_affix_from_dict(data.get('active_affix')),
        extra_affixes=tuple(
            affix
            for affix in (
                _affix_from_dict(item) for item in data.get('extra_affixes') or ()
            )
            if affix is not None
        ),
        backpack=tuple(_paint_from_dict(item) for item in data.get('backpack') or ()),
        current_anchor=(int(anchor[0]), int(anchor[1])) if anchor else None,
        current_buff_id=int(data.get('current_buff_id') or 1),
        current_rule_pending=bool(data.get('current_rule_pending')),
        observed_scope_cells=tuple(
            (int(cell[0]), int(cell[1])) for cell in data.get('observed_scope_cells') or ()
        ),
    )


@dataclass(frozen=True)
class StepRecord:
    """一步棋的完整快照，够用来回放。

    回放只需要 ``kind`` + ``row`` + ``col`` + ``backpack_index``：背包在每次安装后
    会按规则重排，但只要重放的顺序完全一致，同一 seed 的随机数消耗也一致，
    结果就必然可复现。

    ``paint`` 是落子那一刻的颜料签名，**不参与重放**，只用于回放后校验
    "这一步落下去的确实是同一枚颜料"。少了它，背包重排导致的错位会静默通过。
    """

    index: int
    kind: str
    row: int
    col: int
    level_before: int
    level_after: int
    board_value_before: float
    board_value_after: float
    note: str = ''
    backpack_index: int = -1
    paint: str = ''
    affix_id: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {
            'index': self.index,
            'kind': self.kind,
            'row': self.row,
            'col': self.col,
            'level_before': self.level_before,
            'level_after': self.level_after,
            'board_value_before': round(self.board_value_before, 6),
            'board_value_after': round(self.board_value_after, 6),
            'note': self.note,
            'backpack_index': self.backpack_index,
            'paint': self.paint,
            'affix_id': self.affix_id,
        }


@dataclass
class PracticeSession:
    """一局练习：放锚点、装颜料、强制调色，以及**结算后选词缀**。

    ``rules`` 是这局最多结算几条规则；不传真无限（由 ``max_steps`` 兜住）。

    词缀层（``choose_affix``）是后补的：早期版本只有落子动作，
    导致 6 条 ``grants_bonus`` 词缀永远不生效、地块加成恒为 0。
    """

    state: gm.PuzzleState
    seed: int
    pet_level: int = gm.DEFAULT_PET_LEVEL
    weights: EvalWeights = DEFAULT_WEIGHTS
    max_steps: int = 60
    rules: int | None = 3
    _rng: random.Random = field(init=False, repr=False)
    _steps: list[StepRecord] = field(default_factory=list, init=False)
    _force_used: int = field(default=0, init=False)
    _refresh_used: int = field(default=0, init=False)
    _start_rule: int = field(default=0, init=False)
    _start_state: gm.PuzzleState = field(init=False, repr=False)
    _finished: bool = field(default=False, init=False)
    _affix_options: tuple[gm.AffixRule, ...] | None = field(default=None, init=False)
    _affix_scenario: gm.BrushChoiceScenario | None = field(default=None, init=False)
    _affix_rng: random.Random = field(init=False, repr=False)
    _settled_seen: int = field(default=0, init=False)
    #: 刚结算那条规则的初始颜料。选词缀时 ``current_rule`` 已推进，
    #: 只有它能回答"刚才打的是不是宝箱局"。
    _settled_rule_paint: gm.Paint | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_rng', random.Random(self.seed))
        # 词缀相关的随机数**必须走独立流**。
        # 早期版本把"生成词缀选项"也从 _rng 里取，结果只要多抽一次，
        # 之后的 force_complete 炸格位置就整体错位——同一份存档重放出
        # 不同终局。实测 1782 局里有 389 局因此对不上，全部集中在
        # 含 force_complete 的对局。分开之后主随机流保持逐字节稳定。
        object.__setattr__(self, '_affix_rng', random.Random((self.seed << 8) ^ 0x9E3779B9))
        self._start_rule = self.state.current_buff_id
        object.__setattr__(self, '_start_state', self.state)

    # ---- 只读视图 ----------------------------------------------------
    @property
    def steps(self) -> tuple[StepRecord, ...]:
        return tuple(self._steps)

    @property
    def force_remaining(self) -> int:
        return max(0, gm.MAX_FORCE_TIMES - self._force_used)

    @property
    def refresh_remaining(self) -> int:
        total = gm.PET_REFRESH_BY_LEVEL.get(int(self.pet_level), 0)
        return max(0, total - self._refresh_used)

    @property
    def value(self) -> float:
        return board_value(self.state, self.weights)

    @property
    def finished(self) -> bool:
        return self._finished

    def resources(self) -> gm.ResourceState:
        return gm.resources_for_pet_level(
            self.pet_level,
            refresh_used=self._refresh_used,
            force_used=self._force_used,
        )

    # ---- 结束判定 ----------------------------------------------------
    def _settled_rules(self) -> int:
        return max(0, self.state.current_buff_id - self._start_rule)

    def done(self) -> bool:
        if self.state.current_rule is None:
            return True
        if len(self._steps) >= self.max_steps:
            return True
        if self.rules is not None and self._settled_rules() >= self.rules:
            return True
        return False

    # ---- 动作 --------------------------------------------------------
    def _record(
        self,
        kind: str,
        row: int,
        col: int,
        before: gm.PuzzleState,
        after: gm.PuzzleState,
        note: str = '',
        *,
        backpack_index: int = -1,
        paint: str = '',
        affix_id: str = '',
    ) -> None:
        level_before = gm.tile_final_level(before.tile(row, col)) if row >= 0 else 0
        level_after = gm.tile_final_level(after.tile(row, col)) if row >= 0 else 0
        self._steps.append(
            StepRecord(
                index=len(self._steps),
                kind=kind,
                row=row,
                col=col,
                level_before=level_before,
                level_after=level_after,
                board_value_before=board_value(before, self.weights),
                board_value_after=board_value(after, self.weights),
                note=note,
                backpack_index=backpack_index,
                paint=paint,
                affix_id=affix_id,
            )
        )

    # ---- 词缀层 ------------------------------------------------------
    @property
    def affix_pending(self) -> bool:
        """是否正等着选词缀。"""
        return self._affix_options is not None

    @property
    def affix_options(self) -> tuple[gm.AffixRule, ...]:
        """当前可选的词缀（没有则为空）。"""
        return self._affix_options or ()

    @property
    def settled_rule_paint(self) -> gm.Paint | None:
        """刚结算那条规则的初始颜料。

        选词缀时 ``current_rule`` 已经推进到下一条，宝箱相关判断
        （替换优先、锁堆砌、用刷新刷替换）都必须读这个。
        """
        return self._settled_rule_paint

    @property
    def remaining_rules(self) -> int | None:
        """还剩几条规则要打（= 还剩几次选词缀的机会）。

        ``rules is None`` 表示不限条数，返回 ``None`` 让推荐器按"收尾"处理。
        真实一局大约 4~5 条规则：36 格地里空白+阻挡约 20 格，
        每条规则要 1 枚初始颜料 + 3~5 个连接。
        """
        if self.rules is None:
            return None
        return max(0, int(self.rules) - self._settled_rules())

    def _offer_affix(self) -> None:
        """结算完一个调色目标后，按官方流程给出一组词缀供选。

        真实流程是：选画笔类型（转锋/堆砌/扩散）→ 从该类型里选一条词缀。
        这里用 :func:`game_model.random_brush_choice_scenario` 生成同样的结构，
        并把 ``stage2_options`` 作为可选项暴露出来——**这正是
        ``recommend_affixes`` 要评判的东西**。

        **宝箱局第一阶段选堆砌**（用户规则 2026-11-12）。原因很硬：
        第二阶段只从"已选中的画笔类型"里抽词条，而「替换为自身」是堆砌词条。
        第一阶段若是转锋/扩散，这条词缀**根本不会出现**——而第一阶段不可刷新，
        选错就整局无缘。所以初始颜料是宝箱时必须锁堆砌。
        """
        scenario = gm.random_brush_choice_scenario(self._affix_rng.randrange(1 << 30))
        # 用**刚结算那条规则**的颜料判断，不能用 current_rule——它已经推进了。
        settled = self._settled_rule_paint
        if (
            settled is not None
            and settled.is_treasure
            and gm.BrushType.STACK in scenario.stage1_options
            and scenario.chosen_stage1 is not gm.BrushType.STACK
        ):
            # 按"玩家会选堆砌"重建场景：第二阶段改为从堆砌词条里抽。
            catalog = [a for a in gm.build_affix_catalog() if a.brush_type is gm.BrushType.STACK]
            picks = self._affix_rng.sample(catalog, min(2, len(catalog)))
            scenario = gm.BrushChoiceScenario(
                stage1_options=scenario.stage1_options,
                chosen_stage1=gm.BrushType.STACK,
                stage2_options=tuple(picks),
            )
        self._affix_scenario = scenario
        self._affix_options = tuple(scenario.stage2_options)

    def choose_affix(self, index: int) -> bool:
        """选第 ``index`` 条词缀并应用。"""
        if not self._affix_options or not 0 <= index < len(self._affix_options):
            return False
        affix = self._affix_options[index]
        before = self.state
        result = gm.apply_affix_to_state(self.state, affix, seed=self._affix_rng.randrange(1 << 30))
        if not result.success:
            return False
        self._affix_options = None
        self._affix_scenario = None
        self.state = result.state
        self._record(
            'choose_affix', -1, -1, before, self.state, result.message,
            affix_id=affix.name,
        )
        self._sync_finished(before)
        return True


    def place_anchor(self, row: int, col: int) -> bool:
        """放初始颜料（只在 ``current_rule_is_pending`` 时有效）。"""
        if self.done() or not gm.current_rule_is_pending(self.state):
            return False
        before = self.state
        result = gm.place_current_rule_anchor(self.state, row, col, rng=self._rng)
        if not result.success:
            return False
        self.state = result.state
        self._record('place_anchor', row, col, before, self.state, result.message)
        self._sync_finished(before)
        return True

    def install(self, backpack_index: int, row: int, col: int) -> bool:
        """把背包第 ``backpack_index`` 枚颜料落在 ``(row, col)``。"""
        if self.done() or gm.current_rule_is_pending(self.state):
            return False
        if not 0 <= backpack_index < len(self.state.backpack):
            return False
        before = self.state
        chosen = self.state.backpack[backpack_index]
        result = gm.install_backpack_paint(
            self.state, backpack_index, row, col, rng=self._rng
        )
        if not result.success:
            return False
        self.state = result.state
        self._record(
            'install', row, col, before, self.state, result.message,
            backpack_index=backpack_index, paint=paint_signature(chosen),
        )
        self._sync_finished(before)
        return True

    def force_complete(self) -> bool:
        """强制调色。摧毁颜料数按已用次数递增（1/2/3）。"""
        if self.done() or gm.current_rule_is_pending(self.state):
            return False
        if self._force_used >= gm.MAX_FORCE_TIMES:
            return False
        before = self.state
        result = gm.force_complete_current_rule(self.state, self._force_used, rng=self._rng)
        if not result.success:
            return False
        self._force_used += 1
        self.state = result.state
        cells = ' '.join(f'({r},{c})' for r, c in result.destroyed_cells)
        # 规则是第 N 次炸 N 个；棋盘颜料不够时会被截断，这里如实写出来，
        # 免得复盘看到"第三次只炸 2 个"以为是规则错了。
        expected = min(3, self._force_used)
        shortfall = expected - len(result.destroyed_cells)
        note = f'摧毁 {len(result.destroyed_cells)} 个颜料: {cells}'
        if shortfall > 0:
            note += f'（应炸 {expected} 个，棋盘上颜料不足，少炸 {shortfall} 个）'
        self._record(
            'force_complete',
            -1,
            -1,
            before,
            self.state,
            note,
        )
        self._sync_finished(before)
        return True

    def refresh_rule(self) -> bool:
        """刷新**第二阶段词缀选项**（重抽一组词缀）。

        实机确认（2026-11-12）：

        * 刷新次数按宠物等级：Lv1–2 为 0、Lv3–4 为 1、Lv5–6 为 2；
        * **只能刷新第二阶段（词缀）**，第一阶段（画笔类型：转锋/堆砌/扩散）
          不可刷新。

        所以这个动作要求当前正处于"等选词缀"阶段：没有待选词缀时无从刷新。
        """
        if self.done() or self.refresh_remaining <= 0:
            return False
        if not self.affix_pending:
            return False
        scenario = self._affix_scenario
        if scenario is None:
            return False
        before = self.state
        new_scenario = gm.random_brush_choice_scenario(self._affix_rng.randrange(1 << 30))
        self._refresh_used += 1
        # 只换第二阶段的选项；第一阶段（画笔类型）保持不变——它本来就刷不了。
        self._affix_scenario = gm.BrushChoiceScenario(
            stage1_options=scenario.stage1_options,
            chosen_stage1=scenario.chosen_stage1,
            stage2_options=new_scenario.stage2_options,
        )
        self._affix_options = tuple(self._affix_scenario.stage2_options)
        self._record(
            'refresh_rule', -1, -1, before, self.state, '刷新第二阶段词缀选项',
        )
        return True

    def _sync_finished(self, settled_from: gm.PuzzleState | None = None) -> None:
        """结算数增加时，按官方流程弹一次词缀选择。

        ``settled_from`` 是**动作发生前**的状态。这一点很关键：
        ``install``/``force_complete`` 返回的新状态里 ``current_rule`` 已经
        推进到下一条，所以"刚打完的是不是宝箱局"只能从旧状态读。
        之前直接读新状态，导致所有宝箱专属判断（替换优先、锁堆砌、
        用刷新去刷替换）在真实流程里全部落空。
        """
        settled = self._settled_rules()
        if settled > self._settled_seen:
            self._settled_seen = settled
            if settled_from is not None and settled_from.current_rule is not None:
                self._settled_rule_paint = settled_from.current_rule.paint
            if self.state.current_rule is not None:
                self._offer_affix()
        if self.state.current_rule is None:
            self._finished = True

    # ---- 回放 / 导出 --------------------------------------------------
    def replay_steps(self) -> list[dict[str, Any]]:
        return [step.as_dict() for step in self._steps]

    def snapshot(self) -> dict[str, Any]:
        """完整存档：开局局面 + 走法序列 + 参数。

        ``start_state`` 是**开局那一刻**（还没放锚点）的局面，与 ``seed`` 一起
        构成还原的全部前提。``steps`` 只保留可执行字段，``paint`` 用于校验。
        """
        return {
            'schema': PRACTICE_SNAPSHOT_SCHEMA,
            'seed': self.seed,
            'pet_level': int(self.pet_level),
            'rules': self.rules,
            'max_steps': int(self.max_steps),
            'force_used': self._force_used,
            'refresh_used': self._refresh_used,
            'start_rule': self._start_rule,
            'start_state': serialize_state(self._start_state),
            'steps': [
                {
                    'kind': step.kind,
                    'row': step.row,
                    'col': step.col,
                    'backpack_index': step.backpack_index,
                    'paint': step.paint,
                    'affix_id': step.affix_id,
                }
                for step in self._steps
            ],
            'final_board_value': round(self.value, 6),
        }

    def as_dict(self) -> dict[str, Any]:
        treasure7, any7 = level7_counts(self.state)
        return {
            'seed': self.seed,
            'pet_level': self.pet_level,
            'rules_settled': self._settled_rules(),
            'steps': self.replay_steps(),
            'force_used': self._force_used,
            'refresh_used': self._refresh_used,
            'board_value': round(self.value, 6),
            'treasure7': treasure7,
            'any7': any7,
            'board': _board_rows(self.state),
        }


def _board_rows(state: gm.PuzzleState) -> list[str]:
    """6 行文本，用于落盘后肉眼复查。"""
    rows: list[str] = []
    for row in range(gm.BOARD_SIZE):
        cells: list[str] = []
        for col in range(gm.BOARD_SIZE):
            tile = state.tile(row, col)
            level = gm.tile_final_level(tile)
            if tile.state is gm.TileState.PAINT and tile.paint is not None:
                mark = 'T' if tile.paint.is_treasure else (
                    'M' if tile.paint.category is gm.PaintCategory.MONSTER else 'P'
                )
                cells.append(f'{mark}{level}')
            elif tile.state is gm.TileState.BLANK:
                cells.append('..')
            elif tile.state is gm.TileState.VOID:
                cells.append('##')
            elif tile.state is gm.TileState.FADED:
                cells.append('xx')
            else:
                cells.append('XX')
        rows.append(' '.join(cells))
    return rows


#: CLI / GUI 共用的公开别名（``_board_rows`` 是历史命名，保留以免动旧调用点）。
board_rows = _board_rows


def start_practice(
    seed: int | None = None,
    *,
    pet_level: int = gm.DEFAULT_PET_LEVEL,
    rules: int | None = 3,
    max_steps: int = 60,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    wait_for_anchor: bool = True,
) -> PracticeSession:
    """开一局练习。棋盘/规则/背包全部走 ``random_state``（官方权重）。

    ``wait_for_anchor=True``（默认）会把局面退回到「刚抽到规则、还没落初始颜料」，
    这样练习从**你自己选锚点**开始——锚点位置是这局最关键的一步决策，
    交给随机数就练不到东西了。
    """
    actual_seed = int(seed) if seed is not None else random.randrange(1 << 30)
    state = gm.random_state(actual_seed)
    if wait_for_anchor and state.current_anchor is not None:
        state = _rewind_to_pending(state)
    return PracticeSession(
        state=state,
        seed=actual_seed,
        pet_level=pet_level,
        rules=rules,
        max_steps=max_steps,
        weights=weights,
    )


def restore_practice(snapshot: dict[str, Any]) -> PracticeSession:
    """从 :meth:`PracticeSession.snapshot` 的存档还原出一局。

    **只还原局面，不重放**：``state`` 落在开局那一刻，走法序列由
    :func:`replay_session` 单独施加。拆成两步是为了能"回放到第 N 步"
    （复盘的常见需求）而不必每次都从头发车。
    """
    schema = snapshot.get('schema')
    if schema is not None and schema != PRACTICE_SNAPSHOT_SCHEMA:
        raise ValueError(f'不认识的练习存档 schema：{schema}')
    start_state = deserialize_state(snapshot['start_state'])
    session = PracticeSession(
        state=start_state,
        seed=int(snapshot.get('seed') or 0),
        pet_level=int(snapshot.get('pet_level') or gm.DEFAULT_PET_LEVEL),
        rules=snapshot.get('rules'),
        max_steps=int(snapshot.get('max_steps') or 60),
    )
    # 关键：随机数起点必须与原始对局一致，否则次数相关的词缀/刷新会漂。
    object.__setattr__(session, '_rng', random.Random(session.seed))
    # 词缀随机流同样要从同一个起点重建，否则回放时词缀选项会变。
    object.__setattr__(
        session, '_affix_rng', random.Random((session.seed << 8) ^ 0x9E3779B9)
    )
    return session


def apply_step(session: PracticeSession, step: dict[str, Any]) -> bool:
    """把一步存档施加到 ``session`` 上。"""
    kind = str(step.get('kind') or '')
    row = int(step.get('row', -1))
    col = int(step.get('col', -1))
    if kind == 'place_anchor':
        return session.place_anchor(row, col)
    if kind == 'install':
        return session.install(int(step.get('backpack_index', -1)), row, col)
    if kind == 'force_complete':
        return session.force_complete()
    if kind == 'refresh_rule':
        return session.refresh_rule()
    if kind == 'choose_affix':
        # 选项由结算时的随机数生成，所以这里按**名字**匹配，而不是按下标：
        # 下标会随选项生成顺序漂移，名字才是稳定的身份。
        wanted = str(step.get('affix_id') or '')
        options = session.affix_options
        if not options:
            return False
        for index, affix in enumerate(options):
            if affix.name == wanted:
                return session.choose_affix(index)
        return False
    raise ValueError(f'未知的走法类型：{kind}')


@dataclass(frozen=True)
class ReplayCheck:
    """回放校验结果。``ok=False`` 时 ``mismatch`` 指向第一处分歧。"""

    ok: bool
    steps_replayed: int
    expected_value: float
    actual_value: float
    mismatch: str = ''
    paint_mismatch: tuple[int, str, str] | None = None
    board: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'steps_replayed': self.steps_replayed,
            'expected_value': round(self.expected_value, 6),
            'actual_value': round(self.actual_value, 6),
            'mismatch': self.mismatch,
            'paint_mismatch': (
                list(self.paint_mismatch) if self.paint_mismatch else None
            ),
            'board': list(self.board),
        }


def replay_session(
    snapshot: dict[str, Any],
    *,
    until: int | None = None,
    verify_paint: bool = True,
) -> tuple[PracticeSession, ReplayCheck]:
    """按存档重放对局，并校验终局是否与原记录一致。

    ``until`` 用于只重放到第 N 步（0 = 只摆好开局，不落子）。
    校验分两层：

    1. **落子一致性** —— 每步实际落下的颜料签名要等于存档里的 ``paint``。
       背包会按规则重排，若还原逻辑有偏差，这里会先炸；
    2. **终局价值** —— 回放完的 ``board_value`` 要等于存档记录值。
       浮点容差 1e-6，超过就是真有状态没还原到。
    """
    session = restore_practice(snapshot)
    steps = list(snapshot.get('steps') or ())
    limit = len(steps) if until is None else max(0, min(int(until), len(steps)))
    paint_mismatch: tuple[int, str, str] | None = None
    mismatch = ''
    replayed = 0
    for index in range(limit):
        step = steps[index]
        before_count = len(session.steps)
        try:
            ok = apply_step(session, step)
        except ValueError as exc:
            mismatch = str(exc)
            break
        if not ok:
            mismatch = f'第 {index} 步（{step.get("kind")}）执行失败'
            break
        replayed += 1
        if verify_paint and step.get('kind') == 'install' and paint_mismatch is None:
            recorded = str(step.get('paint') or '')
            actual_step = session.steps[before_count] if len(session.steps) > before_count else None
            actual = actual_step.paint if actual_step is not None else ''
            if recorded and actual and recorded != actual:
                paint_mismatch = (index, recorded, actual)
                mismatch = f'第 {index} 步落下的颜料与记录不符'
                break
    expected = float(snapshot.get('final_board_value') or 0.0)
    actual = session.value
    if not mismatch and replayed == len(steps) and abs(actual - expected) > 1e-6:
        mismatch = f'终局分不一致：记录 {expected:.6f}，回放 {actual:.6f}'
    return session, ReplayCheck(
        ok=not mismatch,
        steps_replayed=replayed,
        expected_value=expected,
        actual_value=actual,
        mismatch=mismatch,
        paint_mismatch=paint_mismatch,
        board=tuple(_board_rows(session.state)),
    )


def _rewind_to_pending(state: gm.PuzzleState) -> gm.PuzzleState:
    """把已放好锚点的开局退回「等待放锚点」。

    只撤掉锚点那一格，规则/背包/地形都保留 —— 这样同一 seed 下
    "AI 自动放" 与 "人手放" 面对的是同一张牌。
    """
    anchor = state.current_anchor
    if anchor is None:
        return state
    row, col = anchor
    tiles = list(state.tiles)
    index = row * gm.BOARD_SIZE + col
    tile = tiles[index]
    # 锚点格原本若被颜料占用，撤成它放之前的空白状态。
    tiles[index] = replace(tile, state=gm.TileState.BLANK, paint=None,
                           buff_applied_id=None, affix_applied_id=None)
    return replace(
        state,
        tiles=tuple(tiles),
        current_anchor=None,
        current_rule_pending=True,
    )


# ---------------------------------------------------------------------------
# 复盘：为什么高分 / 有哪些技巧
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreBreakdown:
    """分数拆解，每一项都能对上棋盘上的格子。"""

    board_value: float
    treasure_value: float
    gameplay_value: float
    monster_value: float
    empty_penalty: float
    treasure_count: int
    gameplay_count: int
    monster_count: int
    level7_total: int
    level7_treasure: int
    mean_level: float

    def as_dict(self) -> dict[str, Any]:
        return {
            'board_value': round(self.board_value, 4),
            'treasure_value': round(self.treasure_value, 4),
            'gameplay_value': round(self.gameplay_value, 4),
            'monster_value': round(self.monster_value, 4),
            'empty_penalty': round(self.empty_penalty, 4),
            'treasure_count': self.treasure_count,
            'gameplay_count': self.gameplay_count,
            'monster_count': self.monster_count,
            'level7_total': self.level7_total,
            'level7_treasure': self.level7_treasure,
            'mean_level': round(self.mean_level, 3),
        }


def score_breakdown(
    state: gm.PuzzleState,
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> ScoreBreakdown:
    """把 :func:`board_value` 拆成颜料类别贡献 + 空地惩罚。"""
    treasure = gameplay = monster = 0.0
    empty_penalty = 0.0
    counts = {'T': 0, 'P': 0, 'M': 0}
    levels: list[int] = []
    level7 = 0
    level7_treasure = 0
    for tile in state.tiles:
        effective = gm.tile_effective_paint(tile)
        if effective is not None and tile.state is gm.TileState.PAINT:
            value = weights.paint_value(effective)
            if effective.is_treasure:
                treasure += value
                counts['T'] += 1
            elif effective.category is gm.PaintCategory.GAMEPLAY:
                gameplay += value
                counts['P'] += 1
            else:
                monster += value
                counts['M'] += 1
            levels.append(effective.level)
            if effective.level >= gm.MAX_LEVEL:
                level7 += 1
                if effective.is_treasure:
                    level7_treasure += 1
        else:
            empty_value = weights.empty_tile_value(tile.state, gm.tile_bonus_level(tile))
            if empty_value < 0:
                empty_penalty += empty_value
    return ScoreBreakdown(
        board_value=board_value(state, weights),
        treasure_value=treasure,
        gameplay_value=gameplay,
        monster_value=monster,
        empty_penalty=empty_penalty,
        treasure_count=counts['T'],
        gameplay_count=counts['P'],
        monster_count=counts['M'],
        level7_total=level7,
        level7_treasure=level7_treasure,
        mean_level=(sum(levels) / len(levels)) if levels else 0.0,
    )


def _open_neighbour_count(state: gm.PuzzleState, row: int, col: int) -> int:
    """锚点四邻里"能放东西"的格子数（空白）。"""
    count = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = row + dr, col + dc
        if not (0 <= r < gm.BOARD_SIZE and 0 <= c < gm.BOARD_SIZE):
            continue
        if state.tile(r, c).state in (gm.TileState.BLANK, gm.TileState.PAINT):
            count += 1
    return count


def _bonus_tiles(state: gm.PuzzleState) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for row in range(gm.BOARD_SIZE):
        for col in range(gm.BOARD_SIZE):
            bonus = gm.tile_bonus_level(state.tile(row, col))
            if bonus > 0:
                out.append((row, col, bonus))
    return out


def _low_level_paints(state: gm.PuzzleState) -> int:
    return sum(
        1
        for tile in state.tiles
        if tile.state is gm.TileState.PAINT
        and tile.paint is not None
        and gm.tile_final_level(tile) <= 2
    )


def _blank_count(state: gm.PuzzleState) -> int:
    return sum(1 for tile in state.tiles if tile.state is gm.TileState.BLANK)


def _scope_blank_count(state: gm.PuzzleState, rule: gm.CurrentPaintRule) -> int:
    """词缀作用范围内还能落子的空格数；算不出来返回 -1。

    这是判断"这局还能不能凑够连接数"的依据：锚点本身占 1 格，
    所以还需要 ``match_required - 1`` 个空格才可能完成。
    """
    try:
        cells = gm._placement_scope_cells(state, rule)  # noqa: SLF001
    except (AttributeError, TypeError, ValueError):
        return -1
    total = 0
    for row, col in cells:
        if state.tile(row, col).state is gm.TileState.BLANK:
            total += 1
    return total


def _anchor_from_steps(
    state: gm.PuzzleState, steps: Sequence[StepRecord] | None
) -> tuple[int, int] | None:
    """取本局的锚点位置：优先当前局面，其次走法记录。

    ``wait_for_anchor=True`` 开局时 :func:`_rewind_to_pending` 会把
    ``current_anchor`` 置空，只看局面会让"锚点类"技巧全部判不出来。
    """
    anchor = state.current_anchor
    if anchor is not None:
        return anchor
    if not steps:
        return None
    for step in reversed(steps):
        if step.kind == 'place_anchor' and step.row >= 0 and step.col >= 0:
            return (step.row, step.col)
    return None


def find_tactics(
    state: gm.PuzzleState,
    *,
    steps: Sequence[StepRecord] | None = None,
) -> list[TacticFinding]:
    """从棋盘 + 走棋记录里挑出被触发的技巧。"""
    findings: list[TacticFinding] = []

    # ① 宝箱 + 空间不够 → 放 1 枚就强制调色
    #
    # 只在**锚点已落下且处于作画阶段**时判定：作用范围是相对锚点定义的，
    # 开局未落锚点、或规则已推进到"待放锚点"时，范围都无定义、空位数恒为 0，
    # 那样会让这条技巧对任何带宝箱的局面都成立（假阳性）。
    rule = state.current_rule
    anchor = _anchor_from_steps(state, steps)
    has_treasure = any(paint.is_treasure for paint in state.backpack)
    if (
        anchor is not None
        and rule is not None
        and not gm.current_rule_is_pending(state)
        and has_treasure
        and rule.target_object is not gm.PaintObject.MONSTER
    ):
        row, col = anchor
        neighbours = _open_neighbour_count(state, row, col)
        blanks = _scope_blank_count(state, rule)
        cramped_neighbours = neighbours <= 2
        # 锚点占掉 1 格，剩余空格凑不满连接数 → 这局本来就做不完
        impossible = blanks >= 0 and blanks < max(0, rule.match_required - 1)
        cramped_scope = 0 <= blanks <= 2
        if cramped_neighbours or impossible or cramped_scope:
            title, _cond, advice = TACTIC_LIBRARY[TACTIC_FORCE_ON_ISOLATED_TREASURE]
            if impossible:
                reason = (f'作用范围里只剩 {blanks} 格空位，凑不满需连接的 '
                          f'{rule.match_required} 个')
            elif cramped_scope:
                reason = f'作用范围里只剩 {blanks} 格空位'
            else:
                reason = f'锚点 ({row},{col}) 四邻只有 {neighbours} 格可放'
            findings.append(TacticFinding(
                key=TACTIC_FORCE_ON_ISOLATED_TREASURE,
                title=title,
                evidence=(
                    f'{reason}，且背包里有宝箱颜料 → '
                    f'先落 1 枚背包颜料（棋盘上一共 2 枚）再强制调色'
                ),
                advice=advice,
                weight=1.4,
            ))

    # ② 先叠加成再补高级颜料
    bonuses = _bonus_tiles(state)
    if bonuses:
        top = max(bonuses, key=lambda item: item[2])
        title, _cond, advice = TACTIC_LIBRARY[TACTIC_STACK_BONUS_THEN_LOW_PAINT]
        findings.append(TacticFinding(
            key=TACTIC_STACK_BONUS_THEN_LOW_PAINT,
            title=title,
            evidence=f'有 {len(bonuses)} 格带地块加成，最高 +{top[2]} 在 ({top[0]},{top[1]})',
            advice=advice,
            weight=0.9,
        ))

    # ③ 锚点放中腹
    if anchor is not None:
        row, col = anchor
        if row in (0, gm.BOARD_SIZE - 1) or col in (0, gm.BOARD_SIZE - 1):
            title, _cond, advice = TACTIC_LIBRARY[TACTIC_ANCHOR_CENTER]
            findings.append(TacticFinding(
                key=TACTIC_ANCHOR_CENTER,
                title=title,
                evidence=f'锚点 ({row},{col}) 在外圈',
                advice=advice,
                weight=-1.0,  # 负权重 = 这是要避免的做法
            ))

    # ④ 宝箱堆到 7 级
    treasure7, any7 = level7_counts(state)
    if treasure7 >= 1:
        title, _cond, advice = TACTIC_LIBRARY[TACTIC_TREASURE_HIGH_LEVEL]
        findings.append(TacticFinding(
            key=TACTIC_TREASURE_HIGH_LEVEL,
            title=title,
            evidence=f'终局有 {treasure7} 个 7 级宝箱（任意 7 级共 {any7} 个）',
            advice=advice,
            weight=1.6,
        ))

    # ⑤ 别塞满低级颜料
    low = _low_level_paints(state)
    blanks = _blank_count(state)
    if low >= 3 and blanks <= 3:
        title, _cond, advice = TACTIC_LIBRARY[TACTIC_AVOID_OVERFILL]
        findings.append(TacticFinding(
            key=TACTIC_AVOID_OVERFILL,
            title=title,
            evidence=f'Lv1~Lv2 颜料 {low} 枚，空白只剩 {blanks} 格',
            advice=advice,
            weight=-0.8,
        ))

    # ⑥ 炸颜料保住加成
    if steps:
        forced = [step for step in steps if step.kind == 'force_complete']
        if forced and bonuses:
            title, _cond, advice = TACTIC_LIBRARY[TACTIC_KEEP_BONUS_TILE]
            findings.append(TacticFinding(
                key=TACTIC_KEEP_BONUS_TILE,
                title=title,
                evidence=f'强制调色 {len(forced)} 次，之后仍有 {len(bonuses)} 格加成留存',
                advice=advice,
                weight=0.7,
            ))

    return findings


@dataclass(frozen=True)
class HighScoreExplanation:
    """一局的高分解读。"""

    score: float
    grade: str
    breakdown: ScoreBreakdown
    tactics: tuple[TacticFinding, ...]
    summary: tuple[str, ...]
    session: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            'score': round(self.score, 4),
            'grade': self.grade,
            'breakdown': self.breakdown.as_dict(),
            'tactics': [item.as_dict() for item in self.tactics],
            'summary': list(self.summary),
            'session': self.session,
        }

    def format_text(self) -> str:
        lines = [f'得分 {self.score:.4f}（{self.grade}）']
        lines.extend(f'  · {line}' for line in self.summary)
        if self.tactics:
            lines.append('触发技巧：')
            for item in self.tactics:
                sign = '✔' if item.weight >= 0 else '✘'
                lines.append(f'  {sign} {item.title}')
                lines.append(f'      证据：{item.evidence}')
                lines.append(f'      建议：{item.advice}')
        return '\n'.join(lines)


# 阶级阈值，按 ``board_value`` 的**绝对分**定。
#
# 2026-11-23 重新标定：旧阈值是 1.5 / 0.9 / 0.4 / 0.1，那是给**归一化分数**
# 用的刻度；而 ``board_value`` 早就换成了绝对分（7 级宝箱 70 分、7 级怪物 7 分），
# 于是一局正常棋（5 万局均值 120）永远 >= 1.5，**结果 100% 全是 S 级**，
# 阶级等于没有。归档的旧模拟库 3000 条也都是清一色 S 级，说明这个错配
# 比本次改动更早。
#
# 新阈值取自 5 万局 greedy 模拟的分位数（runtime/monte_50k_result.json）：
#   p90=196  p70=148  p40=101  p15=56
# 换策略（lookahead / 融合专线）或换权重后应当重新标定，不要照抄。
GRADE_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (196, 'S 顶级'),   # 前 ~10%
    (148, 'A 高分'),   # 前 ~30%
    (101, 'B 中等'),   # 前 ~60%
    (56, 'C 偏低'),    # 前 ~85%
)


def _grade(score: float) -> str:
    for threshold, name in GRADE_THRESHOLDS:
        if score >= threshold:
            return name
    return 'D 低分'


def explain_high_score(
    state: gm.PuzzleState,
    *,
    steps: Sequence[StepRecord] | None = None,
    session: PracticeSession | None = None,
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> HighScoreExplanation:
    """给一局打分并解释「为什么是这个分」。"""
    breakdown = score_breakdown(state, weights)
    tactics = find_tactics(state, steps=steps)
    summary: list[str] = []
    if breakdown.level7_treasure:
        summary.append(
            f'7 级宝箱 {breakdown.level7_treasure} 个，单独贡献了 '
            f'{breakdown.treasure_value:.3f} 分（占总分的大头）'
        )
    if breakdown.treasure_count:
        summary.append(
            f'宝箱颜料 {breakdown.treasure_count} 枚，合计 {breakdown.treasure_value:.3f} 分'
        )
    if breakdown.gameplay_count:
        summary.append(
            f'玩法颜料 {breakdown.gameplay_count} 枚，合计 {breakdown.gameplay_value:.3f} 分'
        )
    if breakdown.monster_count:
        summary.append(
            f'怪物颜料 {breakdown.monster_count} 枚，合计 {breakdown.monster_value:.3f} 分'
        )
    summary.append(f'颜料平均等级 {breakdown.mean_level:.2f} 级')
    if breakdown.empty_penalty < 0:
        summary.append(f'空地惩罚 {breakdown.empty_penalty:.3f} 分（失色/阻挡/虚空格）')
    positive = [item for item in tactics if item.weight > 0]
    negative = [item for item in tactics if item.weight < 0]
    if positive:
        summary.append('正向技巧：' + '、'.join(item.title for item in positive))
    if negative:
        summary.append('待改进：' + '、'.join(item.title for item in negative))

    payload = session.as_dict() if session is not None else None
    return HighScoreExplanation(
        score=breakdown.board_value,
        grade=_grade(breakdown.board_value),
        breakdown=breakdown,
        tactics=tuple(tactics),
        summary=tuple(summary),
        session=payload,
    )


# ---------------------------------------------------------------------------
# 自动练习：给"AI 模拟练习"用的批量跑局
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PracticeSummary:
    runs: int
    mean_score: float
    best_score: float
    mean_level: float
    level7_treasure_total: int
    tactic_hits: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            'runs': self.runs,
            'mean_score': round(self.mean_score, 4),
            'best_score': round(self.best_score, 4),
            'mean_level': round(self.mean_level, 3),
            'level7_treasure_total': self.level7_treasure_total,
            'tactic_hits': dict(sorted(self.tactic_hits.items())),
        }


def summarize_runs(explanations: Iterable[HighScoreExplanation]) -> PracticeSummary:
    items = list(explanations)
    if not items:
        return PracticeSummary(0, 0.0, 0.0, 0.0, 0, {})
    hits: dict[str, int] = {}
    for item in items:
        for tactic in item.tactics:
            hits[tactic.key] = hits.get(tactic.key, 0) + 1
    return PracticeSummary(
        runs=len(items),
        mean_score=sum(item.score for item in items) / len(items),
        best_score=max(item.score for item in items),
        mean_level=sum(item.breakdown.mean_level for item in items) / len(items),
        level7_treasure_total=sum(item.breakdown.level7_treasure for item in items),
        tactic_hits=hits,
    )


def auto_play(
    session: PracticeSession,
    *,
    prefer: str = 'lookahead',
    rng: random.Random | None = None,
    explore: float = 0.0,
    fusion_explore: bool = False,
) -> PracticeSession:
    """让策略把这局打完（原地推进 ``session``）。

    ``prefer``:

    * ``'lookahead'`` —— 用 ``HeuristicPolicy(include_lookahead=True)``，
      会算下一步的收益，慢但下得更好；
    * ``'greedy'`` —— 只吃当前格子的即时收益，快，适合刷大量局看分布。

    ``explore`` 是**探索率**（0~1）：以这个概率把某次词缀选择换成随机选，
    而不是一律听推荐器。

    为什么必须有它：推荐器只会沿着它已知的偏好走，于是"初始颜料是宝箱 +
    作用范围全图 + 替换为自身"这种需要三个小概率同时成立的高分路线，
    在大批量模拟里几乎永远采不到（实测 4000 局里 0 次，50000 局理论也只有
    约 2 次）。纯 greedy 跑多少次都只是在重复同一个盲区。

    ``fusion_explore``: **融合专线**（2026-11-23）。默认策略永远不会走到
    融合颜料上去，因为融合要三个条件同时成立：

    1. 用掉一次强制调色，并且随机炸毁恰好炸到**当前锚点格**（炸到才留效果包）；
    2. 下一轮抽到「可放置于失色地块中」(官方 3017) —— 失色格默认放不了锚点；
    3. 在这一格放新锚点，才把地块上的效果包并进新规则。

    实测 5 万局纯 greedy **0 次融合**，等于新规则完全没被验证。打开这个开关后，
    策略会主动凑这条路：有带效果包的失色格时优先刷出 3017 并去那格落笔。

    探索用的随机流是独立的，不影响 ``_rng``，所以回放仍然确定。
    """
    from .engine.actions import ActionKind
    from .engine.policy import HeuristicPolicy
    from .engine.rules import RuleEngine

    engine = RuleEngine()
    policy = HeuristicPolicy(include_lookahead=(prefer == 'lookahead'))
    rng = rng or random.Random(session.seed ^ 0x5EED)
    # 探索决策单独一条流：不能动 _rng，否则回放会漂。
    explore_rng = random.Random((session.seed << 16) ^ 0xB5297A4D)
    explore = max(0.0, min(1.0, float(explore)))

    def _kept_effect_cell() -> tuple[int, int] | None:
        """找一格「炸毁后留下效果包的失色格」—— 融合的唯一入口。"""
        for row in range(gm.BOARD_SIZE):
            for col in range(gm.BOARD_SIZE):
                tile = session.state.tile(row, col)
                if tile.retained_effects:
                    return row, col
        return None

    def _place_on_kept_effect() -> bool:
        """把新初始颜料落在留有效果包的失色格上，完成一次融合。"""
        cell = _kept_effect_cell()
        if cell is None:
            return False
        return session.place_anchor(*cell)

    guard = 0
    while not session.done():
        guard += 1
        if guard > session.max_steps * 4:
            break
        # 结算后先选词缀：这一步直接用生产环境的 recommend_affixes 决策，
        # 模拟跑出来的分布才等于"推荐器实际会怎么打"。
        if session.affix_pending:
            # 宝箱局的特殊打法（用户规则 2026-11-12）：初始颜料是宝箱时
            # 第一优先是「替换为自身」（拿更多宝箱），并且**用掉刷新次数去刷它**。
            # 只要还没刷出替换、且还有刷新次数，就先刷再说。
            settled = session.settled_rule_paint
            is_treasure_round = settled is not None and settled.is_treasure
            # 融合专线：棋盘上已经有留效果包的失色格，就必须拿到「可放置于失色
            # 地块中」才能进那格，否则那条效果包永远白留着。优先级高于宝箱刷替换
            # ——否则两个都在抢刷新次数，融合永远排不上。
            fusion_pending = fusion_explore and _kept_effect_cell() is not None
            if fusion_pending:
                wanted_fusion = gm.AffixEffect.ALLOW_PLACE_ON_FADED
                has_it = any(a.effect is wanted_fusion for a in session.affix_options)
                while not has_it and session.refresh_remaining > 0:
                    if not session.refresh_rule():
                        break
                    has_it = any(a.effect is wanted_fusion for a in session.affix_options)
                if has_it:
                    options = session.affix_options
                    pick_fusion = next(
                        i for i, a in enumerate(options) if a.effect is wanted_fusion
                    )
                    if session.choose_affix(pick_fusion):
                        if _place_on_kept_effect():
                            continue
                    continue
            if not (explore and explore_rng.random() < explore):
                wanted = gm.AffixEffect.COPY_SELF_TO_SCOPE
                hunting = (
                    is_treasure_round
                    and session.refresh_remaining > 0
                    and not any(a.effect is wanted for a in session.affix_options)
                )
                if hunting and session.refresh_rule():
                    continue
            options = session.affix_options
            pick = 0
            if explore and explore_rng.random() < explore:
                # 探索：不管推荐分数，随机挑一条，把冷门词缀也走一遍
                pick = explore_rng.randrange(len(options)) if options else 0
            else:
                ranked = gm.recommend_affixes(
                    session.state,
                    options,
                    settled_paint=settled,
                    remaining_rules=session.remaining_rules,
                )
                if ranked:
                    pick = next(
                        (index for index, affix in enumerate(options) if affix == ranked[0].affix),
                        0,
                    )
            if not session.choose_affix(pick):
                break
            continue
        actions = engine.legal_actions(session.state)
        if not actions:
            break
        # 融合专线第二部分：手上已经拿到「可放置于失色地块中」、且棋盘上留着
        # 效果包，就直接把锚点放进那一格，别听策略的（它看不懂融合的价值）。
        if (
            fusion_explore
            and gm.current_rule_is_pending(session.state)
            and gm._kept_effect_placeable(session.state)
        ):
            if _place_on_kept_effect():
                continue
        # 融合专线第三部分：想让锚点格被炸到、留下效果包，就得**主动用掉强制
        # 完成**。策略自己几乎不会选它（贪心只看即时收益）。这里在一局的前半段
        # 主动消耗一次，把融合的种子埋下去；炸不炸得中锚点是随机的（实测约 51%），
        # 中不中交给种子分布去覆盖。
        if (
            fusion_explore
            and not gm.current_rule_is_pending(session.state)
            and gm.backpack_paint_placed(session.state)
            and _kept_effect_cell() is None
            and session._force_used == 0
        ):
            if session.force_complete():
                continue
        action = policy.choose(session.state, engine, rng, force_used=session._force_used)
        if action is None:
            break
        if action.kind is ActionKind.PLACE_ANCHOR:
            ok = session.place_anchor(action.row, action.col)
        elif action.kind is ActionKind.INSTALL:
            ok = session.install(action.backpack_index, action.row, action.col)
        else:
            ok = session.force_complete()
        if not ok:
            break
    return session


def simulate_practice(
    *,
    runs: int = 50,
    base_seed: int = 1,
    pet_level: int = gm.DEFAULT_PET_LEVEL,
    rules: int | None = 2,
    prefer: str = 'greedy',
    wait_for_anchor: bool = False,
) -> PracticeSummary:
    """批量自动对局，返回分布统计。用于「AI 模拟练习」看官方几率下的期望。"""
    found: list[HighScoreExplanation] = []
    for index in range(max(1, int(runs))):
        session = start_practice(
            base_seed + index,
            pet_level=pet_level,
            rules=rules,
            wait_for_anchor=wait_for_anchor,
        )
        auto_play(session, prefer=prefer)
        found.append(
            explain_high_score(
                session.state,
                steps=session.steps,
                session=session,
                weights=session.weights,
            )
        )
    return summarize_runs(found)
