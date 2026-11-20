"""状态编码字典：把「棋盘 / 颜料 / 词缀」换成短编号，供记录与回放共用。

## 为什么需要它

实测 926 步里只有 283 种不同棋盘状态（重复率 69%，最多的一个状态出现 205 次）。
每步重写完整 36 格是纯浪费。编号之后每步只写 ``B12``，回放时查字典还原。

## 为什么用序号而不是哈希

哈希虽然免维护，但样本攒多了没法「从编号反查内容」，也统计不出「这个局面见过
几次」。字典是双向的：既能压体积，又能当索引——样本足够多之后 ``B1234`` 本身
就是那个棋盘。

## 三类编号

- **棋盘** ``B1``：``states`` + 每格颜料 + 每格地块加成 + 当前调色目标，四者
  合起来才算「一个局面」。只按 ``states`` 编号会把「空地和不同颜料」混为一谈。
- **颜料** ``sr1``：子类型首字母 + 类别 + 等级。子类型没识别出来时退化成
  ``M1``（现有 ``paint_code`` 的写法），所以两套编码可以共存。
- **词缀** ``ZF-1``：转锋笔触的词缀编号，方向改写类单独成族。

## 还原约定

``tiles`` 是 36 项数组，``null`` 表示空格。有颜料的那格存
``{"id": "sr1", "b": 2, "au": 1}``——``b`` 是地块加成、``au`` 是光环强度、
``al`` 是派生缓存（不存，回放时由 :func:`huahua_assistant.game_model._recompute_auras`
重算）。这样「初始颜料落在某格」这件事在还原后自带 buff。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: 字典文件格式版本。改了编码规则就要 +1，旧文件会被拒绝加载而不是静默错读。
CODEBOOK_VERSION = 1

#: 子类型首字母。没有识别出子类型时用 ``None``，退回 ``{类别}{等级}`` 的老写法。
#:
#: 规则：取家族名**前两个字的拼音首字母**（鼠人 shu-ren → ``sr``）。
#: 26 个家族里有 2 组会撞号，按「首字母 + 首字拼音的第二个字母」区分，
#: 并在下面单独标注——这两条是唯一不守通则的地方：
#:
#: * 觉醒者 jue-xing → ``jx`` ｜ 机械之神 ji-xie → ``ji``（撞 ``jx``）
#: * 英灵 ying-ling → ``yl`` ｜ 幽灵 you-ling → ``yo``（撞 ``yl``）
#:
#: ``赛季`` sai-ji → ``sj``：它是**独立的玩法家族**，只有 Lv3/Lv4 两级，
#: 有自己的一张图标（解包里 ``ColorPlay`` 共 15 张，第 15 张没被任何 PlotItem
#: 记录引用，就是它）。所以照常给两字母编码，不退回 ``G3``。
_SUBTYPE_INITIALS = {
    # 怪物颜料 13 种
    "鼠人": "sr",
    "矮人": "ar",
    "觉醒者": "jx",
    "蛊虫": "gc",
    "龙骨": "lg",
    "灾烬": "zj",
    "实验体": "sy",
    "黑帮": "hb",
    "伊吉": "yj",
    "沙怪": "sg",
    "邪影": "xy",
    "英灵": "yl",
    "幽灵": "yo",
    # 玩法颜料 13 种
    "宝箱": "bx",
    "存活": "ch",
    "机械之神": "ji",
    "巨力之神": "jl",
    "狩猎之神": "sl",
    "征战之神": "zz",
    "黑潮": "hc",
    "黑帆": "hf",
    "魔方": "mf",
    "迷城": "mc",
    "梦魇": "my",
    "怪谈": "gt",
    "玩偶": "wo",
    # 赛季家族（特殊：只有 Lv3/Lv4 两级）
    "赛季": "sj",
}

#: 类别首字母，和 ``codes.PAINT_CATEGORY_CODES`` 保持一致。
_CATEGORY_INITIALS = {
    "monster": "m",
    "gameplay": "g",
    "unknown": "u",
}


def board_signature(states: str, tiles: Iterable[dict | None], current_code: str | None) -> str:
    """一个局面的指纹：地块状态 + 每格颜料与加成 + 当前调色目标。

    刻意**不含**候选列表和背包——那些是「这个局面下算出来的东西」，属于派生量，
    换台机器重算也一致，没必要进签名。
    """

    parts: list[str] = [states or ""]
    for tile in tiles:
        if not tile:
            parts.append("-")
            continue
        parts.append(
            "{code}:{bonus}:{aura}".format(
                code=tile.get("code") or tile.get("id") or "?",
                bonus=int(tile.get("b") or 0),
                aura=int(tile.get("au") or 0),
            )
        )
    parts.append(current_code or "-")
    return "|".join(parts)


def paint_id(
    category: str | None,
    level: int | None,
    subtype: str | None,
    *,
    is_treasure: bool = False,
) -> str:
    """``sr1`` = 鼠人 lv1；``M1`` = 没识别出种类的怪物 lv1。

    宝箱沿用现有的 ``T{等级}``——它本身就是唯一一类，不需要子类型前缀。
    """

    if is_treasure:
        return f"T{int(level or 0)}"
    initial = _SUBTYPE_INITIALS.get(subtype or "")
    if initial:
        return f"{initial}{int(level or 0)}"
    return f"{(_CATEGORY_INITIALS.get(category or '') or 'u').upper()}{int(level or 0)}"


def affix_id(brush_code: str | None, direction_code: str | None, index: int = 1) -> str:
    """``ZF-1`` = 转锋第 1 条词缀。

    转锋（``brush_code == "1"``）的词缀会改写方向，所以编号里带上方向码，
    回放时不用再查词缀表就能知道「这条词缀把方向改成了什么」。
    """

    prefix = {"1": "ZF", "2": "DQ", "3": "KS"}.get(brush_code or "", "QT")
    if direction_code:
        return f"{prefix}-{direction_code}{index}"
    return f"{prefix}-{index}"


@dataclass
class Codebook:
    """棋盘 / 颜料 / 词缀的双向字典。

    只增不减：一个编号一旦分配就不会回收，否则旧记录会指向错的内容。
    """

    version: int = CODEBOOK_VERSION
    boards: dict[str, dict[str, Any]] = field(default_factory=dict)
    """编号 → 局面定义。"""
    board_index: dict[str, str] = field(default_factory=dict)
    """签名 → 编号。和 ``boards`` 互为反查。"""
    paints: dict[str, dict[str, Any]] = field(default_factory=dict)
    affixes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _dirty: bool = field(default=False, repr=False)

    # --- 编号分配 ------------------------------------------------------

    def intern_board(
        self,
        signature: str,
        payload: dict[str, Any],
    ) -> str:
        """给一个局面分配编号。已见过就直接返回旧编号。"""

        existing = self.board_index.get(signature)
        if existing is not None:
            return existing
        code = f"B{len(self.boards) + 1}"
        self.boards[code] = payload
        self.board_index[signature] = code
        self._dirty = True
        return code

    def intern_paint(self, code: str, payload: dict[str, Any]) -> str:
        if code not in self.paints:
            self.paints[code] = payload
            self._dirty = True
        return code

    def intern_affix(self, code: str, payload: dict[str, Any]) -> str:
        if code not in self.affixes:
            self.affixes[code] = payload
            self._dirty = True
        return code

    # --- 反查 ----------------------------------------------------------

    def resolve_board(self, code: str) -> dict[str, Any] | None:
        return self.boards.get(code)

    def resolve_paint(self, code: str) -> dict[str, Any] | None:
        return self.paints.get(code)

    def resolve_affix(self, code: str) -> dict[str, Any] | None:
        return self.affixes.get(code)

    # --- 持久化 --------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "boards": self.boards,
            "board_index": self.board_index,
            "paints": self.paints,
            "affixes": self.affixes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8",
        )
        self._dirty = False

    @classmethod
    def load(cls, path: Path) -> "Codebook":
        """读字典。文件不存在、损坏、版本不符时返回空字典而不是抛异常——
        丢一次索引不该让工具起不来。"""

        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict) or data.get("version") != CODEBOOK_VERSION:
            return cls()
        return cls(
            version=CODEBOOK_VERSION,
            boards=dict(data.get("boards") or {}),
            board_index=dict(data.get("board_index") or {}),
            paints=dict(data.get("paints") or {}),
            affixes=dict(data.get("affixes") or {}),
        )

    # --- 统计 ----------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "boards": len(self.boards),
            "paints": len(self.paints),
            "affixes": len(self.affixes),
        }
