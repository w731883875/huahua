"""棋盘等级/种类的时序稳定。

用户 2026-11-22 实机反馈：点击地块查看加成时，游戏浮窗会挡住格子，等级读数在
``1↔7``、``3↔6`` 之间来回抖。抖动帧如果直接进模型，会同时污染三处：

* 观测/轨迹库（存进错误等级，训练证据变脏）；
* 逐格对账（拿抖动值去比，报出一堆假 mismatch）；
* 地块加成反推（``_rebase_observed_levels`` 用「实测 - 加成」反推颜料等级，
  实测抖一下就会把加成算错）。

所以这里做两件事：

1. **连续 N 帧读数一致才采纳变化**（默认 2 帧）。不一致时先沿用上一次确认值。
2. **一局结算后冻结**：等级和种类不再变，只有加成可能还在变。

为什么不放更多帧：连拍节拍是 100ms，2 帧≈200ms，够挡住单帧遮挡，又不会让真实
落子慢半拍。真实变化（落子/升级）通常会连续多帧稳定，代价可以接受。
"""

from __future__ import annotations

from dataclasses import replace

from .models import BoardObservation, TileObservation, TileState

#: 连续这么多帧读数一致，才采纳一次等级/种类变化。
DEFAULT_STABLE_FRAMES = 2

#: 一格的「身份」：等级 + 种类 + 宝箱标记。只有这些变了才算变化。
CellIdentity = tuple[object, object, object, object]


class LevelStabilizer:
    """按格子维护等级/种类的时序稳定器。"""

    def __init__(self, stable_frames: int = DEFAULT_STABLE_FRAMES) -> None:
        self.stable_frames = max(1, int(stable_frames))
        self._committed: dict[tuple[int, int], CellIdentity] = {}
        self._pending: dict[tuple[int, int], tuple[CellIdentity, int]] = {}
        self._frozen = False

    # ---- 生命周期 ---------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """一局结算后调用：此后等级与种类不再接受变化。"""
        self._frozen = True

    def reset(self) -> None:
        """新一局开始：清空历史并解除冻结。"""
        self._committed.clear()
        self._pending.clear()
        self._frozen = False

    # ---- 主流程 -----------------------------------------------------
    def observe(self, board: BoardObservation) -> BoardObservation:
        """返回等级/种类经过稳定处理的棋盘。不修改入参。"""

        tiles: list[TileObservation] = []
        for tile in board.tiles:
            key = (tile.row, tile.col)
            if tile.state is not TileState.PAINT:
                # 不是颜料就没有等级可言，顺手丢掉它的待定读数。
                self._pending.pop(key, None)
                tiles.append(tile)
                continue
            read = self._identity(tile)
            if self._is_blank_read(read):
                # 这一帧压根没读出颜料：不参与表决，也不算「变化」。
                # 首帧读不出来是常见情况（遮挡/动画中间态），不该被当成新落笔。
                tiles.append(tile)
                continue
            committed = self._committed.get(key)
            if committed is None:
                self._committed[key] = read
                self._pending.pop(key, None)
                tiles.append(tile)
                continue
            if read == committed:
                self._pending.pop(key, None)
                tiles.append(tile)
                continue
            if self._frozen:
                tiles.append(self._apply(tile, committed))
                continue
            same, count = self._pending.get(key, (None, 0))
            count = count + 1 if same == read else 1
            if count >= self.stable_frames:
                self._committed[key] = read
                self._pending.pop(key, None)
                tiles.append(tile)
            else:
                self._pending[key] = (read, count)
                tiles.append(self._apply(tile, committed))
        return replace(board, tiles=tuple(tiles))

    def committed_level(self, row: int, col: int) -> int | None:
        """这一格当前被确认的等级（没有就返回 ``None``）。"""
        identity = self._committed.get((row, col))
        if identity is None:
            return None
        level = identity[2]
        return int(level) if isinstance(level, int) else None

    def adopt(self, row: int, col: int, tile: TileObservation) -> None:
        """直接把某格的读数确认为基准（用于锚点这类高置信来源）。"""
        identity = self._identity(tile)
        if self._is_blank_read(identity):
            return
        self._committed[(row, col)] = identity
        self._pending.pop((row, col), None)

    # ---- 内部 -------------------------------------------------------
    @staticmethod
    def _identity(tile: TileObservation) -> CellIdentity:
        return (
            tile.paint_category,
            tile.paint_subtype,
            tile.paint_level,
            tile.paint_is_treasure,
        )

    @staticmethod
    def _is_blank_read(identity: CellIdentity) -> bool:
        """这一帧有没有真的读出颜料内容。"""

        category, subtype, level, _treasure = identity
        return category is None and subtype is None and level is None

    @staticmethod
    def _apply(tile: TileObservation, identity: CellIdentity) -> TileObservation:
        category, subtype, level, treasure = identity
        return replace(
            tile,
            paint_category=category,
            paint_subtype=subtype,
            paint_level=level,
            paint_is_treasure=treasure,
        )
