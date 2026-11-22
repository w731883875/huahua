"""笔触面板 ROI 的右边界必须单独算，不能跟背包共用的 ``right``。

用户 2026-11-22 在 1917x1077 窗口下逐像素量过（Canny 轮廓 + Sobel 边缘 + 逐列
采样，并用项目里的 ``card_bands_ok`` / ``gutter_bands`` 实跑验证）：

* 棋盘 rect = (336, 242, 630, 630)，格距 105 = 630/6；
* 词缀卡区真值 = x ∈ [1048, 1619]、y ∈ [430, 650]；
* 即 x ∈ [b.right+0.13·b.w, b.right+1.04·b.w]。

而 ``right = width - b.w·0.06`` 假设「面板右边距窗口右缘只有一点点」，这一屏右侧
实际空出约 298px（≈0.47·b.w）雪地，ROI 右边界于是多出约 260px（31%）。混进大片
空白后 ``card_bands_ok`` 的「上下两张卡」分带判据崩掉，面板检测不出来——实测
17:30 之后 ``brush_options`` 一条都没有（此前累计 390 条）。

``right`` 本身不能动：BACKPACK 区域要用它（content_x0 到 right 是背包面板宽度）。
"""

from __future__ import annotations

import unittest

from huahua_assistant.models import BoardObservation, Rect


def _brush_roi(board: Rect, width: int) -> tuple[int, int]:
    """复算笔触面板 ROI 的 (左边, 右边)，与 vision._estimate_regions 同式。"""

    x0 = board.right + int(board.w * 0.10)
    right = min(width, width - int(board.w * 0.06))
    brush_right = min(right, board.right + int(board.w * 1.04) + 1)
    return x0, brush_right


class BrushRoiTests(unittest.TestCase):
    def test_right_edge_tracks_card_area_not_window_edge(self) -> None:
        board = Rect(336, 242, 630, 630)
        left, right = _brush_roi(board, 1917)

        # 实测卡区 1048~1619。ROI 要盖住它，但不能把右侧雪地全吃进来。
        self.assertLessEqual(left, 1048)
        self.assertGreaterEqual(right, 1619)
        self.assertLessEqual(right, 1650, "右边界不能再跑到窗口右缘附近")

    def test_shared_right_would_overshoot(self) -> None:
        """保留这条是为了说明「为什么不能直接用共用的 right」。"""

        board = Rect(336, 242, 630, 630)
        width = 1917
        shared_right = min(width, width - int(board.w * 0.06))
        _, brush_right = _brush_roi(board, width)

        self.assertGreater(shared_right - brush_right, 200, "旧边界至少多出 200px 空白")

    def test_roi_stays_sane_on_smaller_window(self) -> None:
        """窗口变小时 brush_right 不能超过右边界，也不能小于左边。"""

        board = Rect(180, 120, 360, 360)
        left, right = _brush_roi(board, 1000)

        self.assertLess(left, right)
        self.assertLessEqual(right, 1000)


if __name__ == "__main__":
    unittest.main()
