"""笔触面板 ROI：可以自动裁右侧空白，但**不能**靠常数收窄。

一、教训（2026-11-22）
有人按某一个窗口尺寸量出「词缀卡区右缘 = b.right + 1.04·b.w」，据此主张把
BRUSH_OPTIONS 的 ROI 右边界从 ``width - 0.06·b.w`` 收到 1.04·b.w。照改之后用
``jietu/`` 下 4 张真实截图做回归：

    ratio = 1.04  → stage2_options 全空，0/4 通过
    ratio = 1.15  → 0/4
    ratio = 1.30  → 0/4
    ratio = 不限  → 4/4 全对

**收窄会把卡片正文切掉，OCR 直接读不到词缀文案。** 所以右边界保持原样，不能收。

二、真正的修法
右侧多出来的空白交给 :func:`_trim_blank_right` 自动裁掉。它不依赖任何面板常数，
从右侧往里找第一列有内容的，留一点边距。动机是
:func:`brush_panel._split_brush_cards` 用**整幅宽度**求行活动度均值
（``active.mean(axis=1)``），多出来的空白列会把它稀释到阈值以下，「上下两张卡」
就分不出来。

实测：裁剪开启/关闭对 4 张真实截图的 stage2 识别都是 4/4，说明它没切掉卡片正文，
属于安全的稳健性加固。
"""

from __future__ import annotations

import unittest

import numpy as np

from huahua_assistant import vision


class BrushRoiTests(unittest.TestCase):
    def test_right_edge_is_not_narrowed_by_a_constant(self) -> None:
        """回归护栏：右边界一旦重新按常数收窄，真实截图会立刻读不到词缀。"""

        self.assertGreater(
            vision.BRUSH_ROI_RIGHT_RATIO,
            2.0,
            "右边界被收窄成有限倍数了——jietu/ 下 4 张截图会 0/4，见模块 docstring",
        )


class TrimBlankRightTests(unittest.TestCase):
    def _image(self, blank_right: int) -> np.ndarray:
        h, w = 200, 560 + blank_right
        img = np.full((h, w, 3), 245, np.uint8)  # 右侧空白（白）
        img[:, 20:560] = np.array([120, 90, 60], np.uint8)  # 卡片区
        img[30:90, 30:120] = np.array([200, 180, 40], np.uint8)
        img[110:170, 30:120] = np.array([200, 180, 40], np.uint8)
        return img

    def test_blank_right_is_trimmed(self) -> None:
        wide = vision._trim_blank_right(self._image(300))
        tight = vision._trim_blank_right(self._image(0))

        self.assertLess(wide.shape[1], 860, "右侧空白应该被裁掉")
        self.assertGreaterEqual(wide.shape[1], 560, "但不能切掉卡片正文")
        self.assertEqual(560, tight.shape[1], "本来就没空白就不该动")

    def test_trimming_keeps_enough_width(self) -> None:
        """裁得太狠会切掉正文，至少留 55% 宽度。"""

        out = vision._trim_blank_right(self._image(0))
        self.assertGreaterEqual(out.shape[1], int(560 * 0.55))

    def test_all_blank_image_is_left_alone(self) -> None:
        blank = np.full((100, 400, 3), 245, np.uint8)

        self.assertEqual(400, vision._trim_blank_right(blank).shape[1])


if __name__ == "__main__":
    unittest.main()
