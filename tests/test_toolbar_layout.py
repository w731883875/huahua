"""工具栏布局守卫：控件必须落在窗口里，不许再被挤到窗口右边外面。

这条测试是给一个**反复犯**的 bug 准备的：工具栏只有一行时整条 reqwidth 实测
1893px，而 1180 宽的窗口只有 1152px 可用，于是「纠错识别」及其后面的
识别统计/导出数据/合并数据/强制完成/宠物等级/强制已用/刷新已用/局面
全排在窗口外——按钮画出来了、看得见一部分、就是点不到，也不会报错。
桌宠的两个开关第一次就是这么「加了但看不见」的。

所以这里锁两条不变量：

1. 结构上：工具栏里的控件必须挂在 ``primary_row`` / ``secondary_row`` 上，
   不许直接挂在 ``toolbar`` 上绕过分行（静态检查源码，不需要开窗口）；
2. 尺寸上：每一行的 reqwidth 都必须小于窗口可用宽度（真开一次窗口去量）。
"""

from __future__ import annotations

import re
import tkinter as tk
import unittest
from pathlib import Path

APP_SOURCE = Path(__file__).resolve().parents[1] / "huahua_assistant" / "app.py"

#: 工具栏外层 pack 的左右留白是 ``padx=14``，可用宽度要扣掉两边。
TOOLBAR_PAD = 14


class ToolbarStructureTests(unittest.TestCase):
    """静态检查：不许有控件直接挂到 toolbar 上。"""

    def test_no_widget_is_parented_to_the_toolbar_directly(self) -> None:
        source = APP_SOURCE.read_text(encoding="utf-8")
        start = source.index("primary_row = tk.Frame(toolbar")
        end = source.index("self.game_log_label = tk.Label(")
        block = source[start:end]

        offenders = re.findall(
            r"tk\.(Button|Checkbutton|Spinbox|Label|Entry|Menubutton)\(\s*toolbar\b",
            block,
        )
        self.assertEqual(
            offenders,
            [],
            "工具栏控件必须挂到 primary_row / secondary_row 上："
            f"直接挂 toolbar 会排到窗口外面、点不到，发现 {offenders}",
        )

    def test_toolbar_has_exactly_two_inner_rows(self) -> None:
        source = APP_SOURCE.read_text(encoding="utf-8")
        self.assertIn("primary_row = tk.Frame(toolbar", source)
        self.assertIn("secondary_row = tk.Frame(toolbar", source)


class ToolbarWidthTests(unittest.TestCase):
    """尺寸检查：每一行都得装得进窗口。"""

    def test_every_row_fits_inside_the_window(self) -> None:
        from huahua_assistant.app import HuahuaApp

        app = HuahuaApp()
        try:
            app.update()
            frames = [w for w in app.winfo_children() if isinstance(w, tk.Frame)]
            self.assertTrue(frames, "找不到工具栏 Frame")
            toolbar = frames[0]
            available = app.winfo_width() - TOOLBAR_PAD * 2

            rows = [w for w in toolbar.winfo_children() if w.winfo_manager() == "pack"]
            self.assertEqual(len(rows), 2, "工具栏应该正好两行")

            for index, row in enumerate(rows, start=1):
                self.assertLessEqual(
                    row.winfo_reqwidth(),
                    available,
                    f"工具栏第 {index} 行要 {row.winfo_reqwidth()}px，"
                    f"窗口只有 {available}px 可用——右边那些控件会被裁掉、点不到",
                )

            # 负例自检：如果这两行合并成一行，一定装不下（说明拆行不是多余的）。
            merged = sum(row.winfo_reqwidth() for row in rows)
            self.assertGreater(
                merged,
                available,
                f"两行合计只有 {merged}px，比可用宽度 {available}px 还小，"
                "说明这条守卫已经测不出东西了（或窗口宽得离谱）",
            )
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
