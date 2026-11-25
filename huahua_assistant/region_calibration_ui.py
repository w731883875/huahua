"""可拖动的标定覆盖层：把四块识别区域画成绿框，拖到正确位置后固定。

窗口行为
--------
* 跟随游戏窗口（``overrideredirect`` + 置顶），坐标系为窗口物理像素；
* **半透明**：整层默认 30% 不透明，游戏画面透过来看得见。一开始这里忘了设
  ``-alpha``，窗口是不透明纯黑，用户 2026-11-25 报「整块黑屏挡住了游戏」；
* **可穿透**：按 ``T`` 切成点击穿透，只用来对着游戏核对位置，此时不能拖；
* 不抠色（``-transparentcolor``）——抠掉的像素会连鼠标一起穿透，那样框内就
  拖不动了。半透明能在「看得见游戏」和「抓得住框」之间取平衡。

操作
----
* 拖框内部：平移
* 拖右下角小方块：缩放
* ``[`` / ``]``：调暗 / 调亮（0.05 ~ 0.85）
* ``T``：切换点击穿透（穿透时只观察，不能拖）
* ``F5`` / 按钮：重新识别一次并把新结果画上去
* ``Ctrl+S`` / 按钮：固定（写 ``runtime/calibration.json``）
* ``Esc``：关闭，不保存
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from typing import Callable

from . import region_calibration as calib
from .region_calibration import REGION_LABELS, Region

GREEN = "#22ff66"
GREEN_DIM = "#0f8a33"
LABEL_BG = "#04300f"
WHITE = "#eafff0"
GRIP_SIZE = 10
DRAG_THRESHOLD = 2

#: 默认不透明度。太低看不见绿框，太高挡住游戏；0.30 实测两边都能用。
DEFAULT_ALPHA = 0.30
MIN_ALPHA = 0.05
MAX_ALPHA = 0.85
ALPHA_STEP = 0.07


class CalibrationOverlay:
    """标定用覆盖层。``parent`` 是 Tk 根窗口。"""

    def __init__(
        self,
        parent: tk.Misc,
        regions: dict[str, Region],
        *,
        on_recalibrate: Callable[[], dict[str, Region] | None] | None = None,
        on_saved: Callable[[dict[str, Region]], None] | None = None,
    ) -> None:
        self.regions = dict(regions)
        self.on_recalibrate = on_recalibrate
        self.on_saved = on_saved
        self._drag: dict[str, object] | None = None
        self._alpha = DEFAULT_ALPHA
        self._clickthrough = False
        self._parent = parent
        self._parent_state: str | None = None

        self.window = tk.Toplevel(parent)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.configure(bg=GREEN_DIM)
        # 关键：不设 -alpha 就是一块不透明纯黑，把游戏整个盖住
        # （用户 2026-11-25 实测反馈）。这里默认 30% 透明。
        try:
            self.window.attributes("-alpha", self._alpha)
        except tk.TclError:
            pass

        self.canvas = tk.Canvas(
            self.window, bg="#000000", highlightthickness=0, cursor="fleur"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._place_over_window(parent)
        # 缺哪块就补默认值：漏画一块会让人以为「这个功能没有它」，而实际上
        # 只是没拿到初始值（用户 2026-11-25：笔触卡片 1、2 都未出现）。
        self.regions = calib.merge_regions(self.overlay_w, self.overlay_h, self.regions)
        self._inset_scope_off_board()
        self._active_key = "board"
        self._build_toolbar(parent)
        self._bind_events()
        self.redraw()
        self._hide_parent()
        if getattr(self, "_game_window_missing", False):
            self._flash("没找到游戏窗口（最小化了？）已按整屏定位，先把游戏切回来")

    # ---- 窗口摆放 ---------------------------------------------------
    def _game_window_rect(self, parent: tk.Misc) -> tuple[int, int, int, int] | None:
        """取游戏窗口的屏幕矩形（物理像素）。拿不到就退化成整屏。"""
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            target = None
            title_wanted = "torchlight"

            @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def enum_proc(hwnd, _lparam):
                nonlocal target
                length = user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if title_wanted in buf.value.lower():
                    target = hwnd
                    return False
                return True

            user32.EnumWindows(enum_proc, 0)
            if target is None:
                return None
            # 最小化的窗口不能用来定位：实测 GetWindowRect 会回
            # (-25600, -25600, 159, 27) —— 位置在屏幕外、尺寸只剩标题栏。
            # 照着建覆盖层就会得到 159x27 的一小块，四块区域全被最小尺寸夹成
            # 24x24，看起来就是「绿框全没了」（2026-11-25 实测到）。
            if user32.IsIconic(target):
                return None
            rect = wintypes.RECT()
            user32.GetWindowRect(target, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if rect.left <= -20000 or rect.top <= -20000 or width < 320 or height < 240:
                return None
            return (rect.left, rect.top, width, height)
        except Exception:
            return None

    def _place_over_window(self, parent: tk.Misc) -> None:
        rect = self._game_window_rect(parent)
        if rect is None:
            width = self.window.winfo_screenwidth()
            height = self.window.winfo_screenheight()
            self.window.geometry("%dx%d+0+0" % (width, height))
            self._game_window_missing = True
        else:
            left, top, width, height = rect
            self.window.geometry("%dx%d+%d+%d" % (width, height, left, top))
            self._game_window_missing = False
        # 尺寸直接来自刚算出的值，**不要回读 geometry()**：窗口还没映射时
        # Tk 回的是 "1x1+0+0"，会把整层算成 1x1，四块框全挤在左上角重叠
        # （2026-11-24 冒烟测试踩到过：overlay=1x1，命中测试全落到别的框上）。
        self.overlay_w = max(1, int(width))
        self.overlay_h = max(1, int(height))

    def _inset_scope_off_board(self) -> None:
        """棋盘与调色范围默认完全重合，视觉上分不开。

        「调色范围」本来就是棋盘里的一片格子，初始两者必然重叠。把调色范围
        往内缩一圈，两块框都能看见，也都能抓到边去拖。
        """
        board = self.regions.get("board")
        scope = self.regions.get("scope")
        if board is None or scope is None:
            return
        if (scope.x_ratio, scope.y_ratio, scope.w_ratio, scope.h_ratio) != (
            board.x_ratio,
            board.y_ratio,
            board.w_ratio,
            board.h_ratio,
        ):
            return
        bx, by, bw, bh = board.to_pixels(self.overlay_w, self.overlay_h)
        pad = max(6, int(round(min(bw, bh) * 0.04)))
        self.regions["scope"] = Region.from_pixels(
            "scope", bx + pad, by + pad, bw - 2 * pad, bh - 2 * pad,
            self.overlay_w, self.overlay_h,
        )

    # ---- 工具栏 -----------------------------------------------------
    def _build_toolbar(self, parent: tk.Misc) -> None:
        bar = tk.Frame(self.window, bg=LABEL_BG)
        bar.place(relx=0.5, y=6, anchor="n")
        tk.Label(
            bar,
            text="区域标定：拖动绿框到正确位置，右下角小方块缩放",
            bg=LABEL_BG,
            fg=WHITE,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(8, 10), pady=4)
        for text, command in (
            ("重新识别 (F5)", self.recalibrate),
            ("固定 (Ctrl+S)", self.save),
            ("关闭 (Esc)", self.close),
            ("变暗 [", lambda: self.nudge_alpha(-ALPHA_STEP)),
            ("变亮 ]", lambda: self.nudge_alpha(ALPHA_STEP)),
            ("穿透 (T)", self.toggle_clickthrough),
        ):
            tk.Button(
                bar,
                text=text,
                command=command,
                bg="#0b6b2a",
                fg=WHITE,
                activebackground="#12933a",
                activeforeground=WHITE,
                relief=tk.FLAT,
                font=("Segoe UI", 9),
            ).pack(side=tk.LEFT, padx=3, pady=4)

        # 「当前编辑哪一块」——棋盘和调色范围本来就重叠，靠点选区分，不靠
        # 命中顺序猜。选中那块才响应拖动，避免拖到一半换了对象。
        picker = tk.Frame(self.window, bg=LABEL_BG)
        picker.place(relx=0.5, y=44, anchor="n")
        tk.Label(
            picker, text="当前编辑：", bg=LABEL_BG, fg=WHITE, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=(8, 4))
        self._active_var = tk.StringVar(value=self._active_key)
        for key, label in REGION_LABELS.items():
            tk.Radiobutton(
                picker,
                text=label.split("（")[0],
                value=key,
                variable=self._active_var,
                command=lambda k=key: self._select(k),
                bg=LABEL_BG,
                fg=WHITE,
                selectcolor="#0b6b2a",
                activebackground=LABEL_BG,
                activeforeground=WHITE,
                font=("Segoe UI", 9),
            ).pack(side=tk.LEFT, padx=4)

    def _select(self, key: str) -> None:
        self._active_key = key
        self.redraw()

    # ---- 透明度 / 穿透 ----------------------------------------------
    def nudge_alpha(self, delta: float) -> None:
        """调不透明度。太低看不见绿框，太高挡住游戏，所以给玩家自己调。"""
        self.set_alpha(self._alpha + delta)

    def set_alpha(self, value: float) -> None:
        self._alpha = max(MIN_ALPHA, min(MAX_ALPHA, round(value, 3)))
        try:
            self.window.attributes("-alpha", self._alpha)
        except tk.TclError:
            pass
        self._flash("不透明度 %.0f%%（[ 变暗 / ] 变亮）" % (self._alpha * 100))

    def toggle_clickthrough(self) -> None:
        """切成点击穿透：只用来对着游戏核对位置，穿透时拖不动。"""
        self._clickthrough = not self._clickthrough
        self._apply_clickthrough(self._clickthrough)
        self._flash(
            "点击穿透：开（只能看，拖不动；再按 T 关）"
            if self._clickthrough
            else "点击穿透：关（可以拖框了）"
        )

    def _apply_clickthrough(self, enabled: bool) -> None:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE = 0x08000000
            hwnd = int(self.window.winfo_id())
            # overrideredirect 的 Toplevel，父窗口才是真正的顶层窗口
            parent = user32.GetParent(hwnd)
            target = parent or hwnd
            style = user32.GetWindowLongW(target, GWL_EXSTYLE)
            if enabled:
                style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE
            else:
                style &= ~WS_EX_TRANSPARENT
                style &= ~WS_EX_NOACTIVATE
            user32.SetWindowLongW(target, GWL_EXSTYLE, style)
        except Exception:
            pass

    # ---- 事件 -------------------------------------------------------
    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.window.bind("<KeyPress-F5>", lambda _e: self.recalibrate())
        self.window.bind("<Control-s>", lambda _e: self.save())
        self.window.bind("<Escape>", lambda _e: self.close())
        self.window.bind("<KeyPress-bracketleft>", lambda _e: self.nudge_alpha(-ALPHA_STEP))
        self.window.bind("<KeyPress-bracketright>", lambda _e: self.nudge_alpha(ALPHA_STEP))
        self.window.bind("<KeyPress-t>", lambda _e: self.toggle_clickthrough())
        self.window.bind("<KeyPress-T>", lambda _e: self.toggle_clickthrough())
        self.window.focus_force()

    def _region_at(self, x: int, y: int) -> tuple[str, str] | None:
        """只对**当前选中的那块**做命中测试。

        棋盘和调色范围天然重叠（调色范围就是棋盘里的一片格子），按绘制顺序
        猜会让其中一块永远拖不到。所以拖动对象由「当前编辑」单选框决定。
        先判右下角把手，再判框内。
        """
        region = self.regions.get(self._active_key)
        if region is None:
            return None
        rx, ry, rw, rh = region.to_pixels(self.overlay_w, self.overlay_h)
        if rx + rw - GRIP_SIZE <= x <= rx + rw and ry + rh - GRIP_SIZE <= y <= ry + rh:
            return self._active_key, "grip"
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return self._active_key, "body"
        return None

    def _on_press(self, event: tk.Event) -> None:
        hit = self._region_at(event.x, event.y)
        if hit is None:
            return
        key, part = hit
        self._drag = {
            "key": key,
            "part": part,
            "x": event.x,
            "y": event.y,
            "start": self.regions[key],
            "moved": False,
        }
        self.canvas.delete("status")

    def _on_motion(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        key = str(self._drag["key"])
        part = str(self._drag["part"])
        start = self._drag["start"]
        assert isinstance(start, Region)
        dx = event.x - int(self._drag["x"])
        dy = event.y - int(self._drag["y"])
        if abs(dx) > DRAG_THRESHOLD or abs(dy) > DRAG_THRESHOLD:
            self._drag["moved"] = True
        if part == "grip":
            self.regions[key] = start.resized_by(dx, dy, self.overlay_w, self.overlay_h)
        else:
            self.regions[key] = start.moved_by(dx, dy, self.overlay_w, self.overlay_h)
        self.redraw()

    def _on_release(self, _event: tk.Event) -> None:
        self._drag = None

    # ---- 绘制 -------------------------------------------------------
    def redraw(self) -> None:
        self.canvas.delete("all")
        for key in REGION_LABELS:
            region = self.regions.get(key)
            if region is None:
                continue
            active = key == self._active_key
            color = GREEN if active else GREEN_DIM
            x, y, w, h = region.to_pixels(self.overlay_w, self.overlay_h)
            if active:
                # 只用点阵填充标出「当前编辑」的那块，不整片涂黑，否则又挡住游戏
                self.canvas.create_rectangle(
                    x, y, x + w, y + h,
                    fill=GREEN, stipple="gray12", outline="", tags=("fill", key),
                )
            self.canvas.create_rectangle(
                x, y, x + w, y + h,
                outline=color, width=3 if active else 1, tags=("frame", key),
            )
            # 右下角缩放把手（只有选中的那块给把手，避免多个把手互相干扰）
            if active:
                self.canvas.create_rectangle(
                    x + w - GRIP_SIZE,
                    y + h - GRIP_SIZE,
                    x + w,
                    y + h,
                    fill=GREEN,
                    outline=GREEN_DIM,
                    tags=("grip", key),
                )
            self.canvas.create_text(
                x + 4,
                y - 8,
                text=("▶ " if active else "") + REGION_LABELS[key],
                fill=color,
                anchor="w",
                font=("Segoe UI", 9, "bold" if active else "normal"),
                tags=("label", key),
            )
        self._draw_board_grid()
        self.canvas.create_text(
            10,
            self.overlay_h - 10,
            text=(
                "绿框=识别区域 | 选「当前编辑」后拖框平移 · 拖右下角缩放 | "
                "[ ] 调透明度 · T 点击穿透 | F5 重新识别 · Ctrl+S 固定 · Esc 关闭"
            ),
            fill=WHITE,
            anchor="sw",
            font=("Segoe UI", 9),
            tags=("status",),
        )

    def _draw_board_grid(self) -> None:
        board = self.regions.get("board")
        if board is None:
            return
        board_px = board.to_pixels(self.overlay_w, self.overlay_h)
        for index, (cx, cy, cw, ch) in enumerate(calib.cells_of(board_px)):
            width = 1 if index % 2 else 1
            self.canvas.create_rectangle(
                cx, cy, cx + cw, cy + ch,
                outline=GREEN_DIM, width=width, tags=("grid",),
            )

    # ---- 动作 -------------------------------------------------------
    def recalibrate(self) -> None:
        if self.on_recalibrate is None:
            return
        updated = self.on_recalibrate()
        if updated:
            self.regions = dict(updated)
            self.redraw()
            self._flash("已按当前识别结果重画绿框")

    def save(self) -> Path | None:
        path = calib.save_regions(self.regions, width=self.overlay_w, height=self.overlay_h)
        if self.on_saved is not None:
            self.on_saved(self.regions)
        self._flash("已固定到 %s" % path.name)
        return path

    def _flash(self, text: str) -> None:
        self.canvas.delete("flash")
        item = self.canvas.create_text(
            10,
            self.overlay_h - 30,
            text=text,
            fill="#ffe680",
            anchor="sw",
            font=("Segoe UI", 10, "bold"),
            tags=("flash",),
        )
        self.window.after(2600, lambda: self.canvas.delete(item))

    def close(self) -> None:
        self._restore_parent()
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def _hide_parent(self) -> None:
        """标定期间把工具窗口最小化。

        工具窗口和游戏窗口几乎完全重叠（实测工具 958x642 @ (26,26)、游戏
        1294x758 @ (121,29)），覆盖层置顶后会同时压住两者，玩家既看不全游戏
        也点不到工具上的按钮，看起来就像「标定不了」。
        """
        try:
            self._parent_state = self._parent.state()
            if self._parent_state == "normal":
                self._parent.iconify()
        except Exception:
            self._parent_state = None

    def _restore_parent(self) -> None:
        try:
            if getattr(self, "_parent_state", None) == "normal":
                self._parent.deiconify()
                self._parent.lift()
        except Exception:
            pass


def open_calibration(
    parent: tk.Misc,
    *,
    width: int,
    height: int,
    on_recalibrate: Callable[[], dict[str, Region] | None] | None = None,
    on_saved: Callable[[dict[str, Region]], None] | None = None,
) -> CalibrationOverlay:
    """打开标定层。已固定的区域会顶替默认值。"""
    regions = calib.load_regions()
    if not regions:
        regions = calib.default_regions(width, height)
    return CalibrationOverlay(
        parent, regions, on_recalibrate=on_recalibrate, on_saved=on_saved
    )