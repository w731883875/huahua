# 笔触面板阶段二读不出时的排查顺序

2026-11-22 排查记录。`affix_card_clicked` 长期为 0，根因不在点击判定，而在
`brush_options.stage2_options` 为空——没有候选，点击就无处映射。

## 一、先分清「面板没检出」和「文字没读出」

两者现象一样（都是 `brush_options` 为空），但修法完全不同。

用**工具自己**的实时能力判，不要另起进程：

```python
scene = analyze_scene_realtime(frame)
scene.board.rect          # 棋盘定位对不对
scene.brush_options        # None = 面板没检出
scene.brush_options.panel_rect
scene.brush_options.stage1_options   # 有值 = 模板匹配这条路通
scene.brush_options.stage2_options   # 空 = 卡在文字识别
```

2026-11-22 18:20 实机一帧的结果：

```
board.rect = (338, 250, 618, 618)  score 0.808
panel      = (1018, 392, 865, 297)
stage1     = ('stack',)
stage2     = ()
card_bands_ok = True
gutter_bands  = [(33,144), (155,266)]
```

说明棋盘、面板、分带**全都是好的**，唯独 stage2 文案读不出来。

## 二、不要用独立脚本判断 OCR 好坏

`windows_ocr.recognize_chinese_lines` 走的是**子进程 + 管道 stdio**。在独立脚本
里跑会返回 `()`，看起来像「OCR 坏了」，其实是被环境限制住了：

* 同一个脚本对 `jietu/转锋后选择前.png`（视觉回归里明明能读出正确文案）也返回 0 行；
* 所以「0 行」这个结果本身不能作为 OCR 失效的证据。

要验 OCR，只能在工具进程里看 `runtime/observations.jsonl` 的 `brush_options`，
或者看 `affix_choice_observed` 里的 `options` 有没有真实中文文案
（2026-11-22 17:21:39 那条读到过「相邻随机阻挡地块变为空白地块」等，说明当时 OCR 是通的）。

## 三、不要靠「量出卡区位置」来收窄 ROI

有人按某个窗口尺寸量出「卡区右缘 = b.right + 1.04·b.w」，据此收窄 ROI。用
`jietu/` 下 4 张真实截图做参数扫描：

| 右边界 | 通过 |
|---|---|
| `1.04·b.w` | 0/4 |
| `1.15·b.w` | 0/4 |
| `1.30·b.w` | 0/4 |
| 不限 | **4/4** |

收窄会把卡片正文切掉。右侧多余空白交给 `vision._trim_blank_right` 自动裁，
`BRUSH_ROI_RIGHT_RATIO` 保持 99.0。`tests/test_brush_roi.py` 有护栏。

## 四、另一条容易踩的坑：截图尺寸

PowerShell 里 `CopyFromScreen` 抓出来是 1536×864（DPI 虚拟化），而工具实际捕获
是 1920×1080。用错尺寸的图去跑 `analyze_scene_realtime` 会得到
`board.rect=(350,250,608,608) score=0.697`（低于 0.7 门槛）这种假失败。

抓图前必须声明 DPI 感知：

```powershell
[DpiFix]::SetProcessDPIAware()   # SetProcessDPIAware, 否则尺寸被缩放
```

同一帧正确尺寸下 score 是 0.808。
