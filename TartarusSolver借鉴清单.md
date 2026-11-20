# TartarusSolver 借鉴清单

对标仓库：<https://github.com/glcoder/TartarusSolver>（MIT，C#/.NET 9，火炬之光：无限 S16「塔尔塔罗斯监狱」沙盒棋盘）

> 同一个游戏引擎、同一个玩法大类（棋盘落子）、同一套"读屏 + 规则搜索 + 提示"的架构。
> 但它已经把三件我们还在硬扛的事做完了：**抓被遮挡的窗口**、**读游戏日志**、**训小模型**。

---

## 一、它这次更新了什么（0.7.38 → 0.7.44，2026-11-02 ~ 11-04）

这段时间的更新**几乎全是抓屏兼容性**，不是模型：

| 版本 | 内容 | 对我们的意义 |
|---|---|---|
| 0.7.44 | 下载断点续传；语言列表加 "Automatic" | 无关 |
| 0.7.43 | 界面/overlay 全语言，简体中文来自 `xingkinglb/TartarusSolver-zh` | 无关 |
| **0.7.42** | **Windows 抓不了窗口时退回"屏幕副本"，此时必须游戏在前台** | **和我们撞的是同一个坑** |
| **0.7.41** | 窗口最大化超出屏幕顶部时帧被裁到显示器，标题栏留在帧里 | 我们改分辨率也会踩 |
| **0.7.40** | 抓屏失败不释放 D3D 设备 → 每次泄漏 28MB/570 句柄，约 850 次后显存耗尽 `0x8876017C` | 长期挂机必踩 |
| 0.7.39 | 从选卡卡片反推 UI 缩放，直到沙盒读出来前每秒重试 | UI 缩放问题 |
| 0.7.38 | `-full` 包内置 OCR 需要的 VC++ 运行时 | 打包时注意 |

**结论：它的"更新"主要在补抓屏的坑。而抓屏正是我们当前卡死的地方。**

---

## 二、它的模型体系（三层，全部开源）

### 1. CellNet —— 视觉，格子分类

```
输入  image (N,3,96,84) RGB 0..1  +  cell (N,16) 位置 one-hot
主干  4 个卷积块（24/40/64/96 通道，每块 2 个 3×3 conv + BN + ReLU），3 次 MaxPool
池化  全局 mean ⊕ 全局 amax  →  96*2 + 16 = 208 维
      ↑ 用 amax 是刻意的：巢穴、"!" 这类小目标会被均值抹掉
头部  Linear(208→128) + ReLU + Dropout(0.2)
输出  5 个多任务头
      kind    4 类  empty / unopened / inmate / covered
      code    4 类  鼠/狼/狮/猛犸
      rarity  5 类  普通/魔法/稀有/传说/终极
      visible 6 类  none + 5 种稀有度（未翻开时显示的）
      flags   6 个  consumable / nest / incident / scratch / tusk / prey
参数量 约 26 万
训练   AdamW lr=2e-3 wd=1e-4，OneCycleLR max_lr=3e-3，batch 64，40 epoch，CPU 十几分钟
```

关键技巧（都可直接抄）：

- **每个头只学自己适用的格子**，不适用的标签填 `IGNORE = -100`，`CrossEntropyLoss(ignore_index=...)` 自然屏蔽。
- **类别权重 = 按频率平方根反比，裁剪到 `[0.5, 8.0]`**——稀有类加权但不至于淹没其他类。
- **flags 用 `BCEWithLogitsLoss` + `pos_weight = clip(sqrt(neg/pos), 1, 8)`**。
- **数据增强不做色相**——颜色就是类别编码（金色=传说），改了就是改标签。
- **验证集按帧分块划分**（8 帧一块整体同侧）——相邻帧高度相似，随机划分会泄漏，验证精度虚高。
- **稀有标记用半合成样本补**：把游戏原图图标按记录的位置/缩放贴到干净格子上，亮度 ×[0.55,1.05] 模拟弹窗遮挡。

### 2. ArrowNet —— 视觉，定位一个特殊标记

48×48 输入，3 个卷积块，输出单个 logit，16 格里取最高分。**跑在 CPU 上**，因为太小。

### 3. ValueNet / PolicyNet —— 决策，不是图像

```
输入  Tartarus.Solver.Features —— 规则特征（从狱长顺时针的格子 + 本局全局量），不是像素
结构  MLP 256 → 128 → 4 头
目标  效用 / 异色数 / 终极数 / 传说数（权重 1.0 / 0.3 / 0.1 / 0.05）
数据  tartarus-sim selfplay --runs 20000 生成，不是人打的
训练  12 epoch，AdamW lr=1e-3，CosineAnnealing，batch 1024
导出  自定义二进制 "TVNT" + 每层 (out,in,weights,biases) float32
```

**注意这一层的输入是"规则特征"而不是"屏幕"**——它和视觉完全解耦，可以单独替换。

---

## 三、它的数据从哪来（这才是重点）

两条腿，都值得抄：

### 腿 A：视觉数据 = 游戏内采帧 + 手标

```
热键 Ctrl+Shift+D 存当前帧（--debug 下还有标签编辑页）
tartarus-scan export dataset     → dataset/train/{cells,arrows}.csv + 裁剪图
tartarus-scan review             → 网页版标注复核
```

历史规模（从 CHANGELOG 倒推）：

| 版本 | 动作 | 标注量 |
|---|---|---|
| 0.7.19 | 重训 CellNet | **840 个手工校验格子** |
| 0.7.20 | 增训 | +200 格 |
| 0.7.22 | 重训 tusk 标记 | ~170 格 |
| 0.7.28 | 重训 prey 标记（此前只有 20 个） | 164 格 |

**840 格 ≈ 23 张整板截图**（36 格/板）。这是它能出可用模型的最小数据量。
100 张板（3600 格）就是宽裕了。

### 腿 B：决策数据 = 模拟器自对弈，不用人打

```
tartarus-sim selfplay --runs 20000 --out values-1.bin
tartarus-sim policydata --runs 20000 --policy vn:64:3 --out actions-1.bin
tartarus-sim compare      # 4000 局配对比较，判断改动好坏
```

它自己写的注释点破了关键：

> the search's value is far less noisy than one run's outcome
> （搜索给出的价值比"单局实际结果"噪声小得多）

所以价值网的训练目标是**搜索的评估值**，不是真实结局——用搜索当老师，网络学它，然后网络再加速搜索。这是 AlphaZero 那套的实用版。

---

## 四、我们能不能训？—— 能，但要分清训哪个

| 模型 | 数据需求 | 训练成本 | 收益 | 结论 |
|---|---|---|---|---|
| **视觉小模型（类 CellNet）** | 25~100 张整板截图（900~3600 格） | CPU 十几分钟 | **替代现在的网格线检测 + 模板匹配**，抗缩放、抗 UI 换肤、抗局部遮挡；推理 ~15ms | ✅ **最该做** |
| 价值网 | 需要完整规则模拟器 + 隐藏信息分布 | 分钟级，但数据要先造 | 给搜索当加速器 | ⚠️ 第二步 |
| 策略网 | 同上 | 同上 | 让 playout 更准 | ⚠️ 第三步 |
| 大模型（LLM） | — | — | 已经是兜底，11/11 成功 | ❌ 不训 |

### 为什么视觉小模型值得

我们现在 `analyze_scene_realtime` 实测 **295ms**，其中大头是**找棋盘**（扫多个候选框 × 多个缩放档做网格打分）。这条路的天花板就在那里：

- 6×6 棋盘内部任意 6×6 子区域都能凑出 7 条等周期网格线 → 只能靠"最外圈判据"区分，我们为此加了 `_outer_edge_penalty` 才把偏差从 1.08 格压到 0.07 格；
- 换分辨率/换 UI 缩放就要重新标定；
- 工具窗口一盖就全盘皆输。

**如果改成"布局由 UI 缩放直接算出来 + 36 个格子各自过 CNN"**：
- 不再需要找棋盘（`PageScale` 从日志读，布局是确定的）；
- 36 格批量推理一次前向 ~15ms，比 295ms 快一个数量级；
- 单格分类抗遮挡、抗缩放，比全局网格打分稳。

### 为什么价值网要等一等

我们 `engine/` 已经有 1178 行（rules/search/value/policy/beam/bench），思路和它同源。但我们的隐藏信息是**"下一条规则长什么样"**，不是"门后是什么"。要自对弈，得先把这个先验分布写出来（现在 `_estimated` 是拍脑袋的先验值）。这是独立的建模工作，不能靠抄。

---

## 五、可以借鉴的（按 ROI 排序）

### P0-1　用 Windows Graphics Capture 抓被遮挡的窗口　✅ 已完成

它的做法（`src/Tartarus.Capture/WindowCapture.cs`）：

```
D3D11CreateDevice
  → CreateDirect3D11DeviceFromDXGIDevice
  → IGraphicsCaptureItemInterop::CreateForWindow(hwnd)   ← 关键：只抓这个窗口自身内容
  → Direct3D11CaptureFramePool.CreateFreeThreaded(...)
  → framePool.CreateCaptureSession(item).StartCapture()
```

**为什么能解决问题**：`CreateForWindow` 抓的是**窗口自身的合成表面**，不是屏幕区域。
别的窗口盖在上面**不会**出现在帧里。这就是我们的死穴——`PrintWindow` 对 DirectX 窗口返回空，退回 `mss` 截屏就截到自己。

已装好 `windows-capture`（`.venv` 里，支持 `window_hwnd=` 参数，底层就是同一套 WinRT API）。接进 `capture.py` 作为被遮挡时的兜底后端即可。

#### 落地情况（`huahua_assistant/capture.py`）

| 项 | 实现 |
|---|---|
| 后端 | `WindowsGraphicsCaptureBackend`（`windows-capture` 2.0.1，`window_hwnd=` 走同一套 WinRT API） |
| 会话 | **按 hwnd 持久化**，不每次新建。反复建/销会漏 D3D 设备（它 0.7.40 才修这个） |
| 取帧 | 回调里只在「有人要」时拷贝，避免 60fps × 6MB 的无谓内存带宽 |
| 新鲜度 | 记的是**拷贝时刻**而不是窗口呈现时刻——否则持续渲染的游戏会让旧图看起来是新的 |
| 冷启动 | `prewarm()` 提前建会话，把 300ms 的 Direct3D 初始化挪出第一帧 |
| 顺序 | 内容抓取**排在激活窗口之前**，手动识别不再抢游戏焦点 |
| 边界 | 改用 DWM 的 `DWMWA_EXTENDED_FRAME_BOUNDS`，去掉 `GetWindowRect` 那圈 9px 隐形边框，帧尺寸与窗口尺寸严格一致 |
| 降级 | WGC 失败 → PrintWindow → 截屏，逐级退回；缺模块只报一次，不反复重试 |
| 遮挡 | `window_content_capture_ready()` 一旦为真，`_app_overlaps_game_window()` 直接返回 False，自动跟随不再因为工具窗口挡着而暂停 |
| 释放 | `close_capture_backends()` 挂在窗口关闭里，会话持有的 D3D 设备不会跨进程残留 |

**实测**（本机，用浏览器/工具窗口做对照，游戏未开）：

- 遮挡前后**像素差 0.00**：微信盖在上面，抓到的仍是目标窗口自己的画面；
- 帧尺寸 = 窗口尺寸（1920×1020），不再有 9px 偏差；
- 连续抓取 4~11ms，冷启动 300ms（已用 `prewarm` 挪走）；
- 动态窗口测试：12 次抓取里有 5 次以上内容确实变了，确认不是一直返回旧图。

诊断脚本：`.venv\Scripts\python.exe _wgc_probe.py`，看 `source` 里有没有 `method:wgc`。

### P0-2　读游戏日志（**本次最大的发现**）　✅ 已完成

它读 `<安装目录>\UE_game\TorchLight\Saved\Logs\UE_game.log`，150ms 增量轮询，靠日志里的 `-click` / 音频事件判断"界面要变了"，触发截图。

**我确认了：我们的游戏装在同一位置，日志存在且完整。**

```
H:\TapTap\PC Games\TorchlightInfinite\UE_game\TorchLight\Saved\Logs\UE_game.log
（6.3 MB，最后写入 2026-11-04 23:14）
```

里面直接有画中雪原的全套阶段事件（`HZXY` = 画中雪原）：

| 日志事件 | 含义 | 出现次数（单局） |
|---|---|---|
| `S8_HZXY_DrawPaper_Pigment_Puzzle` | 一次颜料落子 | 6（间隔 ~2.2s） |
| `S8_HZXY_DrawPaper_Brushwork` / `_Click` | 笔锋面板 / 选择笔锋 | 8 / 8 |
| `S8_HZXY_DrawPaper_Draw_Finish` | 作画完成 | 2 |
| `S8_HZXY_DrawPaper_Into` | 入画 | 2 |
| `S8_HZXY_DrawPaper_TearUp` | 撕开 | 2 |
| `S8_HZXY_DrawPaper_Select` | 选择 | 2 |
| `S8_HZXY_DrawPaper_Unfold` | 展开 | 2 |
| `1782_S8Paint Anim_Shuzi Start/End` | 数字滚动动画（结算） | — |
| `PageViewSetUIScale PageScale = 0.809` | **UI 缩放** | — |

时间戳精确到毫秒，形如 `[2026.11.04-14.26.47:122]`（UTC）。

**这一条能一次解决三个问题**：

1. **"自动识别不触发"** → 不再盲轮询。日志一出现 `Pigment_Puzzle` 或 `Brushwork`，说明画面刚变，立刻抓一帧。事件驱动，不会漏，也不会空转。
2. **"速度起不来"** → 现在是无脑定时抓 + 全量识别；改成"只在事件后抓"，抓的次数从每秒 N 次降到每局 ~30 次，且每次都是有效帧。
3. **"布局对不上 / 缩放不同就崩"** → `PageScale` 直接读到，布局按缩放算，不用再猜候选框。

而且这条**完全不依赖视觉**，可以立刻做，风险极低。

#### 落地情况（`huahua_assistant/game_log.py`）

| 项 | 实现 |
|---|---|
| 定位 | 配置 `game_log.path` → 运行中游戏进程的 exe 往上找 → 缓存 `runtime/game_log.json` |
| 读取 | 150ms 增量轮询，**接上时直接跳到文件末尾**，不解析几 MB 历史 |
| 半行 | 按 `\n` 切，最后一段留到下一轮补齐（日志是边写边刷的） |
| 轮转 | 文件变短或 inode 变了 → 从头重读（游戏重启会换一个新文件） |
| 去重 | 跳过 `requested row` 行：每个事件都会配一条音频查表失败的回声，不跳就会数成两次 |
| 事件 | `pigment` / `brush` / `brush_click` / `finish` / `into` / `tearup` / `select` / `unfold` / `score`；未登记的 `S8_*` 记成 `other`，仍算「画面变了」 |
| 缩放 | `PageScale` 解析进 `watcher.ui_scale`（实测 0.809） |

#### 采集节拍怎么跟着变

```
没有事件            → 1.5s 一帧（棋盘在两笔之间根本不动，盲截是纯浪费）
日志说画面变了      → 0.1s 连拍 + 等 220ms 再叫醒采集线程（避开动画中间态）
最后一次事件后 0.9s → 退回 1.5s
日志找不到 / 挂了    → 退回用户设的固定节拍，功能不降级
```

`FramePump` 为此加了 `wake()` / `set_interval()`：原来只 `stop_event.wait(remaining)`，
只能被停止唤醒；现在能提前叫醒。`set_interval` **不**主动唤醒——改节拍不代表画面变了。

诊断：`.venv\Scripts\python.exe _log_probe.py`（加 `--simulate` 可以没游戏也能验证链路）。

### P1　数据集采集 + 标注链路

照它的形状做一套最小的：

```
采集：热键存帧 → runtime/frames/YYYYMMDD-HHMMSS.png
导出：export → dataset/train/cells.csv + cells/*.png（84×96 裁剪）
标注：本地网页（tkinter 也行）逐格点选，写回 csv，带 trusted 标记
训练：tools/train/cellnet.py（照抄结构，改类别表）
```

### P1　按帧分块划分验证集

```python
# 相邻帧高度相似，必须整块同侧
frames = sorted({r["frame"] for r in rows})
blocks = [frames[i:i+8] for i in range(0, len(frames), 8)]
```

我们现在没有训练，所以还没踩这个坑；一旦开始训，这是最容易把验证精度做假的地方。

### P2　CellNet 的具体结构细节

见第二节。可以直接抄的：mean+max 双池化、多任务头 + `ignore_index=-100`、类别权重 sqrt 反比裁剪、不做色相增强、`pos_weight` 公式。

### P2　半合成稀有样本

把游戏原图里的图标（笔锋图标、颜料图标）按记录的位置和缩放贴到干净格子上，亮度随机压暗模拟弹窗遮挡。样本少的那几类靠这个补，比硬标快得多。

### P3　价值网 + 自对弈 + 配对比较

`tartarus-sim compare` 的 4000 局配对比较是个好习惯：判断一个改动好坏，不看单局体感，看配对胜率。我们 `engine/bench.py` 已经有类似的东西，可以对齐。

---

## 六、不建议照搬的

| 它的做法 | 为什么我们不适用 |
|---|---|
| determinization（门后内容每个 playout 重抽） | 它的隐藏信息是"门后是什么"，我们的是"下一条规则"，分布模型完全不同 |
| C#/WPF overlay | 我们是 Python/tkinter，重写不值 |
| 31 种审讯器具 / 17 种监狱事件的规则库 | 玩法不同，只能借架构不能借内容（MIT 可以借代码，但没意义） |
| PaddleOCR ONNX | 我们已有 `windows_ocr.py`，够用 |

---

## 七、建议的执行顺序

1. ✅ **接 WGC 抓屏** —— 直接解当前"工具窗口一盖就瘫"的死结
2. ✅ **接日志事件触发** —— 解"自动识别不触发 / 速度起不来"
3. ✅ 放宽遮挡暂停逻辑，完整识别不再截到自己（随 1 一起做掉）
4. **采帧 + 标注工具**（1~2 天）
5. 标 25 张板跑通训练链路，看验证精度（半天，训练本身十几分钟）
6. 标到 100 张板，替换现有视觉链路（1 天）
7. 再谈价值网 —— 前提是先有"下一条规则"的先验分布

前 3 步是**修 bug**，后 4 步才是**训模型**。当前卡住用户的不是模型不够好，是抓屏和触发。
