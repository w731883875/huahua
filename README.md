# 火炬之光画画拼图助手 v0.1

这是第一版原型，目标不是直接完成全自动策略，而是先验证最关键的视觉链路：

- 优先截取标题包含 `torchlight_infinite` 的游戏窗口，找不到时回退到当前主屏幕截图。
- 自动定位左侧 6x6 棋盘。
- 切分 36 个格子。
- 粗分空白、阻挡、虚空、失色、已放置颜料和未知状态。
- 根据棋盘位置推导当前初始颜料、背包、强制完成、刷新词缀区域。
- 输出调试图，方便继续调阈值。

## 安装依赖

```powershell
.\scripts\install_deps.ps1
```

如果想手动安装，也可以使用国内镜像：

```powershell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行桌面原型

```powershell
python run_app.py
```

## 命令行截图识别

```powershell
python -m huahua_assistant capture --image runtime\capture.png --debug runtime\debug.png
```

如果游戏窗口标题不是默认值，可以指定：

```powershell
python -m huahua_assistant capture --window torchlight_infinite --image runtime\capture.png --debug runtime\debug.png
```

列出当前可见窗口，排查是否能找到游戏窗口：

```powershell
python -m huahua_assistant windows
```

如果看到 `torchlight_infinite.exe` 但 `minimized=True` 或 `captureable=False`，说明游戏窗口处于最小化/离屏状态。先把游戏恢复到前台，再运行 Capture。

强制要求只截游戏窗口，不允许回退到桌面：

```powershell
python -m huahua_assistant capture --window torchlight_infinite --no-fallback --image runtime\capture.png --debug runtime\debug.png
```

临时允许游戏不可截取时回退到整屏：

```powershell
python -m huahua_assistant capture --window torchlight_infinite --fallback-uncaptureable --image runtime\capture.png --debug runtime\debug.png
```

## 分析已有截图

```powershell
python -m huahua_assistant analyze path\to\screenshot.png --debug runtime\debug.png
```

## 本地随机模拟

不截图、不进游戏，直接生成随机 6x6 地图、随机初始颜料，并输出推荐位置：

```powershell
python -m huahua_assistant simulate --seed 42
```

桌面原型里也可以点 `Simulate` 按钮验证高亮推荐。这个模式和游戏识别模式共用同一套规则引擎，后续笔触/背包识别完成后会直接接入。

模拟模式会按真实开局结构生成：随机地图上已有一个初始颜料锚点，金色框表示初始颜料作用范围；背包会生成足够的目标颜料，并给出 `Install Plan`，按 1/2/3 步提示把哪些背包颜料放进金色范围来完成匹配。

带上随机阶段1/阶段2笔触选项，并输出笔触推荐：

```powershell
python -m huahua_assistant simulate --seed 42 --brush
```

如果有5级宠物的2次刷新词缀机会，可以打开刷新评估：

```powershell
python -m huahua_assistant simulate --seed 42 --brush --refresh
```

强制完成默认按本局还没使用计算；如果已经用过1次或2次，可以传入次数，风险会按下一次破坏数量增加：

```powershell
python -m huahua_assistant simulate --seed 42 --brush --force-used 1
```

阶段1笔触大类二选一会按经验顺序纠错：`转锋 > 堆砌 > 扩散`。例如识别到 `堆砌 + 转锋` 时，规则层会归一化为 `转锋 + 堆砌`：

```powershell
python -m huahua_assistant brush-order stack transform
```

当前落点评分按以下优先级计算：

- 最高优先级：完成匹配并把宝箱升到7级。
- 高优先级：完成匹配、把当前颜料或范围内颜料升到7级。
- 中优先级：提升宝箱、玩法/怪物高等级颜料、增加后续可连锁收益。
- 低优先级：清理阻挡/虚空/失色、增加未来可用空间。

这个优先级参考了成熟 match-3/合成类玩法的通用策略：先完成关卡目标，再制造高价值特殊棋子，最后考虑连锁与空间管理。

## 真实学习与配置

真实 Capture 会写入轻量学习记录，模拟模式不会进入学习数据。低置信度截图仍会显示和允许纠错，但不会保存为高分轨迹。

- `runtime/observations.jsonl`：真实识别事件、质量分、规则异常提醒。
- `runtime/corrected_observations.jsonl`：`Verify Capture` 保存的人工纠错。
- `runtime/high_score_trajectories.jsonl`：只自动保存高价值真实轨迹，例如出现6/7级宝箱或多个7级。
- `runtime/stats.json`：从以上原始数据重建的统计摘要。

当真实高分轨迹样本足够后，工具会把常用落点、常选大类、常选词缀作为“经验先验”轻量加到推荐分里。这个加分不会覆盖硬规则：低级宝箱覆盖高级宝箱、强制完成高风险等仍按规则扣分。

轨迹动作短码：

- `C`：放置新的初始颜料。
- `P`：从背包安装颜料。
- `F`：强制完成。
- `R1` / `R2`：一阶段/二阶段刷新词缀。
- `B`：二阶段词条选项变化。
- `T`：地形变化。
- `N`：出现新的初始颜料。

可以新建 `runtime/strategy_config.json` 覆盖默认阈值，例如：

```json
{
  "quality": {
    "min_overall": 0.58,
    "min_board": 0.48
  },
  "autosave": {
    "min_internal_score": 3000.0,
    "min_treasure6": 1,
    "min_lv7": 3
  },
  "learning": {
    "placement_min_samples": 30,
    "placement_max_bonus": 180.0
  }
}
```

桌面按钮：

- `纠错识别`：纠错本次真实识别，并把错误字段计入统计。
- `识别统计`：查看识别高风险字段、词条冲突、真实轨迹数量。
- `导出数据` / `合并数据`：导出或合并多个用户的真实学习数据。

## 打包 exe

```powershell
.\scripts\build_exe.ps1
```

打包结果在 `dist\HuahuaAssistant.exe`。

## 当前限制

- 当前只是 v0.1 视觉原型，识别规则基于截图颜色和格子纹理，后续需要用更多实际截图校准。
- 当前初始颜料面板已能切出 `runtime/current_paint_panel.png`、`runtime/current_paint_icon.png`、`runtime/current_paint_text.png`。颜料小类、等级、数量、词缀 OCR 仍需接入中文 OCR 后稳定识别。
- 第一批初始颜料关键词清单在 `templates/current_paint/keywords.txt`。后续可从真实截图裁剪关键词模板，替代系统字体模板，提高 `纵向/相邻/对角斜向/所有/提升1级/提升2级/宝箱` 的识别稳定性。
- 自适应定位先以 6x6 棋盘为核心，不依赖固定分辨率，但仍需要实际游戏截图来调优。
