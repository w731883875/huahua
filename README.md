# 火炬之光画画拼图助手 v0.1

这是第一版原型，目标不是直接完成全自动策略，而是先验证最关键的视觉链路：

- 优先截取标题包含 `torchlight_infinite` 的游戏窗口，找不到时回退到当前主屏幕截图。
- 自动定位左侧 6x6 棋盘。
- 切分 36 个格子。
- 粗分空白、阻挡、虚空、失色、已放置颜料和未知状态。
- 根据棋盘位置推导当前初始颜料、背包、强制完成、刷新词缀区域。
- 输出调试图，方便继续调阈值。

## 编码与 PowerShell

项目文本文件统一按 `UTF-8` 保存，编辑器配置见 `.editorconfig`，Git 换行规范见 `.gitattributes`。Windows 上建议使用 PowerShell 7：

```powershell
pwsh
```

如果当前终端仍显示中文乱码，先在项目根目录加载 UTF-8 初始化脚本：

```powershell
. .\scripts\use_utf8.ps1
```

该脚本会把当前控制台切到 UTF-8 代码页，并设置 `[Console]::InputEncoding`、`[Console]::OutputEncoding` 和 `$OutputEncoding`。项目自带的 `scripts/*.ps1` 已经自动加载它。

编码体检命令：

```powershell
python scripts\check_encoding.py .
```

查看中文文件时，PowerShell 5.1 必须显式指定编码；PowerShell 7 通常不需要，但保留这个写法也安全：

```powershell
Get-Content -Encoding UTF8 .\三层高分轨迹方案完成清单.md
```

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

高分局会保存完整 `state -> action -> next_state -> final_value` 轨迹和初始地块快照；普通/失败局不会保存完整步骤，只会把可信终局压缩为 `compressed_run_summary`，记录初始地形、终局价值、内部评分和摘要类型，用来降低只看高分局造成的幸存者偏差。

`runtime/stats.json` 会额外汇总真实识别到的词条出现概率、未知/新词条候选、普通/失败局压缩摘要，以及可信高分轨迹的模仿分析。模仿分析会统计高分局里的常用落点、笔触大类、已选词条、推荐命中率和样本是否达标；样本未达标时只显示趋势，不应视为强策略。

词缀统计以稳定的 `canonical_affix_id` 聚合。界面展示名和 OCR 长短文本只作为别名；历史 `1-n`、`2-n`、`3-n` 代码继续可读。新记录保留原有 `stage2_codes`，同时写入 `stage2_ids`，因此不需要重写已有 JSONL。

当真实高分轨迹样本足够后，工具会把常用落点、常选大类、常选词缀作为“经验先验”轻量加到推荐分里。这个加分不会覆盖硬规则：低级宝箱覆盖高级宝箱、强制完成高风险等仍按规则扣分。

学习统计还会给出专家样本阶段和地形先验。阶段1以 `6级箱子/任意7级颜料` 为门槛，阶段2关注 `7级箱子/多个7级/6级箱子+多个7级`，阶段3样本足够后才适合筛选前20%专家样本。地形先验只在真实样本达到阈值后轻量影响推荐，用来判断“类似初始地形是否值得投入”，不会覆盖宝箱、7级、覆盖风险等硬规则。地形会按开放格、阻挡、虚空和中心四格可用性做粗分桶，避免全局平均把不同地图混在一起。

`识别统计` 会展示地形分桶分析，把样本分为“值得投入”“谨慎投入”“中性观察”“样本不足”。这部分只使用真实记录，用来辅助判断某类地图是否值得消耗刷新/强制资源。

实战识别后，主界面会额外显示“当前地图分桶”，直接给出当前地图所属桶、历史高分比例和投入建议。样本不足时只提示继续收集，不会影响硬规则推荐。

刷新/强制完成的建议会联动当前地图桶：历史表现偏差的地图会更保守；历史表现偏好且当前存在宝箱、6级/7级或明确冲7机会时，资源使用会略积极。这个调整只改变资源建议权重，不会自动点击或接管游戏。

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
