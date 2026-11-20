# 火炬之光画画拼图助手 v0.1

这是第一版原型，目标不是直接完成全自动策略，而是先验证最关键的视觉链路：

- 优先截取标题包含 `torchlight_infinite` 的游戏窗口，找不到时回退到当前主屏幕截图。
  抓窗口走 Windows.Graphics.Capture（读窗口自己渲染的表面），**工具窗口盖在游戏上也不会截到自己**；
  失败才退回截屏幕。
- 自动跟随按游戏日志里的事件触发识别（`<安装目录>\UE_game\TorchLight\Saved\Logs\UE_game.log`），
  不是按固定节拍盲截图：落子/开笔锋/结算才截，空闲时 1.5 秒一帧。日志找不到就自动退回固定节拍。
  日志监听是**常驻**的，不依赖「自动跟随」勾选框：工具栏右侧一直显示「日志：已接上，等待落子」/
  「日志事件 N」，放完颜料即使没开跟随也会明确提示一次，不会出现「什么都看不出来」。
- 遮挡判断只在退回截屏幕时才成立。窗口内容抓取可用时，工具窗口盖住游戏**不会**暂停自动跟随——
  否则一暂停就停掉采集线程，而抓取会话只能由采集线程建立，会把自己锁死。
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

## 搜索式决策引擎（`huahua_assistant.engine`）

规则层收成"合法动作 / 应用 / 终局"三件套，价值函数集中可调，并带同种子配对评估。
设计与取舍说明见 **`搜索式决策引擎.md`**。

```powershell
# 单个局面的候选动作排名（束搜索）
python -m huahua_assistant.engine hint --seed 42

# 同种子配对比较：现有启发式 vs 搜索
python -m huahua_assistant.engine compare --episodes 30 --method beam --width 8
```

价值单位是"1 个 7 级宝箱"，权重集中在 `engine/value.py` 的 `EvalWeights`。
当前实测：束搜索与现有启发式打平（30 局 0 胜 0 负 30 平），瓶颈在价值函数与整局模型，
不在搜索深度，详见设计文档第 4、5 节。

## 视觉层：坐标优先

对齐 TartarusSolver 的识别思路，详见 **`视觉层对齐说明.md`**。

- `huahua_assistant/layout.py`：棋盘位置由窗口宽高换算（`LayoutRect` + 候选布局），
  不再在实时链路里写死百分比魔数；36 格坐标由棋盘矩形算出。
  实测标定图见 `runtime/board_alignment_check.png`（红线与格子边界重合）。
- `huahua_assistant/presence.py`：梯度模板判断"这块区域是不是棋盘"（184×104 Sobel 点积），
  模板在 `assets/layout/board_gradient.npy`。
- 实时链路改用「6×6 网格线强度」作为主判据（内容无关），梯度模板兜底。

实测：`_score_board_rect` 90ms → 23ms，`analyze_scene_realtime` 282ms → 179ms，
11 张实机截图评分最大变化 0.002、格子状态完全一致。

## 官方取证：地块等级与笔触结算

详见 **`解包资源对照_地块等级与笔触结算.md`**（含可复核的 Lua 常量池路径）。

- `huahua_assistant/s8_plot_effect.py`：官方 27 条「调色目标 / 调色效果」表。
  目标与效果的方向、类型始终一致，提升级数只有 1 或 2，**且包含「全图」**。
  规则生成改为只从这 27 组里抽（旧实现永远抽不到全图）。
- 「额外提升」类笔触（2004–2008、2014–2016）改为**落笔瞬间结算**：
  旧实现在选词缀时就结算，作用范围里还没有新颜料，等于白选（实测"范围内 0 个颜料"）。
  修复后同一次落笔可同时吃到调色效果与笔触效果，`Lv2 + 2 + 3 = 7`。

游戏里「等级」与「等级加成」是两套数（`Tips1` / `Tips2`，服务端下发
`RawLevel` / `FacadeLevel` / `FinalItemLevel` / `MaxPlotLevel`），
最终等级 = 等级 + 加成，按上限 7 封顶。**加成挂在「地块」上，不挂在颜料上**——
颜料被覆盖、炸毁、清空后加成仍留在原地，所以"先把一格加成叠满，再落一枚颜料"
可以直接到 7 级。已建模：

- `Tile.bonus_level` 存加成，`tile_final_level()` / `tile_effective_paint()` 算最终等级；
- 官方「额外提升 N 级」类笔触写 `bonus_level`（`AffixRule.grants_bonus`），
  普通「提升 N 级」写 `paint.level`；
- 所有清空 / 炸毁 / 覆盖路径都保留 `bonus_level`；
- 估值、`level7_counts`、候选评分全部改用最终等级。

视觉侧读的是地块上显示的数字，那本身就是最终等级，所以从截图建状态时
`level` = 最终等级、`bonus_level` = 0，不会重复计算。

## 背包识别

实机背包一屏只显示 3×5 = **15 格**，容量 **16**（面板右下角 `1/16`），所以单帧必然
有 1 枚被挤在屏外；落一枚颜料后整列前移，**第二帧才读全**。合并规则见
`huahua_assistant/backpack.py`：以最新一帧为准、按枚数保留、没落笔时把漏识别补回来。
会话键是"一条规则"而不是棋盘状态，否则落笔就会清空累计，正好抹掉第二次识别。
详见 `背包识别说明.md`。

每局背包的颜料是满足的，所以不需要补给模型：`RuleEngine(assume_supply=True)`
在推演中背包见底时按真实背包画像补一枚，跨规则推演不会卡在"没有可用颜料"。

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

## 官方规则依据（S8 画中雪原 · 作画）

规则层不再只依赖实机观察，已从客户端配置表取证：

- 完整规则与证据链：**`S8画中雪原_作画玩法_官方规则.md`**
- 机器可读官方数据：`huahua_assistant/s8_official_data.py`
  - `S8_SELECTABLE_EFFECTS`：59 条可选笔触记录（收敛为 52 种语义）
  - `S8_PLOT_ITEMS`：168 条颜料（26 个家族 × 等级）
  - `S8_DIFFICULTIES` / `S8_REVIVES` / `S8_UI_TEXT`：难度、强制调色、玩法界面原文
  - `affix_option_distribution(id)`：**词缀随机档位的官方概率**（如 `4005` 随机等级：7 级仅 1 %，期望 3.54 级）
- 词缀目录已按官方补齐：`1-n/2-n/3-n` 代码保持历史兼容，同时每条新增
  - `official_ids`：官方记录 ID（`resolve_affix("4005")` 可直接解析）
  - `official_text`：官方 `EffectDes` 原文（自动进入 `aliases`，可按游戏内原文反查）
  - `verified`：官方表中是否有出处；`False` 的 9 条为早期推断产物，详见规则文档第 4.6 节

关键纠错：

- 转锋**没有「作用方向改为全图」**；「全图」是作用范围取值（堆砌/扩散使用）。
- 全图升级类（2020/2021/2022）官方**只有提升 1 级**。
- 「与自身内容相同」的加成官方是**全图**范围，没有"作用范围内同内容"。
- 官方**没有「全图随机 1 个失色地块」**；失色相关只有「范围内 / 可放置 / 距离最近 1 个」。
- 3002/3003/3004 官方是「**最近** 1 个」而非「随机 1 个」。
- 阻挡地块数量按难度取 `[BlockMin, BlockMax]` 区间，**不是固定值**；棋盘尺寸**由服务器下发**，客户端无配置。

## 收尾模式（调色目标凑不满时）

调色目标的进度（如「纵向 · 所有颜料 · 2/4」）是判断局面能不能继续铺的硬指标。
`game_model.assess_finish_mode()` 会在两种情况下改口径：

- **范围内已经没有能落子的格**（实机常见：那条纵向线被颜料占满，只剩虚空格）→
  不再按分数推落点，直接给收尾建议：别用强制调色（点名盘上的 Lv7 在哪格）→
  把剩余颜料塞进列出的空格 → 点「完成绘制」结算。
- **调色范围没识别出来**（金框高亮没抓到、锚点也推不出）→ 只说「范围没识别出来」，
  **不**断言凑不满，避免把玩家往收尾上推。

依据：官方 `config_s8_gameplay_revive` 的强制调色是「随机摧毁地图上 1/2/3 个颜料连同地块」，
**摧毁对象不可指定**；而 `4005` 权重表抽到 7 级只有 1%（期望 3.54 级），Lv7 宝箱又是通关目标。
所以盘上只要有 Lv7，强制调色就是拿必然存在的摧毁风险去换一次凑满 —— 期望必亏。

## 换局重置（「重开一局还是上一局的棋盘」）

旧局状态原来只有**结算成功**（`real_session_finished`）才会清。但用户点「重绘」
开新局根本不走结算，于是 `previous_state` 一直被继承，调色目标、锚点、背包全
是上一局的 —— 界面上看起来就是「换局了还显示旧棋盘」。

现在改成**日志驱动**：日志里出现 `S8_HZXY_DrawPaper_Into`（入画）就置
`pending_new_run`，下一帧无条件开新局，**且放在「没找到棋盘」的提前返回之前**
——选画纸、集雪这些界面本来就看不清棋盘，若等识别成功才清，旧棋盘会一直挂着。

实测时间线（两局各一次，且都紧跟在开局）：

```
15.36.37 Draw_Finish
15.37.17 Into → 15.37.18 TearUp → 15.37.19 Select → 15.37.20 Unfold
```

修之前这一局从「入画」到工具真正开新局隔了 **106 秒**（23:37:17 → 23:39:03），
那两分钟里界面一直是旧棋盘。

顺带修掉一个背包跨局累积：累积器的会话键原来只由**调色目标**构成，新局抽到
同样的目标（比如又是「纵向·所有·2/4」）时键完全一样，上一局的颜料继续叠进来。
现在键前面拼上 `real_session_id`，一局一清。

## 游戏日志能复现棋局吗？——不能，但观测记录可以

全量核过 `UE_game.log`：

| 想找的 | 日志里有什么 | 结论 |
| --- | --- | --- |
| 棋盘布局 | `PushMgr:RecvPushCombineMsg ProtoId = 3932, SynId = N`，**只有 ID** | payload 没打印，复现不了 |
| 背包颜料 | `BagMgr@` 只记道具页 `PageId = 100/102/103` | S8 玩法颜料不在里面 |
| 颜料增减 | `ItemChange@` 只有货币 `330001/330002` | 与作画无关 |
| 界面动作 | `AudioEnum == S8_HZXY_DrawPaper_*` | 只有「点了什么」 |

`ProtoId = 3932` 是 S8 玩法的主推送（47 次），但后面跟的是
`TLNetBase: warning: MsgFromData msg is empty` —— 内容从来没落盘。

**能复现的是工具自己记的 `runtime/observations.jsonl`**。每一步
`quick_capture_step` 都带完整棋盘（`tiles` 里每格有类别/等级/是否宝箱）、
背包指纹、调色目标、推荐打分和本步的 `changed_cells`。回放：

```bash
python _replay_session.py                    # 列出所有局
python _replay_session.py --last             # 回放最后一局
python _replay_session.py run-20261105-233903
```

这也是第 4 步「采帧 + 标注」的数据源 —— 不用再额外录一遍。

## 记录编码（棋盘 / 颜料 / 词缀编号）

一局记录原始 221 KB、1000 局 221 MB。但**空间不在棋盘上**——926 步的分项实测：

| 字段 | 占比 | 说明 |
|---|---|---|
| `backpack_cumulative` | 34% | 每帧重写十几个颜料的完整 dict |
| `backpack_visible` | 34% | 和上面内容基本重复 |
| `candidates` | 11% | 含 explanation 全文 |
| `inferred_action` | 10% | 变化格明细 |
| `top` | 7% | `candidates` 的前 3 个，纯冗余 |
| **`tiles`（棋盘）** | **3%** | 换成编号最多省这么多 |

所以「给棋盘编号」收益只有 3%，真正的大头是**背包**和**候选**。`codebook.py`
就是干这个的：

- **棋盘** `B12`：`states` + 每格颜料 + 每格地块加成 + 当前调色目标，四者合起来
  才算一个局面。实测 926 步只有 283 种（重复率 69%）。
- **颜料** `sr1`：子类型首字母 + 等级，`sr1` = 鼠人 lv1。子类型没识别出来时退回
  `M1`（老的「类别+等级」写法），两套编码共存。
- **词缀** `ZF-H1`：转锋第 1 条，方向码带上，回放不用查表就知道改成了什么方向。

### 地块 buff 必须落记录

初始颜料落在某格之后，那一格会**永久**带加成（`Tile.bonus_level`），颜料被覆盖、
被炸掉都不会消失。`tiles` 里因此多了 `b`（加成）和 `au`（光环强度）；`al` 是从
光环吃到的派生等级，**不记**——它由 `_recompute_auras` 从全场 `au` 重算，存了
反而容易和真实状态打架。

### 步骤 ↔ 帧一一对应

每步带 `frame: {id, at, source}`：

- `id` = `{局号}#{步号}`，全局唯一，将来要补图就按它回查同一帧
- `at` = 精确时间戳，可以拿去录屏里按时间找
- `source` = 采集链路（窗口内容 / 屏幕 / 暂停）

现在不存图，但定位线索必须先留——否则训练数据要补图时对不上号。

### 归档

```bash
python _compact_log.py --dry-run   # 只看数字
python _compact_log.py             # 写 runtime/observations.compact.jsonl
```

实测 12.62 MB → 5.74 MB（46%）。再套 gzip 约 0.2 MB。

`_replay_session.py` 两种格式都能读：

```bash
python _replay_session.py --last              # 原始格式
python _replay_session.py --last --compact    # 紧凑格式（查 codebook 还原棋盘）
```

## 当前限制

- 当前只是 v0.1 视觉原型，识别规则基于截图颜色和格子纹理，后续需要用更多实际截图校准。
- 当前初始颜料面板已能切出 `runtime/current_paint_panel.png`、`runtime/current_paint_icon.png`、`runtime/current_paint_text.png`。颜料小类、等级、数量、词缀 OCR 仍需接入中文 OCR 后稳定识别。
- 第一批初始颜料关键词清单在 `templates/current_paint/keywords.txt`。后续可从真实截图裁剪关键词模板，替代系统字体模板，提高 `纵向/相邻/对角斜向/所有/提升1级/提升2级/宝箱` 的识别稳定性。
- 自适应定位先以 6x6 棋盘为核心，不依赖固定分辨率，但仍需要实际游戏截图来调优。
