# S8 画中雪原 · pak 数据核实报告（2026-11-11）

本文只记录**从客户端 pak 常量池读出来的事实**，以及明确标注为「未取到」的项。
方法：`lua54_undump.py` 解出 Lua 5.4 字节码 → 只做常量池静态分析（游戏把 opcode 表
打乱了，每个 proto 末条指令恒为 opcode 0，无法执行/反汇编）。

数据源：`F:\tools\reverse-toolkit\luac_decrypted\Config\LuaTable\DataCfg\config_s8_gameplay_*.lua.bin`

---

## 1. 权重串归属规则（本次修正）

常量池里一条记录固定长成：

```
<int:Id>  <权重串>  <S8_Gameplay_<表>|<字段>|Id>
```

所以权重归属于**它前面那个 int**，并用紧跟其后的锚点 ID 做校验。
早先按「最近见过的锚点」归属会整体偏到上一条记录（表现为 1006→1007、4005→4004），**是错的**。

工具：`.workbuddy\tmp\s8_probe_weights.py`（输出 `runtime\s8_weights_scan.json`）。
`confirm` 字段为 `int+anchor` 表示 int 与锚点双重确认。

---

## 2. 词缀几率权重（✅ 已核实，与官方文案一致）

扫描六张表，**只有 `selectableeffect` 存在权重串，且只有 3 条**：

| 记录 | 权重串 | 合计 | 归属依据 | 语义 |
| --- | --- | --- | --- | --- |
| `1006` | `1:1000;2:1000;3:100` | 2100 | int+anchor | 作用对象改为另一类型 → 47.62 % / 47.62 % / 4.76 % |
| `2019` | `1:1000;2:1000;3:1000;4:1000;5:100` | 4100 | int+anchor | 随机增加一种作用范围 → 各 24.39 %，全图 2.44 % |
| `4005` | `1:700;2:1800;3:2000;4:3000;5:2000;6:400;7:100` | 10000 | int+anchor | 随机转变自身等级 → 7/18/20/30/20/4/1 %，期望 3.54 |

> `4005` 合计正好 10000（万分比），是「这串就是概率权重」的强证据。
> 其余 56 条 selectableeffect 记录**没有**权重串。

---

## 3. 颜料出现几率（✅ 结论：不存在权重抽签）

`config_s8_gameplay_plotitem.lua.bin`：**168 条记录，零条权重串**。

字段全集（锚点字段 `ItemName` / `ItemGameplay_Des` / `collect_des` + 普通字段）：

```
Id, ItemType, ItemLevel, ItemSeriesId, ItemQuality,
S8MapRes, Icon, EffectLibraryId, ItemGameplay
```

**没有 `Weight` 字段。** 也就是说客户端里没有任何「颜料按权重抽」的表。

记录 ID = 家族 4 位 + 等级 2 位，等级档位恒为 `01..07`（7 档，与 `4005` 的 7 档对应）：

| 家族 | 等级档位 | 说明 |
| --- | --- | --- |
| `1001,1002,1003,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014` | 01–07 全 | 怪物颜料（图标 `ColorMonster*`） |
| `2001,2003,2004,2005,2006,2007` | 01–07 全 | 玩法颜料（图标 `ColorPlay*`） |
| `2008,2009,2010,2011,2012,2013,2014` | **03–07** | 玩法颜料，**没有 Lv1/Lv2** |

合计 13×7 + 6×7 + 7×5 = 91 + 42 + 35 = **168** ✅

> **对项目的直接影响**：`_expected_randomized_paint_value` 里的 `treasure_chance = 0.03`
> 在 pak 里找不到任何依据，应当保留「本项目自设参数」的标注，**不能当成官方几率**。
> 真正的出现几率由服务端下发的棋盘决定，客户端只有「颜料池 + 等级上限」。

---

## 4. 地块 buff 规则（✅ 结论：服务端下发，客户端无配置表）

符号索引 `.workbuddy\refs\_bonus_hits.txt` 佐证：

| 符号 | 命中数 | 位置 |
| --- | --- | --- |
| `TileBonus` / `CellBonus` / `PlotBonus` / `SnowBonus` / `AuraLevel` / `AuraAmount` | **0** | — |
| `IsBonus` | 4 | `S8DrawModel.lua.bin`、`S8Paint.lua.bin`、`S8SnowField.lua.bin`、`CommandUtils.lua.bin` |
| `SetIsBonus` | 2 | `S8Paint.lua.bin`、`S8SnowField.lua.bin` |
| `BonusGrid` | 2 | `Support\Net\PushMgr.lua.bin`、`Plugin\PBC\ProtoEnum.lua.bin` |
| `BlockMin` / `BlockMax` | 各 1 | 仅 `config_s8_gameplay_difficulty.lua.bin`（且只是字段名常量） |

结论：**地块 buff 不由配置表定义**，而是随服务端推送的棋盘数据（`BonusGrid`）到达，
客户端只在视图层通过 `SetIsBonus` 打标记。因此练习模式里若要还原 buff，必须按
「服务端可能给出的形状」建模，而不能照抄某张表。

---

## 5. 地块道具（itemeffect）的 Weight

`config_s8_gameplay_itemeffect.lua.bin`：**27 条记录**。

- `Weight` 这个**字段名只出现 1 次**，全表共用；其后紧跟 `int:1000` → 是全表默认值，
  **不是逐条权重表**，也没有 `档位:权重` 串。
- 字段全集：`Id, TaskId, TaskType, TaskDirection, TaskNum, ActionType, ActionParam,
  ActionDirection, EffectType, UnlockDifficulty, Weight, LibraryId`。
- 另有 `EffectDes` / `TaskDes` 两条锚点字段。
- 27 条记录 ID：`1101-1108`（8 条）、`2001-2008` `2010-2013`（12 条）、
  `3001-3006`（6 条）、`4001`（1 条）—— 按前缀分四组。
- `UnlockDifficulty` 在常量池里紧跟 `int:-1`，即默认不限难度。

> 含义：地块道具的出现同样是**服务端决定**，客户端只有一个统一权重常量 1000。
> `UnlockDifficulty` 是逐条存在的，可用于按难度限制道具池（这是本表真正可用的信息）。

---

## 6. 难度表（difficulty，6 条）

记录 ID：`101`（无色雪纸，教学）、`3`、`4`、`5`、`6`、`7`。

✅ 已确认的数值/结构：

- `MapRule`：教学档 `1901 -> 500`、`2101 -> 80`
- `Cost`：`330001`，各档 `80 / 110 / 90 / 200 / 130`
- `Rule`：`'86' / '83' / '90' / '89'`
- `AvailableDir`：`2301`、`2501`
- 两条区间权重串（未归属到具体字段）：
  - `5:1|9;15:10|19;25:20|29;35:30|39;45:40|49;55:50|1000`
  - `5:1|9;15:10|19;25:20|29;35:30|39;45:40|49;55:50|59;65:60|69;75:70|100`

❌ **`BlockMin` / `BlockMax` 的数值不在客户端**（只在常量池里作为字段名出现，值通过
`__index` 元表取，即服务端下发）。

---

## 7. 对练习模式（任务 B 复刻）的落地口径

| 项 | 口径 | 依据 |
| --- | --- | --- |
| 颜料池 | 27 个家族 × 等级 01–07（2008–2014 只从 03 起） | §3 |
| 颜料等级分布 | 用 `4005` 官方权重 7/18/20/30/20/4/1 % | §2 |
| 词缀作用对象 | `1006` 权重 47.62/47.62/4.76 % | §2 |
| 词缀作用范围 | `2019` 权重 24.39 %×4 + 2.44 % | §2 |
| 地块 buff | 服务端形状建模，**不可照抄配置表** | §4 |
| 阻挡地块数量 | `BlockMin/BlockMax` 未知，需实机统计 | §6 |
| 宝箱出现几率 | **无官方依据**，保持 `treasure_chance` 标注为自设参数 | §3 |

### 7.1 强制调色的前置条件（实机确认，2026-11-11）

**必须先放入一枚颜料，强制调色的按钮才可用。** 只放了初始颜料（锚点）时不能强制调色。
所以"开局直接强制调色"的真实操作序列是：

1. 放初始颜料（锚点）→ 棋盘 1 枚；
2. **再落 1 枚背包颜料** → 棋盘 **2 枚**（1 枚初始 + 1 枚背包）；
3. 此时才能强制调色。

代码已按此约束实现：

- `game_model.backpack_paint_placed(state)` —— 判断锚点之外是否已有玩家放下的颜料；
- `game_model.force_complete_current_rule` —— 不满足时返回 `success=False`；
- `engine/rules.py::legal_actions` —— 不满足时**不再把 `FORCE_COMPLETE` 列为合法动作**，
  避免搜索把"点了没反应"的空操作当成一条可走的路；
- `practice.TACTIC_FORCE_ON_ISOLATED_TREASURE` —— 技巧文案与证据已改写为
  "先落 1 枚背包颜料（棋盘上一共 2 枚）再强制调色"，且只在**锚点已落下**后判定
  （开局未落锚点时作用范围未定义，判定会产生假阳性）。

---

## 8. 7 级宝箱复盘（模拟批次，2026-11-12）

对 10000 个棋盘跑满自动完成，挑出**终局棋盘上有 7 级宝箱**的局：

| 项 | 数值 |
| --- | --- |
| 总跑局 | 10000 |
| 命中 | 1750 局（17.50%） |
| 耗时 | 208 s（8 进程，约 21 ms/局） |
| 命中局分数 | 最高 4.5380，平均 2.4102，最低 1.1720 |
| 7 级宝箱个数 | 1 个：1529 局；2 个：199 局；3 个：22 局 |

产物：

- `runtime/practice_sessions/level7/level7_seed<seed>.json` —— 1750 份复盘快照
- `runtime/practice_sessions/level7/INDEX.json` —— 索引（按分数降序）
- `runtime/level7_scan_10000.json` —— 扫描汇总
- 练习库 1741 条（`source=simulation`）

### 8.1 来源必须区分（硬约束）

⚠ **以上全部是模拟复盘，不是实机复盘。**

| origin | 含义 | 能否用于实机结论 |
| --- | --- | --- |
| `simulated` | 本地按官方等级分布生成棋盘后自动完成（`simulation` / `manual` / `practice`） | ❌ 只能研究技巧本身 |
| `real` | 实机截图识别录制（`real_capture` / `game_log` / `real_run`） | ✅ 唯一可对账官方的数据 |

实现：

- `practice_library.origin_of(source)` 归类；
- 每条库记录带 `origin` 字段；
- **`record_signature` 纳入 origin** —— 否则模拟与实机用同一 seed、同一串走法时，
  后写的那条会被当成重复丢弃，模拟复盘会把真实复盘覆盖掉；
- `PracticeLibrary.split_by_origin()` + 报表分列 `[实机]` / `[模拟]`。


