# 解包资源里的「加成归属」证据（2026-11-23）

用户 2026-11-23 的要求：**把 buff 从「地块加成」改成「颜料加成」**，改之前先去解包资源里
看游戏自己是怎么做的。

本文只记**从客户端资源里直接读到的事实**，不掺推断。推断与结论另起一节，并且明确标注。

数据来源（全部为本机已解密的客户端文件，不是截图、不是猜测）：

| 来源 | 路径 |
|---|---|
| Lua 字节码（已解密） | `F:\tools\reverse-toolkit\luac_decrypted\Module\S8GamePlay\` |
| 协议枚举 | `...\luac_decrypted\Plugin\PBC\ProtoEnum.lua.bin` |
| 完整本地化表 | `F:\tools\reverse-toolkit\gamecfg\out\names.json` |
| 反编译常量池转储 | `F:\tools\huahua\.workbuddy\refs\_drawmodel.txt` |

`_drawmodel.txt` 里的操作数（B/C 列）被脱壳器错位了，**不能逐条读指令语义**；
但**常量池的字段名和行号是可靠的**，下面所有字段名都来自常量池，并做了交叉验证。

---

## 一、地块（Plot）的数据结构

`S8DrawModel.DealServerPlot`（`sub#23`，L1038-L1065）逐字段读出来的地块结构：

| 字段 | 含义 |
|---|---|
| `FinalType` | 最终地块类型 |
| `FinalItemId` | 最终颜料物品 ID |
| `RawLevel` | **颜料自身**的等级 |
| `FacadeLevel` | **显示**等级（含全部加成） |
| `MaxPlotLevel` | 等级上限 |
| `OriItemList` | **参与这一格的全部来源清单** |
| `EffFroms` / `effectFroms` | 效果来源**类型**列表 |
| `facadeItemId` | 显示用物品 ID |
| `facadePlotType` / `rawPlotType` | 显示 / 原始画型 |

L1044 出现封顶运算：`math.min(DrawData.MaxPlotLevel, facadeLevel)`。

**三种等级并存**是这里最关键的事实：`RawLevel`（自身）≠ `FacadeLevel`（显示）。

## 二、`OriItemList` 每一项的字段

`S8DrawModel.DealServerItem`（`sub#21`，L936-L956）——这就是「一条来源记录」的构造：

| 字段 | 含义 |
|---|---|
| `UniqueId` | **颜料实例唯一 ID** |
| `rawLevel` | 该来源颜料自身的等级 |
| `itemId` | 物品模板 ID |
| `installSort` | 安装顺序 |
| `Task` | 任务关联 |
| `FinalItemLevel` | 该项的最终等级 |

注意 `UniqueId` —— 是**实例**级 ID，不是物品模板 ID。同一个「怪物颜料-幽灵 Lv3」
在背包里有两枚，它们的 `UniqueId` 不同。

`S8DrawInstallItem` 协议（`S8GamePlayMgr` L638）发送的正是 `itemUniqueId`，
说明服务端就是按实例来追踪颜料的。

## 三、加成来源的类型枚举

`ProtoEnum.lua.bin` 中 `S8EffectFromType` 的完整成员：

```
S8EffectFromType_None
S8_Item_Effect        ← 颜料（物品）效果
S8_Selectale_Effect   ← 可选效果（即笔触词缀）
S8_Revive_Effect      ← 复活效果
```

地块类型 `S8PlotType`：

```
S8PlotType_None
S8_Free           空地
S8_Nihility       虚空
S8_Ruins          废墟
S8_Internal_Block 内部阻挡
S8_Border_Block   边界阻挡
```

**加成的来源被明确分类成「颜料效果 / 可选效果 / 复活效果」三种**，
而不是「这一格攒了多少级」。

## 四、点击地块时的提示框内容

`S8DrawModel.GetPlotTipsDataByIndex`（`sub#35`，L1831-L1900）用到的本地化键，
去 `names.json` 里查到的真实文案：

| 本地化键 | 文案 |
|---|---|
| `Func_S8Gameplay_UI_Tips3` | **调色目标** |
| `Func_S8Gameplay_UI_Tips4` | **调色效果** |
| `Func_S8Gameplay_UI_Tips6` | **笔触效果** |
| `Func_S8Gameplay_UI_Tips1` | 等级 |
| `Func_S8Gameplay_UI_Tips2` | 等级加成 |
| `Func_S8Gameplay_UI_Tips5` | 颜料内容 |

**注意：`Tips1`/`Tips2` 在 `S8DrawModel.lua.bin` 的字符串表里都没有出现**，
只有 `Tips3/4/5/6` 被引用。也就是说 S8 的地块提示框**不显示单独的「等级加成 +N」数字**，
而是：

```lua
L1869  FacadeLevel                        -- 取显示等级
L1873  'LV' .. Level                      -- 拼 "LV{等级}"
L1878  '+'                                -- 拼加号
L1881  for _, sub in ipairs(plot.OriItemList) do
L1882      sub.EffectDesc , '\n\n'        -- 每一条来源的效果说明，空行分隔
```

即：**显示 `LV{最终等级}`，然后把 `OriItemList` 里每一条来源的效果说明逐行列出来。**

这正好解释用户口述的现象「点击地块看到 `相邻的玩法颜料提升1级` ×2 +
`对角斜向的玩法颜料提升1级`」——那几行就是 `OriItemList` 的各项。

## 五、棋盘上每枚颜料的加成标记

`S8Paint.lua.bin`（棋盘颜料 widget）的字段全集里，加成是**独立的一套 widget**：

```
SetLevel / Level        等级
SetIsBonus / bIsBonus   是否处于加成状态
BonusIcon               加成图标
Bonus                   加成显示
Cai_Glow                彩光
Punish                  惩罚
```

`S8SnowField.lua.bin` 里也有对应的 `SetIsBonus`、`PlayBonusAni`（播放加成动画）、
`SetbOnEffect`。

## 六、服务端下发的字段名（游戏管理器侧）

`S8GamePlayMgr.lua.bin` 的 `S8ChangedField` 分支里出现的字段：

```
BonusPlotList     （大写，服务器字段）
bonusPlotList     （小写，客户端字段）
bonusItemId       （S8TileCollect / S8CurrentState 里都有）
plotItemId
allPlotNum / allFreePlotNum / fixedPlotNum / donePlotNum
levelPlots
```

`bonusItemId` 同时出现在 `S8CurrentState`、`S8TileCollect` 两个结构里 —— 
**「加成来自哪个物品」是随地块收集数据一起下发的。**

---

## 七、结论

### 可以直接确认的（资源里读到的）

1. 游戏**按 `RawLevel`（颜料自身等级）** 存底数，另有 `FacadeLevel`（显示值），
   二者之间隔着「全部来源的加成」，并受 `MaxPlotLevel` 封顶。
2. 一格的加成来源是一张**清单**（`OriItemList`），每一条指向一个**颜料实例**
   （`UniqueId`），并各自带自己的 `rawLevel` 和效果说明。
3. 加成来源被分类为「颜料效果 / 可选效果 / 复活效果」`S8EffectFromType`。
4. 棋盘和提示框都是**按来源逐项展示**加成，不存在「地块自己攒了 N 级」这种字段。

### 由此得到的判断（推断，需实机验证）

工具现在把加成存成地块上的一个整数 `Tile.bonus_level`，**结构上就和客户端不一致**：
客户端没有这个字段，只有来源清单。

改成「颜料加成」在数据模型上是对的方向，但**行为差异只在一种场景下才显现**：
加成来源消失时（销毁/覆盖），清单里那一条是否被移除。

`docs/bonus_attribution.md` 记录的替换场景实测结论是「加成还在」——
这并不矛盾：替换（内容/等级被改写）时来源清单里的那一条仍然在，只是那条的
`rawLevel` 变了；真正会移除条目的是**销毁**，而这一场景至今没有实测样本。

### 尚未从资源确认的

* `OriItemList` 里一条的记录在**销毁**时是否真的被移除（需要服务端逻辑或实测）。
* `bonusItemId` 与 `OriItemList[i].UniqueId` 的对应关系（字段名对得上，但没有
  找到把两者关联起来的代码）。
* proto 的字段序号：proto 描述是运行时从二进制加载的（`Plugin/PBC/ProtoLoad`），
  本地没有 `.proto` 文本。

---

## 复现方式

```powershell
# 1. 取地块/来源结构的字段名（常量池可靠）
.\.venv\Scripts\python.exe -B -c "import re;t=open('.workbuddy/refs/_drawmodel.txt',encoding='utf-8',errors='replace').read();print(t[t.find('=== sub#21'):t.find('=== sub#22')])"

# 2. 取效果来源枚举
.\.venv\Scripts\python.exe -B -c "import re;s=[x.decode() for x in re.findall(rb'[\x20-\x7e]{3,}',open(r'F:\tools\reverse-toolkit\luac_decrypted\Plugin\PBC\ProtoEnum.lua.bin','rb').read())];i=s.index('S8EffectFromType');print(s[i:i+6])"

# 3. 取提示框文案
.\.venv\Scripts\python.exe -B -c "import json;d=json.load(open(r'F:\tools\reverse-toolkit\gamecfg\out\names.json',encoding='utf-8'));print([ (k,v) for k,v in d.items() if 'S8Gameplay_UI_Tips' in k ])"
```
