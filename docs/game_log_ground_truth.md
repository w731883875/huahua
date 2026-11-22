# 游戏日志里的 S8 操作真值

`H:\TapTap\PC Games\TorchlightInfinite\UE_game\TorchLight\Saved\Logs\UE_game.log`
里有一整套**服务端驱动**的事件行，比视觉和点击坐标硬得多。日志是 UTC，本地 = UTC+8。

## 效果动画词表（全日志只有这 5 种）

行格式：`[2026.11.22-11.45.40:452][  81]TLLua: Display: [Game] 5042_S8SnowField Anim_Fangzhi Start`

| 动画 | 含义 |
|---|---|
| `Anim_Huizhi` | 绘制（落笔）。开局一次刷 36 条（4996→5066，步长 2）= 整盘 6×6 |
| `Anim_PutongShengji` | 普通升级（+1 级） |
| `Anim_CaiShengji` | 彩升级 = 「转为同等级的颜料」那类改写 |
| `Anim_Fangzhi` | 放置 |
| `Anim_Zudang` | 阻挡（生成阻挡地块） |

前面的数字（`5042`）是**格子单位 ID**，同一个格子每次都是同一个号，可以用来定位是哪一格动了。

## 点击事件

| 行 | 含义 |
|---|---|
| `UIButton@ normal-click BtnSelect` | 点卡片（阶段一大类 **或** 阶段二词缀，两者同名） |
| `UIButton@ normal-click BtnConfirm` | 点确认 |
| `UIButton@ normal-click Btn` / `BtnClose` / `BtnReset` | 面板其它按钮 |
| `4993_S8ChooseAblityMini Anim_Show Start` | 词缀面板弹出 |
| `4905_S8EnterDraw` / `4985_S8ChoosePaper` / `4991_S8DrawMap` | 入画 / 选纸 / 绘制盘面 |

`BtnSelect` 阶段一和阶段二**同名**，单看按钮名分不出是哪一步；但配合
「`S8ChooseAblityMini` 弹出的时刻」和点击的先后顺序可以拆开：面板弹出前的那次是
阶段一，弹出后的是阶段二。

## 实测节选（2026-11-22 20:27 那局）

```
12.27.31  36 条 Anim_Huizhi ← 开局整盘绘制
12.27.40  BtnSelect          ← 点词缀（开局后 9s）
12.27.42  PutongShengji ×3   ← 效果结算
12.27.46/47/48  BtnSelect ×3 ← 连点，每秒一下
12.27.52  BtnConfirm         ← 确认 → Fangzhi + PutongShengji ×5
12.28.01/02/03  BtnSelect ×3
12.28.05  BtnConfirm         ← Fangzhi + PutongShengji ×1
12.28.18  BtnConfirm
12.28.22  CaiShengji ×3      ← 「转为同等级」类改写
```

由此得到的节奏结论：**每次操作本身 1~2 秒，慢的部分全是效果动画（3~6 秒）**。
用户 2026-11-22 问「我的速度如何（这个速度还能加快）」——按日志，用户的手速不是瓶颈。

## 另一个实测发现：`run_finished` 会误判

同一局，工具的 `run_started` 是 20:27:41（**比游戏开局晚 10 秒**），`run_finished`
在 20:28:23，`reason="no_blank_tiles"`，`duration=41.7s`。

但游戏日志显示 20:28:22 还在 `CaiShengji` 结算、20:28:18 还有一次 `BtnConfirm`，之后
仍在继续。该局 `terrain_counts` 从 `blank=18` 变成 `blank=0`（`void=11`、`blocked=6`
全程没变），于是「没有空白格」这条判据把还在进行的局判成了结束。

**判据本身有缺陷**：`void` 格在真实棋盘上也可能是可落笔的，或者空白被误读成颜料。
不能只看 `blank == 0`。