# 游戏协议笔触 ID 对照试验

这是独立诊断入口，尚未接入主程序的自动记录循环。原视觉记录和历史存档不改写。

## 已实测的接收链路

2026-11-26 使用本机 Wireshark/Npcap，只采集经进程连接表确认的游戏连接。
TCP 重组后可以解析 `[type:1][length:3, big-endian][payload]`。
Type 2 可用 Wireshark 自带的 zstd DLL 解压；解压后的消息头第 14～15 字节是大端协议编号，正文从第 16 字节开始。

已在真实回包中验证 `3915 = rpc.S8DrawMapDataAck`：36 个地块编号为 1～36，背包颜料也能读取。
`3922 = rpc.S8DrawPushChangeNotify` 来自客户端 ProtoEnum，尚需在可解码的真实回包上验证。
Type 1 暂不解码；不能将它直接按 protobuf 解析或凭数值相似猜笔触 ID。

协议描述符来自本机保存的 `proto_pb_all_20260715`，属于历史版本。字段成功解析也不等于所有字段语义已验证。
尤其 `rawLevel`、`facadeLevel`、`facadeItemId` 应先原样保存，本诊断不把 `facadeLevel` 当加成相加。

## 使用方式

先从游戏进程的 TCP 连接确认当前服务器、端口和本地端口，再用 dumpcap 对这一个连接限时采集。
不能复用文档中的旧 IP/端口，也不要抓全机流量。开始诊断前启动采集，完成后停止；不发送游戏协议请求。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\probe_game_protocol.py runtime\protocol_probe\session.pcapng `
  --descriptor-dir '本机描述符目录\proto' `
  --server-port 当前游戏服务器端口 `
  --output runtime\protocol_probe\report.json
```

需要本机已有 Wireshark（默认 `C:\Program Files\Wireshark`），以及可选 Python 包 `protobuf`。
这些只由诊断命令加载，不会成为主程序启动的强制依赖。不包含自动安装行为。

输出包括抓包文件哈希、消息时间、TCP 流、原始包编号、地块、颜料唯一标识、调色效果 ID、
笔触效果 ID 和候选 ID。缺包、冲突重传、截断帧、解压或解析失败会明确报告，不用零字节填补。
同一完整状态可能被服务器返回两次；`nonzero_brush_records` 是消息内记录条数，不是独立选择次数。

## 如何认定一致

必须先证明属于同一局、同一枚颜料或同一次选择，再比较两边独立保存的 ID：

| 状态 | 含义 |
| --- | --- |
| exact | 视觉唯一 ID 与协议 ID 相同 |
| candidate_contains | 视觉有多个候选，协议 ID 在候选中；不是唯一识别成功 |
| mismatch | 同一事件已配对，但 ID 不一致 |
| unpaired | 还没有充分证据配对；仅时间接近不能认定 |
| ocr_unknown | 协议有 ID，视觉没有可靠 ID |
| protocol_no_brush | 协议状态此时没有笔触；不能拿 0 算匹配 |
| protocol_unavailable | 没有可用协议数据 |

`S8Effect.effectId` 是调色效果，`selectableEffectId` 才是笔触效果。
颜料名称、地块编号、候选内容、画面证据、消息时序可以支持配对，但不能用视觉候选反推协议真值。
主程序对照接入前还需解决持续读取 Type 1 变更或其它可靠的完整状态来源；不应让用户每次选择后退出重进。

## 本次实测结论（2026-11-26）

三段限时采集分别收到 537、775、1883 个网络包，采集器报告丢包为 0。
前两段各解析出两条开局完整回包，每条 36 个地块、31 枚背包颜料；第三段没有可解码的 S8 回包。
两条重复的开局回包不能作为两次独立选择。

三段共有 451 个尚未解码的 Type 1 帧，它们不一定全部属于 S8。
已解码状态中的非零笔触 ID 为 0，配对数量为 0，一致率为 `null`，不是 0% 或 100%。
本次证明了完整状态的接收和解析入口可行，没有证明连续选择记录已经接通。

本地证据保存在 `runtime/protocol_probe/`：原始 `live_20261126_*.pcapng`、
`report_01.json` 至 `report_03.json`，以及 `comparison_summary.json`。
这些运行数据不纳入源码，也不并入正式高分样本。

验证：协议诊断与已记录笔触复盘的针对性测试 25 项通过；完整测试集运行 1409 项，
结果为 16 项失败、6 项错误、1 项跳过。失败涉及现有视觉、规则目录、工具栏及宝箱优先级用例，
不能宣称项目全量测试通过。详细输出为 `runtime/protocol_probe/test_suite.log`。
