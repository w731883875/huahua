"""Generate the local, read-only brush evidence review page."""
import html
import json
from datetime import datetime
from .codes import resolve_affix, BRUSH_NAMES


def write_report(store, output):
    records = sorted(store.records().values(), key=lambda r:r.get("observed_at", 0))
    report = store.audit()
    accuracy = f'{report["accuracy"]:.2%}' if report["accuracy"] is not None else "尚无样本"
    e = lambda value:html.escape(str(value))
    rows = []
    for record in records:
        if not record.get("selected") and record.get("canonical_affix_id"):
            rule = resolve_affix(record["canonical_affix_id"])
            if rule is not None:
                record["selected"] = rule.display_name
        images = []
        for sample in record.get("samples", []):
            for index, relative in enumerate(sample.get("images", [])):
                if relative:
                    path = store.root / relative
                    if path.is_file():
                        images.append(f'<figure><img src="{e(path.resolve().as_uri())}"><figcaption>卡片 {index+1}</figcaption></figure>')
        status = {'confirmed': '已确认', 'observed': '等待选择或识别', 'unconfirmed': '证据不足', 'unread': '文字未读全'}
        stage = {'effect': '第二阶段·笔触效果', 'direction': '第一阶段·笔触类别', 'unknown': '阶段待核对'}
        selected = BRUSH_NAMES.get(record.get('brush_type'), record.get('selected'))
        stamp = datetime.fromtimestamp(record.get('observed_at', 0)).strftime('%m-%d %H:%M:%S')
        rows.append(f'<article><h2>{e(stamp)} · {e(stage.get(record.get("stage"), "未知"))}</h2>'
                    f'<p>局：{e(record.get("session"))} · '
                    f'状态：{e(status.get(record.get("status"), "未知"))} · 选择：{e(selected or "未知")} · '
                    f'效果 ID：{e(record.get("canonical_affix_id") or "未确认")} · '
                    f'官方 ID：{e(record.get("brush_effect_id") or "待唯一确定")} · '
                    f'官方 ID 候选：{e(record.get("brush_effect_ids") or "未知")}</p>'
                    f'<div class="images">{"".join(images)}</div>'
                    f'<details><summary>点击、OCR 与独立核对证据</summary><pre>{e(json.dumps(record, ensure_ascii=False, indent=2))}</pre></details></article>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('<!doctype html><meta charset="utf-8"><title>笔触记录核对</title>'
                      '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;background:#f4f6f8;color:#172332}'
                      'article{background:white;padding:20px;margin:20px 0;border-radius:12px}h2{font-size:18px}'
                      '.images{display:flex;flex-wrap:wrap}figure{margin:8px}img{max-width:520px}pre{white-space:pre-wrap}</style>'
                      '<h1>笔触记录核对</h1><p>第二阶段确认需要选卡点击或连续亮框、确认点击及画面切换；'
                      '第一阶段还可由随后完整的第二阶段卡片核实类别。选中未确认时结果保持空值。'
                      'OCR 相似分不代表准确率；未知和漏记必须纳入人工核对。官方 ID 可能有多个候选。</p>'
                      f'<p>图片空间：{store.image_bytes/1024/1024:.1f} / {store.max_bytes/1024/1024:.0f} MiB。'
                      f'当前模式：{"仅保存 ID" if store.mode == "id_only" else "保存 ID，并附局部图供核对"}。'
                      '到达上限后继续记录文字与 ID，停止新增图片。</p>'
                      f'<p>人工核对 {report["reviewed"]} 次，正确 {report["correct"]} 次；'
                      f'准确率：{accuracy}。'
                      f'仅存 ID 条件：{"已满足" if report["eligible"] else "尚未满足"}。</p>'
                      f'<details><summary>核对样本与准确率门槛</summary><pre>{e(json.dumps(report, ensure_ascii=False, indent=2))}</pre></details>'
                      + ''.join(rows), encoding='utf-8')
    return report
