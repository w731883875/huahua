"""Read-only replay of saved capture snapshots, with provenance and gaps.

Run: .venv/Scripts/python.exe scripts/replay_recent_runs.py --count 5
No gameplay simulation, OCR correction, or writes to source archives occur.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huahua_assistant.learning import expand_trajectory_payload
from huahua_assistant.line_capacity import analyze_lines
from huahua_assistant.replay_effects import scope_comparison, selection_evidence
from scripts.replay_auto_run import _brush_event_line, split_runs, verify

LOCAL = timezone(timedelta(hours=8))
STATES = {'B': '空白', 'K': '阻挡', 'V': '虚空', 'F': '失色', 'P': '颜料', '?': '未知'}
CATEGORIES = {'monster': '怪物', 'gameplay': '玩法'}
FIELDS = ('c', 's', 'l', 't', 'b', 'au', 'ra', 'ral')
ACTIONS = {'new_current_paint': '换初始颜料', 'install_paint': '安装颜料',
           'place_current_paint': '放置初始颜料', 'capture_update': '状态更新',
           'session_start': '开始记录', 'terrain_changed': '地形变化',
           'force_complete': '强制完成', 'choose_affix': '选择词缀'}


def clock(ts):
    return datetime.fromtimestamp(float(ts), LOCAL).strftime('%Y-%m-%d %H:%M:%S')


def read_rows(path):
    if not path.exists():
        return
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                yield number, None
                continue
            yield number, value if isinstance(value, dict) else None


def fingerprint(step):
    """Same board/paint evidence; ignore timing, recommendations and UI caches."""
    tiles = [({k: tile.get(k) for k in FIELDS} if isinstance(tile, dict) else None)
             for tile in step.get('tiles', [])]
    return json.dumps([step.get('board', {}).get('states'), tiles], sort_keys=True, ensure_ascii=False)


def paint_label(paint):
    if not paint:
        return '颜料详情缺失'
    if not paint.get('c'):
        return f"保留效果 Lv{paint.get('ral', '?')}" if paint.get('ra') else '无颜料详情'
    category = CATEGORIES.get(paint.get('c'), '类别未知')
    return f"{paint.get('s') or category+'·家族未知'} Lv{paint.get('l', '?')}"


def cell_label(char, tile):
    base = paint_label(tile) if char == 'P' else STATES.get(char, '未知')
    if tile and char != 'P' and tile.get('c'):
        base += '（颜料明细冲突：'+paint_label(tile)+'）'
    if tile and tile.get('b'):
        base += f" / 加成+{tile['b']}"
    if tile and tile.get('au'):
        base += f" / 光环{tile['au']}"
    if tile and tile.get('ra'):
        base += f" / 保留效果{tile.get('ral', '?')}级"
    return base


def cells(step):
    states = str(step.get('board', {}).get('states') or '')
    tiles = step.get('tiles') or []
    return [(states[i] if i < len(states) else '?', tiles[i] if i < len(tiles) else None)
            for i in range(36)]


def diff_steps(before, after):
    changes = []
    for i, (left, right) in enumerate(zip(cells(before), cells(after))):
        lp = {k: (left[1] or {}).get(k) for k in FIELDS}
        rp = {k: (right[1] or {}).get(k) for k in FIELDS}
        if left[0] == right[0] and lp == rp:
            continue
        detail_only = (left[0] == right[0] == 'P' and lp['l'] == rp['l'] and lp['c'] == rp['c']
                       and not lp['s'] and bool(rp['s'])
                       and all(lp[k] == rp[k] for k in ('t', 'b', 'au', 'ra', 'ral')))
        changes.append({'cell': i, 'row': i//6+1, 'col': i%6+1,
                        'before': cell_label(*left), 'after': cell_label(*right),
                        'detail_only': detail_only, 'added': left[0] != 'P' and right[0] == 'P'})
    return changes


def select_archives(path, count):
    latest, ignored, bad = {}, [], []
    for line, record in read_rows(path):
        if record is None:
            bad.append(line)
            continue
        payload = record.get('payload') or record
        if not payload.get('session') or not payload.get('finish_seen'):
            continue
        key = payload['session']
        rank = (float(record.get('ts') or 0), len(payload.get('steps') or []), line)
        if key not in latest or rank > latest[key][0]:
            latest[key] = (rank, payload, line)
    candidates = []
    for _, raw, line in latest.values():
        payload = expand_trajectory_payload(copy.deepcopy(raw))
        steps = payload.get('steps') or []
        item = {'session': payload['session'], 'archive_line': line,
                'started_at': payload.get('started_at'), 'finished_at': payload.get('finished_at'),
                'snapshot_count': len(steps)}
        if len({fingerprint(s) for s in steps}) < 2:
            ignored.append({**item, 'reason': '只有单一局面，不能展示过程'})
            continue
        candidates.append((float(payload.get('finished_at') or 0), payload, line))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:count], ignored, bad


def snapshot_issues(step):
    issues = []
    if len(step.get('board', {}).get('states') or '') != 36:
        issues.append('地形状态不是完整的 36 格')
    if len(step.get('tiles') or []) != 36:
        issues.append('颜料明细不是完整的 36 格')
    for i, (char, tile) in enumerate(cells(step)):
        if char not in STATES or char == '?':
            issues.append(f'{i//6+1}行{i%6+1}列：地形未知')
        if (char == 'P') != bool(tile and tile.get('c')):
            issues.append(f'{i//6+1}行{i%6+1}列：地形与颜料明细冲突')
    current = step.get('current') or {}
    if not (current.get('paint') or {}).get('s'):
        issues.append('初始颜料家族未识别')
    if current.get('direction') in (None, 'unknown'):
        issues.append('作用方向未识别')
    if current.get('target') in (None, 'unknown'):
        issues.append('作用目标未识别')
    quality = step.get('quality') or {}
    if not quality.get('ok'):
        issues.append('原记录未通过采集质量检查')
    return issues


def compact_event(record, first_ts):
    name = record.get('event')
    payload = record.get('payload') or {}
    label = _brush_event_line(name, payload)
    if name == 'affix_options_observed':
        label = '出现词缀候选：' + ' / '.join(payload.get('options') or [])
    if label is None and name in ('paint_installed', 'initial_paint_placed', 'paint_upgraded'):
        label = {'paint_installed': '安装颜料', 'initial_paint_placed': '放置初始颜料',
                 'paint_upgraded': '颜料升级'}[name] + f"（录制器归因，{payload.get('count', '?')} 格）"
    if label is None:
        return None
    source = payload.get('source') or ('captured_click' if name in
             ('affix_card_clicked', 'brush_stage1_card_clicked') else 'unspecified')
    clicked = source in ('captured_click', 'buffered_click')
    return {'label': label, 'kind': name, 'source': source,
            'evidence': '点击记录，卡片文字仍来自识别' if clicked else '识别或推断，非点击证据',
            'ts': record['ts'], 'offset': round(record['ts']-first_ts, 3),
            'session': record.get('session'), 'line': record.get('_line'),
            'sequence': record.get('sequence'), 'payload': payload}


def make_run(payload, archive_line, observations, recorder_records, brush_records=None):
    steps = copy.deepcopy(payload['steps'])
    start = float(payload.get('started_at') or 0)
    first_ts = start+float(steps[0].get('t') or 0)
    last_ts = start+float(steps[-1].get('t') or 0)
    session = payload['session']
    obs = observations.get(session, [])
    accepted = [x for x in obs if x.get('event') == 'quick_capture_step'
                and (x.get('payload') or {}).get('session') == session]
    contemporaneous = [x for x in recorder_records if first_ts <= float(x.get('ts') or 0) <= last_ts]
    events = [e for record in contemporaneous if (e := compact_event(record, first_ts))]
    previous = None
    seen_embedded_events = set()
    output_steps = []
    for index, step in enumerate(steps):
        ts = start+float(step.get('t') or 0)
        bundle = step.get('recording_evidence') or {}
        modern = bundle.get('brush_choices')
        if isinstance(modern, dict) and brush_records:
            # Exact recorder session + choice identity only. Late OCR is evidence
            # about captured pixels, not a new action at the time replay is built.
            session_id = bundle.get('session')
            merged = {r['choice_id']: r for r in modern.get('choices') or []}
            from huahua_assistant.brush_recording import choice_at_time
            for key, record in brush_records.items():
                if record.get('session') == session_id and float(record.get('observed_at') or 0) <= ts:
                    merged[key] = choice_at_time(record, ts)
            modern['choices'] = sorted(merged.values(), key=lambda r:r.get('observed_at', 0))
        changes = diff_steps(previous, step) if previous is not None else []
        gaps = ts-(start+float(previous.get('t') or 0)) if previous else 0
        issues = snapshot_issues(step)
        if gaps < 0:
            issues.append('存档时间倒退，保留原始记录顺序')
        if gaps > 8:
            issues.append(f'距上一快照 {gaps:.1f} 秒，期间可能包含多次操作')
        matches = [x for x in accepted if x['payload'].get('step_index') == step.get('step_index')
                   and fingerprint(x['payload']) == fingerprint(step)]
        evidence = {'archive_line': archive_line, 'archive_step': index,
                    'observation_lines': [x['_line'] for x in matches]}
        if not matches:
            issues.append('未在同局采集日志中找到相同棋盘与颜料明细，当前帧仅有轨迹存档来源')
        action = step.get('inferred_action') or {}
        embedded_events = []
        for record in (step.get('recording_evidence') or {}).get('events') or []:
            identity = (record.get('session'), record.get('sequence'))
            if identity in seen_embedded_events:
                continue
            seen_embedded_events.add(identity)
            if event := compact_event(record, first_ts):
                embedded_events.append(event)
        period_events = [e for e in events if (previous is None or e['ts'] > output_steps[-1]['ts']) and e['ts'] <= ts]
        identities = {(e['session'], e['sequence']) for e in embedded_events}
        period_events = [e for e in period_events if (e['session'], e['sequence']) not in identities]
        inferred = ACTIONS.get(action.get('type'), action.get('type') or '无动作推断')
        if previous is None:
            title = '第一张已保存局面'
        elif changes:
            added = sum(c['added'] for c in changes)
            title = f'新增 {added} 格颜料 · {len(changes)} 格读数变化' if added else f'{len(changes)} 格读数变化'
        else:
            title = '棋盘相同 · 面板或背包记录更新'
        output_steps.append({
            'index': index, 'source_index': step.get('step_index'), 'clock': clock(ts), 'ts': ts,
            'offset': round(ts-first_ts, 3), 'gap': round(gaps, 3), 'title': title,
            'board': step.get('board'), 'tiles': step.get('tiles'), 'current': step.get('current'),
            'line_capacity': analyze_lines((step.get('board') or {}).get('states') or '', step.get('tiles') or []),
            'anchor': step.get('anchor'), 'brush': step.get('brush'), 'resources': step.get('resources'),
            'pending': step.get('pending'), 'active_affix': step.get('active_affix'),
            'selection': selection_evidence(step),
            'scope_comparison': scope_comparison(previous, step, gaps),
            'observed_tiles': step.get('observed_tiles'),
            'recording_evidence': step.get('recording_evidence'),
            'backpack': step.get('backpack'),
            'backpack_visible': [{k:v for k,v in paint.items() if k in ('i','c','s','l','t','confidence')}
                                 for paint in step.get('backpack_visible') or []],
            'changes': changes, 'issues': issues, 'evidence': evidence,
            'inferred_action': {'label': inferred, 'confidence': action.get('confidence')},
            'events': embedded_events + period_events,
            'quality': {'ok': (step.get('quality') or {}).get('ok'),
                        'reason': (step.get('quality') or {}).get('reason')},
        })
        previous = step
    for i in range(1, len(steps)-1):
        for change in output_steps[i]['changes']:
            cell = change['cell']
            origin = cells(steps[i-1])[cell]
            if any(cells(steps[j])[cell] == origin and
                   output_steps[j]['ts']-output_steps[i]['ts'] <= 8
                   for j in range(i+1, min(len(steps), i+3))):
                output_steps[i]['issues'].append(f"{cell//6+1}行{cell%6+1}列短时间恢复原读数，可能是识别抖动")
    record_runs = split_runs(recorder_records)
    associated = [r for r in record_runs if any(first_ts <= x.get('ts', 0) <= last_ts for x in r)]
    chain_issues = [{'session': r[0].get('session'), 'issues': verify(r)} for r in associated]
    clicks = {(e['session'], e['sequence'] if e['sequence'] is not None else e['line'], e['kind'])
              for step in output_steps for e in step['events']
              if e['source'] in ('captured_click', 'buffered_click')
              and e['kind'] in ('affix_card_clicked', 'brush_stage1_card_clicked')}
    statistics = {
        'snapshot_count': len(steps), 'board_changes': sum(bool(x['changes']) for x in output_steps),
        'first_paints': steps[0].get('board', {}).get('paint'),
        'last_paints': steps[-1].get('board', {}).get('paint'),
        'duration': round(last_ts-first_ts, 3), 'max_gap': max(x['gap'] for x in output_steps),
        'missing_family': sum(not ((x.get('current') or {}).get('paint') or {}).get('s') for x in steps),
        'missing_rule': sum((x.get('current') or {}).get('direction') in (None,'unknown') or
                            (x.get('current') or {}).get('target') in (None,'unknown') for x in steps),
        'conflict_frames': sum(any('明细冲突' in issue for issue in x['issues']) for x in output_steps),
        'corroborated': sum(bool(x['evidence']['observation_lines']) for x in output_steps),
        'accepted_observations': len(accepted),
        'ignored_observations': sum(x.get('event') == 'quick_capture_ignored' for x in obs),
        'click_records': len(clicks),
    }
    notes = ['按存档逐帧重建；状态变化不等于一次真实操作。',
             '两份日志来自同一识别链；相互吻合只说明记录一致，不证明画面识别正确。']
    initial_count = statistics['first_paints']
    if initial_count is not None and initial_count > 1:
        notes.append(f'首帧已有 {initial_count} 枚颜料，之前的落子未存档。')
    elif initial_count == 1:
        notes.append('首帧只有 1 枚颜料，接近开局；更早的界面和点击未记录。')
    else:
        notes.append('首帧尚未记录到明确的初始颜料，不能判断是否完整覆盖开局。')
    if float(steps[0].get('t') or 0) > 8:
        notes.append(f"会话建立后 {steps[0]['t']:.1f} 秒才有首帧，不能把此前等待时间算成已还原操作。")
    if not clicks:
        notes.append('同期没有笔触卡片点击记录；旧动作归因只能作为线索。')
    if any(x['issues'] for x in chain_issues):
        notes.append('同期动作录制链存在缺口；采用此局保存的完整快照播放，不跨来源补造步骤。')
    notes.append('档案没有逐帧原图路径，不能用新识别器重新识别这些历史步骤。')
    if any((s.get('recording_evidence') or {}).get('truncated') for s in steps):
        notes.append('部分帧携带的事件超过 256 条；更早的完整事件需查看动作日志。')
    return {'session': session, 'first_clock': clock(first_ts), 'last_clock': clock(last_ts),
            'started_clock': clock(start), 'archive_line': archive_line,
            'stats': statistics, 'notes': notes, 'steps': output_steps,
            'recorder_checks': chain_issues, 'terminal': payload.get('terminal')}


def build_report(runtime, count=5):
    from huahua_assistant.brush_recording import EvidenceStore
    brush_records = EvidenceStore(runtime/'brush_evidence').records()
    archives = runtime/'high_score_trajectories.jsonl'
    selected, ignored, bad = select_archives(archives, count)
    if not selected:
        raise ValueError('没有找到已结束且保存了多个局面的轨迹')
    wanted = {p['session'] for _,p,_ in selected}
    observations = {s: [] for s in wanted}
    bad_obs = 0
    for line, record in read_rows(runtime/'observations.jsonl'):
        if record is None:
            bad_obs += 1
            continue
        if record.get('event') not in ('quick_capture_step', 'quick_capture_ignored'):
            continue
        session = (record.get('payload') or {}).get('session')
        if session in observations:
            observations[session].append({**record, '_line': line})
    recorder = [{**r, '_line': line} for line,r in read_rows(runtime/'auto_run_records.jsonl') if r is not None]
    report = {'schema': 'huahua-archive-replay-v1', 'generated_at': datetime.now(LOCAL).isoformat(),
              'runs': [make_run(p, line, observations, recorder, brush_records) for _,p,line in selected],
              'excluded_single_state': sorted(ignored, key=lambda x: x.get('finished_at') or 0, reverse=True),
              'invalid_archive_lines': bad, 'invalid_observation_lines': bad_obs,
              'sources': [{'name': name, 'bytes': (runtime/name).stat().st_size}
                          for name in ('high_score_trajectories.jsonl','observations.jsonl','auto_run_records.jsonl')]}
    return report


def render_html(report):
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf8')
    # User/game text must never terminate the inert JSON script element.
    payload = json.dumps(report, ensure_ascii=False).replace('<', '\\u003c').replace('&', '\\u0026')
    return template.replace('__REPLAY_DATA__', payload)


def render_markdown(report):
    lines = ['# 最近几局存档复盘', '',
             '可以逐帧重建已保存的棋盘、面板和可见背包；不能保证还原每次点击及两帧之间的操作顺序。', '',
             '| 第一张快照（北京时间） | 快照 | 棋盘变化帧 | 首帧颜料 | 最大间隔 | 未知初始家族 | 规则缺失 | 明细冲突帧 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        s = run['stats']
        lines.append(f"| {run['first_clock']} | {s['snapshot_count']} | {s['board_changes']} | {s['first_paints']} | {s['max_gap']:.1f}s | {s['missing_family']} | {s['missing_rule']} | {s['conflict_frames']} |")
    lines += ['', '## 首帧的分类型连线容量', '',
              '按整条横、竖、斜线的空白格 + 同类已安装颜料计数，异类/空洞不切段。类型只分怪物和玩法。',
              '下表为确定的恰好 4/5/6 格线；未知或冲突可能影响的线不计入确定条数。首帧未必是开局。', '',
              '| 首帧时间 | 怪物 4/5/6 格线 | 玩法 4/5/6 格线 | 待核对格数 |',
              '|---|---|---|---:|']
    for run in report['runs']:
        capacity = run['steps'][0]['line_capacity']
        totals = [' / '.join(str(capacity['counts'][kind]['exact'][str(n)]) for n in (4,5,6))
                  for kind in ('monster','gameplay')]
        lines.append(f"| {run['first_clock']} | {totals[0]} | {totals[1]} | {len(capacity['uncertain_cells'])} |")
    lines += ['', '所有“未知”均保留原样。近期修好的 OCR 无法回填没有逐帧原图的历史档案。', '',
              '## 具体效果、位置与等级核对', '',
              '页面显示保存到的面板效果原文、各条调色规则、笔触选择及来源、初始位置。',
              '“上一帧调色范围的等级核对”逐条列出范围内目标格的前后等级；重复效果保留，目标不混合。',
              '读数上升不等于已证实由这条笔触引起。身份不明、冲突、低置信度或长间隔显示待核对。',
              '旧档缺少具体词缀则显示未知；旧档的选择值未验证，可能含历史界面的默认候选。', '',
              '同期动作日志只按时间关联，未用它们覆盖轨迹里的棋盘。点击记录与识别推断在页面中分别标注。', '',
              '单一局面档案另列，不冒充完整对局过程。原存档不修改。', '']
    for run in report['runs']:
        lines += [f"## {run['session']}", '', *['- '+note for note in run['notes']], '',
                  f"逐帧来源匹配：{run['stats']['corroborated']}/{run['stats']['snapshot_count']}；点击记录：{run['stats']['click_records']}。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=ROOT/'runtime')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'runtime/replays_recent')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count 必须大于 0')
    report = build_report(args.runtime, args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'index.html').write_text(render_html(report), encoding='utf8')
    (args.output/'replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (args.output/'README.md').write_text(render_markdown(report), encoding='utf8')
    print(json.dumps({'output': str(args.output), 'runs': [dict(session=r['session'], **r['stats']) for r in report['runs']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
