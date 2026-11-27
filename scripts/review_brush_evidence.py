"""Local card evidence report and explicit human audit labels (no game control)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huahua_assistant.brush_recording import EvidenceStore, ROOT, recognizer_revision
from huahua_assistant.codes import resolve_affix
from huahua_assistant.brush_review import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--choice', help='Choice UUID, or missed-<unique> for an unrecorded real choice')
    parser.add_argument('--expected', help='Human-verified effect name or canonical ID')
    parser.add_argument('--index', type=int, choices=[1, 2], help='Human-verified card slot (1 or 2)')
    parser.add_argument('--official-ids', help='Human-verified official candidate IDs, comma separated')
    parser.add_argument('--reviewer', help='Required human reviewer name; no automatic labels')
    parser.add_argument('--not-choice', action='store_true', help='Human verified direction page/cancelled offer, not an effect choice')
    args = parser.parse_args()
    store = EvidenceStore(args.root)
    if args.choice:
        if not args.reviewer or (not args.not_choice and (not args.expected or args.index is None or not args.official_ids)):
            parser.error('label requires --expected, --index, --official-ids and --reviewer')
        record = store.records().get(args.choice)
        if record is None and not args.choice.startswith('missed-'):
            parser.error('unknown choice; use missed-<unique> for a missed real choice')
        matched = resolve_affix(args.expected) if args.expected else None
        if not args.not_choice and matched is None:
            parser.error('expected must be an exact catalog effect name or canonical ID')
        expected = matched.canonical_affix_id if matched is not None else None
        label = {'choice_id': args.choice, 'expected_canonical_id': expected,
                 'expected_index': args.index-1 if args.index else None, 'reviewer': args.reviewer,
                 'expected_kind': 'not_effect' if args.not_choice else 'effect',
                 'expected_official_ids': sorted(int(x) for x in (args.official_ids or '').split(',') if x),
                 'recognizer_revision': recognizer_revision()}
        target = store.root / 'reviews.json'
        labels = json.loads(target.read_text(encoding='utf-8')) if target.exists() else []
        labels = [r for r in labels if r['choice_id'] != args.choice] + [label]
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(target)
    output = args.output or store.root / 'review.html'
    print(json.dumps(write_report(store, output), ensure_ascii=False, indent=2))
    print(output.resolve())


if __name__ == '__main__':
    main()
