"""Offline real-image checks. Output is diagnostic, never a human accuracy audit."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huahua_assistant.brush_recording import read_card, detect_cards
from huahua_assistant.codes import canonical_affix_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', default='1,.75,1.2')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root/'tests/fixtures/brush_panel_labels.json').read_text(encoding='utf-8'))
    results = []
    for case in fixture['cases']:
        original = cv2.imdecode(np.fromfile(str(root/case['image']), np.uint8), cv2.IMREAD_COLOR)
        if original is None:
            continue
        for scale in map(float, args.scales.split(',')):
            frame = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            grid, found = detect_cards(frame)
            expected = case['brush_types'] if case['stage'] == 'direction' else [canonical_affix_id(s) for s in case['affixes']]
            row = {'image': case['image'], 'scale': scale, 'grid': bool(grid), 'cards': [], 'confirm': False,
                   'expected': expected}
            if found:
                board, cards, confirm = found
                row['confirm'] = confirm is not None
                for rect in cards:
                    result, attempts = read_card(frame[rect.y:rect.bottom, rect.x:rect.right])
                    row['cards'].append({'attempts': attempts, **result})
            predicted = [r['name'] if case['stage'] == 'direction' else r['canonical_affix_id'] for r in row['cards']]
            row['correct_slots'] = sum(a == b and a is not None for a, b in zip(predicted, expected))
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
