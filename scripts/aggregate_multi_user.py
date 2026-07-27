from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huahua_assistant.multi_user import BundleValidationError, aggregate_multi_user_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate validated Huahua bundles locally without network access.")
    parser.add_argument("bundles", nargs="+", type=Path, help="Bundle or local queue JSON files.")
    parser.add_argument("--output", required=True, type=Path, help="Atomic aggregate JSON output path.")
    args = parser.parse_args()
    try:
        aggregate = aggregate_multi_user_bundles(args.bundles, output_path=args.output)
    except (BundleValidationError, OSError) as exc:
        parser.error(str(exc))
    counts = aggregate["counts"]
    print(f"Aggregate: {args.output}")
    print(
        f"bundles={counts['bundles']} installations={counts['installations']} "
        f"sessions={counts['sessions']} accepted={counts['accepted_events']} "
        f"quarantined={counts['quarantined_events']}"
    )
    print(f"SHA256: {aggregate['content_sha256']}")


if __name__ == "__main__":
    main()
