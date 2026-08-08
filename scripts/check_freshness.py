from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geekbrain_rag.governance import build_record, load_policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail when governed sources are stale or near review due"
    )
    parser.add_argument("--warning-days", type=int, default=30)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    source_dir = ROOT / "data_package" / "knowledge_base"
    policy = load_policy()
    today = datetime.now(UTC).date()
    warning_date = today + timedelta(days=args.warning_days)
    stale, warning = [], []
    for path in sorted(source_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in policy["allowed_extensions"]:
            continue
        record, _ = build_record(path, policy, today=today)
        if record.status != "CURRENT":
            continue
        expiry = datetime.fromisoformat(record.expires_at).date()
        if expiry < today:
            stale.append(record)
        elif expiry <= warning_date:
            warning.append(record)
    for record in stale:
        print(f"STALE {record.path}: owner={record.owner}; expired={record.expires_at}")
    for record in warning:
        print(f"DUE_SOON {record.path}: owner={record.owner}; expires={record.expires_at}")
    print(f"Freshness summary: stale={len(stale)}, due_soon={len(warning)}")
    if stale or (warning and args.fail_on_warning):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
