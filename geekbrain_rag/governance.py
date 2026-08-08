from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
SUSPICIOUS_INSTRUCTIONS = re.compile(
    r"(?i)(ignore\s+(all|any|the)\s+(previous|prior|system)|system\s+prompt|"
    r"reveal\s+.*(secret|credential|prompt)|<\s*(system|assistant)\s*>)"
)


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    path: str
    title: str
    owner: str
    status: str
    version: str
    reviewed_at: str
    review_interval_days: int
    expires_at: str
    checksum_sha256: str
    approval_status: str
    is_stale: bool
    sensitivity: str = "INTERNAL"
    suspicious_instruction_count: int = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title[:200],
            "owner": self.owner[:100],
            "status": self.status,
            "version": self.version[:50],
            "reviewed_at": self.reviewed_at,
            "expires_at": self.expires_at,
            "is_stale": self.is_stale,
            "sensitivity": self.sensitivity,
            "approval_status": self.approval_status,
            "checksum": self.checksum_sha256,
        }


def load_policy(path: Path | None = None) -> dict:
    policy_path = path or ROOT / "config" / "governance.yaml"
    return yaml.safe_load(policy_path.read_text(encoding="utf-8"))


def parse_frontmatter(text: str) -> tuple[dict, str]:
    match = FRONTMATTER.match(text)
    if not match:
        return {}, text
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML frontmatter: {exc}") from exc
    return metadata, text[match.end() :]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def _match_rule(filename: str, rules: list[dict], key: str, default: Any) -> Any:
    for rule in rules:
        patterns = str(rule.get("pattern", "")).split("|")
        if any(fnmatch.fnmatch(filename.lower(), pattern.lower()) for pattern in patterns):
            return rule.get(key, default)
    return default


def _as_date(value: Any, fallback: date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            pass
    return fallback


def build_record(
    path: Path, policy: dict, *, today: date | None = None
) -> tuple[SourceRecord, str]:
    raw = path.read_text(encoding="utf-8-sig")
    frontmatter, body = parse_frontmatter(raw)
    cleaned = clean_text(body)
    today = today or datetime.now(UTC).date()
    fallback_review = datetime.fromtimestamp(path.stat().st_mtime, UTC).date()
    filename = path.name
    owner = str(
        frontmatter.get("owner")
        or frontmatter.get("department")
        or frontmatter.get("author")
        or _match_rule(
            filename,
            policy.get("ownership_rules", []),
            "owner",
            policy["default_owner"],
        )
    )
    status = str(frontmatter.get("status", "CURRENT")).upper()
    status = {
        "ACTIVE": "CURRENT",
        "FINAL": "CURRENT",
        "PUBLISHED": "CURRENT",
        "DEPRECATED": "ARCHIVED",
    }.get(status, status)
    if "archived" in filename.lower():
        status = "ARCHIVED"
    if status not in policy["status_values"]:
        raise ValueError(f"{filename}: unsupported status {status!r}")
    reviewed = _as_date(
        frontmatter.get("reviewed_at")
        or policy.get("migration_reviewed_at")
        or frontmatter.get("last_updated")
        or frontmatter.get("date"),
        fallback_review,
    )
    interval = int(
        frontmatter.get("review_interval_days")
        or _match_rule(
            filename,
            policy.get("cadence_rules", []),
            "review_interval_days",
            policy["default_review_interval_days"],
        )
    )
    expires = reviewed + timedelta(days=interval)
    source_id = str(frontmatter.get("source_id") or path.stem.lower().replace("_", "-"))
    title = str(frontmatter.get("title") or path.stem.replace("_", " ").title())
    version = str(frontmatter.get("version") or "1.0")
    checksum = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    suspicious_count = len(SUSPICIOUS_INSTRUCTIONS.findall(cleaned))
    try:
        relative_path = path.relative_to(ROOT)
    except ValueError:
        relative_path = Path(path.name)
    record = SourceRecord(
        source_id=source_id,
        path=str(relative_path).replace("\\", "/"),
        title=title,
        owner=owner,
        status=status,
        version=version,
        reviewed_at=reviewed.isoformat(),
        review_interval_days=interval,
        expires_at=expires.isoformat(),
        checksum_sha256=checksum,
        approval_status=str(frontmatter.get("approval_status", "APPROVED")).upper(),
        is_stale=status == "CURRENT" and expires < today,
        sensitivity=str(frontmatter.get("sensitivity", "INTERNAL")).upper(),
        suspicious_instruction_count=suspicious_count,
    )
    return record, cleaned


def build_registry(
    source_dir: Path,
    build_dir: Path,
    *,
    include_archived: bool = False,
    allow_stale: bool = False,
) -> list[SourceRecord]:
    policy = load_policy()
    records: list[SourceRecord] = []
    clean_dir = build_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in policy["allowed_extensions"]:
            continue
        if path.stat().st_size > policy["max_source_bytes"]:
            errors.append(f"{path.name}: exceeds maximum source size")
            continue
        try:
            record, cleaned = build_record(path, policy)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        records.append(record)
        if record.suspicious_instruction_count:
            errors.append(
                f"{path.name}: quarantined; detected {record.suspicious_instruction_count} "
                "document-level prompt instruction(s)"
            )
            continue
        if record.status == "ARCHIVED" and not include_archived:
            continue
        if record.is_stale and not allow_stale:
            errors.append(f"{path.name}: review expired on {record.expires_at}")
            continue
        target = clean_dir / path.name
        target.write_text(cleaned, encoding="utf-8", newline="\n")
        sidecar = {"metadataAttributes": record.metadata()}
        (clean_dir / f"{path.name}.metadata.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    registry_path = build_dir / "source_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": policy["schema_version"],
                "generated_at": datetime.now(UTC).isoformat(),
                "sources": [asdict(record) for record in records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if errors:
        raise ValueError("Governance validation failed:\n- " + "\n- ".join(errors))
    return records
