from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import boto3

from .config import Settings


@dataclass(slots=True)
class Evidence:
    kind: str
    content: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    citation_id: int = 0


def _parse_expiry(value: object) -> date | None:
    """Parse governed date-only or RFC 3339 expiry values; malformed values fail closed."""
    if isinstance(value, datetime):
        parsed_datetime = value
    elif isinstance(value, date):
        return value
    elif isinstance(value, str):
        text = value.strip()
        for date_format in ("%Y-%m-%d", "%m-%d-%Y"):
            try:
                return datetime.strptime(text, date_format).replace(tzinfo=UTC).date()
            except ValueError:
                pass
        try:
            parsed_datetime = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(tzinfo=UTC)
    return parsed_datetime.astimezone(UTC).date()


class KnowledgeBaseRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = boto3.client("bedrock-agent-runtime", region_name=settings.aws_region)

    def retrieve(
        self,
        query: str,
        *,
        include_archived: bool = False,
        include_drafts: bool = False,
        top_k: int | None = None,
    ) -> list[Evidence]:
        if not self.settings.bedrock_kb_id:
            return []
        vector_config: dict[str, Any] = {
            "numberOfResults": top_k or self.settings.rag_retrieval_top_k,
            "overrideSearchType": "SEMANTIC",
        }
        if not include_archived:
            if include_drafts:
                vector_config["filter"] = {
                    "orAll": [
                        {"equals": {"key": "status", "value": "CURRENT"}},
                        {"equals": {"key": "status", "value": "DRAFT"}},
                    ]
                }
            else:
                vector_config["filter"] = {"equals": {"key": "status", "value": "CURRENT"}}
        response = self.client.retrieve(
            knowledgeBaseId=self.settings.bedrock_kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": vector_config},
        )
        results: list[Evidence] = []
        today = datetime.now(UTC).date()
        allowed_statuses = {"CURRENT"}
        if include_drafts:
            allowed_statuses.add("DRAFT")
        if include_archived:
            allowed_statuses.add("ARCHIVED")
        for item in response.get("retrievalResults", []):
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or not 0 <= score <= 1:
                continue
            if score < self.settings.rag_min_score:
                continue
            metadata = item.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                continue
            status = str(metadata.get("status", "")).strip().upper()
            if status not in allowed_statuses:
                continue
            if self.settings.rag_strict_freshness:
                expires = _parse_expiry(metadata.get("expires_at"))
                stale = metadata.get("is_stale")
                explicitly_stale = stale is True or (
                    isinstance(stale, str) and stale.strip().lower() in {"true", "1", "yes"}
                )
                if explicitly_stale or expires is None or expires < today:
                    continue
            location = item.get("location", {})
            if not isinstance(location, dict):
                continue
            s3_location = location.get("s3Location", {})
            if not isinstance(s3_location, dict):
                continue
            s3_uri = str(s3_location.get("uri", "unknown"))
            filename = s3_uri.rsplit("/", 1)[-1]
            title = metadata.get("title")
            source = f"{filename} ({title})" if title and title != filename else filename
            content_payload = item.get("content", {})
            if not isinstance(content_payload, dict):
                continue
            content = str(content_payload.get("text", "")).strip()
            if content:
                results.append(
                    Evidence("DOCUMENT", content, str(source), score=score, metadata=metadata)
                )
        return results
