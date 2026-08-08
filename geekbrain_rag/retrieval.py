from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
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
                    "notEquals": {"key": "status", "value": "ARCHIVED"}
                }
            else:
                vector_config["filter"] = {"equals": {"key": "status", "value": "CURRENT"}}
        response = self.client.retrieve(
            knowledgeBaseId=self.settings.bedrock_kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": vector_config},
        )
        results: list[Evidence] = []
        today = datetime.now(UTC).date().isoformat()
        for item in response.get("retrievalResults", []):
            score = float(item.get("score", 0))
            if score < self.settings.rag_min_score:
                continue
            metadata = item.get("metadata", {}) or {}
            if self.settings.rag_strict_freshness:
                expires = str(metadata.get("expires_at", "9999-12-31"))
                if metadata.get("is_stale") is True or expires < today:
                    continue
            location = item.get("location", {})
            s3_uri = location.get("s3Location", {}).get("uri", "unknown")
            filename = s3_uri.rsplit("/", 1)[-1]
            title = metadata.get("title")
            source = f"{filename} ({title})" if title and title != filename else filename
            content = item.get("content", {}).get("text", "").strip()
            if content:
                results.append(
                    Evidence("DOCUMENT", content, str(source), score=score, metadata=metadata)
                )
        return results
