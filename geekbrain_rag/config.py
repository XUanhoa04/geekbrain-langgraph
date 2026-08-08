from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_FILE = ROOT / "config" / "aws_resources.local.json"
FALLBACK_RESOURCE_FILE = ROOT / "config" / "aws_resources.json"


def _resource_defaults() -> dict:
    path = RESOURCE_FILE if RESOURCE_FILE.exists() else FALLBACK_RESOURCE_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "aws_region": raw.get("region"),
        "bedrock_kb_id": raw.get("knowledge_base_id"),
        "bedrock_data_source_id": raw.get("data_source_id"),
        "bedrock_guardrail_id": raw.get("guardrail_id"),
        "bedrock_guardrail_version": raw.get("guardrail_version"),
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    aws_region: str = "us-east-1"
    bedrock_kb_id: str = ""
    bedrock_data_source_id: str = ""
    bedrock_guardrail_id: str = ""
    bedrock_guardrail_version: str = "DRAFT"
    bedrock_model_id: str = "amazon.nova-pro-v1:0"
    bedrock_planner_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_fallback_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    rag_retrieval_top_k: int = Field(8, ge=1, le=100)
    rag_min_score: float = Field(0.25, ge=0, le=1)
    rag_max_tool_iterations: int = Field(6, ge=1, le=12)
    rag_conversation_context_chars: int = Field(12_000, ge=2_000, le=50_000)
    rag_source_timeout_seconds: float = Field(20.0, ge=1, le=60)
    rag_strict_freshness: bool = True
    monitoring_api_url: str = "http://localhost:8000"
    analytics_db_path: Path = ROOT / "data_package" / "scripts" / "geekbrain.db"
    operations_db_path: Path = ROOT / "rag_ops.db"

    @model_validator(mode="before")
    @classmethod
    def merge_resource_file(cls, values: dict) -> dict:
        merged = {k: v for k, v in _resource_defaults().items() if v not in (None, "")}
        merged.update(values)
        return merged

    @property
    def model_id(self) -> str:
        model = self.bedrock_model_id
        if model.startswith(("us.", "eu.", "apac.")):
            return model
        if model.startswith("anthropic."):
            return f"us.{model}"
        return model

    @property
    def planner_model_id(self) -> str:
        return self.bedrock_planner_model_id

    @property
    def fallback_model_id(self) -> str:
        return self.bedrock_fallback_model_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
