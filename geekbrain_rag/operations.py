from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def initialize_operations_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_id TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                intents TEXT NOT NULL,
                tools_used TEXT NOT NULL,
                citation_count INTEGER NOT NULL,
                abstained INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                knowledge_base_id TEXT,
                data_source_id TEXT,
                ingestion_job_id TEXT,
                document_count INTEGER NOT NULL DEFAULT 0,
                stale_count INTEGER NOT NULL DEFAULT 0,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_id TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                rating INTEGER CHECK (rating BETWEEN 1 AND 5),
                comment TEXT
            );
            """
        )


def log_query(path: Path, payload: dict[str, Any]) -> None:
    initialize_operations_db(path)
    columns = (
        "created_at",
        "session_id",
        "query_hash",
        "intents",
        "tools_used",
        "citation_count",
        "abstained",
        "latency_ms",
        "error",
    )
    values = [payload.get(column) for column in columns]
    values[0] = values[0] or datetime.now(UTC).isoformat()
    values[3] = json.dumps(values[3] or [])
    values[4] = json.dumps(values[4] or [])
    with sqlite3.connect(path, timeout=2) as conn:
        conn.execute(
            f"INSERT INTO query_audit ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            values,
        )


def log_ingestion(path: Path, payload: dict[str, Any]) -> None:
    initialize_operations_db(path)
    columns = (
        "started_at",
        "completed_at",
        "status",
        "knowledge_base_id",
        "data_source_id",
        "ingestion_job_id",
        "document_count",
        "stale_count",
        "details_json",
    )
    values = [payload.get(column) for column in columns]
    values[0] = values[0] or datetime.now(UTC).isoformat()
    values[-1] = json.dumps(values[-1] or {}, default=str)
    with sqlite3.connect(path, timeout=2) as conn:
        conn.execute(
            f"INSERT INTO ingestion_runs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            values,
        )
