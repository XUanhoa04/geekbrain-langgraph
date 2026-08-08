from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geekbrain_rag.config import ROOT
from geekbrain_rag.governance import build_registry
from geekbrain_rag.operations import log_ingestion


def load_resources() -> dict:
    path = ROOT / "config" / "aws_resources.local.json"
    if not path.exists():
        raise RuntimeError("Run scripts/provision_aws.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def publish(build_dir: Path, resources: dict, *, dry_run: bool = False) -> tuple[int, int]:
    s3 = boto3.client("s3", region_name=resources["region"])
    bucket = resources["source_bucket"]
    prefix = resources["source_prefix"]
    clean_dir = build_dir / "clean"
    files = sorted(path for path in clean_dir.iterdir() if path.is_file())
    expected = {prefix + path.name for path in files}
    existing: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        existing.update(item["Key"] for item in page.get("Contents", []))
    removed = existing - expected
    if dry_run:
        return len(files), len(removed)
    for path in files:
        extra = {"ServerSideEncryption": "AES256"}
        if path.name.endswith(".metadata.json"):
            extra["ContentType"] = "application/json"
        elif path.suffix.lower() == ".md":
            extra["ContentType"] = "text/markdown; charset=utf-8"
        s3.upload_file(str(path), bucket, prefix + path.name, ExtraArgs=extra)
    if removed:
        # Scope is exact and recoverable because bucket versioning is enabled.
        for start in range(0, len(removed), 1000):
            s3.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [{"Key": key} for key in sorted(removed)[start : start + 1000]],
                    "Quiet": True,
                },
            )
    registry_key = f"manifests/source-registry-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    s3.upload_file(
        str(build_dir / "source_registry.json"),
        bucket,
        registry_key,
        ExtraArgs={"ServerSideEncryption": "AES256", "ContentType": "application/json"},
    )
    return len(files), len(removed)


def start_and_wait(resources: dict) -> dict:
    client = boto3.client("bedrock-agent", region_name=resources["region"])
    response = client.start_ingestion_job(
        knowledgeBaseId=resources["knowledge_base_id"],
        dataSourceId=resources["data_source_id"],
        description=f"Governed publish {datetime.now(UTC).isoformat()}",
    )
    job_id = response["ingestionJob"]["ingestionJobId"]
    while True:
        job = client.get_ingestion_job(
            knowledgeBaseId=resources["knowledge_base_id"],
            dataSourceId=resources["data_source_id"],
            ingestionJobId=job_id,
        )["ingestionJob"]
        print(f"Ingestion {job_id}: {job['status']}")
        if job["status"] in {"COMPLETE", "FAILED", "STOPPED"}:
            return job
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="source -> clean -> chunk -> index -> publish")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data_package" / "knowledge_base")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build" / "knowledge_base")
    parser.add_argument(
        "--allow-stale", action="store_true", help="Publish stale sources with is_stale metadata"
    )
    parser.add_argument("--include-archived", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()
    resources = load_resources()
    records = build_registry(
        args.source_dir,
        args.build_dir,
        include_archived=args.include_archived,
        allow_stale=args.allow_stale,
    )
    uploaded, removed = publish(args.build_dir, resources, dry_run=args.dry_run)
    stale = sum(record.is_stale for record in records)
    print(
        f"Validated {len(records)} sources; publish files={uploaded}; removed={removed}; stale={stale}"
    )
    if not args.dry_run and not args.no_sync:
        job = start_and_wait(resources)
        print(json.dumps(job, indent=2, default=str))
        log_ingestion(
            ROOT / "rag_ops.db",
            {
                "started_at": str(job.get("startedAt", "")),
                "completed_at": str(job.get("updatedAt", "")),
                "status": job["status"],
                "knowledge_base_id": resources["knowledge_base_id"],
                "data_source_id": resources["data_source_id"],
                "ingestion_job_id": job["ingestionJobId"],
                "document_count": len(records),
                "stale_count": stale,
                "details_json": job.get("statistics", {}),
            },
        )
        if job["status"] != "COMPLETE":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
