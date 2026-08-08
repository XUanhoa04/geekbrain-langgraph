from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import boto3

s3 = boto3.client("s3")
cloudwatch = boto3.client("cloudwatch")
sns = boto3.client("sns")


def handler(_event, _context):
    bucket = os.environ["SOURCE_BUCKET"]
    prefix = os.environ.get("SOURCE_PREFIX", "published/current/")
    topic_arn = os.environ.get("ALERT_TOPIC_ARN", "")
    today = datetime.now(UTC).date()
    warning_date = today + timedelta(days=30)
    stale, due_soon, checked = [], [], 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(".metadata.json"):
                continue
            payload = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            metadata = payload.get("metadataAttributes", {})
            if metadata.get("status") != "CURRENT":
                continue
            checked += 1
            expiry = datetime.fromisoformat(str(metadata["expires_at"])).date()
            source = metadata.get("source_id", key)
            owner = metadata.get("owner", "unknown")
            if expiry < today:
                stale.append({"source": source, "owner": owner, "expires_at": expiry.isoformat()})
            elif expiry <= warning_date:
                due_soon.append({"source": source, "owner": owner, "expires_at": expiry.isoformat()})
    cloudwatch.put_metric_data(
        Namespace="GeekBrain/RAG",
        MetricData=[
            {"MetricName": "StaleDocuments", "Value": len(stale), "Unit": "Count"},
            {"MetricName": "DocumentsDueSoon", "Value": len(due_soon), "Unit": "Count"},
            {"MetricName": "DocumentsChecked", "Value": checked, "Unit": "Count"},
        ],
    )
    if topic_arn and (stale or due_soon):
        sns.publish(
            TopicArn=topic_arn,
            Subject=f"GeekBrain RAG freshness: {len(stale)} stale, {len(due_soon)} due soon",
            Message=json.dumps({"stale": stale, "due_soon": due_soon}, indent=2),
        )
    return {"checked": checked, "stale": len(stale), "due_soon": len(due_soon)}
