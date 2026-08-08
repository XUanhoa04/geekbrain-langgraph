from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]


def wait_for(getter, *, timeout: int = 180, desired: str = "AVAILABLE"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = getter()
        if value.get("status") == desired:
            return value
        if value.get("status") in {"FAILED", "DELETE_UNSUCCESSFUL"}:
            raise RuntimeError(json.dumps(value, default=str))
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for {desired}")


def ensure_source_bucket(s3, name: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=name)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        kwargs = {"Bucket": name}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)
    s3.put_bucket_versioning(Bucket=name, VersioningConfiguration={"Status": "Enabled"})
    s3.put_bucket_encryption(
        Bucket=name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    s3.put_public_access_block(
        Bucket=name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_lifecycle_configuration(
        Bucket=name,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "RetainRecoverableVersions",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "published/"},
                    "NoncurrentVersionTransitions": [
                        {"NoncurrentDays": 30, "StorageClass": "STANDARD_IA"}
                    ],
                    "NoncurrentVersionExpiration": {
                        "NoncurrentDays": 365,
                        "NewerNoncurrentVersions": 10,
                    },
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                }
            ]
        },
    )
    s3.put_bucket_tagging(
        Bucket=name,
        Tagging={
            "TagSet": [
                {"Key": "Project", "Value": "GeekBrainRAG"},
                {"Key": "ManagedBy", "Value": "provision_aws.py"},
            ]
        },
    )


def ensure_vector_store(client, bucket_name: str, index_name: str) -> tuple[str, str]:
    try:
        bucket = client.get_vector_bucket(vectorBucketName=bucket_name)["vectorBucket"]
    except client.exceptions.NotFoundException:
        client.create_vector_bucket(
            vectorBucketName=bucket_name,
            encryptionConfiguration={"sseType": "AES256"},
            tags={"Project": "GeekBrainRAG", "ManagedBy": "provision_aws.py"},
        )
        bucket = client.get_vector_bucket(vectorBucketName=bucket_name)["vectorBucket"]
    try:
        index = client.get_index(vectorBucketName=bucket_name, indexName=index_name)["index"]
    except client.exceptions.NotFoundException:
        client.create_index(
            vectorBucketName=bucket_name,
            indexName=index_name,
            dataType="float32",
            dimension=1024,
            distanceMetric="euclidean",
            metadataConfiguration={
                "nonFilterableMetadataKeys": ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]
            },
            tags={"Project": "GeekBrainRAG", "ManagedBy": "provision_aws.py"},
        )
        index = client.get_index(vectorBucketName=bucket_name, indexName=index_name)["index"]
    return bucket["vectorBucketArn"], index["indexArn"]


def ensure_role(iam, role_name: str, region: str, source_bucket: str, index_arn: str) -> str:
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": ACCOUNT_ID},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:{region}:{ACCOUNT_ID}:knowledge-base/*"
                    },
                },
            }
        ],
    }
    try:
        role = iam.get_role(RoleName=role_name)["Role"]
        iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(trust))
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Least-privilege Bedrock Knowledge Base role for GeekBrain S3 Vectors",
            Tags=[{"Key": "Project", "Value": "GeekBrainRAG"}],
        )["Role"]
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadGovernedSources",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{source_bucket}",
                "Condition": {"StringLike": {"s3:prefix": ["published/current/*"]}},
            },
            {
                "Sid": "ReadGovernedSourceObjects",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": f"arn:aws:s3:::{source_bucket}/published/current/*",
            },
            {
                "Sid": "InvokeEmbeddingModel",
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0",
            },
            {
                "Sid": "S3VectorIndexReadWrite",
                "Effect": "Allow",
                "Action": [
                    "s3vectors:PutVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:DeleteVectors",
                    "s3vectors:QueryVectors",
                    "s3vectors:GetIndex",
                ],
                "Resource": index_arn,
            },
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="GeekBrainKnowledgeBaseAccess",
        PolicyDocument=json.dumps(policy),
    )
    return role["Arn"]


def put_vector_bucket_policy(
    client, vector_bucket_name: str, vector_bucket_arn: str, role_arn: str, index_arn: str
) -> None:
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowOnlyGeekBrainBedrockRole",
                "Effect": "Allow",
                "Principal": {"AWS": role_arn},
                "Action": [
                    "s3vectors:PutVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:DeleteVectors",
                    "s3vectors:QueryVectors",
                    "s3vectors:GetIndex",
                ],
                "Resource": index_arn,
            }
        ],
    }
    client.put_vector_bucket_policy(
        vectorBucketArn=vector_bucket_arn,
        policy=json.dumps(policy, separators=(",", ":")),
    )


def ensure_knowledge_base(
    agent,
    name: str,
    role_arn: str,
    region: str,
    vector_bucket_arn: str,
    index_arn: str,
):
    existing = next(
        (
            item
            for item in agent.list_knowledge_bases(maxResults=100)["knowledgeBaseSummaries"]
            if item["name"] == name
        ),
        None,
    )
    if existing:
        return agent.get_knowledge_base(knowledgeBaseId=existing["knowledgeBaseId"])[
            "knowledgeBase"
        ]
    response = agent.create_knowledge_base(
        name=name,
        description="Governed GeekBrain RAG knowledge base backed by low-cost Amazon S3 Vectors",
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0",
                "embeddingModelConfiguration": {
                    "bedrockEmbeddingModelConfiguration": {
                        "dimensions": 1024,
                        "embeddingDataType": "FLOAT32",
                    }
                },
            },
        },
        storageConfiguration={
            "type": "S3_VECTORS",
            "s3VectorsConfiguration": {
                "vectorBucketArn": vector_bucket_arn,
                "indexArn": index_arn,
            },
        },
        tags={
            "Project": "GeekBrainRAG",
            "VectorStore": "S3Vectors",
            "ManagedBy": "provision_aws.py",
        },
    )
    kb_id = response["knowledgeBase"]["knowledgeBaseId"]
    return wait_for(
        lambda: agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"],
        desired="ACTIVE",
    )


def ensure_data_source(agent, kb_id: str, source_bucket: str):
    name = "geekbrain-governed-documents"
    existing = next(
        (
            item
            for item in agent.list_data_sources(knowledgeBaseId=kb_id, maxResults=100)[
                "dataSourceSummaries"
            ]
            if item["name"] == name
        ),
        None,
    )
    if existing:
        return agent.get_data_source(knowledgeBaseId=kb_id, dataSourceId=existing["dataSourceId"])[
            "dataSource"
        ]
    response = agent.create_data_source(
        knowledgeBaseId=kb_id,
        name=name,
        description="Only validated CURRENT sources under the immutable publish prefix",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{source_bucket}",
                "inclusionPrefixes": ["published/current/"],
                "bucketOwnerAccountId": ACCOUNT_ID,
            },
        },
        dataDeletionPolicy="DELETE",
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {"maxTokens": 500, "overlapPercentage": 15},
            }
        },
    )
    ds_id = response["dataSource"]["dataSourceId"]
    return wait_for(
        lambda: agent.get_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id)["dataSource"]
    )


def ensure_guardrail(bedrock, name: str) -> tuple[str, str]:
    desired_revision = "Production policy v2 - calibrated grounding thresholds"
    existing = next(
        (
            item
            for item in bedrock.list_guardrails(maxResults=100).get("guardrails", [])
            if item["name"] == name
        ),
        None,
    )
    if existing:
        versions = bedrock.list_guardrails(guardrailIdentifier=existing["id"], maxResults=100).get(
            "guardrails", []
        )
        published = [item for item in versions if str(item.get("version", "")).isdigit()]
        if published:
            latest = max(published, key=lambda item: int(item["version"]))
            if latest.get("description") == desired_revision:
                return existing["id"], str(latest["version"])
        bedrock.update_guardrail(
            guardrailIdentifier=existing["id"],
            name=name,
            description="Prompt-injection, sensitive-data and contextual-grounding controls for GeekBrain RAG",
            blockedInputMessaging="Yêu cầu đã bị chặn bởi chính sách an toàn của GeekBrain.",
            blockedOutputsMessaging="Câu trả lời không đạt ngưỡng an toàn hoặc độ bám nguồn cần thiết.",
            contentPolicyConfig={
                "filtersConfig": [
                    {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"}
                ]
                + [
                    {"type": kind, "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"}
                    for kind in ["HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT"]
                ]
            },
            sensitiveInformationPolicyConfig={
                "piiEntitiesConfig": [
                    {"type": "AWS_ACCESS_KEY", "action": "BLOCK"},
                    {"type": "AWS_SECRET_KEY", "action": "BLOCK"},
                    {"type": "PASSWORD", "action": "BLOCK"},
                ]
            },
            contextualGroundingPolicyConfig={
                "filtersConfig": [
                    {"type": "GROUNDING", "threshold": 0.6},
                    {"type": "RELEVANCE", "threshold": 0.4},
                ]
            },
        )
        version = bedrock.create_guardrail_version(
            guardrailIdentifier=existing["id"], description=desired_revision
        )["version"]
        return existing["id"], str(version)
    response = bedrock.create_guardrail(
        name=name,
        description="Prompt-injection, sensitive-data and contextual-grounding controls for GeekBrain RAG",
        blockedInputMessaging="Yêu cầu đã bị chặn bởi chính sách an toàn của GeekBrain.",
        blockedOutputsMessaging="Câu trả lời không đạt ngưỡng an toàn hoặc độ bám nguồn cần thiết.",
        contentPolicyConfig={
            "filtersConfig": [
                {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"}
            ]
            + [
                {"type": kind, "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"}
                for kind in ["HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT"]
            ]
        },
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {"type": "AWS_ACCESS_KEY", "action": "BLOCK"},
                {"type": "AWS_SECRET_KEY", "action": "BLOCK"},
                {"type": "PASSWORD", "action": "BLOCK"},
            ]
        },
        contextualGroundingPolicyConfig={
            "filtersConfig": [
                {"type": "GROUNDING", "threshold": 0.6},
                {"type": "RELEVANCE", "threshold": 0.4},
            ]
        },
        tags=[
            {"key": "Project", "value": "GeekBrainRAG"},
            {"key": "ManagedBy", "value": "provision_aws.py"},
        ],
    )
    guardrail_id = response["guardrailId"]
    version = bedrock.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description=desired_revision,
    )["version"]
    return guardrail_id, str(version)


def ensure_freshness_monitor(session, source_bucket: str, source_prefix: str) -> dict:
    iam = session.client("iam")
    lambda_client = session.client("lambda")
    events = session.client("events")
    cloudwatch = session.client("cloudwatch")
    sns = session.client("sns")
    region = session.region_name
    role_name = "GeekBrainRAGFreshnessLambdaRole"
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        role = iam.get_role(RoleName=role_name)["Role"]
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Daily source freshness monitor for GeekBrain RAG",
            Tags=[{"Key": "Project", "Value": "GeekBrainRAG"}],
        )["Role"]
    topic_arn = sns.create_topic(
        Name="geekbrain-rag-freshness-alerts",
        Tags=[{"Key": "Project", "Value": "GeekBrainRAG"}],
    )["TopicArn"]
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": f"arn:aws:logs:{region}:{ACCOUNT_ID}:log-group:/aws/lambda/geekbrain-rag-freshness:*",
            },
            {
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{source_bucket}",
                "Condition": {"StringLike": {"s3:prefix": [f"{source_prefix}*"]}},
            },
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{source_bucket}/{source_prefix}*",
            },
            {
                "Effect": "Allow",
                "Action": "cloudwatch:PutMetricData",
                "Resource": "*",
                "Condition": {"StringEquals": {"cloudwatch:namespace": "GeekBrain/RAG"}},
            },
            {"Effect": "Allow", "Action": "sns:Publish", "Resource": topic_arn},
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="GeekBrainRAGFreshnessAccess",
        PolicyDocument=json.dumps(policy),
    )
    source = (ROOT / "aws" / "lambda_freshness.py").read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("lambda_function.py", source)
    function_name = "geekbrain-rag-freshness"
    environment = {
        "Variables": {
            "SOURCE_BUCKET": source_bucket,
            "SOURCE_PREFIX": source_prefix,
            "ALERT_TOPIC_ARN": topic_arn,
        }
    }
    try:
        function = lambda_client.get_function(FunctionName=function_name)["Configuration"]
        lambda_client.update_function_code(
            FunctionName=function_name, ZipFile=buffer.getvalue(), Publish=True
        )
        lambda_client.get_waiter("function_updated_v2").wait(FunctionName=function_name)
        function = lambda_client.update_function_configuration(
            FunctionName=function_name,
            Role=role["Arn"],
            Handler="lambda_function.handler",
            Runtime="python3.12",
            Timeout=30,
            MemorySize=128,
            Environment=environment,
        )
    except lambda_client.exceptions.ResourceNotFoundException:
        time.sleep(10)
        function = lambda_client.create_function(
            FunctionName=function_name,
            Description="Daily owner/freshness audit for every governed RAG source",
            Runtime="python3.12",
            Role=role["Arn"],
            Handler="lambda_function.handler",
            Code={"ZipFile": buffer.getvalue()},
            Timeout=30,
            MemorySize=128,
            Publish=True,
            Environment=environment,
            Tags={"Project": "GeekBrainRAG", "ManagedBy": "provision_aws.py"},
        )
    rule_name = "geekbrain-rag-daily-freshness"
    rule_arn = events.put_rule(
        Name=rule_name,
        Description="Daily RAG source ownership and freshness check",
        ScheduleExpression="rate(1 day)",
        State="ENABLED",
        Tags=[{"Key": "Project", "Value": "GeekBrainRAG"}],
    )["RuleArn"]
    events.put_targets(
        Rule=rule_name, Targets=[{"Id": "FreshnessLambda", "Arn": function["FunctionArn"]}]
    )
    try:
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId="AllowDailyEventBridge",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn,
        )
    except lambda_client.exceptions.ResourceConflictException:
        pass
    alarm_name = "geekbrain-rag-stale-documents"
    cloudwatch.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription="At least one CURRENT RAG source has passed its owner review deadline",
        Namespace="GeekBrain/RAG",
        MetricName="StaleDocuments",
        Statistic="Maximum",
        Period=86400,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="breaching",
        AlarmActions=[topic_arn],
        Tags=[{"Key": "Project", "Value": "GeekBrainRAG"}],
    )
    lambda_client.get_waiter("function_active_v2").wait(FunctionName=function_name)
    lambda_client.invoke(FunctionName=function_name, InvocationType="Event", Payload=b"{}")
    return {
        "function_name": function_name,
        "rule_arn": rule_arn,
        "alarm_name": alarm_name,
        "alert_topic_arn": topic_arn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision idempotent GeekBrain RAG infrastructure"
    )
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    region = args.region
    source_bucket = f"geekbrain-rag-source-{ACCOUNT_ID}-{region}"
    vector_bucket = f"geekbrain-rag-vectors-{ACCOUNT_ID}"
    index_name = "geekbrain-rag-index"

    session = boto3.Session(region_name=region)
    s3 = session.client("s3")
    s3vectors = session.client("s3vectors")
    iam = session.client("iam")
    agent = session.client("bedrock-agent")
    bedrock = session.client("bedrock")

    ensure_source_bucket(s3, source_bucket, region)
    vector_bucket_arn, index_arn = ensure_vector_store(s3vectors, vector_bucket, index_name)
    role_arn = ensure_role(
        iam, "GeekBrainBedrockKnowledgeBaseRole", region, source_bucket, index_arn
    )
    put_vector_bucket_policy(s3vectors, vector_bucket, vector_bucket_arn, role_arn, index_arn)
    time.sleep(10)
    kb = ensure_knowledge_base(
        agent,
        "geekbrain-rag-s3-vectors",
        role_arn,
        region,
        vector_bucket_arn,
        index_arn,
    )
    ds = ensure_data_source(agent, kb["knowledgeBaseId"], source_bucket)
    guardrail_id, guardrail_version = ensure_guardrail(bedrock, "geekbrain-rag-guardrail")
    freshness = ensure_freshness_monitor(session, source_bucket, "published/current/")
    resources = {
        "region": region,
        "account_id": ACCOUNT_ID,
        "knowledge_base_id": kb["knowledgeBaseId"],
        "data_source_id": ds["dataSourceId"],
        "guardrail_id": guardrail_id,
        "guardrail_version": guardrail_version,
        "source_bucket": source_bucket,
        "source_prefix": "published/current/",
        "vector_bucket": vector_bucket,
        "vector_bucket_arn": vector_bucket_arn,
        "vector_index": index_name,
        "vector_index_arn": index_arn,
        "role_arn": role_arn,
        "freshness_monitor": freshness,
    }
    target = ROOT / "config" / "aws_resources.local.json"
    target.write_text(json.dumps(resources, indent=2), encoding="utf-8")
    print(json.dumps(resources, indent=2))


if __name__ == "__main__":
    main()
