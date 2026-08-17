"""
Bonus C — Knowledge Base Sync
Upload local documents to S3, then trigger Bedrock KB re-ingestion.
Usage:
    python sync_kb.py                  # Sync all docs from knowledge_base folder
    python sync_kb.py --file FILE_PATH # Sync a single file only
"""
import argparse
import os
import time
from pathlib import Path

import boto3

# ================== CONFIG ==================
KB_ID = "WZXVNGT9JA"
DATA_SOURCE_ID = "S1CCUQSWUS"
REGION = "us-east-1"

# Đường dẫn thư mục chứa tài liệu trên máy local
LOCAL_KB_DIR = os.path.join(os.path.dirname(__file__), "data_package", "knowledge_base")

def get_s3_bucket_and_prefix(kb_id: str, data_source_id: str, region: str) -> tuple:
    """Tự động lấy S3 bucket name và prefix từ Bedrock KB config"""
    client = boto3.client('bedrock-agent', region_name=region)

    ds = client.get_data_source(
        knowledgeBaseId=kb_id,
        dataSourceId=data_source_id
    )['dataSource']

    s3_config = ds['dataSourceConfiguration']['s3Configuration']
    bucket = s3_config['bucketArn'].split(':::')[-1]
    prefix = s3_config.get('inclusionPrefixes', [''])[0] if s3_config.get('inclusionPrefixes') else ''

    return bucket, prefix

def upload_files_to_s3(bucket: str, prefix: str, files: list, region: str):
    """Upload danh sách file lên S3 bucket"""
    s3 = boto3.client('s3', region_name=region)

    uploaded = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        s3_key = f"{prefix}{filename}" if prefix else filename

        print(f"   Uploading: {filename} → s3://{bucket}/{s3_key}")
        s3.upload_file(file_path, bucket, s3_key)
        uploaded += 1

    print(f"\nUploaded {uploaded} file(s) to S3.")
    return uploaded

def start_sync(kb_id: str, data_source_id: str, region: str):
    """Trigger Bedrock KB re-ingestion và theo dõi trạng thái"""
    client = boto3.client('bedrock-agent', region_name=region)

    print("\n Starting Bedrock Knowledge Base sync...")

    response = client.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=data_source_id,
        description="Auto-sync from sync_kb.py script"
    )

    job_id = response['ingestionJob']['ingestionJobId']
    print(f"   Job ID: {job_id}")

    # Poll trạng thái
    while True:
        job_info = client.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=data_source_id,
            ingestionJobId=job_id
        )['ingestionJob']

        status = job_info['status']
        print(f"   Status: {status}")

        if status in ['COMPLETE', 'FAILED']:
            if status == 'COMPLETE':
                stats = job_info.get('statistics', {})
                print("\n Sync hoàn tất!")
                print("   BÁO CÁO:")
                print(f"   - Tổng file đã quét:   {stats.get('numberOfDocumentsScanned', 0)}")
                print(f"   - File thêm mới:        {stats.get('numberOfNewDocumentsIndexed', 0)}")
                print(f"   - File đã cập nhật:     {stats.get('numberOfModifiedDocumentsIndexed', 0)}")
                print(f"   - File bị xóa:          {stats.get('numberOfDeletedDocuments', 0)}")
                if stats.get('numberOfDocumentsFailed', 0) > 0:
                    print(f"   - ⚠️ File bị lỗi:      {stats.get('numberOfDocumentsFailed', 0)}")
            else:
                print("Sync thất bại!")
                if 'failureReasons' in job_info:
                    print(f"   Lý do: {job_info['failureReasons']}")
            break

        time.sleep(3)

def main():
    parser = argparse.ArgumentParser(description="Sync Knowledge Base: Upload local docs to S3 → Trigger Bedrock re-ingestion")
    parser.add_argument('--file', type=str, help='Sync a single file only (path to .md file)')
    parser.add_argument('--sync-only', action='store_true', help='Skip upload, only trigger re-ingestion')
    args = parser.parse_args()

    print("=" * 50)
    print("GeekBrain KB Sync Tool (Bonus C)")
    print("=" * 50)

    try:
        # Bước 0: Lấy thông tin S3 bucket từ Bedrock config
        print("\n📡 Fetching S3 bucket info from Bedrock KB config...")
        bucket, prefix = get_s3_bucket_and_prefix(KB_ID, DATA_SOURCE_ID, REGION)
        print(f"   Bucket: {bucket}")
        print(f"   Prefix: {prefix or '(root)'}")

        # Bước 1: Upload files lên S3
        if not args.sync_only:
            if args.file:
                # Upload single file
                if not os.path.exists(args.file):
                    print(f" File not found: {args.file}")
                    return
                files = [args.file]
                print("\n📁 Uploading single file...")
            else:
                # Upload all .md files from knowledge_base folder
                files = sorted(Path(LOCAL_KB_DIR).glob("*.md"))
                if not files:
                    print(f" No .md files found in: {LOCAL_KB_DIR}")
                    return
                files = [str(f) for f in files]
                print(f"\n📁 Uploading {len(files)} files from knowledge_base/...")

            upload_files_to_s3(bucket, prefix, files, REGION)
        else:
            print("\n⏭️  Skipping upload (--sync-only mode)")

        # Bước 2: Trigger re-ingestion
        start_sync(KB_ID, DATA_SOURCE_ID, REGION)

    except (boto3.exceptions.Boto3Error, OSError, ValueError) as e:
        print(f"\n Error: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"\n Unexpected error: {e}")

if __name__ == "__main__":
    main()
