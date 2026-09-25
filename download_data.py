"""Загрузка исходных parquet-файлов проекта из S3 в папку data/."""
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

DATA_DIR = Path('data')
FILES = (
    'wb_products_raw_sample.parquet',
    'wb_products_dedup.parquet',
    'queries_synthetic_train.parquet',
    'queries_synthetic_test.parquet',
)


def main():
    load_dotenv()
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['S3_ENDPOINT'],
        aws_access_key_id=os.environ['S3_ACCESS_KEY'],
        aws_secret_access_key=os.environ['S3_SECRET_KEY'],
    )
    bucket = os.environ['S3_BUCKET']

    # Ищем файлы по имени, чтобы не зависеть от префикса внутри бакета
    keys = {
        Path(obj['Key']).name: obj['Key']
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket)
        for obj in page.get('Contents', [])
    }

    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        path = DATA_DIR / name
        if path.exists():
            print(f'{name}: уже загружен')
        elif name not in keys:
            print(f'{name}: не найден в бакете {bucket}')
        else:
            s3.download_file(bucket, keys[name], str(path))
            print(f'{name}: {path.stat().st_size / 2**20:.1f} МБ')


if __name__ == '__main__':
    main()
