import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import boto3
from botocore.exceptions import ClientError
import hashlib
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler

from dataset import config as config_dataset

s3 = boto3.client('s3')
bucket_name = 'emj-data-lab-alpha'

def get_s3_object_md5(key):
    """
    Return the MD5 hash of the S3 object's content
    """
    s3_object = s3.Object(bucket_name, key)
    return s3_object.e_tag.strip('"')

def run_ticker(t: str):
    Path(config_dataset.YFINANCE_S3_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    file_path = config_dataset.yfinance_s3_path(t)
    key = os.path.join('dataset/time_series', config_dataset.YFINANCE_DIR_NAME, f'{t}.csv')

    print(f'Downloading file_path: {file_path} key: {key}')

    try:
        s3_object = s3.head_object(Bucket=bucket_name, Key=key)
    except ClientError as e:
        print(f'Key does not exist: {key}')
        return

    # Check if the file already exists and has the same content
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            local_md5 = hashlib.md5(f.read()).hexdigest()

        s3_md5 = s3_object['ETag'].strip('"')
        if local_md5 == s3_md5:
            print(f"Skipping s3://{bucket_name}/{key} because the content has not changed.")
            return

    # Download the file if it doesn't exist or has different content
    with open(file_path, 'wb') as f:
        s3.download_fileobj(bucket_name, key, f)
    print(f"Downloaded s3://{bucket_name}/{key} to {file_path}")


def run(initial=False):
    dataset_dir = config_dataset.root_path
    if initial and os.path.exists(dataset_dir):
        print(f'skipping initial download because {dataset_dir} already exists')
        return
    else:
        print('---------starting download run-------')

    print('bucket_name: ', bucket_name)
    response = s3.list_objects_v2(Bucket=bucket_name)
    Path(config_dataset.YFINANCE_S3_FOLDER_PATH).mkdir(parents=True, exist_ok=True)


    for obj in response['Contents']:
        key = obj['Key']
        if config_dataset.YFINANCE_DIR_NAME not in key:
            continue

        folder_path = config_dataset.root_s3_path
        file_path = os.path.join(folder_path, key)
        print('file_path: ', file_path)

        if os.path.isdir(file_path):
            continue

        # Check if the file already exists and has the same content
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                local_md5 = hashlib.md5(f.read()).hexdigest()
            s3_object = s3.head_object(Bucket=bucket_name, Key=key)
            s3_md5 = s3_object['ETag'].strip('"')
            if local_md5 == s3_md5:
                print(f"Skipping s3://{bucket_name}/{key} because the content has not changed.")
                continue

        # Download the file if it doesn't exist or has different content
        with open(file_path, 'wb') as f:
            s3.download_fileobj(bucket_name, key, f)
        print(f"Downloaded s3://{bucket_name}/{key} to {file_path}")

    print('---------finished download run-------')


if __name__=='__main__':
    print('executed download')
    if 'INIT' in os.environ:
        run(True)
    elif 'SCHEDULE' in os.environ:
        minutes = int(os.environ['SCHEDULE'])
        print(f'scheduling run every {minutes} minutes')
        scheduler = BlockingScheduler()
        scheduler.add_job(run, 'interval', minutes=minutes)
        scheduler.start()
    else:
        run()
