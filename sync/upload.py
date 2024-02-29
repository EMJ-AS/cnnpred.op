import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import boto3
import os
import hashlib
from dataset import config as config_dataset

IS_TEST = True

s3 = boto3.resource('s3')
bucket_name = 'emj-data-lab-alpha'
local_folder_path = config_dataset.local_path
s3_folder_path = 'dataset/'

bucket = s3.Bucket(bucket_name)

def get_s3_object_md5(key):
    """
    Return the MD5 hash of the S3 object's content
    """
    s3_object = s3.Object(bucket_name, key)
    return s3_object.e_tag.strip('"')

def run_ticker(t: str):
    local_file_path = os.path.join(config_dataset.yfinance_local_path(t), f'{t}.csv')
    s3_file_path = os.path.join(s3_folder_path, local_file)
    try:
        s3_object_md5 = get_s3_object_md5(s3_file_path)
    except:
        s3_object_md5 = ''
    with open(local_file_path, 'rb') as f:
        local_file_md5 = hashlib.md5(f.read()).hexdigest()
    if s3_object_md5 != local_file_md5:
        if not IS_TEST:
            bucket.upload_file(local_file_path, s3_file_path)
        print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_file_path}")
    else:
        print(f"Skipping upload of {local_file_path} because it has not changed")



def upload_folder(local_path, s3_prefix):
    for local_file in os.listdir(local_path):

        local_file_path = os.path.join(local_path, local_file)
        s3_file_path = s3_prefix + local_file

        if os.path.isfile(local_file_path):
            try:
                s3_object_md5 = get_s3_object_md5(s3_file_path)
            except:
                s3_object_md5 = ''
            with open(local_file_path, 'rb') as f:
                local_file_md5 = hashlib.md5(f.read()).hexdigest()
            if s3_object_md5 != local_file_md5:
                if not IS_TEST and config_dataset.YFINANCE_DIR_NAME in s3_file_path:
                    bucket.upload_file(local_file_path, s3_file_path)
                    print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_file_path}")
                else:
                    print(f"Skipped uploading {local_file_path} to s3://{bucket_name}/{s3_file_path}")

            else:
                print(f"Skipping upload of {local_file_path} because it has not changed")
        elif os.path.isdir(local_file_path):
            upload_folder(local_file_path, s3_file_path + '/')
            print(f"Created folder s3://{bucket_name}/{s3_file_path}")

def run():
    upload_folder(local_folder_path, s3_folder_path)
    print(f"All files in {local_folder_path} have been uploaded to s3://{bucket_name}/{s3_folder_path}")

if __name__=='__main__':
    run()
