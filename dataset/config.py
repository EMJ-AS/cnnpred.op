import os

root_s3_path = os.path.join(os.path.dirname(__file__), '../s3')
root_path = os.path.join(root_s3_path, 'dataset')
local_path = os.path.join(os.path.dirname(__file__), '../local/dataset')

YFINANCE_DIR_NAME='yfinance_3'

YFINANCE_S3_FOLDER_PATH=os.path.join(root_path, 'time_series', YFINANCE_DIR_NAME)
YFINANCE_LOCAL_FOLDER_PATH=os.path.join(local_path, 'time_series' ,YFINANCE_DIR_NAME)

def yfinance_s3_path(ticker: str):
    return os.path.join(YFINANCE_S3_FOLDER_PATH, f'{ticker}.csv')

def yfinance_local_path(ticker: str):
    return os.path.join(YFINANCE_LOCAL_FOLDER_PATH, f'{ticker}.csv')
