from pathlib import Path
import pandas as pd

from dataset import yfwrap
from dataset import config as dataset_config
from sync import download as sync_download
'''
    Ensure latest is loaded from s3
    Load ticker from disk
    Load ticker from net
    Concat all rows from net to disk version.
'''

def ticker(t: str):
    sync_download.run_ticker(t)
    try:
        df = yfwrap.load_disk(t)
        df_latest = yfwrap.load_net(t)
        new_rows = df_latest[~df_latest.index.isin(df.index)]
        combined_df = pd.concat([df, new_rows]).sort_index()
    except FileNotFoundError as e:
        combined_df = yfwrap.load_net(t)

    if len(combined_df) > 0:
        Path(dataset_config.YFINANCE_LOCAL_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(dataset_config.yfinance_local_path(t))

        Path(dataset_config.YFINANCE_S3_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(dataset_config.yfinance_s3_path(t))
