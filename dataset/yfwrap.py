import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import yfinance as yf
from ratelimit import limits, sleep_and_retry

'''
A wrapper for Yahoo Finance API so that we can ensure non-deterministic output from multiple runs.

* to download data to disk beforehand; run `sync/download.py`

* looks for a cached version on disk and if present loads it

* if not then throws an error

* if `attempt_update` is True; then it will call sync/update
    * get latest from Yahoo Finance
    * ensure we have the latest from S3 on disk
    * update the cache on disk with rows that do not already exist in cache
    * upload the updated version of cache to s3 if it has changed

'''

SYNC_YF_PATH = 's3/dataset/time_series/yfinance'
from dataset import config

cache = {}
def load_disk(ticker: str, use_cache:bool = True):
    if ticker not in cache:
        path = config.yfinance_s3_path(ticker)
        if not os.path.exists(path):
            raise FileNotFoundError(f'The path {path} does not exist.')
        df = pd.read_csv(path, index_col='Date', parse_dates=True, float_precision='round_trip')
        if use_cache:
            cache[ticker] = df

    if use_cache:
        return cache[ticker]
    else:
        return df

'''
@sleep_and_retry
@limits(calls=1, period=1)
'''
def load_net(ticker: str):
    return yf.download(ticker, period='10y')


if __name__=='__main__':
    load_disk('0700.HK')
