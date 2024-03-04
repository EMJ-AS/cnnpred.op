import os

import pandas as pd
from dataset import stock_scope


def get_path(ticker: str, dir_suffix):
    return os.path.join(f'data/forecasts_30_day.{dir_suffix}', f'{ticker}.csv')


def get_dfs(ticker: str):
    old_path = get_path(ticker, 'old')
    new_path = get_path(ticker, 'new')
    if os.path.exists(old_path) and os.path.exists(new_path):
        old_df = pd.read_csv(old_path)
        new_df = pd.read_csv(new_path)
        return new_df, old_df
    else:
        return None, None

if __name__=='__main__':
    for t in stock_scope.ALL_TICKERS:
        new_df, old_df = get_dfs(t)
        if new_df is None or old_df is None:
            continue
        old_len = len(old_df)
        new_df = new_df.iloc[:old_len].copy()
        diff_rows = new_df.compare(old_df)
        if len(diff_rows) > 1:
            print(f'{t}: {len(diff_rows)} rows different')
