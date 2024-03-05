import os
import glob
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_FORECAST_DIR = 'data'
PREDICTED_LABEL = 'predicted_label_future_buy_sell'
TARGET_LABEL = 'target_label_future_buy_sell'

def get_forecast_dir(days):
    return os.path.join(ROOT_FORECAST_DIR, f'forecasts_{days}_day')

def calculate_auc(df):
    return roc_auc_score(df[TARGET_LABEL], df['predicted'])

def get_auc(days, forecast_paths):
    total_df = None
    for path in forecast_paths:
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df])

    total_df.dropna(inplace=True)
    if len(total_df) == 0:
        print(f'No valid forecasts found for {days} days. Skipping.')
        return None
    return calculate_auc(total_df)

if __name__=='__main__':
    for d in [1, 3, 5, 10, 20, 30]:
        forecast_paths = glob.glob(os.path.join(get_forecast_dir(d), '*.csv'))
        auc = get_auc(d, forecast_paths)
        print(f'{d} days: {auc}')