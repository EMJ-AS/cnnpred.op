import os
import pandas as pd
import numpy as np
from dataset import yfwrap
import matplotlib.pyplot as plt
import pandas_ta as ta
import warnings
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
from dataset import stock_scope
import config

LOAD_RAW_FROM_NET = True
RAW_DIR = 'data/raw'

#tickers  = ['AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AMAT', 'AMBA', 'AMD', 'AMZN', 'ANET', 'ARKK', 'ASML', 'ATER', 'AVAV', 'AVGO', 'AYX', 'BABA', 'BB', 'BIDU', 'BILI', 'BKNG', 'BL', 'BLUE', 'BOX', 'BSX', 'BYND', 'CCJ', 'CDNS', 'CDW', 'CHGG', 'CHKP', 'CHWY', 'CMCSA', 'CORT', 'CRM', 'CRSP', 'CRWD', 'CSCO', 'CSIQ', 'CVNA', 'CYBR', 'DBX', 'DIS', 'DKNG', 'DNN', 'DOCU', 'DT', 'DXCM', 'EA', 'EB', 'EBAY', 'EDIT', 'ENPH', 'ESTC', 'ETSY', 'EXAS', 'EXPE', 'FATE', 'FCEL', 'FI', 'FIS', 'FSLY', 'FTCH', 'FTNT', 'FUBO', 'FUTU', 'FVRR', 'GDS', 'GLOB', 'GME', 'GNRC', 'GOGO', 'GOOGL', 'GPRO', 'GRPN', 'HIMX', 'HPE', 'HUBS', 'IAC', 'ILMN', 'IMAX', 'INTC', 'INTU', 'IONS', 'ISRG', 'JD', 'KLAC', 'KOPN', 'KURA', 'KWEB', 'LC', 'LITE', 'LOGI', 'LRCX', 'LULU', 'LYFT', 'MARA', 'MCHP', 'MDB', 'MELI', 'META', 'MGNI', 'MRNA', 'MRVL', 'MSFT', 'MSTR', 'MU', 'MVIS', 'MVST', 'NFLX', 'NICE', 'NIO', 'NKLA', 'NOW', 'NTAP', 'NTDOY', 'NTES', 'NTLA', 'NTNX', 'NVAX', 'NVDA', 'NVTA', 'NXPI', 'NYT', 'OKTA', 'ON', 'ORCL', 'PACB', 'PANW', 'PARA', 'PAYC', 'PD', 'PDD', 'PENN', 'PINS', 'PLUG', 'PSTG', 'PYPL', 'QCOM', 'RDFN', 'REAL', 'RNG', 'ROKU', 'RVLV', 'SABR', 'SAP', 'SBGI', 'SE', 'SFIX', 'SFTBY', 'SGBI', 'SGML', 'SHOP', 'SMAR', 'SMCI', 'SMH', 'SNAP', 'SNPS', 'SOHU', 'SONO', 'SONY', 'SPCE', 'SPLK', 'SPOT', 'SQ', 'STM', 'T', 'TDC', 'TDOC', 'TEAM', 'TENB', 'TIGR', 'TNDM', 'TSLA', 'TTD', 'TTWO', 'TWLO', 'TXN', 'UBER', 'UPWK', 'VEEV', 'VIPS', 'VZ', 'W', 'WB', 'WBD', 'WDAY', 'WDC', 'WIX', 'XBI', 'YELP', 'YEXT', 'Z', 'ZM', 'ZS']
#tickers = ['AAPL', 'NVDA']
tickers = stock_scope.ALL_TICKERS
#tickers = ['MNDY', 'MCHP', 'WDC', 'TTD']

def create_features(ticker=None, data=None):
    if ticker is None or data is None:
        return None

    # Aberration
    aberration_df = ta.aberration(data['High'], data['Low'], data['Close'], length=5, \
                                    atr_length=14)
    data = pd.concat([data, aberration_df], axis=1)

    # ATR = Average True Range
    atr_df = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data = pd.concat([data, atr_df], axis=1)

    # BBands = Bollinger Bands
    bbands_df = ta.bbands(data['Close'], length=5, std=2, mamode='sma')
    data = pd.concat([data, bbands_df], axis=1)

    # AO = Awesome Oscillator
    data['AO_5_34'] = ta.ao(data['High'], data['Low'], fast=5, slow=34)
    
    # BOP = Balance Of Power
    data['BOP'] = ta.bop(data['Open'], data['High'], data['Low'], data['Close'])

    # MOM = Momentum
    data['MOM_1'] = ta.mom(data['Close'], length=1)
    data['MOM_2'] = ta.mom(data['Close'], length=2)
    data['MOM_4'] = ta.mom(data['Close'], length=4)
    data['MOM_6'] = ta.mom(data['Close'], length=6)

    # SMA = Simple Moving Average
    data['SMA_30'] = ta.sma(data['Close'], length=30)
    data['SMA_60'] = ta.sma(data['Close'],length=60)
    data['SMA_90'] = ta.sma(data['Close'],length=90)

    # RSI = Relative Strength Index
    data['RSI_14'] = ta.rsi(close=data['Close'], length=14)

    # STOCH = Stochastic Oscillator
    stoch = ta.stoch(high=data['High'], low=data['Low'], close=data['Close'], k=14, d=3, smooth_k=3)
    # Add the stochastic oscillator values to the DataFrame
    data['%K_14_3_3'] = stoch['STOCHk_14_3_3']
    data['%D_14_3_3'] = stoch['STOCHd_14_3_3']

    # CMF = Chaikin Money Flow
    data['CMF_20'] = ta.cmf(high=data['High'], low=data['Low'], close=data['Close'], \
                            volume=data['Volume'], length=20)
    # AD = Accumulation/Distribution
    data['AD'] = ta.ad(high=data['High'], low=data['Low'], close=data['Close'], \
                            volume=data['Volume'])


    data = data.astype('float64')
    data = data.add_prefix(f'{ticker}_')
    # Print the updated DataFrame with Aberration values
    data_filtered = data.iloc[100:]
    

    return data_filtered

index_tickers = stock_scope.INDEX_TICKERS
treas_tickers = stock_scope.TREAS_TICKERS
macro_tickers = stock_scope.MACRO_TICKERS
period = "10y"

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for i in range(len(tickers)):
            ticker = tickers[i]
            if LOAD_RAW_FROM_NET:
                print(f'{i:3d}:{ticker}')
                data = yfwrap.load_disk(ticker)
                data_filtered = create_features(ticker=ticker, data=data)

                all_df = data_filtered
                print(f"ticker: {ticker}")
                print(f"range: {all_df.index[0]} {all_df.index[-1]}")


                # Add indices
                for idx in index_tickers:
                    data = yfwrap.load_disk(idx)
                    print(f"index: {data.index[0]} {data.index[-1]}")
                    data_filtered = create_features(ticker=idx, data=data)
                    # Perform inner join on the index
                    result = all_df.join(data_filtered, how='inner', lsuffix='_left', rsuffix='_right')
                    print(f"all: {result.index[0]} {result.index[-1]}")
                    
                    # assert all_df.shape[0] == data_filtered.shape[0], \
                    print(f"dataset lengths (existing vs new): {all_df.shape[0]}, {data_filtered.shape[0]}")

                    all_df = result

                # Add treasury yields
                for treas in treas_tickers:
                    print(f"treas: {treas}")
                    data = yfwrap.load_disk(treas)
                    data_filtered = create_features(ticker=treas, data=data)

                    # Perform inner join on the index
                    result = all_df.join(data_filtered, how='inner', lsuffix='_left', rsuffix='_right')
                    print(f"all: {result.index[0]} {result.index[-1]}")
                    
                    # assert all_df.shape[0] == data_filtered.shape[0], \
                    print(f"dataset lengths (existing vs new): {all_df.shape[0]}, {data_filtered.shape[0]}")

                    all_df = result

                # Add macro tickers
                for macro in macro_tickers:
                    if ticker == macro:
                        continue
                    print(f"treas: {macro}")
                    data = yfwrap.load_disk(macro)
                    data_filtered = create_features(ticker=macro, data=data)

                    # Perform inner join on the index
                    result = all_df.join(data_filtered, how='inner', lsuffix='_left', rsuffix='_right')
                    print(f"all: {result.index[0]} {result.index[-1]}")
                    
                    # assert all_df.shape[0] == data_filtered.shape[0], \
                    print(f"dataset lengths (existing vs new): {all_df.shape[0]}, {data_filtered.shape[0]}")

                    all_df = result


                
                # Get the columns with only one unique value
                single_value_cols = all_df.columns[all_df.nunique() == 1]

                # Drop the columns with only one unique value
                all_df = all_df.drop(single_value_cols, axis=1)

                all_df = all_df.dropna(axis=1)
                # print(all_df.isnull().sum())

                Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
                all_df.to_csv(os.path.join(RAW_DIR, f'{ticker}.csv'),)
            else:
                all_df = pd.read_csv(os.path.join(RAW_DIR, f'{ticker}.csv'), index='Date', parse_dates=True)


            # Compute mutual information up until start of test period
            X_mi = all_df[all_df.index < config.TRAIN_TEST_CUTOFF]
            # if there are no rows before train-test cutoff, skip this ticker
            if X_mi.shape[0] == 0:
                continue
            y_mi = (X_mi[f'{ticker}_Close'].pct_change().shift(-1) > 0).astype(int)
            X_mi = X_mi.drop([f'{ticker}_Close'], axis=1)

            mi_scores = mutual_info_classif(X_mi, y_mi, random_state=1)

            # Sort features by mutual information value
            sorted_features = sorted(zip(X_mi.columns, mi_scores), key=lambda x: x[1], reverse=True)

            # Print the sorted features and their mutual information scores
            # for feature, score in sorted_features:
            #     print(f'{feature}: {score}')

            # Get the columns with mutual information above 0.01, excluding the off-limits column
            selected_columns = X_mi.columns[mi_scores > 0.003]

            # Filter the DataFrame based on selected columns
            df_filtered = all_df[selected_columns]
            df_filtered = df_filtered.copy()

            # df_filtered.loc[:, 'Name'] = ticker
            df_filtered.loc[:, 'Name'] = ticker
            df_filtered[f'{ticker}_Close'] = all_df[f'{ticker}_Close']
            Path(f'./data/data-2d/{ticker}').mkdir(parents=True, exist_ok=True)
            df_filtered.to_csv(f'./data/data-2d/{ticker.upper()}/{ticker.upper()}.csv', index=True)
except AssertionError as e:
    print(f'Exception: {e}')
    different_index_values = all_df.index.symmetric_difference(data_filtered.index)

    # Create a new DataFrame indicating which original DataFrame each index value is missing
    missing_indicator = pd.DataFrame(index=different_index_values)
    missing_indicator['In_all_df'] = missing_indicator.index.isin(all_df.index)
    missing_indicator['In_data_filtered'] = missing_indicator.index.isin(data_filtered.index)

    print(missing_indicator)


