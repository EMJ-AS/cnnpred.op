import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta as ta
import warnings
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path


#tickers  = ['AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AMAT', 'AMBA', 'AMD', 'AMZN', 'ANET', 'ARKK', 'ASML', 'ATER', 'AVAV', 'AVGO', 'AYX', 'BABA', 'BB', 'BIDU', 'BILI', 'BKNG', 'BL', 'BLUE', 'BOX', 'BSX', 'BYND', 'CCJ', 'CDNS', 'CDW', 'CHGG', 'CHKP', 'CHWY', 'CMCSA', 'CORT', 'CRM', 'CRSP', 'CRWD', 'CSCO', 'CSIQ', 'CVNA', 'CYBR', 'DBX', 'DIS', 'DKNG', 'DNN', 'DOCU', 'DT', 'DXCM', 'EA', 'EB', 'EBAY', 'EDIT', 'ENPH', 'ESTC', 'ETSY', 'EXAS', 'EXPE', 'FATE', 'FCEL', 'FI', 'FIS', 'FSLY', 'FTCH', 'FTNT', 'FUBO', 'FUTU', 'FVRR', 'GDS', 'GLOB', 'GME', 'GNRC', 'GOGO', 'GOOGL', 'GPRO', 'GRPN', 'HIMX', 'HPE', 'HUBS', 'IAC', 'ILMN', 'IMAX', 'INTC', 'INTU', 'IONS', 'ISRG', 'JD', 'KLAC', 'KOPN', 'KURA', 'KWEB', 'LC', 'LITE', 'LOGI', 'LRCX', 'LULU', 'LYFT', 'MARA', 'MCHP', 'MDB', 'MELI', 'META', 'MGNI', 'MRNA', 'MRVL', 'MSFT', 'MSTR', 'MU', 'MVIS', 'MVST', 'NFLX', 'NICE', 'NIO', 'NKLA', 'NOW', 'NTAP', 'NTDOY', 'NTES', 'NTLA', 'NTNX', 'NVAX', 'NVDA', 'NVTA', 'NXPI', 'NYT', 'OKTA', 'ON', 'ORCL', 'PACB', 'PANW', 'PARA', 'PAYC', 'PD', 'PDD', 'PENN', 'PINS', 'PLUG', 'PSTG', 'PYPL', 'QCOM', 'RDFN', 'REAL', 'RNG', 'ROKU', 'RVLV', 'SABR', 'SAP', 'SBGI', 'SE', 'SFIX', 'SFTBY', 'SGBI', 'SGML', 'SHOP', 'SMAR', 'SMCI', 'SMH', 'SNAP', 'SNPS', 'SOHU', 'SONO', 'SONY', 'SPCE', 'SPLK', 'SPOT', 'SQ', 'STM', 'T', 'TDC', 'TDOC', 'TEAM', 'TENB', 'TIGR', 'TNDM', 'TSLA', 'TTD', 'TTWO', 'TWLO', 'TXN', 'UBER', 'UPWK', 'VEEV', 'VIPS', 'VZ', 'W', 'WB', 'WBD', 'WDAY', 'WDC', 'WIX', 'XBI', 'YELP', 'YEXT', 'Z', 'ZM', 'ZS']
# Exclude ADPT, ATVI, NEWR, WWE, TRUE, FTCH
tickers = [
    "0700.HK",
    "1024.HK",
    "1263.HK",
    "1810.HK",
    "2330.TW",
    "2454.TW",
    "3668.T",
    "3690.HK",
    "7832.T",
    "9684.T",
    "9697.T",
    "9992.HK",
    "AAPL",
    "ACCD",
    "ADBE",
    "ADE.OL",
    "ADI",
    "ADP",
#    "ADPT",
    "ADSK",
    "ADYEN.AS",
    "AFRM",
    "AI",
    "AMAT",
    "AMBA",
    "AMD",
    "AMZN",
    "ANET",
    "ARKK",
    "ARRY",
    "ASAN",
    "ASML",
    "ATER",
#    "ATVI",
    "ATZ.TO",
    "AVAV",
    "AVGO",
    "AYX",
    "BABA",
    "BB",
    "BBD-B.TO",
    "BEAM",
    "BIDU",
    "BIGC",
    "BILI",
    "BILL",
    "BITO",
    "BKNG",
    "BL",
    "BLND",
    "BLUE",
    "BMBL",
    "BNTX",
    "BOX",
    "BSX",
    "BTC-USD",
    "BYND",
    "BZFD",
    "CCJ",
    "CDNS",
    "CDW",
    "CHGG",
    "CHKP",
    "CHWY",
    "CMCSA",
    "COIN",
    "CORT",
    "CPNG",
    "CRM",
    "CRSP",
    "CRWD",
    "CSCO",
    "CSIQ",
    "CSU.TO",
    "CVNA",
    "CYBR",
    "DASH",
    "DBX",
    "DDOG",
    "DHER.DE",
    "DIS",
    "DKNG",
    "DLO",
    "DNN",
    "DOCS",
    "DOCU",
    "DSY.PA",
    "DT",
    "DUOL",
    "DXCM",
    "EA",
    "EB",
    "EBAY",
    "EDIT",
    "EDR",
    "ENPH",
    "ENVX",
    "ESTC",
    "ETH-USD",
    "ETSY",
    "EXAS",
    "EXPE",
    "FATE",
    "FCEL",
    "FI",
    "FIS",
    "FLTR.L",
    "FREY",
    "FSLY",
    #"FTCH",
    "FTNT",
    "FUBO",
    "FUTU",
    "FVRR",
    "GCT",
    "GDRX",
    "GDS",
    "GFS",
    "GLBE",
    "GLOB",
    "GLXY.TO",
    "GME",
    "GNRC",
    "GOGO",
    "GOOGL",
    "GPRO",
    "GRAB",
    "GRPN",
    "GTLB",
    "HCP",
    "HFG.DE",
    "HIMX",
    "HOOD",
    "HPE",
    "HUBS",
    "IAC",
    "IBM",
    "ILMN",
    "IMAX",
    "INTC",
    "INTU",
    "IONS",
    "IPX",
    "ISRG",
    "JD",
    "JOBY",
    "KAHOT.OL",
    "KC",
    "KLAC",
    "KOPN",
    "KURA",
    "KWEB",
    "LC",
    "LCID",
    "LI",
    "LITE",
    "LMND",
    "LOGI",
    "LRCX",
    "LSPD",
    "LULU",
    "LYFT",
    "MARA",
    "MBLY",
    "MCHP",
    "MDB",
    "ME",
    "MELI",
    "META",
    "MGNI",
    "MNDY",
    "MQ",
    "MRNA",
    "MRVL",
    "MSFT",
    "MSTR",
    "MU",
    "MVIS",
    "MVST",
    "NCNO",
    "NET",
#    "NEWR",
    "NFLX",
    "NICE",
    "NIO",
    "NKLA",
    "NOW",
    "NTAP",
    "NTDOY",
    "NTES",
    "NTLA",
    "NTNX",
    "NVAX",
    "NVDA",
    "NVTA",
    "NXPI",
    "NYT",
    "OKTA",
    "ON",
    "OPAD",
    "OPEN",
    "ORCL",
    "OTLY",
    "PACB",
    "PANW",
    "PARA",
    "PATH",
    "PAYC",
    "PD",
    "PDD",
    "PENN",
    "PINS",
    "PLTK",
    "PLTR",
    "PLUG",
    "PSTG",
    "PTON",
    "PYPL",
    "QCOM",
    "QS",
    "RBLX",
    "RDFN",
    "REAL",
    "RENT",
    "RIVN",
    "RNG",
    "ROKU",
    "ROOT",
    "RSI",
    "RVLV",
    "S",
    "SABR",
    "SANA",
    "SAP",
    "SBGI",
    "SDGR",
    "SE",
    "SFIX",
    "SFTBY",
    "SGBI",
    "SGML",
    "SHOP",
    "SMAR",
    "SMCI",
    "SMH",
    "SMR",
    "SNAP",
    "SNOW",
    "SNPS",
    "SOFI",
    "SOHU",
    "SONO",
    "SONY",
    "SPCE",
    "SPLK",
    "SPOT",
    "SQ",
    "STM",
    "T",
    "TBLA",
    "TDC",
    "TDOC",
    "TEAM",
    "TENB",
    "TIGR",
    "TKWY.AS",
    "TNDM",
    "TOST",
#    "TRUE",
    "TSLA",
    "TTD",
    "TTWO",
    "TWLO",
    "TXN",
    "U",
    "UBER",
    "UBI.PA",
    "UPST",
    "UPWK",
    "V",
    "VEEV",
    "VIPS",
    "VIV.PA",
    "VRT",
    "VTEX",
    "VZ",
    "W",
    "WB",
    "WBD",
    "WDAY",
    "WDC",
    "WISH",
    "WIX",
    "WLDS",
    "WNDR.TO",
#    "WWE",
    "XBI",
    "XPEV",
    "XRO.AX",
    "YELP",
    "YEXT",
    "Z",
    "ZAL.DE",
    "ZI",
    "ZM",
    "ZS"
]


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

index_tickers = ["^DJI", "^GSPC", "^IXIC"]
treas_tickers = ["^TNX", "^FVX"]
macro_tickers = ['ARKK', 'IWM', 'QQQ', 'AAPL']
period = "10y"

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for i in range(len(tickers)):
            ticker = tickers[i]
            print(f'{i:3d}:{ticker}')
            data = yf.download(ticker, period=period)
            data_filtered = create_features(ticker=ticker, data=data)

            all_df = data_filtered
            print(f"ticker: {ticker}")
            print(f"range: {all_df.index[0]} {all_df.index[-1]}")


            # Add indices
            for idx in index_tickers:
                data = yf.download(idx, period=period)
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
                data = yf.download(treas, period=period)
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
                data = yf.download(macro, period=period)
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

            y = (all_df[f'{ticker}_Close'].pct_change().shift(-1) > 0).astype(int)
            X = all_df.drop([f'{ticker}_Close'], axis=1)


            # Compute mutual information
            mi_scores = mutual_info_classif(X, y)

            # Sort features by mutual information value
            sorted_features = sorted(zip(X.columns, mi_scores), key=lambda x: x[1], reverse=True)

            # Print the sorted features and their mutual information scores
            # for feature, score in sorted_features:
            #     print(f'{feature}: {score}')

            # Get the columns with mutual information above 0.01, excluding the off-limits column
            selected_columns = X.columns[mi_scores > 0.0001]

            # Filter the DataFrame based on selected columns
            df_filtered = X[selected_columns]
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


