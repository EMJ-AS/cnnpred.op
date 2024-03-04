import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, \
    recall_score, roc_auc_score

import matplotlib.pyplot as plt
import warnings
import lightgbm as lgb
from lightgbm.plotting import plot_tree, plot_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from datetime import datetime

from IPython.display import clear_output
import seaborn as sns

import config
from dataset import stock_scope


#tickers = ['NVDA']
# tickers = ['QCOM', 'ASML', 'MU', 'ON', 'AMD', 'NVDA', 'INTC']
# tickers = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'JNJ', 'JPM', 'SPY', 'BRK-B', 'XOM', 'V', 'PG', 'HD', 'NVDA', 'CVX', 'META', 'PFE', 'MRK', 'PEP', 'ABBV', 'UNH', 'DIS', 'CSCO', 'COST', 'MCD', 'TSLA', 'VZ', 'KO', 'WMT', 'BAC', 'ABT', 'MA', 'LLY', 'INTC', 'IVV', 'QQQ', 'BMY', 'IBM', 'AMGN', 'UNP', 'T', 'ADBE', 'NEE', 'QCOM', 'ORCL', 'RTX', 'TMO', 'NKE', 'SBUX', 'HON', 'CVS', 'LOW', 'CMCSA', 'CAT', 'LMT', 'UPS', 'CRM', 'AVGO', 'BA', 'TXN', 'COP', 'DE', 'ACN', 'DHR', 'WFC', 'MMM', 'AXP', 'PM', 'ADP', 'MDT', 'NFLX', 'TGT', 'GLD', 'AMD', 'MO', 'GS', 'MDLZ', 'BLK', 'DUK', 'EMR', 'AMAT', 'TJX', 'WM', 'SO', 'AMT', 'MS', 'GILD', 'F', 'USB', 'SYK', 'C', 'SCHW', 'CL', 'FDX', 'KMB', 'BX', 'INTU', 'SPGI', 'GIS', 'GE']
# tickers  = ['AAPL', 'ADBE', 'ADI', 'ADP', 'ADPT', 'ADSK', 'AMAT', 'AMBA', 'AMD', 'AMZN', 'ANET', 'ARKK', 'ASML', 'ATER', 'AVAV', 'AVGO', 'AYX', 'BABA', 'BB', 'BIDU', 'BILI', 'BKNG', 'BL', 'BLUE', 'BOX', 'BSX', 'BYND', 'CCJ', 'CDNS', 'CDW', 'CHGG', 'CHKP', 'CHWY', 'CMCSA', 'CORT', 'CRM', 'CRSP', 'CRWD', 'CSCO', 'CSIQ', 'CVNA', 'CYBR', 'DBX', 'DIS', 'DKNG', 'DNN', 'DOCU', 'DT', 'DXCM', 'EA', 'EB', 'EBAY', 'EDIT', 'ENPH', 'ESTC', 'ETSY', 'EXAS', 'EXPE', 'FATE', 'FCEL', 'FI', 'FIS', 'FSLY', 'FTCH', 'FTNT', 'FUBO', 'FUTU', 'FVRR', 'GDS', 'GLOB', 'GME', 'GNRC', 'GOGO', 'GOOGL', 'GPRO', 'GRPN', 'HIMX', 'HPE', 'HUBS', 'IAC', 'ILMN', 'IMAX', 'INTC', 'INTU', 'IONS', 'ISRG', 'JD', 'KLAC', 'KOPN', 'KURA', 'KWEB', 'LC', 'LITE', 'LOGI', 'LRCX', 'LULU', 'LYFT', 'MARA', 'MCHP', 'MDB', 'MELI', 'META', 'MGNI', 'MRNA', 'MRVL', 'MSFT', 'MSTR', 'MU', 'MVIS', 'MVST', 'NFLX', 'NICE', 'NIO', 'NKLA', 'NOW', 'NTAP', 'NTDOY', 'NTES', 'NTLA', 'NTNX', 'NVAX', 'NVDA', 'NVTA', 'NXPI', 'NYT', 'OKTA', 'ON', 'ORCL', 'PACB', 'PANW', 'PARA', 'PAYC', 'PD', 'PDD', 'PENN', 'PINS', 'PLUG', 'PSTG', 'PYPL', 'QCOM', 'RDFN', 'REAL', 'RNG', 'ROKU', 'RVLV', 'SABR', 'SAP', 'SBGI', 'SE', 'SFIX', 'SFTBY', 'SGBI', 'SGML', 'SHOP', 'SMAR', 'SMCI', 'SMH', 'SNAP', 'SNPS', 'SOHU', 'SONO', 'SONY', 'SPCE', 'SPLK', 'SPOT', 'SQ', 'STM', 'T', 'TDC', 'TDOC', 'TEAM', 'TENB', 'TIGR', 'TNDM', 'TSLA', 'TTD', 'TTWO', 'TWLO', 'TXN', 'UBER', 'UPWK', 'VEEV', 'VIPS', 'VZ', 'W', 'WB', 'WBD', 'WDAY', 'WDC', 'WIX', 'XBI', 'YELP', 'YEXT', 'Z', 'ZM', 'ZS']
#import json
#tickers = json.load(open('all_stock_tickers.json'))

tickers = stock_scope.ALL_TICKERS
#tickers = ['ARRY']


# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
# to implement F1 score for validation in a batch
@tf.keras.saving.register_keras_serializable()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

@tf.keras.saving.register_keras_serializable()
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

@tf.keras.saving.register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

@tf.keras.saving.register_keras_serializable()
def f1macro(y_true, y_pred):
    f_pos = f1_m(y_true, y_pred)
    # negative version of the data and prediction
    f_neg = f1_m(1-y_true, 1-K.clip(y_pred,0,1))
    return (f_pos + f_neg)/2



def calculate_metrics(y_true, y_pred, ticker='', filename='./data/performance.csv'):

    # Calculate false positive rate, true positive rate, thresholds for ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
    threshold_opt_roc = thresholds_roc[np.argmax(tpr - fpr)]
    
    auc = roc_auc_score(y_true, y_pred)

    performance = f'{ticker.lower()},'
    performance += f'{auc:.4f},'

    # Convert the predicted probabilities to class labels
    y_pred_labels = [1 if x >= threshold_opt_roc else 0 for x in y_pred] 
    y_pred_labels = np.array(y_pred_labels)

    # Compute the 10th percentile threshold
    bottom_percentile = 10
    bottom_threshold = np.percentile(y_pred, bottom_percentile)

    # Filter the predicted scores and true labels based on the bottom threshold
    bottom_predicted_scores = y_pred_labels[y_pred <= bottom_threshold]
    bottom_y_true = y_true[y_pred <= bottom_threshold]

    # Compute the 90th percentile threshold
    top_percentile = 90
    top_threshold = np.percentile(y_pred, top_percentile)

    # Filter the predicted scores and true labels based on the top threshold
    top_predicted_scores = y_pred_labels[y_pred >= top_threshold]
    top_y_true = y_true[y_pred >= top_threshold]

    bottom_true_labels = np.logical_not(bottom_y_true).astype(int)
    bottom_predicted_scores = np.logical_not(bottom_predicted_scores).astype(int)

    precision_bottom = precision_score(bottom_true_labels, bottom_predicted_scores)
    precision_top = precision_score(top_y_true, top_predicted_scores)

    performance += f'{precision_bottom:.4f},'
    performance += f'{precision_top:.4f},'
    performance += f'{top_threshold:.4f},'
    performance += f'{bottom_threshold:.4f}'
    # print(performance)

    # Open the file in append mode
    with open(filename, 'a') as file:
        # Append the string to the file
        file.write(f'{performance}\n')

    return auc, precision_bottom, precision_top


def calculate_plots(y_true, y_pred, horizon, percent_threshold, ticker=''):
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred)

    # Calculate false positive rate, true positive rate, thresholds for ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)

    # Calculate confusion matrix
    # threshold_opt_pr = thresholds_pr[np.argmax(precision - recall)]
    threshold_opt_roc = thresholds_roc[np.argmax(tpr - fpr)]
    # labels_predicted_pr = (scores >= threshold_opt_pr).astype(int)
    labels_predicted_roc = (y_pred >= threshold_opt_roc).astype(int)
    # confusion_matrix_pr = confusion_matrix(true_labels, labels_predicted_pr)
    confusion_matrix_roc = confusion_matrix(y_true, labels_predicted_roc)

    # Plot the figures in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Plot Precision-Recall curve
    axes[0, 0].plot(recall, precision)
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title(f'Precision-Recall Curve: {ticker},{horizon},{percent_threshold}')   

    # Plot Confusion Matrix for Precision-Recall curve
    # Set the font scale for the seaborn plot
    # sns.set(font_scale=1.5)
    # sns.heatmap(confusion_matrix_roc, annot=True, fmt="d", cmap="Blues")

    axes[0, 1].imshow(confusion_matrix_roc, cmap='rainbow', interpolation='nearest')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(['Negative (pred)', 'Positive (pred)'])
    axes[0, 1].set_yticklabels(['Negative (true)', 'Positive (true)'])
    axes[0, 1].set_title(f'Confusion Matrix: {ticker}')

    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, str(confusion_matrix_roc[i, j]), ha='center', va='center', \
                            color='white' if confusion_matrix_roc[i, j] > confusion_matrix_roc.max() / 2 else 'black', \
                                fontsize=16)
    

    # Plot Score Distribution
    axes[1, 0].hist(y_pred[y_true == 0], bins=10, color='b', alpha=0.5, label='Negative')
    axes[1, 0].hist(y_pred[y_true == 1], bins=10, color='r', alpha=0.5, label='Positive')
    axes[1, 0].set_xlabel('Scores')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].set_title(f'Score Distribution: {ticker}')

    # Plot ROC curve
    axes[1, 1].plot(fpr, tpr)
    axes[1, 1].plot([0, 1], [0, 1], linestyle='--')  # Diagonal line for random classifier
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    auc = roc_auc_score(y_true, y_pred)
    axes[1, 1].set_title(f'ROC Curve: {ticker} (AUC = {auc:.4f})')

    plt.tight_layout()
    plt.show()


def cnnpred_2d(seq_len=60, n_features=82, n_filters=(8,8,8), droprate=0.1):
    "2D-CNNpred model according to the paper"
    model = Sequential([
        Input(shape=(seq_len, n_features, 1)),
        Conv2D(n_filters[0], kernel_size=(1, n_features), activation="relu"),
        Conv2D(n_filters[1], kernel_size=(3,1), activation="relu"),
        MaxPool2D(pool_size=(2,1)),
        Conv2D(n_filters[2], kernel_size=(3,1), activation="relu"),
        MaxPool2D(pool_size=(2,1)),
        Flatten(),
        Dropout(droprate),
        Dense(1, activation="sigmoid")
    ])
    return model


def datagen(data, seq_len, batch_size, targetcol, kind, TRAIN_TEST_CUTOFF=None, TRAIN_VALID_RATIO=0.8):
    """
    A generator to produce samples for Keras model
    """
    batch = []
    while True:
        # Pick one dataframe from the pool
        key = random.choice(list(data.keys()))
        df = data[key]
        input_cols = [c for c in df.columns if c != targetcol]
        index = df.index[df.index < TRAIN_TEST_CUTOFF]
        split = int(len(index) * TRAIN_VALID_RATIO)
        assert split > seq_len, "Training data too small for sequence length {}".format(seq_len)
        if kind == 'train':
            index = index[:split]   # range for the training set
        elif kind == 'valid':
            index = index[split:]   # range for the validation set
        else:
            raise NotImplementedError
        # Pick one position, then clip a sequence length
        while True:
            t = random.choice(index)     # pick one time step
            n = (df.index == t).argmax() # find its position in the dataframe
            if n-seq_len+1 < 0:
                continue # this sample is not enough for one sequence length
            frame = df.iloc[n-seq_len+1:n+1]
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            break
        # if we get enough for a batch, dispatch
        if len(batch) == batch_size:
            X, y = zip(*batch)
            X, y = np.expand_dims(np.array(X), 3), np.array(y)
            yield X, y
            batch = []



def datagen_gbm(data, targetcol, kind, TRAIN_TEST_CUTOFF=None, TRAIN_VALID_RATIO=0.8):
    print('datagen_gbm train_test_cutoff: ', TRAIN_TEST_CUTOFF)
    """
    Creates train and validation datasets for LightGBM
    """
    batch = None
    for key, df in data.items():
        input_cols = [c for c in df.columns if c != targetcol]
        # find the start of test sample
        # t = df.index[df.index < TRAIN_TEST_CUTOFF][0]
        index = df.index[df.index < TRAIN_TEST_CUTOFF]
        split = int(len(index) * TRAIN_VALID_RATIO)
        # if kind == 'train':
        #     index = index[:split]   # range for the training set
        # elif kind == 'valid':
        #     index = index[split:]   # range for the validation set
        # else:
        #     raise NotImplementedError

        n = len(index)
        if kind == 'train':
            frame = df.iloc[0:split]
        else:
            frame = df.iloc[split:n]
        if batch is None:
            batch = frame
        else:
            combined = pd.concat([batch, frame])
            batch = combined
    return batch[input_cols].values, batch[[targetcol]].values.flatten()
    

def datagen_gbm_test(data, targetcol, TRAIN_TEST_CUTOFF):
    """ Creates test datasets for LightGBM. Reads data which is a dictionary of dataframes"""
    batch = None
    for key, df in data.items():
        print('last index: ', df.index[-1])

        input_cols = [c for c in df.columns if c != targetcol]
        # find the start of test sample
        t = df.index[df.index >= TRAIN_TEST_CUTOFF][0]
        u = (df.index == t).argmax()
        # index = df.index[df.index >= TRAIN_TEST_CUTOFF]
        # split = int(len(index))
        # if kind == 'train':
        #     index = index[:split]   # range for the training set
        # elif kind == 'valid':
        #     index = index[split:]   # range for the validation set
        # else:
        #     raise NotImplementedError
        n = len(df)
        frame = df.iloc[u:n]
        if batch is None:
            batch = frame
        else:
            combined = pd.concat([batch, frame])
            batch = combined
    return batch[input_cols].values, batch[[targetcol]].values.flatten()


def testgen(data, seq_len, targetcol, TRAIN_TEST_CUTOFF):
    """
    A generator to produce test samples for Keras model. 
    Reads data which is a dictionary of dataframes
    """
    batch = []
    for key, df in data.items():
        input_cols = [c for c in df.columns if c != targetcol]
        # find the start of test sample
        t = df.index[df.index >= TRAIN_TEST_CUTOFF][0]
        n = (df.index == t).argmax()
        # extract sample using a sliding window
        for i in range(n+1, len(df)+1):
            frame = df.iloc[i-seq_len:i]
            batch.append([frame[input_cols].values, frame[targetcol][-1]])
    X, y = zip(*batch)
    return np.expand_dims(np.array(X),3), np.array(y)


def create_datasets(DATADIR, TRAIN_VALID_RATIO, TRAIN_TEST_CUTOFF, ticker='', \
                    horizon=1, percent_threshold=0.0, for_current_day=False):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Read data into pandas DataFrames.  This creates a dictionary of dataframes, one for each ticker. 
        # Each ticker is represented by a file in the datadir. 
        data = {}
        for filename in os.listdir(DATADIR):
            if not filename.lower().endswith(".csv"):
                continue # read only the CSV files
            filepath = os.path.join(DATADIR, filename)
            X = pd.read_csv(filepath, index_col="Date", parse_dates=True)
            # basic preprocessing: get the name, the classification
            # Save the target variable as a column in dataframe for easier dropna()
            name = X["Name"][0]
            del X["Name"]
            cols = X.columns

            # X["Target"] = (X[f'{ticker}_Close'].pct_change().shift(-20) > 0).astype(int)

            # Calculate the percentage change between the nth and nth - 5th position
            n = horizon
            percentage_change = (X[f'{ticker}_Close'].shift(-n) / X[f'{ticker}_Close'] - 1)  * 100
            positive_percentage_indicator = \
                percentage_change.apply(lambda x: 1 if x > percent_threshold else 0 if not np.isnan(x) else np.nan)



            # Assign the calculated values to the dataframe
            # df['nth_minus_nth_minus_ith_pct_change'] = percentage_change_relative
            X['Target'] = positive_percentage_indicator

            if for_current_day:
                X['Target'].ffill(inplace=True)

            # Drop rows with NaN
            X.dropna(inplace=True)

            # Fit the standard scaler using the training dataset
            X['Target'].astype(int)
        
            # Compute scaling statistics on training rows only.
            index = X.index[X.index < TRAIN_TEST_CUTOFF]
            index = index[:int(len(index) * TRAIN_VALID_RATIO)]
            scaler = StandardScaler().fit(X.loc[index, cols])
            # Save scale transformed dataframe
            X[cols] = scaler.transform(X[cols])
            data[name] = X
    return data


def train_lightgbm(data, ticker=None, TRAIN_TEST_CUTOFF=None, TRAIN_VALID_RATIO=None, model_prefix='gbm'):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        # Filter out the specific warning

        # Load the training and validation datasets
        X_train, y_train = datagen_gbm(data, "Target", "train", \
                                       TRAIN_TEST_CUTOFF=TRAIN_TEST_CUTOFF, \
                                        TRAIN_VALID_RATIO=TRAIN_VALID_RATIO)
        X_val, y_val = datagen_gbm(data, "Target", "valid", \
                                   TRAIN_TEST_CUTOFF=TRAIN_TEST_CUTOFF, \
                                    TRAIN_VALID_RATIO=TRAIN_VALID_RATIO)

        feature_names = [c for c in data[ticker].columns if c != "Target"]

        # Assuming numpy arrays for training and validation datasets. Create the training and the 
        # validation datasets from the numpy arrays.
        # print("X_train shape:", X_train.shape)
        # print("y_train shape:", y_train.shape)
        # print("X_val shape:", X_val.shape)
        # print("y_val shape:", y_val.shape)
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)

        # Set sensible parameters for the LightGBM model
        '''
        params = {
            'objective': 'binary',
            'metric': 'auc', # or binary_logloss
            'is_unbalance':True,
            'boosting':'gbdt',
            'max_depth': 4,
            'num_leaves': 31,
            'learning_rate': 0.2,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        '''

        params = {
            'objective': 'binary',
            'lambda_l1': 0.7,
            'lambda_l2': 0.7,
            'min_gain_to_split': 0.2,
            'max_bin': 127,
            'boosting': 'gbdt',
            'metric': 'auc', # or binary_logloss
            'n_estimators': 50,
            'is_unbalance': True, # use with oversampling too.
            'max_depth': 4,  # was 4
            'min_data_in_leaf': 5, # was 10
            'min_child_samples': 5,
            'early_stopping_round': 20,
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
        }

        model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=400, \
                        callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # Optional: Save the trained model
    model.save_model(f'./models/{model_prefix}-{ticker}-model.txt')

    return model


def describe_lightgbm(model, ticker=''):
    # Plot the architecture of the model
    # lgb.plotting.plot_tree(model)
    # plt.show()


    # Plot the feature importance
    lgb.plot_importance(model, max_num_features=40, importance_type="gain", \
                        title=f'Feature importance: {ticker}', figsize=(10, 10)) 
    plt.show()
    

from datetime import timedelta
from pathlib import Path


def test_lightgbm(model, data, TRAIN_TEST_CUTOFF, horizon, percent_threshold, ticker='', model_prefix='gbm'):
    # Evaluate the lightgbm model.
    print('Creating data for gbm_test')
    X_test, y_test = datagen_gbm_test(data, "Target", TRAIN_TEST_CUTOFF)
    print(type(y_test))
    # Load the trained LightGBM model
    model = lgb.Booster(model_file=f'./models/{model_prefix}-{ticker}-model.txt')
    # Generate predictions on the test dataset
    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred, ticker)

    #calculate_plots(y_test, y_pred, horizon, percent_threshold, ticker)

    # save forecast
    df = data[ticker]
    dates = df.index[df.index >= TRAIN_TEST_CUTOFF]
    forecast = pd.DataFrame({'date': dates, 'predicted_label_future_buy_sell': y_pred, 'target_label_future_buy_sell': y_test})
    # fill the last rows with nan
    total_rows = len(forecast)
    #forecast['target_label_future_buy_sell'][total_rows - horizon:] = np.nan
    data_slice = forecast['target_label_future_buy_sell'][total_rows-horizon:].copy()
    data_slice.loc[:] = np.nan
    forecast.loc[total_rows-horizon:, 'target_label_future_buy_sell'] = data_slice
    FORECAST_DIR = f'./data/forecasts_{horizon}_day'
    Path(FORECAST_DIR).mkdir(parents=True, exist_ok=True)
    f_p = os.path.join(FORECAST_DIR, f'{ticker}.csv')
    forecast.to_csv(f_p, index=False)
    return metrics



def compute_models(tickers, train_test_cutoff='2023-1-18', train_valid_ratio=0.75, \
                   data_prefix='', data_dir='./data/data-2d/', model_prefix='gbm', \
                   model_dir='./models', horizon=1, percent_threshold=0.0, \
                   perf_file='./models/model_performance.csv'):
    print(f'train_test_cutoff: ', train_test_cutoff)

    sum_auc = 0.0
    sum_prec_bottom = 0.0
    sum_prec_top = 0.0
    m = 0
    for ticker in tickers:
        try:
            DATADIR = f'{data_dir}{ticker}'
            data_train = create_datasets(DATADIR=DATADIR, TRAIN_VALID_RATIO=train_valid_ratio, \
                                TRAIN_TEST_CUTOFF=train_test_cutoff, ticker=ticker, horizon=horizon, \
                                    percent_threshold=percent_threshold, for_current_day=False)
            data_test = create_datasets(DATADIR=DATADIR, TRAIN_VALID_RATIO=train_valid_ratio, \
                                TRAIN_TEST_CUTOFF=train_test_cutoff, ticker=ticker, horizon=horizon, \
                                    percent_threshold=percent_threshold, for_current_day=True)

            model = train_lightgbm(data_train, ticker, TRAIN_TEST_CUTOFF=train_test_cutoff, \
                                TRAIN_VALID_RATIO=train_valid_ratio, model_prefix=model_prefix)


            # Combine lists into triplets and filter for positive numbers
            result = [(name, num1, num2) for name, num1, num2 in \
                    zip(data_train[ticker].columns.values, \
                        model.feature_importance(importance_type='gain'), \
                        model.feature_importance(importance_type='split')) if num1 > 0 and num2 > 0]

            # Sort the output by the sum of the numbers in descending order
            result_sorted_descending = sorted(result, key=lambda x: x[1], reverse=True)

            metrics = test_lightgbm(model, data_test, train_test_cutoff, horizon, percent_threshold, \
                                    ticker=ticker)

            print(f'auc: {metrics[0]:.3f}, prec_bottom: {metrics[1]:.3f}, prec_top: {metrics[2]:.3f}')
            #describe_lightgbm(model, ticker=ticker)

            sum_auc += metrics[0]
            sum_prec_bottom += metrics[1]
            sum_prec_top += metrics[2]
            m += 1

        except Exception as e:
            print(e)

    avg_auc = sum_auc / m
    avg_prec_bottom = sum_prec_bottom / m
    avg_prec_top = sum_prec_top / m

    # Open the file in append mode
    with open(perf_file, 'a') as file:
        file.write(f'{datetime.now()},')
        file.write(f'{avg_auc:9.3f},{avg_prec_bottom:10.3f},{avg_prec_top:10.3f}, ')
        file.write(f'{horizon:10d},{percent_threshold:10.3f},{m:10d}')
        file.write('\n')


if __name__=='__main__':

    TRAIN_TEST_CUTOFF = config.TRAIN_TEST_CUTOFF
    TRAIN_VALID_RATIO = 0.75

    perf_file = './models/model_performance.csv'
    with open(perf_file, 'w+') as file:
        # Append the string to the file
        file.write(f'{"timestamp":30s},{"auc":10s},{"prec_bot":10s},{"prec_top":10s},{"horizon":10s},{"percent_thresh":10s}, {"ticker_count":10s}\n')

    def run(horizon, percent_threshold):
        try: 
            compute_models(tickers, train_test_cutoff=TRAIN_TEST_CUTOFF, train_valid_ratio=TRAIN_VALID_RATIO, \
                        horizon=horizon, percent_threshold=percent_threshold, perf_file=perf_file)
            # Clear the output immediately
            #clear_output(wait=False)
        except Exception as e:
            print(e)

    print('running....')
    run(1, 0)
    run(3, 0)
    run(5, 0)
    run(10, 0)
    run(20, 0)
    run(30, 0)

    def run_many():
        for h in [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]:
            for p in [-10, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0]:
                if abs(p) > 3.0 and h < 10 or abs(p) > 5.0 and h < 30:
                    continue

                run(h, p)

    #run_many()
