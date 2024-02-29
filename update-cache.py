from dataset import stock_scope
from collect import update as collect_update
from sync import upload as sync_upload

if __name__=='__main__':
    #collect_update.ticker('AAPL')

    for ticker in stock_scope.ALL_TICKERS:
        collect_update.ticker(ticker)

    for ticker in stock_scope.INDEX_TICKERS:
        collect_update.ticker(ticker)

    for ticker in stock_scope.TREAS_TICKERS:
        collect_update.ticker(ticker)

    for ticker in stock_scope.MACRO_TICKERS:
        collect_update.ticker(ticker)

    sync_upload.run()
