import os
import json

def get_path(name: str):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), f'../collect/{name}_tickers.json'))


ALL_TICKERS_WITHOUT_EXCLUDE = json.load(open(get_path('stock'), 'r'))
EXCLUDED_TICKERS = json.load(open(get_path('exclude'), 'r'))
ALL_TICKERS = [t for t in ALL_TICKERS_WITHOUT_EXCLUDE if t not in EXCLUDED_TICKERS]

INDEX_TICKERS = json.load(open(get_path('index'), 'r'))
TREAS_TICKERS = json.load(open(get_path('treas'), 'r'))
MACRO_TICKERS = json.load(open(get_path('macro'), 'r'))
