import pandas as pd

# Create a sample DataFrame
data = {'value': [100, 100, 100, 100, 100, 120, 125, 130, 40, 50, 60, 70, 20, 90, 100]}
df = pd.DataFrame(data)


# Shift the values in the 'value' column by 1 period
n = 4
percent_threshold = 20.0
df['percent'] = (df['value'].shift(-n) / df['value'] - 1)  * 100

df['Target'] =  df['percent'].apply(lambda x: 1 if x > percent_threshold else 0 if not np.isnan(x) else np.nan)
df['Target'] = df['Target']



print(df)



import pandas as pd
import datetime as dt

horizon = 3

start_date = dt.datetime(2023, 1, 1)
num_days = 31  # Adjust as needed

# Generate the date range excluding weekends
date_range = pd.date_range(start=start_date, periods=num_days, freq='B')
# Create the DataFrame
close_values = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
print(len(close_values))
print(len(date_range))
# close_values[-5] = 0
df = pd.DataFrame({
    'Date': date_range,
    'AAPL_Close': close_values,
    'Name': 'AAPL'
})

datadir = './tests/data/'
# Save the DataFrame to a CSV file
print(len(df))
df.to_csv('./tests/data/AAPL_data.csv', index=False)

data = create_datasets(datadir, 0.75, '2023-1-18', ticker='AAPL', horizon=horizon, percent_threshold=100.0)
data['AAPL']['Sequence'] = list(range(1, len(data['AAPL'])+1))
print(len(data['AAPL']))
