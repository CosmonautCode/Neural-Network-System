import pandas as pd
from datasets import load_dataset

# Load the dataset to get real examples
dataset = load_dataset('Ammok/apple_stock_price_from_1980-2021')
df = pd.DataFrame(dataset['train'])

# Get recent examples (last 10 records)
recent_data = df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

print('Recent Apple Stock Data Examples:')
print('=' * 80)
for i, (idx, row) in enumerate(recent_data.iterrows(), 1):
    print(f'Example {i}:')
    print(f'  Open: ${row["Open"]:.2f}')
    print(f'  High: ${row["High"]:.2f}')
    print(f'  Low: ${row["Low"]:.2f}')
    print(f'  Close: ${row["Close"]:.2f}')
    print(f'  Volume: {row["Volume"]:,}')
    print(f'  Actual Adj Close: ${row["Adj Close"]:.2f}')
    print()

# Also get some historical examples from different price ranges
print('Historical Examples from Different Eras:')
print('=' * 80)

# Early 2000s (low prices)
early_2000s = df[(df.index >= '2000-01-01') & (df.index <= '2000-12-31')].head(3)
print('Early 2000s (Low Price Era):')
for i, (idx, row) in enumerate(early_2000s.iterrows(), 1):
    print(f'  Open: ${row["Open"]:.2f}, High: ${row["High"]:.2f}, Low: ${row["Low"]:.2f}, Close: ${row["Close"]:.2f}, Volume: {row["Volume"]:,}')

# 2010s (medium prices)
mid_2010s = df[(df.index >= '2015-01-01') & (df.index <= '2015-12-31')].head(3)
print('\nMid 2010s (Medium Price Era):')
for i, (idx, row) in enumerate(mid_2010s.iterrows(), 1):
    print(f'  Open: ${row["Open"]:.2f}, High: ${row["High"]:.2f}, Low: ${row["Low"]:.2f}, Close: ${row["Close"]:.2f}, Volume: {row["Volume"]:,}')

# 2020s (high prices)
recent_2020s = df[(df.index >= '2020-01-01') & (df.index <= '2020-12-31')].head(3)
print('\n2020s (High Price Era):')
for i, (idx, row) in enumerate(recent_2020s.iterrows(), 1):
    print(f'  Open: ${row["Open"]:.2f}, High: ${row["High"]:.2f}, Low: ${row["Low"]:.2f}, Close: ${row["Close"]:.2f}, Volume: {row["Volume"]:,}')

print(f'\nDataset Statistics:')
print(f'Price Range: ${df["Adj Close"].min():.2f} - ${df["Adj Close"].max():.2f}')
print(f'Average Volume: {df["Volume"].mean():,.0f}')
