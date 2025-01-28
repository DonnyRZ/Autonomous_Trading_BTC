# MTAM Bitcoin Backtesting (TA-Lib Free)

## Overview
This Python script is designed for backtesting a Bitcoin trading strategy using **Vectorbt** and **Plotly**. It processes **1-minute BTC data**, applies multiple timeframes, calculates indicators, generates trading signals, and performs backtesting from **2012 to 2020**.

## Features
- **Multi-timeframe resampling** (15m, 1H, 4H, 1D)
- **Technical indicators**: EMA, ATR, RSI, Bollinger Bands
- **Dynamic signal generation**
- **Backtesting with Vectorbt**
- **Performance analysis and visualization**

## Requirements
Ensure you have the following libraries installed before running the script:

```bash
pip install vectorbt plotly
```

## Usage Guide
### 1. Upload Your 1-Minute BTC Data
The script requires a CSV file with **1-minute BTC price data**. Use the file upload function in Google Colab to provide your dataset.

```python
from google.colab import files
uploaded = files.upload()
```

Ensure your CSV file contains a **timestamp column** with OHLCV (Open, High, Low, Close, Volume) data.

### 2. Data Preprocessing
The script resamples **1-minute data** into multiple timeframes (15m, 1H, 4H, 1D) using Pandas **resample()** function.

```python
def resample_data(df):
    timeframes = {
        '15T': df.resample('15T').agg(ohlc_dict).dropna(),
        '1H': df.resample('1H').agg(ohlc_dict).dropna(),
        '4H': df.resample('4H').agg(ohlc_dict).dropna(),
        '1D': df.resample('1D').agg(ohlc_dict).dropna()
    }
    return timeframes
```

### 3. Indicator Calculation
Key indicators used in this strategy:
- **4H Timeframe:** EMA21, EMA55 (Trend confirmation)
- **1D Timeframe:** EMA50, EMA200 (Macro trend)
- **1H Timeframe:** ATR, RSI, Dynamic RSI thresholds
- **15m Timeframe:** Bollinger Bands, Bandwidth squeeze

```python
timeframes = calculate_indicators(timeframes)
```

### 4. Signal Generation
The script generates buy/sell signals based on:
- **Trend Filters:** 4H EMA21 > EMA55 and 1D EMA50 > EMA200
- **Entry Trigger:** 1H RSI below a dynamic threshold
- **Exit Trigger:** 1H RSI above a dynamic threshold
- **Confirmation:** Bollinger Bands squeeze on 15m timeframe

```python
signals = generate_signals(timeframes)
```

### 5. Backtesting (2012-2020)
The script tests the strategy performance between **2012-2020** using **Vectorbt**.

```python
portfolio = vbt.Portfolio.from_signals(
    close=timeframes['15T']['close'].loc[train_signals.index],
    entries=train_signals['entries'],
    exits=train_signals['exits'],
    fees=0.001,
    slippage=0.0005,
    freq='15T'
)
```

### 6. Performance Analysis
After backtesting, the script provides performance statistics and visualizes the results:

```python
print(portfolio.stats())
```

#### Equity Curve Plot:
```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio.value.index, y=portfolio.value,
                         mode='lines', name='Strategy'))
fig.add_trace(go.Scatter(x=timeframes['15T']['close'].loc[train_signals.index].index,
                         y=timeframes['15T']['close'].loc[train_signals.index],
                         mode='lines', name='BTC Price'))
fig.show()
```

#### Monthly Returns Plot:
```python
monthly_returns = portfolio.returns.resample('M').sum()
fig = go.Figure(go.Bar(x=monthly_returns.index, y=monthly_returns))
fig.show()
```

## Notes
- This script is **TA-Lib free** and relies solely on Pandas for technical indicators.
- Ensure your CSV file is formatted correctly with **timestamp and OHLCV** columns.
- The **training period is from 2012 to 2020**; adjust as needed.

## Disclaimer
This is a **backtesting script only** and should not be used for live trading. Use at your own risk.
