# Install required libraries
!pip install vectorbt plotly

import numpy as np
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from google.colab import files

# Upload your 1-minute BTC data file
uploaded = files.upload()

# Load data (replace 'btc_1m.csv' with your filename)
df = pd.read_csv('btc_1m.csv', parse_dates=['timestamp'], index_col='timestamp')

# ======================
# DATA PREPROCESSING
# ======================

def resample_data(df):
    """Resample 1m data to multiple timeframes"""
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    timeframes = {
        '15T': df.resample('15T').agg(ohlc_dict).dropna(),
        '1H': df.resample('1H').agg(ohlc_dict).dropna(),
        '4H': df.resample('4H').agg(ohlc_dict).dropna(),
        '1D': df.resample('1D').agg(ohlc_dict).dropna()
    }
    return timeframes

# Resample data
timeframes = resample_data(df)

# ======================
# INDICATOR CALCULATION
# ======================

def calculate_indicators(tf_dict):
    """Calculate all required indicators for each timeframe"""
    # 4H Timeframe
    tf_dict['4H']['ema21'] = tf_dict['4H']['close'].ewm(span=21, adjust=False).mean()
    tf_dict['4H']['ema55'] = tf_dict['4H']['close'].ewm(span=55, adjust=False).mean()
    
    # 1D Timeframe
    tf_dict['1D']['ema50'] = tf_dict['1D']['close'].ewm(span=50, adjust=False).mean()
    tf_dict['1D']['ema200'] = tf_dict['1D']['close'].ewm(span=200, adjust=False).mean()
    
    # 1H Timeframe: Calculate ATR
    high_1h = tf_dict['1H']['high']
    low_1h = tf_dict['1H']['low']
    close_1h = tf_dict['1H']['close']
    
    # True Range Calculation
    tr = pd.DataFrame({
        'h_l': high_1h - low_1h,
        'h_pc': (high_1h - close_1h.shift(1)).abs(),
        'l_pc': (low_1h - close_1h.shift(1)).abs()
    }).max(axis=1)
    
    # ATR (14-period EMA of TR)
    tf_dict['1H']['atr'] = tr.ewm(span=14, adjust=False).mean()
    
    # RSI Calculation
    delta = close_1h.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    tf_dict['1H']['rsi'] = 100 - (100 / (1 + rs))
    
    # Dynamic RSI thresholds
    atr_pct = tf_dict['1H']['atr'] / tf_dict['1H']['close']
    tf_dict['1H']['rsi_low'] = 30 + (atr_pct * 10 * 100)
    tf_dict['1H']['rsi_high'] = 70 - (atr_pct * 10 * 100)
    
    # 15m Timeframe: Bollinger Bands
    close_15T = tf_dict['15T']['close']
    tf_dict['15T']['middle_band'] = close_15T.rolling(20).mean()
    std = close_15T.rolling(20).std()
    tf_dict['15T']['upper_band'] = tf_dict['15T']['middle_band'] + 2 * std
    tf_dict['15T']['lower_band'] = tf_dict['15T']['middle_band'] - 2 * std
    tf_dict['15T']['bandwidth'] = (tf_dict['15T']['upper_band'] - tf_dict['15T']['lower_band']) / tf_dict['15T']['middle_band']
    
    return tf_dict

timeframes = calculate_indicators(timeframes)

# ======================
# SIGNAL GENERATION
# ======================

def generate_signals(tf_dict):
    """Generate trading signals at 15-minute resolution"""
    # Align all timeframes to 15m index
    aligned_signals = pd.DataFrame(index=tf_dict['15T'].index)
    
    # Trend Filters
    aligned_signals['4h_bullish'] = tf_dict['4H']['ema21'] > tf_dict['4H']['ema55']
    aligned_signals['1d_bullish'] = tf_dict['1D']['ema50'] > tf_dict['1D']['ema200']
    
    # Entry Triggers
    aligned_signals['1h_rsi_buy'] = tf_dict['1H']['rsi'] < tf_dict['1H']['rsi_low']
    aligned_signals['1h_rsi_sell'] = tf_dict['1H']['rsi'] > tf_dict['1H']['rsi_high']
    
    # Confirmation Filter
    aligned_signals['15m_squeeze'] = tf_dict['15T']['bandwidth'] < 0.015  # 1.5%
    
    # Forward fill higher timeframe signals
    aligned_signals[['4h_bullish', '1d_bullish']] = aligned_signals[['4h_bullish', '1d_bullish']].ffill()
    
    # Generate final signals
    aligned_signals['entries'] = (
        aligned_signals['4h_bullish'] & 
        aligned_signals['1d_bullish'] & 
        aligned_signals['1h_rsi_buy'] & 
        aligned_signals['15m_squeeze']
    )
    
    aligned_signals['exits'] = (
        (~aligned_signals['4h_bullish'] | ~aligned_signals['1d_bullish']) &
        aligned_signals['1h_rsi_sell']
    )
    
    return aligned_signals

signals = generate_signals(timeframes)

# ======================
# BACKTESTING (2012-2020)
# ======================

# Filter for training period
train_signals = signals.loc['2012-01-01':'2020-12-31']

# Create portfolio
portfolio = vbt.Portfolio.from_signals(
    close=timeframes['15T']['close'].loc[train_signals.index],
    entries=train_signals['entries'],
    exits=train_signals['exits'],
    fees=0.001,  # 0.1% per trade
    slippage=0.0005,  # 0.05%
    freq='15T'
)

# ======================
# PERFORMANCE ANALYSIS
# ======================

print("===== Strategy Performance (2012-2020) =====")
print(portfolio.stats())

# Plot equity curve
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio.value.index, y=portfolio.value,
                         mode='lines', name='Strategy'))
fig.add_trace(go.Scatter(x=timeframes['15T']['close'].loc[train_signals.index].index,
                         y=timeframes['15T']['close'].loc[train_signals.index],
                         mode='lines', name='BTC Price'))
fig.update_layout(title='MTAM Strategy vs BTC Price (Training Period)',
                  yaxis_title='Value', template='plotly_dark')
fig.show()

# Plot monthly returns
monthly_returns = portfolio.returns.resample('M').sum()
fig = go.Figure(go.Bar(x=monthly_returns.index, y=monthly_returns))
fig.update_layout(title='Monthly Returns', yaxis_title='Return', template='plotly_dark')
fig.show()
