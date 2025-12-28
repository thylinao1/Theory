# Financial Trading in Python - Comprehensive Notes

## Table of Contents
1. [Introduction to Financial Trading](#1-introduction-to-financial-trading)
2. [Financial Time Series Data](#2-financial-time-series-data)
3. [Data Exploration and Transformation](#3-data-exploration-and-transformation)
4. [Backtesting with bt Package](#4-backtesting-with-bt-package)
5. [Technical Indicators](#5-technical-indicators)
6. [Trading Signals and Strategies](#6-trading-signals-and-strategies)
7. [Strategy Optimization and Benchmarking](#7-strategy-optimization-and-benchmarking)
8. [Performance Evaluation](#8-performance-evaluation)

---

## 1. Introduction to Financial Trading

### What is Financial Trading?

**Definition**: Financial trading is the buying and selling of financial assets (securities) with the goal of making a profit by taking calculated risks.

### Types of Financial Instruments

Traders work with various financial instruments including:

- **Equities**: Shares of stocks representing ownership in companies
- **Bonds**: Debt instruments issued by governments or corporations
- **Forex**: Foreign exchange market for trading currencies
- **Commodities**: Physical goods like gold, silver, and oil
- **Cryptocurrencies**: Digital assets like Bitcoin

### How Traders Make Profit

Traders can profit in two primary ways:

- **Going Long**: Buying a security at a lower price and selling later at a higher price
- **Going Short**: Selling a borrowed security at a higher price and buying it back at a lower price

### Types of Traders by Holding Period

Traders are categorized by how long they hold their positions:

- **Day Traders**: Hold positions throughout the day but not overnight; trade frequently
- **Swing Traders**: Hold positions from a few days to several weeks
- **Position Traders**: Hold positions from a few months to several years

### Trading vs. Investing

| Aspect | Trading | Investing |
|--------|---------|-----------|
| **Time Horizon** | Days to months | Years to decades |
| **Focus** | Short-term market trends, volatility | Market fundamentals, macroeconomic environment |
| **Positions** | Both long and short | Typically long only |
| **Goal** | Profit from price fluctuations | Ride big trends over years |

### Institutional vs. Retail Traders

- **Institutional Traders** (hedge funds, investment banks): Trade to hedge risks, provide market liquidity, or rebalance portfolios
- **Retail Traders**: Trade for their own accounts, sometimes as a side hustle

---

## 2. Financial Time Series Data

### What is Time Series Data?

**Definition**: A sequence of data points indexed in time order. For financial trading, this typically includes security prices over time.

### Components of Daily Price Data

- **Open**: Price at market open
- **High**: Highest price during the period
- **Low**: Lowest price during the period
- **Close**: Price at market close
- **Volume**: Number of shares/units traded

### Loading Time Series Data in Python

```python
import pandas as pd

# Load CSV data with Date as index
bitcoin_data = pd.read_csv('bitcoin_data.csv',
                           index_col='Date', 
                           parse_dates=True)

# Display top 5 rows
print(bitcoin_data.head())
```

**Key Parameters**:
- `index_col`: Specifies which column to use as the DataFrame index
- `parse_dates=True`: Parses the index in DateTime format

### Visualizing Time Series Data

#### Line Chart

```python
import matplotlib.pyplot as plt

# Plot daily high and low prices
plt.plot(bitcoin_data['High'], color='green')
plt.plot(bitcoin_data['Low'], color='red')
plt.title('Daily high low prices')
plt.show()
```

#### Candlestick Chart

**Definition**: A chart style that displays multiple pieces of price information (Open, High, Low, Close) in a single visualization.

**Components of a Candlestick**:
- **Body**: Shows open and close prices
- **Wick/Shadow**: Shows high and low prices
- **Color**: Green/white = close > open (bullish); Red/black = close < open (bearish)

```python
import plotly.graph_objects as go

# Define candlestick data
candlestick = go.Candlestick(
    x=bitcoin_data.index,
    open=bitcoin_data['Open'],
    high=bitcoin_data['High'],
    low=bitcoin_data['Low'],
    close=bitcoin_data['Close'])

# Create and display figure
fig = go.Figure(data=[candlestick])
fig.update_layout(title='Bitcoin prices')
fig.show()
```

---

## 3. Data Exploration and Transformation

### Resampling Data

**Definition**: Converting time series data from one frequency to another (e.g., hourly to daily).

Traders resample data to match their trading style. For example, a swing trader might prefer daily data over hourly.

```python
# Resample 4-hour data to weekly using mean
eurusd_weekly = eurusd_4h.resample('W').mean()

# Other options: 'D' for daily, 'M' for monthly
# Aggregation: .mean(), .min(), .max(), .sum()
```

### Calculating Returns

**Definition**: The percentage change in price, also known as price return.

```python
# Calculate daily returns (percentage change)
tsla_data['daily_return'] = tsla_data['Close'].pct_change() * 100

# Plot histogram of returns
tsla_data['daily_return'].hist(bins=100, color='red')
plt.ylabel('Frequency')
plt.xlabel('Daily return')
plt.title('Daily return histogram')
plt.show()
```

**Why Histograms Matter**: They reveal the distribution of returns, typical return ranges, and the volatility profile of an asset.

### Simple Moving Average (SMA)

**Definition**: The arithmetic mean of prices over a specified number of periods. Called "moving" because it's recalculated with each new data point using the most recent n periods.

**Purpose**: Creates a smoothing effect that helps identify price direction (upward, downward, or sideways).

```python
# Calculate 50-day SMA using pandas
aapl_data['sma_50'] = aapl_data['Close'].rolling(window=50).mean()

# Plot SMA with close price
plt.plot(aapl_data['sma_50'], color='green', label='SMA_50')
plt.plot(aapl_data['Close'], color='red', label='Close')
plt.legend()
plt.show()
```

---

## 4. Backtesting with bt Package

### What is Backtesting?

**Definition**: A method to assess the effectiveness of a trading strategy by testing it on historical data. The results help determine if a strategy would have been profitable in the past and whether it's viable for future trading.

### What is a Trading Strategy?

**Definition**: A method of buying and selling financial assets based on predefined rules, typically based on technical indicators and signals.

### The bt Package

**Definition**: A flexible Python framework for defining and backtesting trading strategies.

```python
import bt
```

### The 4-Step bt Process

1. **Obtain historical price data**
2. **Define the strategy**
3. **Backtest the strategy with historical data**
4. **Evaluate the results**

### Getting Price Data

```python
# Method 1: Load from CSV
price_data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

# Method 2: Download from Yahoo Finance
bt_data = bt.get('GOOG,AMZN,TSLA', start='2020-01-01', end='2020-06-30')
```

**Notes**:
- By default, `bt.get()` downloads "Adjusted Close" prices (adjusted for stock splits, dividends, etc.)
- Multiple tickers are separated by commas in a single string

### Defining a Strategy with Algos

**Definition**: `bt.Strategy` combines trading logic using individual "algos" - small task forces that perform specific operations.

```python
bt_strategy = bt.Strategy('Trade_Weekly',
    [bt.algos.RunWeekly(),      # When to execute trades
     bt.algos.SelectAll(),       # What data to apply (all securities)
     bt.algos.WeighEqually(),    # How to weight each asset
     bt.algos.Rebalance()])      # Rebalance to maintain weights
```

**Key Algos**:
- `RunWeekly()`, `RunMonthly()`, `RunOnce()`: Trade frequency
- `SelectAll()`: Apply strategy to all data
- `SelectWhere(condition)`: Apply strategy where condition is True
- `WeighEqually()`: Equal capital allocation to each asset
- `WeighTarget(signal)`: Use signal values for position sizing
- `Rebalance()`: Execute the rebalancing

### Running a Backtest

```python
# Create backtest combining strategy and data
bt_test = bt.Backtest(bt_strategy, bt_data)

# Run the backtest
bt_result = bt.run(bt_test)

# Plot results
bt_result.plot(title="Backtest result")
plt.show()

# View transaction details
bt_result.get_transactions()
```

---

## 5. Technical Indicators

### What are Technical Indicators?

**Definition**: Calculations based on historical market data (price, volume, etc.) used to gain insight into past price patterns and anticipate possible future price movements.

**Core Assumption**: Markets are efficient and prices have incorporated all public information.

### Types of Indicators

| Type | Purpose | Examples |
|------|---------|----------|
| **Trend** | Measure direction or strength of a trend | Moving Averages, ADX |
| **Momentum** | Measure velocity of price movement (rate of change) | RSI |
| **Volatility** | Measure magnitude of price deviations | Bollinger Bands |

### The TA-Lib Package

**Definition**: Technical Analysis Library - includes over 150 indicators for technical trading.

```python
import talib
```

### Trend Indicators: Moving Averages

#### Simple Moving Average (SMA)

**Definition**: Arithmetic mean of past n prices, giving equal weight to all data points.

```python
# Calculate SMA with TA-Lib
stock_data['SMA'] = talib.SMA(stock_data['Close'], timeperiod=50)
```

#### Exponential Moving Average (EMA)

**Definition**: Exponentially weighted average of past n prices, giving higher weight to more recent data points.

```python
# Calculate EMAs
stock_data['EMA_12'] = talib.EMA(stock_data['Close'], timeperiod=12)
stock_data['EMA_26'] = talib.EMA(stock_data['Close'], timeperiod=26)
```

#### SMA vs. EMA Comparison

| Aspect | SMA | EMA |
|--------|-----|-----|
| **Weighting** | Equal weight to all periods | Higher weight to recent data |
| **Responsiveness** | Slower to react | Faster to react to price changes |
| **Best For** | Longer-term analysis | Shorter-term, volatile markets |

**Plotting Moving Averages**:

```python
plt.plot(stock_data['SMA'], label='SMA')
plt.plot(stock_data['EMA'], label='EMA')
plt.plot(stock_data['Close'], label='Close')
plt.legend()
plt.title('SMA vs EMA')
plt.show()
```

### Trend Strength Indicator: ADX

**Definition**: Average Directional Movement Index (ADX) measures the strength of a trend (not its direction). Developed by J. Welles Wilder.

**Interpretation**:
- ADX < 25: No clear trend (sideways market)
- ADX > 25: Market is trending
- ADX > 50: Strong trending market

**ADX oscillates between 0 and 100.**

```python
# Calculate ADX (default 14-period)
stock_data['ADX_14'] = talib.ADX(stock_data['High'],
                                  stock_data['Low'],
                                  stock_data['Close'])

# Calculate with custom timeperiod
stock_data['ADX_21'] = talib.ADX(stock_data['High'],
                                  stock_data['Low'],
                                  stock_data['Close'],
                                  timeperiod=21)
```

**Plotting ADX with Price** (using subplots):

```python
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_ylabel('Price')
ax1.plot(stock_data['Close'])
ax2.set_ylabel('ADX')
ax2.plot(stock_data['ADX'], color='red')
ax1.set_title('Price and ADX')
plt.show()
```

### Momentum Indicator: RSI

**Definition**: Relative Strength Index (RSI) measures momentum - the speed of rising or falling prices. Also developed by J. Welles Wilder.

**Formula**:
$$RSI = 100 - \frac{100}{1 + RS}$$

Where RS = (Average of upward price changes) / (Average of downward price changes)

**Interpretation**:
- RSI > 70: **Overbought** - asset may be overvalued, price may reverse
- RSI < 30: **Oversold** - asset may be undervalued, price may rally

**RSI oscillates between 0 and 100.**

```python
# Calculate RSI (default 14-period)
stock_data['RSI_14'] = talib.RSI(stock_data['Close'])

# Custom timeperiod
stock_data['RSI_21'] = talib.RSI(stock_data['Close'], timeperiod=21)
```

**Plotting RSI with Price**:

```python
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_ylabel('Price')
ax1.plot(stock_data['Close'])
ax2.set_ylabel('RSI')
ax2.plot(stock_data['RSI'], color='orangered')
ax1.set_title('Price and RSI')
plt.show()
```

### Volatility Indicator: Bollinger Bands

**Definition**: Envelopes plotted above and below a simple moving average, designed to gauge price volatility (deviations from the mean). Developed by John Bollinger.

**Components**:
- **Middle Band**: n-period SMA (default n=20)
- **Upper Band**: Middle band + k standard deviations (default k=2)
- **Lower Band**: Middle band - k standard deviations (default k=2)

**Interpretation**:
- **Wider bands** = Higher volatility
- **Price near upper band** = Relatively expensive
- **Price near lower band** = Relatively cheap
- With 2 standard deviations, ~95% of price moves stay within bands

```python
# Calculate Bollinger Bands
upper_2sd, mid_2sd, lower_2sd = talib.BBANDS(
    bitcoin_data['Close'],
    nbdevup=2,      # Standard deviations for upper band
    nbdevdn=2,      # Standard deviations for lower band
    timeperiod=20)  # SMA period
```

**Plotting Bollinger Bands**:

```python
plt.plot(bitcoin_data['Close'], color='green', label='Price')
plt.plot(upper_2sd, color='orange', label='Upper 2sd')
plt.plot(lower_2sd, color='orange', label='Lower 2sd')
plt.legend(loc='upper left')
plt.title('Bollinger Bands (2sd)')
plt.show()
```

---

## 6. Trading Signals and Strategies

### What are Trading Signals?

**Definition**: Triggers to buy (long) or sell (short) financial assets based on predetermined criteria. They can be constructed using one or multiple technical indicators combined with market data.

**Purpose**: Used in algorithmic trading to make decisions based on quantitative rules, removing human discretion.

### Two Main Strategy Types

#### 1. Trend-Following (Momentum) Strategies

**Philosophy**: "The trend is your friend" - bet that price trends will continue in the same direction.

**Common Indicators**: Moving averages, ADX

**Example - MA Crossover Strategy**:
- **Long signal**: Short-term EMA crosses above long-term EMA (price gaining momentum)
- **Short signal**: Short-term EMA crosses below long-term EMA (price losing momentum)

#### 2. Mean Reversion Strategies

**Philosophy**: "Buy the fear, sell the greed" - when markets reach overbought/oversold conditions, prices tend to reverse back toward the mean.

**Common Indicators**: RSI, Bollinger Bands

**Example - RSI Strategy**:
- **Long signal**: RSI < 30 (oversold, price may rally)
- **Short signal**: RSI > 70 (overbought, price may reverse)

### Implementing Signals in bt

#### Method 1: SelectWhere (Price Comparison)

```python
# SMA-based signal: go long when price > SMA
sma = price_data.rolling(20).mean()

bt_strategy = bt.Strategy('AboveSMA',
    [bt.algos.SelectWhere(price_data > sma),
     bt.algos.WeighEqually(),
     bt.algos.Rebalance()])
```

#### Method 2: WeighTarget (Custom Signal DataFrame)

```python
# EMA crossover signal
signal = EMA_short.copy()
signal[EMA_short > EMA_long] = 1   # Long position
signal[EMA_short < EMA_long] = -1  # Short position

bt_strategy = bt.Strategy('EMA_crossover',
    [bt.algos.WeighTarget(signal),
     bt.algos.Rebalance()])
```

### Building a Trend-Following Strategy (Complete Example)

```python
# 1. Calculate indicators
EMA_short = talib.EMA(price_data['Close'], timeperiod=10).to_frame()
EMA_long = talib.EMA(price_data['Close'], timeperiod=40).to_frame()

# 2. Construct signal
signal = EMA_short.copy()
signal[:] = 0  # Initialize
signal[EMA_short > EMA_long] = 1   # Long when short EMA > long EMA
signal[EMA_short < EMA_long] = -1  # Short when short EMA < long EMA

# 3. Visualize signal with price
combined_df = bt.merge(signal, price_data, EMA_short, EMA_long)
combined_df.columns = ['signal', 'Price', 'EMA_short', 'EMA_long']
combined_df.plot(secondary_y=['signal'])
plt.show()

# 4. Define and run strategy
bt_strategy = bt.Strategy('EMA_crossover',
    [bt.algos.WeighTarget(signal),
     bt.algos.Rebalance()])

bt_backtest = bt.Backtest(bt_strategy, price_data)
bt_result = bt.run(bt_backtest)
bt_result.plot(title='Backtest result')
plt.show()
```

### Building a Mean Reversion Strategy (Complete Example)

```python
# 1. Calculate RSI indicator
stock_rsi = talib.RSI(price_data['Close']).to_frame()

# 2. Construct signal
signal = stock_rsi.copy()
signal[stock_rsi > 70] = -1   # Short when overbought
signal[stock_rsi < 30] = 1    # Long when oversold
signal[(stock_rsi <= 70) & (stock_rsi >= 30)] = 0  # No position

# 3. Define and run strategy
bt_strategy = bt.Strategy('RSI_MeanReversion',
    [bt.algos.WeighTarget(signal),
     bt.algos.Rebalance()])

bt_backtest = bt.Backtest(bt_strategy, price_data)
bt_result = bt.run(bt_backtest)
bt_result.plot(title='Backtest result')
plt.show()
```

### Important Simplifications in These Examples

1. **Single asset trading**: Multi-asset strategies require considering price correlations for proper position sizing
2. **No slippage**: Real trading has differences between expected and executed prices
3. **No commissions**: Broker fees affect profitability in real trading

---

## 7. Strategy Optimization and Benchmarking

### What is Strategy Optimization?

**Definition**: The process of testing a range of input parameter values to find the ones that give better strategy performance based on historical data.

**Example Question**: What SMA lookback period (10, 20, or 50) results in the most profitable strategy?

### Creating Reusable Strategy Functions

```python
def signal_strategy(price_data, period, name):
    """Create a signal-based strategy with configurable SMA period."""
    # Calculate SMA
    sma = price_data.rolling(period).mean()
    
    # Define the signal-based Strategy
    bt_strategy = bt.Strategy(name,
        [bt.algos.SelectWhere(price_data > sma),
         bt.algos.WeighEqually(),
         bt.algos.Rebalance()])
    
    # Return the backtest
    return bt.Backtest(bt_strategy, price_data)
```

### Running Multiple Backtests for Optimization

```python
# Create backtests with different parameters
sma10 = signal_strategy(price_data, period=10, name='SMA10')
sma30 = signal_strategy(price_data, period=30, name='SMA30')
sma50 = signal_strategy(price_data, period=50, name='SMA50')

# Run all backtests and plot together
bt_results = bt.run(sma10, sma30, sma50)
bt_results.plot(title='Strategy optimization')
plt.show()
```

### What is a Benchmark?

**Definition**: A standard or point of reference against which a strategy can be compared or assessed.

**Common Benchmarks**:
- **Buy and hold strategy**: Passive holding of the asset
- **S&P 500 Index**: For US equity strategies
- **US Treasuries**: For bond strategies

### Creating a Benchmark Strategy

```python
def buy_and_hold(price_data, name):
    """Create a passive buy-and-hold benchmark strategy."""
    bt_strategy = bt.Strategy(name,
        [bt.algos.RunOnce(),      # Execute only once at start
         bt.algos.SelectAll(),
         bt.algos.WeighEqually(),
         bt.algos.Rebalance()])
    return bt.Backtest(bt_strategy, price_data)

# Create and compare with benchmark
benchmark = buy_and_hold(price_data, name='benchmark')
bt_results = bt.run(sma10, sma30, sma50, benchmark)
bt_results.plot(title='Strategy benchmarking')
plt.show()
```

---

## 8. Performance Evaluation

### Obtaining Backtest Statistics

```python
# Get all backtest statistics as a DataFrame
resInfo = bt_result.stats
```

### Return Metrics

#### Basic Returns

**Definition**: The net gain or loss of a portfolio over a specified time period.

```python
# Daily, monthly, and yearly returns
print('Daily return: %.4f' % resInfo.loc['daily_mean'])
print('Monthly return: %.4f' % resInfo.loc['monthly_mean'])
print('Yearly return: %.4f' % resInfo.loc['yearly_mean'])
```

#### Compound Annual Growth Rate (CAGR)

**Definition**: The rate of return required for an asset to grow from its beginning balance to its ending balance, assuming all profits are reinvested at the end of each year. CAGR smooths returns when growth rates are volatile.

```python
print('CAGR: %.4f' % resInfo.loc['cagr'])
```

#### Return Histogram

```python
# Plot weekly return histogram
bt_result.plot_histograms(bins=50, freq='w')
plt.show()
```

#### Comparing Lookback Returns

```python
# Compare returns across multiple strategies
lookback_returns = bt_results.display_lookback_returns()
print(lookback_returns)
```

### Drawdown Metrics

#### What is Drawdown?

**Definition**: A peak-to-trough decline during a specific period, usually expressed as a percentage. Drawdowns measure downside volatility.

**Example**: If an account drops from $1,000 to $900 before recovering, that's a 10% drawdown.

#### Max Drawdown

**Definition**: The maximum observed loss from a peak to a trough before a new peak is established. This is a key indicator of downside risk.

**Example Calculation**: 
- Account peaks at $1,700 (point A)
- Account drops to $800 (point D) before reaching new high
- Max Drawdown = (1,700 - 800) / 1,700 = 52.9%

```python
# Get drawdown statistics
avg_drawdown = resInfo.loc['avg_drawdown']
print('Average drawdown: %.2f' % avg_drawdown)

avg_drawdown_days = resInfo.loc['avg_drawdown_days']
print('Average drawdown days: %.0f' % avg_drawdown_days)

max_drawdown = resInfo.loc['max_drawdown']
print('Maximum drawdown: %.2f' % max_drawdown)
```

### Risk-Adjusted Return Metrics

#### Why Risk-Adjusted Returns?

**Key Insight**: Higher returns aren't always better if they come with proportionally higher risk. Risk-adjusted metrics allow fair comparison between strategies with different risk profiles.

**Example**:
- Strategy 1: 15% return, 30% volatility
- Strategy 2: 10% return, 8% volatility
- Which is better? Strategy 2 has better risk-adjusted performance!

#### Calmar Ratio

**Definition**: A risk-adjusted return measure calculated as CAGR divided by maximum drawdown. Named after Terry Young's company "California Managed Accounts Report."

$$Calmar\ Ratio = \frac{CAGR}{Max\ Drawdown}$$

**Interpretation**: Higher is better. A Calmar ratio > 3 is considered excellent.

```python
# Calculate Calmar ratio manually
cagr = resInfo.loc['cagr']
max_drawdown = resInfo.loc['max_drawdown']
calmar_calc = cagr / max_drawdown * (-1)  # Multiply by -1 for positive result
print('Calmar Ratio: %.2f' % calmar_calc)

# Or get directly
calmar = resInfo.loc['calmar']
```

#### Sharpe Ratio

**Definition**: A risk-adjusted return measure developed by Nobel Laureate William F. Sharpe. Calculated as excess return (over risk-free rate) divided by return volatility (standard deviation).

$$Sharpe\ Ratio = \frac{Return - Risk\ Free\ Rate}{Standard\ Deviation}$$

**Interpretation**: Measures return per unit of total risk. Higher is better.

```python
# Get annual return and volatility
yearly_return = resInfo.loc['yearly_mean']
yearly_vol = resInfo.loc['yearly_vol']

# Calculate Sharpe ratio manually (assuming risk-free rate ≈ 0)
sharpe_ratio = yearly_return / yearly_vol

# Or get directly
print('Sharpe ratio: %.2f' % resInfo.loc['yearly_sharpe'])
```

**Limitation**: Sharpe ratio uses total volatility, treating upside and downside volatility equally. This can unfairly penalize strategies with high upside volatility.

#### Sortino Ratio

**Definition**: A modification of the Sharpe ratio that only penalizes downside volatility, not total volatility.

$$Sortino\ Ratio = \frac{Return - Risk\ Free\ Rate}{Downside\ Deviation}$$

**Interpretation**: Measures return per unit of downside risk. Higher is better. Often provides a more accurate picture of risk-reward efficiency than Sharpe ratio.

```python
# Get Sortino ratios
yearly_sortino = resInfo.loc['yearly_sortino']
monthly_sortino = resInfo.loc['monthly_sortino']
print('Annual Sortino ratio: %.2f' % yearly_sortino)
print('Monthly Sortino ratio: %.2f' % monthly_sortino)
```

### Summary of Key Performance Metrics

| Metric | Formula | What It Measures | Higher = ? |
|--------|---------|------------------|------------|
| **CAGR** | Smoothed annual growth rate | Compound returns | Better |
| **Max Drawdown** | Peak-to-trough decline | Downside risk | Worse |
| **Calmar Ratio** | CAGR / Max Drawdown | Return per drawdown risk | Better |
| **Sharpe Ratio** | Excess Return / Total Volatility | Return per total risk | Better |
| **Sortino Ratio** | Excess Return / Downside Volatility | Return per downside risk | Better |

---

## Quick Reference: Python Packages

| Package | Purpose | Import |
|---------|---------|--------|
| **pandas** | Data manipulation and analysis | `import pandas as pd` |
| **matplotlib** | Data visualization | `import matplotlib.pyplot as plt` |
| **plotly** | Interactive charts | `import plotly.graph_objects as go` |
| **bt** | Backtesting framework | `import bt` |
| **talib** | Technical indicators | `import talib` |

---

## Quick Reference: Key bt Algos

| Algo | Purpose |
|------|---------|
| `RunWeekly()`, `RunMonthly()`, `RunDaily()` | Trade frequency |
| `RunOnce()` | Execute once (buy-and-hold) |
| `SelectAll()` | Apply to all securities |
| `SelectWhere(condition)` | Filter by Boolean condition |
| `WeighEqually()` | Equal weight allocation |
| `WeighTarget(signal)` | Use signal for position sizing |
| `Rebalance()` | Execute portfolio rebalancing |

---

## Quick Reference: Key talib Functions

| Function | Indicator Type | Usage |
|----------|---------------|-------|
| `talib.SMA(close, timeperiod)` | Trend | Simple Moving Average |
| `talib.EMA(close, timeperiod)` | Trend | Exponential Moving Average |
| `talib.ADX(high, low, close, timeperiod)` | Trend Strength | Average Directional Index |
| `talib.RSI(close, timeperiod)` | Momentum | Relative Strength Index |
| `talib.BBANDS(close, timeperiod, nbdevup, nbdevdn)` | Volatility | Bollinger Bands |

---

*Notes compiled from DataCamp's "Financial Trading in Python" course*
