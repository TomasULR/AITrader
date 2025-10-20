# AI Trading Signal Generator

An advanced machine learning system that analyzes NASDAQ market data and generates real-time buy/sell signals with stop loss, take profit, and probability estimates.

## 🆕 NEW: Continuous Learning

The system now **learns from its predictions in real-time**:
- ✅ Tracks every prediction and its actual outcome
- ✅ Records accuracy metrics continuously
- ✅ Automatically retrains the model when enough new data is collected
- ✅ Adapts to changing market conditions
- ✅ Shows live learning metrics (accuracy, model version, etc.)

## Features

- 📊 **Real-time Data Fetching**: Automatically fetches 5-minute candle data from NASDAQ
- 🤖 **Machine Learning Predictions**: Uses Random Forest classifier with 30+ technical indicators
- 💰 **Complete Trading Signals**: Generates signals with:
  - Entry price
  - Stop loss levels
  - Take profit targets
  - Success probability
  - Risk/reward ratio
  - Signal strength rating
- 📈 **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, and more
- ⚡ **Live Monitoring**: Continuous real-time signal generation
- 🧠 **Continuous Learning**: Automatically learns from predictions and retrains model
- 📊 **Performance Tracking**: Real-time accuracy metrics and model versioning
- 💾 **Persistent Learning**: Stores all predictions and outcomes for long-term improvement

## How Continuous Learning Works

1. **Prediction Tracking**: Every time the model makes a prediction, it's recorded with timestamp and confidence
2. **Outcome Recording**: After 10 minutes (2 candles), the system checks if the prediction was correct
3. **Accuracy Monitoring**: Tracks overall accuracy and recent performance (last 50/100 predictions)
4. **Automatic Retraining**: When 50+ new outcomes are recorded, the model automatically retrains with fresh data
5. **Model Versioning**: Each retrain creates a new model version, tracking improvement over time

## Installation

### Step 1: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### Step 2: Install TA-Lib (Windows)

TA-Lib requires special installation on Windows:

1. Download the appropriate wheel file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   - For Python 3.11 64-bit: `TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl`
   - For Python 3.10 64-bit: `TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl`

2. Install the wheel:
```powershell
pip install path\to\downloaded\TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl
```

Alternatively, you can try:
```powershell
pip install TA-Lib-binary
```

## Usage

### 1. Train the Model

First, train the model with historical data:

```powershell
python train_model.py
```

This will:
- Fetch 60 days of historical 5-minute candle data
- Calculate 30+ technical indicators
- Train a Random Forest model
- Display training accuracy and feature importance
- Save the model to `trading_model.pkl`

### 2. Run Live Trading Signals

Once trained, start the live signal generator:

```powershell
python live_trader.py
```

This will:
- Monitor the market every 60 seconds
- Generate predictions for the next candle
- Display BUY/SELL signals with full trading parameters
- Show probability of success and signal strength
- **Track predictions and learn from outcomes**
- **Automatically retrain when enough data is collected**
- **Display live learning metrics**

### 3. View Learning Metrics

The system automatically displays metrics every 50 iterations and when stopped:

```
📊 LEARNING METRICS SUMMARY
======================================================================
Total Predictions: 145
Completed: 120 | Pending: 25
Correct: 68
Overall Accuracy: 56.67%
Recent Accuracy (50): 58.00%
Recent Accuracy (100): 57.50%
Model Version: v2
Retrain Count: 1
Last Retrain: 2025-10-20 15:30:45
======================================================================
```

## Configuration

### Change the Stock Symbol

Edit `train_model.py` or `live_trader.py`:

```python
SYMBOL = 'QQQ'   # NASDAQ-100 ETF
# Or use any other stock:
# SYMBOL = 'AAPL'  # Apple
# SYMBOL = 'MSFT'  # Microsoft
# SYMBOL = 'GOOGL' # Google
# SYMBOL = 'TSLA'  # Tesla
```

### Adjust Risk Parameters

Edit `live_trader.py`:

```python
self.signal_gen = SignalGenerator(
    risk_reward_ratio=2.0,      # Minimum R:R ratio
    atr_multiplier_sl=1.5,      # Stop loss distance (ATR multiplier)
    atr_multiplier_tp=3.0       # Take profit distance (ATR multiplier)
)
```

### Change Check Frequency

Edit `live_trader.py`:

```python
CHECK_INTERVAL = 60  # Check every 60 seconds
```

### Enable/Disable Continuous Learning

Edit `live_trader.py`:

```python
LEARNING_ENABLED = True  # Set to False to disable learning
```

### Adjust Retraining Parameters

Edit `live_trader.py` in the `check_and_retrain` method:

```python
self.tracker.should_retrain(
    min_new_samples=50,          # Minimum outcomes needed before retrain
    max_hours_since_retrain=24   # Maximum hours between retrains
)
```

## Example Output

```
======================================================================
🚨 TRADING SIGNAL GENERATED
======================================================================
Timestamp: 2025-10-20 14:35:00
Direction: BUY
Entry Price: $385.50

📊 PROBABILITIES:
  Success Probability: 68.50%
  Up Probability: 68.50%
  Down Probability: 31.50%
  Signal Strength: MODERATE

💰 TRADE PARAMETERS:
  Stop Loss: $383.25
  Take Profit: $389.75
  Risk Amount: $2.25
  Reward Amount: $4.25
  Risk/Reward Ratio: 1.89

✅ RECOMMENDED ACTION: TAKE THIS TRADE
======================================================================
```

## Project Structure

```
AITrader/
│
├── data_fetcher.py          # Fetches real-time market data
├── feature_engineering.py   # Calculates technical indicators
├── ml_model.py              # Machine learning model with incremental learning
├── signal_generator.py      # Generates trading signals
├── learning_tracker.py      # NEW: Tracks predictions and outcomes
├── train_model.py           # Training script
├── live_trader.py           # Live monitoring with continuous learning
├── learning_data.csv        # NEW: Stores predictions and outcomes
├── model_metrics.json       # NEW: Stores performance metrics
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Learning Data Files

- **`learning_data.csv`**: Contains all predictions with timestamps, entry prices, probabilities, actual outcomes, and correctness
- **`model_metrics.json`**: Stores overall metrics including total predictions, accuracy, retrain count, and model version

## Technical Indicators Used

- **Trend**: SMA (10, 20, 50), EMA (10, 20)
- **Momentum**: RSI, MACD, ROC, Momentum, Williams %R
- **Volatility**: Bollinger Bands, ATR
- **Strength**: ADX, CCI
- **Oscillators**: Stochastic (K, D)
- **Volume**: Volume Ratio

## Risk Disclaimer

⚠️ **IMPORTANT**: This is an educational project for learning purposes only.

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- This system does not guarantee profits
- Always do your own research and consult with a financial advisor
- Use at your own risk

## Notes

- The model requires market hours to fetch live data
- Signals are based on historical patterns and may not predict future movements
- Always use proper risk management
- Consider paper trading first before using real money
- The model should be retrained regularly with fresh data

## Troubleshooting

### "Model file not found" error
Run `python train_model.py` first to train the model.

### No data received
Check if the market is open and the symbol is valid.

### Import errors
Make sure all packages are installed: `pip install -r requirements.txt`

### TA-Lib installation issues
Follow the Windows-specific installation instructions above.

## Future Enhancements

- Add more sophisticated ML models (LSTM, XGBoost)
- Implement backtesting functionality
- Add multi-timeframe analysis
- Include sentiment analysis
- Add automated trade execution
- Create web dashboard for visualization

## License

MIT License - Feel free to use and modify for your own learning and trading.
