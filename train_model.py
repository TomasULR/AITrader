"""
Train Model Script
Fetches historical data and trains the ML model
"""

from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer
from ml_model import TradingModel
import pandas as pd

def train_trading_model(symbol='QQQ', period='60d', interval='5m'):
    """
    Train the trading model with historical data
    
    Args:
        symbol: Stock symbol to train on
        period: Historical period to fetch
        interval: Time interval
    """
    print("=" * 70)
    print("AI TRADING MODEL TRAINING")
    print("=" * 70)
    
    # Step 1: Fetch data
    print(f"\n1. Fetching historical data for {symbol}...")
    fetcher = DataFetcher(symbol=symbol, interval=interval)
    df = fetcher.fetch_live_data(period=period)
    
    if df is None or df.empty:
        print("Error: Could not fetch data!")
        return None
    
    print(f"   Retrieved {len(df)} candles")
    
    # Step 2: Engineer features
    print("\n2. Engineering features and indicators...")
    engineer = FeatureEngineer()
    df = engineer.add_technical_indicators(df)
    print(f"   Added technical indicators")
    
    # Step 3: Create target variable
    print("\n3. Creating target variable...")
    df = engineer.create_target(df, prediction_horizon=1)
    print(f"   Created target for next candle prediction")
    print(f"   Final dataset size: {len(df)} samples")
    
    # Step 4: Prepare features
    feature_cols = engineer.prepare_features(df)
    print(f"\n4. Prepared {len(feature_cols)} features")
    
    # Step 5: Train model
    print("\n5. Training machine learning model...")
    model = TradingModel(model_type='random_forest')
    metrics = model.train(df, feature_cols)
    
    # Step 6: Show feature importance
    print("\n6. Top 10 Most Important Features:")
    importance_df = model.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))
    
    # Step 7: Save model
    print("\n7. Saving model...")
    model.save_model('trading_model.pkl')
    
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return model

if __name__ == "__main__":
    # Train the model
    # You can change the symbol to any NASDAQ stock
    # Popular options: QQQ (NASDAQ ETF), AAPL, MSFT, GOOGL, TSLA, etc.
    
    print("Starting model training...")
    print("Note: This may take a few minutes...\n")
    
    model = train_trading_model(
        symbol='QQQ',      # NASDAQ-100 ETF
        period='60d',      # 60 days of historical data
        interval='5m'      # 5-minute candles
    )
    
    if model:
        print("\n✅ You can now run 'python live_trader.py' to start live trading signals!")
