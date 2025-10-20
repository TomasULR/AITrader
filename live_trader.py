"""
Live Trading Signal Generator with Continuous Learning
Continuously monitors the market, generates real-time signals, and learns from outcomes
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer
from ml_model import TradingModel
from signal_generator import SignalGenerator
from learning_tracker import LearningTracker
import os

class LiveTrader:
    def __init__(self, symbol='QQQ', interval='5m', check_interval=60, learning_enabled=True):
        """
        Initialize live trader with continuous learning
        
        Args:
            symbol: Stock symbol to trade
            interval: Candle interval
            check_interval: How often to check for new signals (seconds)
            learning_enabled: Enable continuous learning
        """
        self.symbol = symbol
        self.interval = interval
        self.check_interval = check_interval
        self.learning_enabled = learning_enabled
        
        # Initialize components
        self.fetcher = DataFetcher(symbol=symbol, interval=interval)
        self.engineer = FeatureEngineer()
        self.model = TradingModel()
        self.signal_gen = SignalGenerator(
            risk_reward_ratio=2.0,
            atr_multiplier_sl=1.5,
            atr_multiplier_tp=3.0
        )
        
        # Initialize learning tracker
        self.tracker = LearningTracker()
        
        self.last_signal_time = None
        self.active_predictions = {}  # Track active predictions
        self.last_outcome_check = datetime.now()
        self.last_retrain_check = datetime.now()
        
    def load_model(self, model_path='trading_model.pkl'):
        """Load the trained model"""
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found: {model_path}")
            print("Please run 'python train_model.py' first to train the model.")
            return False
        
        self.model.load_model(model_path)
        return True
    
    def get_latest_features(self):
        """
        Fetch latest data and compute features
        
        Returns:
            DataFrame with features for latest candle
        """
        # Fetch recent data
        df = self.fetcher.fetch_live_data(period='5d')
        
        if df is None or df.empty:
            return None
        
        # Add technical indicators
        df = self.engineer.add_technical_indicators(df)
        
        if df.empty:
            return None
        
        # Get latest candle
        latest = df.iloc[-1:]
        
        return latest
    
    def generate_live_signal(self):
        """
        Generate a live trading signal and record it for learning
        
        Returns:
            Signal dictionary or None
        """
        # Get latest features
        latest_features = self.get_latest_features()
        
        if latest_features is None:
            print("⚠️  Could not fetch latest data")
            return None
        
        # Get current price
        current_price = latest_features['Close'].values[0]
        atr = latest_features['ATR_14'].values[0]
        
        # Make prediction
        try:
            prediction = self.model.predict(latest_features)
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
        
        # Generate signal
        signal = self.signal_gen.generate_signal(
            current_price=current_price,
            prediction_result=prediction,
            atr=atr
        )
        
        # Record prediction for learning
        if self.learning_enabled:
            pred_id = self.tracker.record_prediction(
                symbol=self.symbol,
                entry_price=current_price,
                prediction=prediction['prediction'],
                probability=prediction['probability']
            )
            # Store for later outcome checking
            self.active_predictions[pred_id] = {
                'timestamp': datetime.now(),
                'entry_price': current_price,
                'features': latest_features
            }
        
        return signal
    
    def check_prediction_outcomes(self):
        """
        Check outcomes of previous predictions for learning
        """
        if not self.learning_enabled:
            return
        
        # Get unrecorded predictions (at least 10 minutes old - 2 candles)
        unrecorded = self.tracker.get_unrecorded_predictions(max_age_minutes=10)
        
        if len(unrecorded) == 0:
            return
        
        # Get current price
        current_price = self.fetcher.get_current_price()
        if current_price is None:
            return
        
        # Record outcomes
        for idx, row in unrecorded.iterrows():
            self.tracker.record_outcome(idx, current_price)
        
        if len(unrecorded) > 0:
            print(f"📝 Recorded {len(unrecorded)} prediction outcomes for learning")
    
    def check_and_retrain(self):
        """
        Check if model should be retrained and do so if needed
        """
        if not self.learning_enabled:
            return
        
        # Check if retraining is needed
        if not self.tracker.should_retrain(min_new_samples=50, max_hours_since_retrain=24):
            return
        
        print("\n" + "=" * 70)
        print("🔄 AUTOMATIC RETRAINING INITIATED")
        print("=" * 70)
        
        try:
            # Fetch fresh historical data
            print("Fetching updated historical data...")
            df = self.fetcher.fetch_live_data(period='60d')
            
            if df is None or df.empty:
                print("❌ Could not fetch data for retraining")
                return
            
            # Add features
            print("Engineering features...")
            df = self.engineer.add_technical_indicators(df)
            df = self.engineer.create_target(df, prediction_horizon=1)
            
            feature_cols = self.engineer.prepare_features(df)
            
            # Retrain model
            print("Retraining model with fresh data...")
            metrics = self.model.train(df, feature_cols)
            
            # Save updated model
            self.model.save_model('trading_model.pkl')
            
            # Mark as retrained
            self.tracker.mark_retrained()
            
            print("=" * 70)
            print("✅ AUTOMATIC RETRAINING COMPLETED")
            print(f"New Model Version: v{self.tracker.metrics['model_version']}")
            print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"Testing Accuracy: {metrics['test_accuracy']:.4f}")
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"❌ Error during retraining: {str(e)}")
    
    def show_learning_metrics(self):
        """Display current learning metrics"""
        if self.learning_enabled:
            self.tracker.print_metrics()
    
    def run(self):
        """
        Run the live trading signal generator with continuous learning
        """
        print("=" * 70)
        print("🤖 AI TRADING SIGNAL GENERATOR - LIVE MODE WITH LEARNING")
        print("=" * 70)
        print(f"Symbol: {self.symbol}")
        print(f"Interval: {self.interval}")
        print(f"Check Frequency: Every {self.check_interval} seconds")
        print(f"Continuous Learning: {'✅ ENABLED' if self.learning_enabled else '❌ DISABLED'}")
        print("=" * 70)
        
        # Show initial metrics
        if self.learning_enabled:
            self.show_learning_metrics()
        
        print("\n⏳ Starting in 3 seconds...")
        time.sleep(3)
        
        print("\n✅ MONITORING STARTED - Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                try:
                    iteration += 1
                    
                    # Generate signal
                    signal = self.generate_live_signal()
                    
                    if signal:
                        # Only show buy signals or strong signals
                        if signal['trade_signal'] and signal['direction'] == 'BUY':
                            output = self.signal_gen.format_signal_output(signal)
                            print(output)
                            self.last_signal_time = datetime.now()
                        else:
                            # Show status update every check
                            now = datetime.now()
                            learning_status = f"[v{self.tracker.metrics['model_version']}]" if self.learning_enabled else ""
                            print(f"[{now.strftime('%H:%M:%S')}] {learning_status} 📊 Monitoring {self.symbol} - "
                                  f"Price: ${signal['price']:.2f} - "
                                  f"Prediction: {signal['direction']} "
                                  f"({signal['probability']*100:.1f}% confidence) - "
                                  f"No strong signal yet...")
                    
                    # Check prediction outcomes every 5 minutes
                    if self.learning_enabled and (datetime.now() - self.last_outcome_check).seconds >= 300:
                        self.check_prediction_outcomes()
                        self.last_outcome_check = datetime.now()
                    
                    # Check if retraining needed every 30 minutes
                    if self.learning_enabled and (datetime.now() - self.last_retrain_check).seconds >= 1800:
                        self.check_and_retrain()
                        self.last_retrain_check = datetime.now()
                    
                    # Show metrics every 50 iterations
                    if self.learning_enabled and iteration % 50 == 0:
                        self.show_learning_metrics()
                    
                except Exception as e:
                    print(f"⚠️  Error: {str(e)}")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("🛑 MONITORING STOPPED")
            if self.learning_enabled:
                print("\nFinal Learning Metrics:")
                self.show_learning_metrics()
            print("=" * 70)

def main():
    """Main function"""
    # Configuration
    SYMBOL = 'QQQ'           # Change to any NASDAQ stock (AAPL, MSFT, GOOGL, etc.)
    INTERVAL = '5m'          # 5-minute candles
    CHECK_INTERVAL = 60      # Check every 60 seconds
    LEARNING_ENABLED = True  # Enable continuous learning
    
    # Create trader with learning enabled
    trader = LiveTrader(
        symbol=SYMBOL,
        interval=INTERVAL,
        check_interval=CHECK_INTERVAL,
        learning_enabled=LEARNING_ENABLED
    )
    
    # Load trained model
    if not trader.load_model('trading_model.pkl'):
        return
    
    print("✅ Model loaded successfully!")
    if LEARNING_ENABLED:
        print("🧠 Continuous learning is ENABLED")
        print("   - Predictions are tracked automatically")
        print("   - Outcomes are recorded after 10 minutes")
        print("   - Model retrains automatically with new data")
    print()
    
    # Run live monitoring
    trader.run()

if __name__ == "__main__":
    main()
