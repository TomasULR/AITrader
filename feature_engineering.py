"""
Feature Engineering Module
Creates technical indicators and features for ML model
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # ATR (Average True Range)
        df['ATR_14'] = self._calculate_atr(df, 14)
        
        # ADX (Average Directional Index)
        df['ADX'] = self._calculate_adx(df, 14)
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        df['Price_Change_10'] = df['Close'].pct_change(periods=10)
        
        # Momentum
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # CCI (Commodity Channel Index)
        df['CCI'] = self._calculate_cci(df, 14)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        # Calculate +DM and -DM
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        atr = self._calculate_atr(df, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_cci(self, df, period=14):
        """Calculate Commodity Channel Index"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci
    
    def create_target(self, df, prediction_horizon=1):
        """
        Create target variable (future price movement)
        
        Args:
            df: DataFrame with features
            prediction_horizon: Number of candles ahead to predict
            
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        
        # Future close price
        df['Future_Close'] = df['Close'].shift(-prediction_horizon)
        
        # Price change percentage
        df['Target_Return'] = (df['Future_Close'] - df['Close']) / df['Close'] * 100
        
        # Binary classification: 1 if price goes up, 0 if down
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
        
        # Remove rows where we don't have future data
        df = df[:-prediction_horizon]
        
        return df
    
    def prepare_features(self, df):
        """
        Select and prepare final feature set
        
        Args:
            df: DataFrame with all indicators
            
        Returns:
            Feature columns list
        """
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20',
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
            'Stoch_K', 'Stoch_D', 'ATR_14', 'ADX',
            'Volume_Ratio', 'Price_Change', 'Price_Change_5', 'Price_Change_10',
            'ROC', 'Momentum', 'Williams_R', 'CCI'
        ]
        
        return feature_cols
