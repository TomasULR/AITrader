"""
Data Fetcher Module
Fetches real-time NASDAQ data with 5-minute intervals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class DataFetcher:
    def __init__(self, symbol='QQQ', interval='5m'):
        """
        Initialize data fetcher
        
        Args:
            symbol: Stock symbol (default QQQ for NASDAQ-100 ETF)
            interval: Time interval (1m, 5m, 15m, 30m, 1h, etc.)
        """
        self.symbol = symbol
        self.interval = interval
        
    def fetch_live_data(self, period='5d'):
        """
        Fetch live market data
        
        Args:
            period: Period to fetch (1d, 5d, 1mo, 3mo, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=self.interval)
            
            if df.empty:
                print(f"Warning: No data received for {self.symbol}")
                return None
                
            # Clean the data
            df = df.dropna()
            
            # Reset index to make datetime a column
            df.reset_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
    
    def get_latest_candle(self):
        """
        Get the most recent completed candle
        
        Returns:
            Series with latest candle data
        """
        df = self.fetch_live_data(period='1d')
        if df is not None and not df.empty:
            return df.iloc[-1]
        return None
    
    def get_current_price(self):
        """
        Get current market price
        
        Returns:
            Current price as float
        """
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return data['Close'].iloc[-1]
        except Exception as e:
            print(f"Error getting current price: {str(e)}")
        return None
