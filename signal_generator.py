"""
Signal Generator Module
Generates buy/sell signals with stop loss, take profit, and probability
"""

import numpy as np
import pandas as pd

class SignalGenerator:
    def __init__(self, risk_reward_ratio=2.0, atr_multiplier_sl=1.5, atr_multiplier_tp=3.0):
        """
        Initialize signal generator
        
        Args:
            risk_reward_ratio: Risk/reward ratio for TP calculation
            atr_multiplier_sl: ATR multiplier for stop loss
            atr_multiplier_tp: ATR multiplier for take profit
        """
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        
    def generate_signal(self, current_price, prediction_result, atr, additional_features=None):
        """
        Generate trading signal with SL and TP
        
        Args:
            current_price: Current market price
            prediction_result: Dictionary with prediction and probability
            atr: Average True Range value
            additional_features: Additional market features (optional)
            
        Returns:
            Dictionary with complete signal information
        """
        signal = {
            'timestamp': pd.Timestamp.now(),
            'price': current_price,
            'direction': 'BUY' if prediction_result['prediction'] == 1 else 'SELL',
            'probability': prediction_result['probability'],
            'prob_up': prediction_result['prob_up'],
            'prob_down': prediction_result['prob_down'],
        }
        
        # Calculate stop loss and take profit based on ATR
        if prediction_result['prediction'] == 1:  # BUY signal
            stop_loss = current_price - (atr * self.atr_multiplier_sl)
            take_profit = current_price + (atr * self.atr_multiplier_tp)
        else:  # SELL signal
            stop_loss = current_price + (atr * self.atr_multiplier_sl)
            take_profit = current_price - (atr * self.atr_multiplier_tp)
        
        signal['stop_loss'] = round(stop_loss, 2)
        signal['take_profit'] = round(take_profit, 2)
        
        # Calculate risk and reward
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        signal['risk_amount'] = round(risk, 2)
        signal['reward_amount'] = round(reward, 2)
        signal['risk_reward_ratio'] = round(reward / risk, 2) if risk > 0 else 0
        
        # Calculate position size based on risk (example: 1% risk)
        # This can be customized based on account size
        signal['suggested_position_size'] = 'Calculate based on account size and risk %'
        
        # Signal strength
        signal['strength'] = self._calculate_signal_strength(prediction_result['probability'])
        
        # Should we trade this signal?
        signal['trade_signal'] = self._should_trade(prediction_result, signal)
        
        return signal
    
    def _calculate_signal_strength(self, probability):
        """
        Calculate signal strength based on probability
        
        Args:
            probability: Prediction probability
            
        Returns:
            String indicating signal strength
        """
        if probability >= 0.70:
            return "STRONG"
        elif probability >= 0.60:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _should_trade(self, prediction_result, signal, min_probability=0.55):
        """
        Determine if we should trade this signal
        
        Args:
            prediction_result: Prediction results
            signal: Generated signal
            min_probability: Minimum probability threshold
            
        Returns:
            Boolean indicating whether to trade
        """
        # Only trade if probability is above threshold
        if prediction_result['probability'] < min_probability:
            return False
        
        # Only trade if risk/reward ratio is favorable
        if signal['risk_reward_ratio'] < 1.5:
            return False
        
        return True
    
    def format_signal_output(self, signal):
        """
        Format signal for console output
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Formatted string
        """
        border = "=" * 70
        
        output = f"\n{border}\n"
        output += f"🚨 TRADING SIGNAL GENERATED\n"
        output += f"{border}\n"
        output += f"Timestamp: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"Direction: {signal['direction']}\n"
        output += f"Entry Price: ${signal['price']:.2f}\n"
        output += f"\n"
        output += f"📊 PROBABILITIES:\n"
        output += f"  Success Probability: {signal['probability']*100:.2f}%\n"
        output += f"  Up Probability: {signal['prob_up']*100:.2f}%\n"
        output += f"  Down Probability: {signal['prob_down']*100:.2f}%\n"
        output += f"  Signal Strength: {signal['strength']}\n"
        output += f"\n"
        output += f"💰 TRADE PARAMETERS:\n"
        output += f"  Stop Loss: ${signal['stop_loss']:.2f}\n"
        output += f"  Take Profit: ${signal['take_profit']:.2f}\n"
        output += f"  Risk Amount: ${signal['risk_amount']:.2f}\n"
        output += f"  Reward Amount: ${signal['reward_amount']:.2f}\n"
        output += f"  Risk/Reward Ratio: {signal['risk_reward_ratio']:.2f}\n"
        output += f"\n"
        
        if signal['trade_signal']:
            output += f"✅ RECOMMENDED ACTION: TAKE THIS TRADE\n"
        else:
            output += f"⚠️  RECOMMENDED ACTION: SKIP THIS TRADE (Low confidence)\n"
        
        output += f"{border}\n"
        
        return output
