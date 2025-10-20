"""
Learning Tracker Module
Tracks predictions and actual outcomes for continuous learning
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

class LearningTracker:
    def __init__(self, data_file='learning_data.csv', metrics_file='model_metrics.json'):
        """
        Initialize learning tracker
        
        Args:
            data_file: CSV file to store predictions and outcomes
            metrics_file: JSON file to store performance metrics
        """
        self.data_file = data_file
        self.metrics_file = metrics_file
        
        # Load or create tracking data
        if os.path.exists(data_file):
            self.data = pd.read_csv(data_file)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        else:
            self.data = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'entry_price', 'prediction',
                'probability', 'actual_direction', 'actual_return',
                'was_correct', 'outcome_recorded'
            ])
        
        # Load or create metrics
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'last_retrain_time': None,
                'retrain_count': 0,
                'model_version': 1
            }
    
    def record_prediction(self, symbol, entry_price, prediction, probability):
        """
        Record a new prediction
        
        Args:
            symbol: Stock symbol
            entry_price: Price at prediction time
            prediction: Predicted direction (1=up, 0=down)
            probability: Prediction confidence
            
        Returns:
            Prediction ID
        """
        new_prediction = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'entry_price': entry_price,
            'prediction': prediction,
            'probability': probability,
            'actual_direction': None,
            'actual_return': None,
            'was_correct': None,
            'outcome_recorded': False
        }
        
        # Add to dataframe
        self.data = pd.concat([self.data, pd.DataFrame([new_prediction])], ignore_index=True)
        
        # Update metrics
        self.metrics['total_predictions'] += 1
        
        # Save
        self.save()
        
        return len(self.data) - 1  # Return index as ID
    
    def record_outcome(self, prediction_id, actual_price):
        """
        Record the actual outcome of a prediction
        
        Args:
            prediction_id: ID of the prediction
            actual_price: Actual price after time period
        """
        if prediction_id >= len(self.data):
            print(f"Warning: Invalid prediction ID {prediction_id}")
            return
        
        row = self.data.iloc[prediction_id]
        
        if row['outcome_recorded']:
            return  # Already recorded
        
        # Calculate actual return
        entry_price = row['entry_price']
        actual_return = (actual_price - entry_price) / entry_price * 100
        actual_direction = 1 if actual_return > 0 else 0
        
        # Check if prediction was correct
        was_correct = (row['prediction'] == actual_direction)
        
        # Update record
        self.data.at[prediction_id, 'actual_direction'] = actual_direction
        self.data.at[prediction_id, 'actual_return'] = actual_return
        self.data.at[prediction_id, 'was_correct'] = was_correct
        self.data.at[prediction_id, 'outcome_recorded'] = True
        
        # Update metrics
        if was_correct:
            self.metrics['correct_predictions'] += 1
        
        # Recalculate accuracy
        completed = self.data[self.data['outcome_recorded'] == True]
        if len(completed) > 0:
            self.metrics['accuracy'] = (completed['was_correct'].sum() / len(completed)) * 100
        
        # Save
        self.save()
    
    def get_unrecorded_predictions(self, max_age_minutes=10):
        """
        Get predictions that haven't had outcomes recorded yet
        
        Args:
            max_age_minutes: Only get predictions older than this
            
        Returns:
            DataFrame with unrecorded predictions
        """
        unrecorded = self.data[self.data['outcome_recorded'] == False].copy()
        
        if len(unrecorded) == 0:
            return unrecorded
        
        # Filter by age
        now = datetime.now()
        unrecorded['age_minutes'] = (now - unrecorded['timestamp']).dt.total_seconds() / 60
        unrecorded = unrecorded[unrecorded['age_minutes'] >= max_age_minutes]
        
        return unrecorded
    
    def get_recent_accuracy(self, n=100):
        """
        Get accuracy of last N predictions
        
        Args:
            n: Number of recent predictions to evaluate
            
        Returns:
            Accuracy percentage
        """
        completed = self.data[self.data['outcome_recorded'] == True]
        if len(completed) == 0:
            return 0.0
        
        recent = completed.tail(n)
        accuracy = (recent['was_correct'].sum() / len(recent)) * 100
        
        return accuracy
    
    def should_retrain(self, min_new_samples=50, max_hours_since_retrain=24):
        """
        Determine if model should be retrained
        
        Args:
            min_new_samples: Minimum new samples needed
            max_hours_since_retrain: Maximum hours since last retrain
            
        Returns:
            Boolean indicating if retraining is needed
        """
        # Check if we have enough new samples
        completed = self.data[self.data['outcome_recorded'] == True]
        new_samples = len(completed)
        
        if new_samples < min_new_samples:
            return False
        
        # Check time since last retrain
        if self.metrics['last_retrain_time']:
            last_retrain = datetime.fromisoformat(self.metrics['last_retrain_time'])
            hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
            
            if hours_since < max_hours_since_retrain and new_samples < min_new_samples * 2:
                return False
        
        return True
    
    def mark_retrained(self):
        """Mark that model has been retrained"""
        self.metrics['last_retrain_time'] = datetime.now().isoformat()
        self.metrics['retrain_count'] += 1
        self.metrics['model_version'] += 1
        self.save()
    
    def get_training_data(self):
        """
        Get completed predictions for retraining
        
        Returns:
            DataFrame with features and targets
        """
        completed = self.data[self.data['outcome_recorded'] == True].copy()
        return completed
    
    def save(self):
        """Save tracking data and metrics to disk"""
        self.data.to_csv(self.data_file, index=False)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metrics_summary(self):
        """
        Get summary of learning metrics
        
        Returns:
            Dictionary with metrics
        """
        completed = len(self.data[self.data['outcome_recorded'] == True])
        pending = len(self.data[self.data['outcome_recorded'] == False])
        
        summary = {
            'total_predictions': self.metrics['total_predictions'],
            'completed_predictions': completed,
            'pending_predictions': pending,
            'correct_predictions': self.metrics['correct_predictions'],
            'overall_accuracy': self.metrics['accuracy'],
            'recent_accuracy_50': self.get_recent_accuracy(50),
            'recent_accuracy_100': self.get_recent_accuracy(100),
            'retrain_count': self.metrics['retrain_count'],
            'model_version': self.metrics['model_version'],
            'last_retrain': self.metrics['last_retrain_time']
        }
        
        return summary
    
    def print_metrics(self):
        """Print metrics summary to console"""
        summary = self.get_metrics_summary()
        
        print("\n" + "=" * 70)
        print("📊 LEARNING METRICS SUMMARY")
        print("=" * 70)
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"Completed: {summary['completed_predictions']} | Pending: {summary['pending_predictions']}")
        print(f"Correct: {summary['correct_predictions']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%")
        print(f"Recent Accuracy (50): {summary['recent_accuracy_50']:.2f}%")
        print(f"Recent Accuracy (100): {summary['recent_accuracy_100']:.2f}%")
        print(f"Model Version: v{summary['model_version']}")
        print(f"Retrain Count: {summary['retrain_count']}")
        if summary['last_retrain']:
            last_retrain = datetime.fromisoformat(summary['last_retrain'])
            print(f"Last Retrain: {last_retrain.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
