"""
Machine Learning Model Module
Trains and makes predictions for trading signals
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class TradingModel:
    def __init__(self, model_type='random_forest'):
        """
        Initialize ML model
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boost')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        
    def create_model(self):
        """Create the ML model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, df, feature_cols, target_col='Target_Direction'):
        """
        Train the model
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            
        Returns:
            Dictionary with training metrics
        """
        self.feature_cols = feature_cols
        
        # Prepare data
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.create_model()
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        self.is_trained = True
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))
        
        return metrics
    
    def predict(self, features):
        """
        Make prediction on new data
        
        Args:
            features: Array or DataFrame with features
            
        Returns:
            Dictionary with prediction and probability
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        # Handle DataFrame input
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_cols].values
        
        # Reshape if single sample
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': int(prediction),  # 1 for up, 0 for down
            'probability': float(probabilities[prediction]),
            'prob_down': float(probabilities[0]),
            'prob_up': float(probabilities[1])
        }
        
        return result
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath='trading_model.pkl'):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='trading_model.pkl'):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def incremental_train(self, new_features, new_targets):
        """
        Incrementally train model with new data (online learning)
        
        Args:
            new_features: New feature data
            new_targets: New target labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained initially before incremental training")
        
        # Scale new features
        new_features_scaled = self.scaler.transform(new_features)
        
        # For Random Forest, we can't do true incremental learning
        # Instead, we'll use warm_start for partial fit
        if hasattr(self.model, 'n_estimators'):
            # Add more trees to existing model
            current_trees = self.model.n_estimators
            self.model.n_estimators = current_trees + 10
            self.model.set_params(warm_start=True)
            self.model.fit(new_features_scaled, new_targets)
            
        print(f"Model incrementally updated with {len(new_features)} new samples")
