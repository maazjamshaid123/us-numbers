#!/usr/bin/env python3
"""
Lottery Number Predictor - XGBoost Implementation
Fast and efficient lottery number prediction using XGBoost.
"""

import os
import warnings
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class LotteryDataPreprocessor:
    """Data preprocessor for lottery data."""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.max_number = 80
        
    def load_data(self, file_path: str, months: int = 1) -> pd.DataFrame:
        """Load lottery data from CSV file and filter to latest N months."""
        print(f"Loading data (latest {months} month(s) only for speed)...")
        df = pd.read_csv(file_path)
        
        # Keep only the columns we need: Draw Date and Winning Numbers
        # Drop all other columns as they're not useful for prediction
        required_columns = ['Draw Date', 'Winning Numbers']
        df = df[required_columns]
        
        df['Draw Date'] = pd.to_datetime(df['Draw Date'])
        
        # Filter to latest N months only (for faster training)
        latest_date = df['Draw Date'].max()
        cutoff_date = latest_date - pd.DateOffset(months=months)
        df = df[df['Draw Date'] >= cutoff_date]
        
        print(f"Loaded dataset: {len(df)} draws")
        print(f"   Date range: {df['Draw Date'].min().date()} to {df['Draw Date'].max().date()}")
        print(f"   Using latest {months} month(s) of data for optimal speed!")
        
        # Parse winning numbers
        df['numbers'] = df['Winning Numbers'].apply(lambda x: [int(num) for num in x.split()])
        df = df.sort_values('Draw Date').reset_index(drop=True)
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features from lottery data."""
        features_data = []
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"Processing draw {idx}/{len(df)}...")
            
            date = row['Draw Date']
            numbers = row['numbers']
            
            # Base temporal features
            feature_row = {
                'time_idx': idx,
                'day_of_week': date.weekday(),
                'day_of_month': date.day,
                'month': date.month,
                'year': date.year,
                'quarter': date.quarter,
                'week_of_year': date.isocalendar()[1],
            }
            
            # Historical frequency features (last 100 draws)
            if idx > 0:
                historical_numbers = []
                for i in range(max(0, idx - 100), idx):
                    if i < len(df):
                        historical_numbers.extend(df.iloc[i]['numbers'])
                
                for num in range(1, self.max_number + 1):
                    feature_row[f'freq_100_{num}'] = (
                        historical_numbers.count(num) / len(historical_numbers) 
                        if historical_numbers else 0.0
                    )
            else:
                for num in range(1, self.max_number + 1):
                    feature_row[f'freq_100_{num}'] = 1.0 / self.max_number
            
            # Recent frequency features (last 10 draws)
            if idx > 0:
                recent_numbers = []
                for i in range(max(0, idx - 10), idx):
                    if i < len(df):
                        recent_numbers.extend(df.iloc[i]['numbers'])
                
                for num in range(1, self.max_number + 1):
                    feature_row[f'freq_10_{num}'] = (
                        recent_numbers.count(num) / len(recent_numbers) 
                        if recent_numbers else 0.0
                    )
            else:
                for num in range(1, self.max_number + 1):
                    feature_row[f'freq_10_{num}'] = 1.0 / self.max_number
            
            # Target encoding (binary indicators for each number)
            for num in range(1, self.max_number + 1):
                feature_row[f'target_{num}'] = 1.0 if num in numbers else 0.0
            
            # Statistical features
            feature_row.update({
                'min_number': float(min(numbers)),
                'max_number': float(max(numbers)),
                'mean_number': float(np.mean(numbers)),
                'std_number': float(np.std(numbers)),
                'sum_numbers': float(sum(numbers)),
                'number_range': float(max(numbers) - min(numbers)),
            })
            
            # Number gaps analysis
            sorted_numbers = sorted(numbers)
            gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(len(sorted_numbers)-1)]
            feature_row.update({
                'mean_gap': float(np.mean(gaps)) if gaps else 0.0,
                'max_gap': float(max(gaps)) if gaps else 0.0,
                'min_gap': float(min(gaps)) if gaps else 0.0,
            })
            
            features_data.append(feature_row)
        
        print("Feature extraction completed!")
        return pd.DataFrame(features_data)


class LotterySystem:
    """Complete lottery prediction system using XGBoost."""
    
    def __init__(self):
        self.models = {}  # Dictionary to store models for each number
        self.preprocessor = None
        self.feature_cols = None
        
    def prepare_data(self, file_path: str, sequence_length: int = 50, months: int = 1):
        """Load and prepare data for training."""
        print("Loading and processing data...")
        
        self.preprocessor = LotteryDataPreprocessor(sequence_length)
        df = self.preprocessor.load_data(file_path, months=months)
        features_df = self.preprocessor.create_features(df)
        
        print("Data preprocessing completed")
        
        return features_df
    
    def train(self, file_path: str, sequence_length: int = 50, prediction_length: int = 5, months: int = 1):
        """Train the lottery prediction model using XGBoost."""
        # Prepare data
        features_df = self.prepare_data(file_path, sequence_length, months=months)
        
        # Separate features and targets
        target_cols = [f'target_{i}' for i in range(1, 81)]
        self.feature_cols = [col for col in features_df.columns 
                            if col not in target_cols + ['time_idx']]
        
        X = features_df[self.feature_cols].values
        y = features_df[target_cols].values
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print("Training XGBoost models...")
        print("This will be much faster than deep learning!")
        
        # Train separate model for each number (1-80)
        for num_idx in range(80):
            if (num_idx + 1) % 10 == 0:
                print(f"Training model for number {num_idx + 1}/80...")
            
            # XGBoost parameters optimized for speed and accuracy
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist',  # Faster training
                'verbosity': 0,
            }
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train[:, num_idx])
            dval = xgb.DMatrix(X_val, label=y_val[:, num_idx])
            
            # Train model
            evals = [(dval, 'eval')]
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=evals,
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            self.models[num_idx] = model
        
        print("Training completed!")
        return features_df
    
    def predict(self, features_df: pd.DataFrame, sequence_length: int = 50, top_k: int = 20, 
                prediction_length: int = 5) -> List[List[int]]:
        """Generate lottery number predictions."""
        if not self.models:
            raise ValueError("No trained models available!")
        
        # Get the latest features
        latest_features = features_df[self.feature_cols].iloc[-1:].values
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(latest_features)
        
        # Predict probabilities for each number
        predictions = []
        for num_idx in range(80):
            model = self.models[num_idx]
            prob = model.predict(dtest)[0]
            predictions.append(prob)
        
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Generate multiple draws
        predicted_draws = []
        for _ in range(prediction_length):
            # Get top-k numbers
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            predicted_numbers = sorted([idx + 1 for idx in top_indices])
            predicted_draws.append(predicted_numbers)
            
            # Add some randomness for variety in predictions
            predictions = predictions + np.random.normal(0, 0.05, size=80)
            predictions = np.clip(predictions, 0, 1)
        
        return predicted_draws
    
    def analyze_and_visualize(self, raw_df: pd.DataFrame, predictions: List[List[int]]):
        """Create analysis and visualizations."""
        print("Creating visualizations...")
        
        # Historical analysis
        all_numbers = []
        for numbers_str in raw_df['Winning Numbers']:
            numbers = [int(x) for x in numbers_str.split()]
            all_numbers.extend(numbers)
        
        # Create analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Number frequency
        freq_series = pd.Series(all_numbers).value_counts().sort_index()
        ax1.bar(freq_series.index, freq_series.values, alpha=0.7, color='steelblue')
        ax1.set_title('Historical Number Frequency', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Most frequent numbers
        most_frequent = freq_series.tail(10)
        ax2.barh(range(len(most_frequent)), most_frequent.values, color='green', alpha=0.7)
        ax2.set_yticks(range(len(most_frequent)))
        ax2.set_yticklabels(most_frequent.index)
        ax2.set_title('Top 10 Most Frequent Numbers', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frequency')
        
        # Recent trends
        recent_draws = raw_df.tail(50)
        recent_numbers = []
        for numbers_str in recent_draws['Winning Numbers']:
            recent_numbers.extend([int(x) for x in numbers_str.split()])
        
        recent_freq = pd.Series(recent_numbers).value_counts().sort_index()
        ax3.bar(recent_freq.index, recent_freq.values, alpha=0.7, color='orange')
        ax3.set_title('Recent Number Frequency (Last 50 Draws)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Prediction visualization
        pred_grid = np.zeros((8, 10))
        for draw in predictions[:3]:  # Show first 3 predictions
            for num in draw:
                row, col = (num - 1) // 10, (num - 1) % 10
                pred_grid[row, col] += 1
        
        sns.heatmap(pred_grid, annot=True, fmt='.0f', cmap='Reds', ax=ax4, cbar=False)
        ax4.set_title('Predicted Numbers Heatmap (First 3 Draws)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Column (0-9)')
        ax4.set_ylabel('Row (Tens)')
        
        plt.tight_layout()
        return fig
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from the models."""
        # Average importance across all models
        importance_dict = {}
        
        for num_idx, model in self.models.items():
            # Get feature importance scores
            importance_scores = model.get_score(importance_type='gain')
            
            # Map feature names to importance
            for feat_idx, feat_name in enumerate(self.feature_cols):
                feat_key = f'f{feat_idx}'
                if feat_key in importance_scores:
                    if feat_name not in importance_dict:
                        importance_dict[feat_name] = []
                    importance_dict[feat_name].append(importance_scores[feat_key])
        
        # Calculate average importance
        avg_importance = {k: np.mean(v) for k, v in importance_dict.items()}
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame(
            list(avg_importance.items()), 
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Lottery Number Predictor (XGBoost)')
    parser.add_argument('--data-path', type=str, default='data/Lottery_Quick_Draw_Winning_Numbers__Beginning_2013_20251013.csv', 
                       help='Path to lottery data')
    parser.add_argument('--sequence-length', type=int, default=50, help='Input sequence length')
    parser.add_argument('--prediction-length', type=int, default=40, help='Number of future draws')
    parser.add_argument('--months', type=int, default=1, help='Number of months of historical data to use (default: 1 for speed)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎲 Lottery Number Predictor - XGBoost Edition")
    print("=" * 60)
    
    # Check data file
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        return
    
    # Initialize system
    system = LotterySystem()
    
    try:
        print(f"\n📊 Training XGBoost model (FAST!)...")
        features_df = system.train(
            args.data_path, 
            args.sequence_length, 
            args.prediction_length,
            months=args.months
        )
        
        # Make predictions
        print(f"\n🔮 Generating {args.prediction_length} lottery predictions...")
        predictions = system.predict(
            features_df, 
            args.sequence_length,
            prediction_length=args.prediction_length
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("🎯 PREDICTED NUMBERS:")
        print("=" * 60)
        for i, numbers in enumerate(predictions, 1):
            print(f"Draw {i}: {' '.join(f'{num:2d}' for num in numbers)}")
        
        # Create analysis
        raw_df = pd.read_csv(args.data_path)
        raw_df = raw_df[['Draw Date', 'Winning Numbers']]
        raw_df['Draw Date'] = pd.to_datetime(raw_df['Draw Date'])
        
        analysis_fig = system.analyze_and_visualize(raw_df, predictions)
        
        # Statistics
        all_predicted = [num for draw in predictions for num in draw]
        print(f"\n📈 STATISTICS:")
        print(f"   Range: {min(all_predicted)}-{max(all_predicted)}")
        print(f"   Most frequent: {max(set(all_predicted), key=all_predicted.count)}")
        print(f"   Average: {np.mean([np.mean(draw) for draw in predictions]):.1f}")
        
        # Feature importance
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        importance_df = system.get_feature_importance(top_n=10)
        for idx, row in importance_df.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.2f}")
        
        print(f"\n✅ Complete! Good luck with your lottery numbers!")
        print("=" * 60)
        
        plt.show()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
