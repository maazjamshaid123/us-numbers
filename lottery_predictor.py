#!/usr/bin/env python3
"""
Lottery Number Predictor - Complete All-in-One System
Temporal LSTM model for lottery number prediction using historical patterns.
"""

import os
import sys
import warnings
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class LotteryDataPreprocessor:
    """Data preprocessor for lottery data."""
    
    def __init__(self, sequence_length: int = 50, prediction_length: int = 5):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.max_number = 80
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load lottery data from CSV file."""
        df = pd.read_csv(file_path)
        df['Draw Date'] = pd.to_datetime(df['Draw Date'])
        
        print(f"Loaded dataset: {len(df)} draws")
        print(f"   Date range: {df['Draw Date'].min().date()} to {df['Draw Date'].max().date()}")
        
        # Parse winning numbers
        df['numbers'] = df['Winning Numbers'].apply(lambda x: [int(num) for num in x.split()])
        df = df.sort_values('Draw Date').reset_index(drop=True)
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features from lottery data."""
        features_data = []
        
        for idx, row in df.iterrows():
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
            
            # Historical frequency features
            if idx > 0:
                historical_numbers = []
                for i in range(max(0, idx - 100), idx):
                    if i < len(df):
                        historical_numbers.extend(df.iloc[i]['numbers'])
                
                for num in range(1, self.max_number + 1):
                    feature_row[f'freq_{num}'] = (
                        historical_numbers.count(num) / len(historical_numbers) 
                        if historical_numbers else 0.0
                    )
            else:
                for num in range(1, self.max_number + 1):
                    feature_row[f'freq_{num}'] = 1.0 / self.max_number
            
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
            
            # Hot/Cold numbers (recent pattern analysis)
            recent_numbers = []
            for i in range(max(0, idx - 10), idx):
                if i < len(df):
                    recent_numbers.extend(df.iloc[i]['numbers'])
            
            hot_numbers = [num for num in range(1, self.max_number + 1) if recent_numbers.count(num) >= 3]
            cold_numbers = [num for num in range(1, self.max_number + 1) if recent_numbers.count(num) == 0]
            
            feature_row.update({
                'hot_numbers_count': float(len(hot_numbers)),
                'cold_numbers_count': float(len(cold_numbers)),
                'hot_numbers_drawn': float(sum(1 for num in numbers if num in hot_numbers)),
                'cold_numbers_drawn': float(sum(1 for num in numbers if num in cold_numbers)),
            })
            
            features_data.append(feature_row)
        
        return pd.DataFrame(features_data)


class LotteryDataset(Dataset):
    """PyTorch dataset for lottery sequences."""
    
    def __init__(self, features_df: pd.DataFrame, sequence_length: int, prediction_length: int):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Separate features and targets
        target_cols = [f'target_{i}' for i in range(1, 81)]
        feature_cols = [col for col in features_df.columns 
                       if col not in target_cols + ['time_idx']]
        
        self.features = features_df[feature_cols].values.astype(np.float32)
        self.targets = features_df[target_cols].values.astype(np.float32)
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        print(f"Dataset: {len(self.features)} samples, {self.features.shape[1]} features")
        
    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        feature_seq = self.features[idx:idx + self.sequence_length]
        target_seq = self.targets[idx + self.sequence_length:idx + self.sequence_length + self.prediction_length]
        return torch.FloatTensor(feature_seq), torch.FloatTensor(target_seq)


class LotteryPredictor(pl.LightningModule):
    """Advanced LSTM-based lottery number predictor."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, learning_rate: float = 0.001, prediction_length: int = 5):
        super().__init__()
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.learning_rate = learning_rate
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-head attention for sequence modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers for each future draw
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 80),  # 80 possible numbers
                nn.Sigmoid()
            ) for _ in range(prediction_length)
        ])
        
        self.criterion = nn.BCELoss()
        
    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use final hidden state for predictions
        last_hidden = attn_out[:, -1, :]
        
        # Generate predictions for each future draw
        predictions = []
        for i in range(self.prediction_length):
            pred = self.output_layers[i](last_hidden)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1)
    
    def training_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self(features)
        loss = self.criterion(predictions, targets)
        
        # Calculate accuracy
        with torch.no_grad():
            pred_binary = predictions > 0.5
            target_binary = targets > 0.5
            accuracy = (pred_binary == target_binary).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self(features)
        loss = self.criterion(predictions, targets)
        
        pred_binary = predictions > 0.5
        target_binary = targets > 0.5
        accuracy = (pred_binary == target_binary).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


class LotterySystem:
    """Complete lottery prediction system."""
    
    def __init__(self, model_params: Dict = None, trainer_params: Dict = None):
        # Optimal parameters for best results (no user configuration needed)
        self.model_params = model_params or {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
        }
        
        # Optimal training parameters for non-technical users
        self.trainer_params = trainer_params or {
            'max_epochs': 30,  # Optimal balance of speed and accuracy
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'gradient_clip_val': 1.0,
        }
        
        self.model = None
        self.trainer = None
        self.preprocessor = None
        
        
    def prepare_data(self, file_path: str, sequence_length: int, prediction_length: int):
        """Load and prepare data for training."""
        print("Loading and processing data...")
        
        self.preprocessor = LotteryDataPreprocessor(sequence_length, prediction_length)
        df = self.preprocessor.load_data(file_path)
        features_df = self.preprocessor.create_features(df)
        
        print("Data preprocessing completed")
        
        # Create dataset
        dataset = LotteryDataset(features_df, sequence_length, prediction_length)
        
        # Train/validation split
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, dataset, features_df
    
    def train(self, file_path: str, sequence_length: int = 50, prediction_length: int = 5):
        """Train the lottery prediction model."""
        # Prepare data
        train_loader, val_loader, dataset, features_df = self.prepare_data(
            file_path, sequence_length, prediction_length
        )
        
        # Get input size
        sample_features, _ = dataset[0]
        input_size = sample_features.shape[1]
        
        print(f"Initializing model (input size: {input_size})")
        
        # Initialize model
        self.model = LotteryPredictor(
            input_size=input_size,
            prediction_length=prediction_length,
            **self.model_params
        )
        
        # Setup trainer
        callbacks = [
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ]
        
        self.trainer = pl.Trainer(
            callbacks=callbacks, 
            enable_checkpointing=False,
            logger=False,
            **self.trainer_params
        )
        
        # Train
        print("Starting training...")
        self.trainer.fit(self.model, train_loader, val_loader)
        
        print("Training completed!")
        return features_df
    
    def predict(self, features_df: pd.DataFrame, sequence_length: int, top_k: int = 20) -> List[List[int]]:
        """Generate lottery number predictions."""
        if self.model is None:
            raise ValueError("No trained model available!")
        
        self.model.eval()
        
        # Prepare latest sequence
        feature_cols = [col for col in features_df.columns 
                       if col not in [f'target_{i}' for i in range(1, 81)] + ['time_idx']]
        
        # Create dataset to get the scaler
        temp_dataset = LotteryDataset(features_df, sequence_length, 1)
        latest_features = temp_dataset.scaler.transform(features_df[feature_cols].values.astype(np.float32))
        latest_sequence = torch.FloatTensor(latest_features[-sequence_length:])
        
        with torch.no_grad():
            predictions = self.model(latest_sequence.unsqueeze(0))
            predictions = predictions.squeeze(0)
            
            predicted_draws = []
            for step in range(self.model.prediction_length):
                step_probs = predictions[step]
                top_indices = torch.topk(step_probs, top_k).indices
                predicted_numbers = sorted([idx.item() + 1 for idx in top_indices])
                predicted_draws.append(predicted_numbers)
        
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
        ax1.bar(freq_series.index, freq_series.values, alpha=0.7)
        ax1.set_title('Historical Number Frequency')
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Most frequent numbers
        most_frequent = freq_series.tail(10)
        ax2.barh(range(len(most_frequent)), most_frequent.values, color='green', alpha=0.7)
        ax2.set_yticks(range(len(most_frequent)))
        ax2.set_yticklabels(most_frequent.index)
        ax2.set_title('Top 10 Most Frequent Numbers')
        
        # Recent trends
        recent_draws = raw_df.tail(50)
        recent_numbers = []
        for numbers_str in recent_draws['Winning Numbers']:
            recent_numbers.extend([int(x) for x in numbers_str.split()])
        
        recent_freq = pd.Series(recent_numbers).value_counts().sort_index()
        ax3.bar(recent_freq.index, recent_freq.values, alpha=0.7, color='orange')
        ax3.set_title('Recent Number Frequency (Last 50 Draws)')
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
        ax4.set_title('Predicted Numbers Heatmap (First 3 Draws)')
        
        plt.tight_layout()
        return fig
    
    
    


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Lottery Number Predictor')
    parser.add_argument('--data-path', type=str, default='data/train.csv', help='Path to lottery data')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--sequence-length', type=int, default=50, help='Input sequence length')
    parser.add_argument('--prediction-length', type=int, default=5, help='Number of future draws')
    parser.add_argument('--hidden-size', type=int, default=128, help='Model hidden size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    print("Lottery Number Predictor")
    print("=" * 25)
    
    # Check data file
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        return
    
    # Initialize system
    model_params = {
        'hidden_size': args.hidden_size,
        'learning_rate': args.learning_rate,
        'num_layers': 2,
        'dropout': 0.2,
    }
    
    trainer_params = {
        'max_epochs': args.epochs,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'gradient_clip_val': 1.0,
    }
    
    system = LotterySystem(model_params, trainer_params)
    
    try:
        print(f"Training model ({args.epochs} epochs)")
        features_df = system.train(
            args.data_path, 
            args.sequence_length, 
            args.prediction_length
        )
        
        # Make predictions
        print(f"Generating {args.prediction_length} lottery predictions...")
        predictions = system.predict(features_df, args.sequence_length)
        
        # Display results
        print("\nPredicted Numbers:")
        print("-" * 30)
        for i, numbers in enumerate(predictions, 1):
            print(f"Draw {i}: {' '.join(f'{num:2d}' for num in numbers)}")
        
        
        # Create analysis
        raw_df = pd.read_csv(args.data_path)
        raw_df['Draw Date'] = pd.to_datetime(raw_df['Draw Date'])
        
        analysis_fig = system.analyze_and_visualize(raw_df, predictions)
        analysis_fig.show()
        
        # Statistics
        all_predicted = [num for draw in predictions for num in draw]
        print(f"\nStatistics:")
        print(f"   Range: {min(all_predicted)}-{max(all_predicted)}")
        print(f"   Most frequent: {max(set(all_predicted), key=all_predicted.count)}")
        print(f"   Average: {np.mean([np.mean(draw) for draw in predictions]):.1f}")
        
        print(f"\nComplete! Good luck with your lottery numbers!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
