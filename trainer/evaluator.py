
import os
import sys
from pathlib import Path
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)



def evaluate_model(model, test_loader, scaler_y):
    """Evaluate model performance"""
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
            predictions.extend(y_pred.numpy())
            actuals.extend(y_batch.numpy())
    
    # Convert to numpy arrays and inverse transform
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    predictions_unscaled = scaler_y.inverse_transform(predictions)
    actuals_unscaled = scaler_y.inverse_transform(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals_unscaled, predictions_unscaled)
    mae = mean_absolute_error(actuals_unscaled, predictions_unscaled)
    r2 = r2_score(actuals_unscaled, predictions_unscaled)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actuals_unscaled - predictions_unscaled) / actuals_unscaled)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'predictions': predictions_unscaled,
        'actuals': actuals_unscaled
    }
