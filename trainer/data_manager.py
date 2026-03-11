import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading dataset...")
    
    try:
        # Load external dataset
        dataset = load_dataset("Ammok/apple_stock_price_from_1980-2021")
        df = pd.DataFrame(dataset["train"])
        
        # Convert Date column to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Filter for recent data (2010 onwards) - use last 3000 records as approximation for 2010+
        df_recent = df.tail(3000)  # This should cover roughly 2010-2021
        print(f"Using recent data: {len(df_recent)} records")
        print(f"Date range: {df_recent.index.min()} to {df_recent.index.max()}")
        
        # Define features and target - predict Close price instead of Adj Close
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]
        target_column = "Close"  # Changed from "Adj Close"
        
        # Prepare data
        x_data = df_recent[feature_columns].fillna(0)
        y_data = df_recent[target_column].fillna(0)
        
        print(f"Target: Predicting {target_column} prices")
        print(f"Price range: ${y_data.min():.2f} - ${y_data.max():.2f}")
        
        return x_data, y_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def create_data_loaders(x_data, y_data, batch_size=32, test_size=0.2):
    """Create train/test data loaders"""
    
    # Scale the data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    x_scaled = scaler_x.fit_transform(x_data)
    y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1, 1))
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_scaled, test_size=test_size, random_state=42
    )
    
    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Create simple data loaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler_x, scaler_y