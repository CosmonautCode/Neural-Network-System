


from pathlib import Path
import pickle
import torch
import numpy as np


def save_model_and_scalers(model, optimizer, scaler_x, scaler_y, metrics):
    """Save model, scalers, and metadata"""
    
    # Get the project root directory and create model folder
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "model"
    model_dir.mkdir(exist_ok=True)
    
    try:
        # Save model checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'metrics': metrics,
            'model_info': {
                'input_size': 5,
                'hidden_sizes': [64, 32, 16, 8, 4],
                'output_size': 1,
                'target': 'Close',  # Updated target
                'data_period': '2010-2021'  # Updated data period
            }
        }
        
        torch.save(checkpoint, model_dir / "stocks_model.pt")

        print(f"\nModel saved to: {model_dir / 'stocks_model.pt'}")
        
        # Save scalers separately for compatibility
        with open(model_dir / "scaler_x.pkl", "wb") as f:
            pickle.dump(scaler_x, f)
        with open(model_dir / "scaler_y.pkl", "wb") as f:
            pickle.dump(scaler_y, f)
        
        print("Scalers saved successfully!")
        
    except Exception as e:
        print(f"Error saving files: {e}")