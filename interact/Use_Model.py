import pickle
import torch
import time
import pandas as pd
from pathlib import Path
import sys

# Add the parent directory to the path to import the model
sys.path.append(str(Path(__file__).resolve().parent.parent))
from trainer.train_model import NN_Regression


def get_model_path():
    """Get the correct path to model files"""
    # Get the project root directory and model folder
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "model"
    return model_dir

def validate_input(prompt):
    """Validate numeric input"""
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Please enter a valid number.")

def load_model_and_scalers():
    """Load model and scalers with error handling"""
    model_dir = get_model_path()
    
    try:
        # Load scalers
        with open(model_dir / "scaler_x.pkl", "rb") as f:
            scaler_x = pickle.load(f)
        with open(model_dir / "scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)
        
        # Load model
        model = NN_Regression()
        checkpoint = torch.load(model_dir / "stocks_model.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        return model, scaler_x, scaler_y
        
    except FileNotFoundError as e:
        print(f"Error: Missing file - {e}")
        print("Please run main.py first to generate the required files.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def use_model():
    """Main prediction function"""
    print("Apple Stock Price Predictor")
    print("=" * 40)
    
    # Load model and scalers
    model, scaler_x, scaler_y = load_model_and_scalers()
    if model is None:
        return
    
    print("Model loaded successfully!")
    print("\nSample examples to try (2010+ data):")
    print("Example 1 (Recent): Open=147.03, High=148.57, Low=144.90, Close=146.14, Volume=71598400")
    print("Example 2 (2015): Open=112.50, High=115.20, Low=111.80, Close=114.30, Volume=45000000")
    print("Example 3 (2010): Open=45.20, High=46.80, Low=44.90, Close=46.30, Volume=120000000")
    print("Model now predicts Close prices! See sample_data.md for more examples!\n")
    
    # Get user input with validation
    try:
        Open = validate_input("Open: ")
        High = validate_input("High: ")
        Low = validate_input("Low: ")
        Close = validate_input("Close: ")
        Volume = validate_input("Volume: ")
        
        # Process input with proper column names
        new_data = [[Open, High, Low, Close, Volume]]
        new_data_df = pd.DataFrame(new_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        new_data_scaled = scaler_x.transform(new_data_df)
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)
        
        print("\nProcessing prediction...")
        time.sleep(1)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(new_data_tensor)
            prediction_unscaled = scaler_y.inverse_transform(prediction.numpy())
        
        # Display result
        predicted_price = prediction_unscaled[0][0]
        print(f"\nPredicted Close Price: ${predicted_price:.2f}")
        print("\nNote: This is for educational purposes only. Do not use for real trading!")
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError during prediction: {e}")


