# Apple Stock Price Prediction Neural Network

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Made with UV](https://img.shields.io/badge/made%20with-uv-ff69b4.svg)](https://github.com/astral-sh/uv)

A **modular neural network system** for predicting Apple stock closing prices using historical data. This project demonstrates a complete machine learning pipeline with data preprocessing, model training, evaluation, and inference - all built from scratch using PyTorch.

---

## **Features**
- **Modular Architecture**: Clean separation of concerns with dedicated modules for training, data management, evaluation, and inference
- **Modern Training Pipeline**: Early stopping, learning rate scheduling, gradient clipping, and dropout regularization
- **Data-Driven Approach**: Trained on 2010-2021 Apple stock data (3,000 records) for realistic market patterns
- **Interactive Prediction**: User-friendly console interface for making stock price predictions
- **Comprehensive Metrics**: MSE, RMSE, MAE, R², and MAPE evaluation metrics
- **Model Persistence**: Complete checkpoint saving with model weights, optimizers, and data scalers

---

## **Project Structure**
```bash
neural-network-system/
├── model/                     # Trained model artifacts (auto-created)
│   ├── stocks_model.pt        # Complete model checkpoint
│   ├── scaler_x.pkl          # Feature scaler
│   └── scaler_y.pkl          # Target scaler
├── trainer/                   # Training pipeline modules
│   ├── train_model.py         # Main training orchestration
│   ├── data_manager.py        # Data loading and preprocessing
│   ├── model.py              # Neural network architecture
│   ├── evaluator.py          # Model evaluation metrics
│   └── save_model.py         # Model persistence utilities
├── interact/                  # User interface modules
│   └── Use_Model.py          # Interactive prediction interface
├── main.py                    # Main entry point
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

---

## **Quick Start**

### 1. Install Dependencies
```bash
uv sync
```

### 2. Train the Model
I recommend deleting the files in model/ as these are from previous training

```bash
uv run python -m main
```

### 3. Make Predictions
After training completes, the system automatically launches the prediction interface. You can also run it separately:
```bash
uv run python interact/Use_Model.py
```

---

## **Model Architecture**

- **Input Layer**: 5 features (Open, High, Low, Close, Volume)
- **Hidden Layers**: 632 -> 16 -> 8 -> 4 neurons with ReLU activations   
- **Output Layer**: 1 neuron (predicted Close price)
- **Regularization**: Dropout (0.1) and L2 weight decay
- **Parameters**: 897 trainable weights   <- This can be adjust eeasily to much higher amounts of weights, reccomend to have 1:10 paramter - data ratio

---

## **Training Performance**

Based on 2010-2021 Apple stock data:
- **R² Score**: 0.9834 (98.34% variance explained)
- **RMSE**: $5.51
- **MAE**: $3.85
- **Training Time**: ~53 epochs with early stopping
- **Data Period**: July 2010 - June 2022

---

## **Usage Examples**

### Sample Predictions to Try:
```
Example 1 (Recent):    Open=147.03, High=148.57, Low=144.90, Close=146.14, Volume=71598400
Example 2 (2015):      Open=112.50, High=115.20, Low=111.80, Close=114.30, Volume=45000000
Example 3 (2010):      Open=45.20,  High=46.80,  Low=44.90,  Close=46.30,  Volume=120000000
```

### Programmatic Usage:
```python
import torch
import pickle
from pathlib import Path
from trainer.model import NN_Regression

# Load model and scalers
model_dir = Path("model")
model = NN_Regression()
checkpoint = torch.load(model_dir / "stocks_model.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

with open(model_dir / "scaler_x.pkl", "rb") as f:
    scaler_x = pickle.load(f)
with open(model_dir / "scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

# Make prediction
import pandas as pd
new_data = pd.DataFrame([[147, 148, 144, 146, 71598400]], 
                        columns=['Open', 'High', 'Low', 'Close', 'Volume'])
scaled_data = scaler_x.transform(new_data)
tensor_data = torch.tensor(scaled_data, dtype=torch.float32)

model.eval()
with torch.no_grad():
    prediction = model(tensor_data)
    price = scaler_y.inverse_transform(prediction.numpy())[0][0]
    
print(f"Predicted Close Price: ${price:.2f}")
```

---

## **Technical Details**

### Data Pipeline
- **Source**: Apple stock price dataset (1980-2021)
- **Filtering**: Last 3,000 records (2010-2022) for modern market patterns
- **Features**: Open, High, Low, Close, Volume
- **Target**: Close price (next day prediction)
- **Preprocessing**: StandardScaler normalization for both features and targets

### Training Configuration
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Validation Split**: 80/20 train-test
- **Early Stopping**: Patience=20 epochs
- **LR Scheduling**: ReduceLROnPlateau (factor=0.5, patience=10)

---

## **Important Notes**

 **Educational Purpose Only**: This system is for learning neural networks and data science concepts. **Do not use for real trading or financial decisions.**

 **Model Limitations**: 
- Trained on historical data only
- Cannot account for real-time market events
- Performance may vary with different market conditions
- Stock prices are influenced by many factors not captured in this model

---

## **Troubleshooting**

### Common Issues:
- **ModuleNotFoundError**: Ensure you're in the project root and run `uv sync`
- **Missing model files**: Run training first with `uv run python -m main`
- **PyTorch weights_only error**: The system handles this automatically with `weights_only=False`
- **Sklearn feature names warning**: Fixed in the current version with proper DataFrame handling

### File Locations:
- **Model files**: Saved to `/model` directory (auto-created)
- **Training logs**: Printed to console during training
- **Predictions**: Interactive console output

---

## **Dependencies**

Core dependencies managed via UV:
- `torch` - Neural network framework
- `pandas` - Data manipulation
- `scikit-learn` - Data preprocessing and metrics
- `datasets` - Apple stock dataset loading
- `numpy` - Numerical operations
- `tqdm` - Progress bars

---

## **License**

MIT License - feel free to use this for educational purposes and learning.
