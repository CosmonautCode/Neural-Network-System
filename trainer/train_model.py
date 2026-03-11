
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from trainer.data_manager import load_and_prepare_data, create_data_loaders
from trainer.model import NN_Regression
from trainer.save_model import save_model_and_scalers
from trainer.evaluator import evaluate_model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)



def train_model():
    """Main training function"""
    
    # Load data
    x_data, y_data = load_and_prepare_data()
    if x_data is None:
        return False
    
    # Create data loaders
    train_loader, test_loader, scaler_x, scaler_y = create_data_loaders(x_data, y_data)
    
    # Initialize model and optimizer
    model = NN_Regression()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add weight decay
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training parameters
    num_epochs = 200
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch.view(-1, 1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, scaler_y)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"MSE:  {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"R²:   {metrics['R2']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    
    # Show sample predictions
    print(f"\nSample Predictions (first 10):")
    print("-" * 40)
    for i in range(min(10, len(metrics['predictions']))):
        pred = metrics['predictions'][i][0]
        actual = metrics['actuals'][i][0]
        diff = abs(pred - actual)
        print(f"Pred: ${pred:8.2f} | Actual: ${actual:8.2f} | Diff: ${diff:6.2f}")
    
    # Save model and scalers
    save_model_and_scalers(model, optimizer, scaler_x, scaler_y, metrics)
    
    return True


