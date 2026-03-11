
import torch.nn as nn



class NN_Regression(nn.Module):
    """Improved Neural Network for stock price prediction"""
    
    def __init__(self, input_size=5, hidden_sizes=[32, 16, 8, 4], output_size=1):
        super(NN_Regression, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            prev_size = hidden_size
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
