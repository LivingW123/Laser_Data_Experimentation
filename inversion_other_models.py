import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------
# Model 1: Random Forest Regressor for Multi-output Regression
# ----------------------------------------------------------------------------
# Random Forests are robust to overfitting, require less tuning than NNs,
# and can capture complex non-linear relationships in the parameter space.

def train_random_forest(X, y):
    """
    Trains a Random Forest Regressor to predict the 6 PFF inversion parameters.
    
    Args:
        X (np.array): Measurements 'b' or b-like spectrum (Shape: [num_samples, length_of_b])
        y (np.array): Target parameters corresponding to a1-a6 (Shape: [num_samples, 6])
    """
    print("Training Random Forest Regressor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We use MultiOutputRegressor to handle predicting 6 outputs simultaneously
    base_rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model = MultiOutputRegressor(base_rf)
    
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Random Forest Test MSE: {mse:.6f}\n")
    
    return model


# ----------------------------------------------------------------------------
# Model 2: 1D Convolutional Neural Network (CNN)
# ----------------------------------------------------------------------------
# 1D CNNs are highly effective for spectral or sequential data (like detector 
# responses), as they can extract local spatial features along the spectrum.

class InversionSpectrum1DCNN(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super(InversionSpectrum1DCNN, self).__init__()
        # Input shape: (Batch, Channels=1, Length=input_dim)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate dimension after convolutions to properly size the Linear layer
        conv_output_length = input_dim // 4
        self.flattened_size = conv_output_length * 32
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x is originally (Batch, Length), needs to be (Batch, Channels, Length) for Conv1d
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(-1, self.flattened_size) # Flatten
        x = self.fc_layers(x)
        return x

def train_cnn(X_train, y_train, epochs=500, batch_size=32, lr=0.001):
    """
    Trains a 1D CNN to predict the 6 PFF inversion parameters.
    """
    print("Training 1D Convolutional Neural Network...")
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = InversionSpectrum1DCNN(input_dim, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"CNN Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
            
    print("CNN Training Complete!\n")
    return model


if __name__ == "__main__":
    # --- Example Usage Script ---
    num_samples = 2000
    b_dim = 24  # Assuming the condensed spectral vector 'b' has 24 dimensions
    
    # Simulated dummy data
    print("Generating simulated dataset...")
    X_dummy = np.random.rand(num_samples, b_dim)
    y_dummy = np.random.rand(num_samples, 6) 
    
    # 1. Train Random Forest
    rf_model = train_random_forest(X_dummy, y_dummy)
    
    # 2. Train 1D CNN
    cnn_model = train_cnn(X_dummy, y_dummy, epochs=300)
    
    print("Both models are ready for parameter prediction!")
