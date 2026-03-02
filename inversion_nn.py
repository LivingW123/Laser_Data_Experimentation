import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# A Multilayer Perceptron designed to infer the 6 arbitrary parameters of the PFF
# f_E formulation: f(x) = a1*exp(-a2*x) + a3*exp(-(x-a4)^2 / (a5*x + a6/x))
# directly from the measured spectrum/detector responses.
class InversionSpectrumMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128, 64, 32], output_dim=6):
        """
        Args:
            input_dim (int): The number of data points in the provided detector response vector 'b'.
            hidden_layers (list): Dimensions of the hidden layers.
            output_dim (int): The number of arbitrary parameters to predict, 6 for the PFF formulation.
        """
        super(InversionSpectrumMLP, self).__init__()
        
        layers = []
        in_features = input_dim
        
        # Build the hidden layers
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            # Optional: Add Batch Normalization or Dropout here if overfitting occurs
            in_features = out_features
            
        # Output layer mapping to the 6 parameters
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_inversion_nn(X_train, y_train, epochs=1000, batch_size=32, lr=0.001):
    """
    Trains the MLP to predict the 6 PFF inversion parameters.
    
    Args:
        X_train (np.array): Measurements 'b' or b-like spectrum (Shape: [num_samples, length_of_b])
        y_train (np.array): Target parameters corresponding to a1-a6 (Shape: [num_samples, 6])
    """
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # Create a DataLoader for batched processing
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model dynamically based on data dimensionality
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = InversionSpectrumMLP(input_dim, output_dim=output_dim)
    
    # Define regression loss and the Adam optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    # Training Loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            
            predictions = model(batch_X)     # Forward pass
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Log training progress
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
            
    return model

if __name__ == "__main__":
    # --- Example Usage Script ---
    # In a real scenario, you can generate X_train (measurements b) by running the forward PFF 
    # model (theory = new_EDRM * input) on numerous samples of randomly initialized true parameter
    # variables (a1-a6) from some feasible domain space (bounds defined by lb and ub).
    
    num_samples = 2000
    b_dim = 24  # Assuming the condensed spectral vector 'b' has 24 dimensions
    
    # Simulated dummy data
    print("Generating simulated dataset...")
    X_dummy = np.random.rand(num_samples, b_dim)
    # Target vectors containing random bounded values for 6 parameters
    y_dummy = np.random.rand(num_samples, 6) 
    
    print("\nInitializing Neural Network (Multilayer Perceptron)...")
    model = train_inversion_nn(X_dummy, y_dummy, epochs=500)
    
    print("\nTraining Complete! You can now pass experimental vector 'b' into the model predictor: ")
    print("predicted_params = model(torch.tensor(b_experiment_vector, dtype=torch.float32))")
