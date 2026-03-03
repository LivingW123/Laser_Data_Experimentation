"""
Model Comparison: MLP vs Random Forest vs 1D CNN
for PFF Inversion Parameter Prediction

This script:
1. Loads the real 200x200 EDRM matrix
2. Translates the PFF forward model from MATLAB into Python
3. Generates realistic synthetic training data by sampling parameter ranges
4. Trains all 3 models on identical data splits
5. Compares performance with metrics and visualizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import time
import os

# Import the model classes from your existing files
from inversion_nn import InversionSpectrumMLP
from inversion_other_models import InversionSpectrum1DCNN

# ============================================================================
# 1. FORWARD MODEL (translated from PFF.m)
# ============================================================================
# f_E(params, xdata) = a1*exp(-a2*x) + a3*exp(-(x-a4)^2 / (a5*x + a6/x))

def f_E(params, xdata):
    """
    PFF forward model translated from MATLAB.
    params: array of 6 parameters [a1, a2, a3, a4, a5, a6]
    xdata:  1D array of x positions (1 to x_ran)
    Returns: the unfolded spectrum vector
    """
    a1, a2, a3, a4, a5, a6 = params
    x = xdata.astype(np.float64)
    
    term1 = a1 * np.exp(-a2 * x)
    
    # Protect against division by zero in the denominator
    denominator = a5 * x + a6 / x
    # Clamp to avoid divide-by-zero or extremely small denominators
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12 * np.sign(denominator + 1e-30), denominator)
    
    term2 = a3 * np.exp(-(x - a4)**2 / denominator)
    
    return term1 + term2


def generate_measurement(params, xdata, new_EDRM, noise_frac=0.05):
    """
    Simulates the measured detector response b = new_EDRM * f_E(params, xdata) + noise
    This mirrors PFF.m: theory = new_EDRM * input; b = theory + noise
    """
    spectrum = f_E(params, xdata)
    theory = new_EDRM @ spectrum
    # Add realistic multiplicative noise (same as PFF.m)
    noise = (np.random.rand(len(theory)) - 0.5) * theory * noise_frac
    return theory + noise


# ============================================================================
# 2. LOAD EDRM & BUILD GROUPED MATRIX (same as PFF.m lines 5-11)
# ============================================================================

def load_edrm(xlsx_path):
    """Load the 200x200 EDRM matrix and group columns by 2 (matching PFF.m)."""
    # MATLAB's readtable skips the header row automatically; replicate that here
    sample = pd.read_excel(xlsx_path)
    x200 = sample.values.astype(np.float64)
    print(f"  Raw x200 shape: {x200.shape}, NaN count: {np.isnan(x200).sum()}")
    EDRM = x200.T  # EDRM = x200'
    
    gp_sz = 2
    size_0 = EDRM.shape[0]
    x_ran = size_0 // gp_sz  # 100
    new_EDRM = np.zeros((size_0, x_ran))
    for i in range(x_ran):
        new_EDRM[:, i] = EDRM[:, gp_sz*i : gp_sz*i + gp_sz].sum(axis=1)
    
    xdata = np.arange(1, x_ran + 1, dtype=np.float64)
    return new_EDRM, xdata, x_ran


# ============================================================================
# 3. GENERATE SYNTHETIC DATASET
# ============================================================================

def generate_dataset(new_EDRM, xdata, num_samples=5000, noise_frac=0.05):
    """
    Generate synthetic (measurement, parameter) pairs for training.
    Parameter ranges are informed by the various x0 initial guesses in PFF.m:
      x0 = [1.8e-7, 0.4, 9, -28.6, 5.1, -5]
      x0 = [4e-7, 0.309, 9, -22.8, 5, -1]
      x0 = [5e-7, 0.4, 8, -12, 8, 10]
    
    a1 is sampled log-uniformly since it spans orders of magnitude (1e-8 to 1e-5).
    All other parameters are sampled uniformly within their bounds.
    The returned y_data stores log10(a1) instead of raw a1 so all targets
    live on comparable scales for the ML models.
    """
    X_data = []  # measurements (b vectors)
    y_data = []  # true parameters (with log10 a1)
    
    valid = 0
    attempts = 0
    while valid < num_samples and attempts < num_samples * 5:
        attempts += 1
        
        # a1: log-uniform between 1e-8 and 1e-5
        a1 = 10 ** np.random.uniform(-8, -5)
        a2 = np.random.uniform(0.05, 0.8)
        a3 = np.random.uniform(1.0, 15.0)
        a4 = np.random.uniform(-35.0, -5.0)
        a5 = np.random.uniform(1.0, 12.0)
        a6 = np.random.uniform(-10.0, 15.0)
        
        params = np.array([a1, a2, a3, a4, a5, a6])
        
        try:
            b = generate_measurement(params, xdata, new_EDRM, noise_frac)
            # Filter out degenerate / non-finite cases
            if np.all(np.isfinite(b)) and np.max(np.abs(b)) < 1e10:
                X_data.append(b)
                # Store log10(a1) so all targets are on comparable scales
                y_data.append(np.array([np.log10(a1), a2, a3, a4, a5, a6]))
                valid += 1
                if valid % 1000 == 0:
                    print(f"  ... generated {valid}/{num_samples} samples")
        except Exception:
            continue
    
    print(f"Generated {valid} valid samples out of {attempts} attempts.")
    return np.array(X_data), np.array(y_data)


# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================

def train_mlp(X_train, y_train, X_val, y_val, epochs=500, batch_size=64, lr=1e-3):
    """Train the MLP model and return model + training history."""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = InversionSpectrumMLP(X_train.shape[1], output_dim=y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        
        train_loss = epoch_loss / len(X_train)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"  MLP Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")
    
    return model, history


def train_cnn(X_train, y_train, X_val, y_val, epochs=500, batch_size=64, lr=1e-3):
    """Train the 1D CNN model and return model + training history."""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = InversionSpectrum1DCNN(X_train.shape[1], output_dim=y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        
        train_loss = epoch_loss / len(X_train)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"  CNN Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")
    
    return model, history


def train_rf(X_train, y_train):
    """Train the Random Forest model."""
    base_rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_rf)
    model.fit(X_train, y_train)
    return model


# ============================================================================
# 5. EVALUATION
# ============================================================================

def evaluate_model(name, model, X_test, y_test, is_torch=False):
    """Evaluate a model and return a dict of metrics."""
    if is_torch:
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            preds = model(X_t).numpy()
    else:
        preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds, multioutput='uniform_average')
    
    # Per-parameter metrics
    param_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    per_param_r2 = {}
    per_param_mse = {}
    for i, pname in enumerate(param_names):
        per_param_r2[pname] = r2_score(y_test[:, i], preds[:, i])
        per_param_mse[pname] = mean_squared_error(y_test[:, i], preds[:, i])
    
    return {
        'name': name,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'per_param_r2': per_param_r2,
        'per_param_mse': per_param_mse,
        'predictions': preds
    }


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def plot_comparison(results, y_test, histories, save_dir):
    """Generate comparison plots."""
    param_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    colors = {'MLP': '#FF6B6B', 'Random Forest': '#4ECDC4', '1D CNN': '#45B7D1'}
    
    # --- Plot 1: Training curves for NN models ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(hist['train_loss'], label='Train', color=colors[name], alpha=0.8)
        ax.plot(hist['val_loss'], label='Validation', color=colors[name], linestyle='--', alpha=0.8)
        ax.set_title(f'{name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Overall metrics bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Overall Model Comparison', fontsize=16, fontweight='bold')
    
    model_names = [r['name'] for r in results]
    bar_colors = [colors[n] for n in model_names]
    
    # MSE
    mses = [r['mse'] for r in results]
    axes[0].bar(model_names, mses, color=bar_colors, edgecolor='white', linewidth=2)
    axes[0].set_title('Mean Squared Error (lower = better)')
    axes[0].set_ylabel('MSE')
    axes[0].grid(axis='y', alpha=0.3)
    
    # MAE
    maes = [r['mae'] for r in results]
    axes[1].bar(model_names, maes, color=bar_colors, edgecolor='white', linewidth=2)
    axes[1].set_title('Mean Absolute Error (lower = better)')
    axes[1].set_ylabel('MAE')
    axes[1].grid(axis='y', alpha=0.3)
    
    # R2
    r2s = [r['r2'] for r in results]
    axes[2].bar(model_names, r2s, color=bar_colors, edgecolor='white', linewidth=2)
    axes[2].set_title('R² Score (higher = better)')
    axes[2].set_ylabel('R²')
    axes[2].set_ylim([min(0, min(r2s) - 0.1), 1.05])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Per-parameter R² heatmap ---
    fig, ax = plt.subplots(figsize=(10, 4))
    r2_matrix = np.array([[r['per_param_r2'][p] for p in param_names] for r in results])
    
    im = ax.imshow(r2_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, fontsize=12)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=12)
    ax.set_title('Per-Parameter R² Score', fontsize=14, fontweight='bold')
    
    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(param_names)):
            val = r2_matrix[i, j]
            color = 'white' if val < 0.3 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='R² Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_param_r2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 4: Predicted vs Actual scatter for each parameter ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    for i, pname in enumerate(param_names):
        ax = axes[i]
        for r in results:
            ax.scatter(y_test[:, i], r['predictions'][:, i], alpha=0.3, s=10,
                      color=colors[r['name']], label=r['name'])
        
        # Perfect prediction line
        lims = [y_test[:, i].min(), y_test[:, i].max()]
        ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.7, label='Perfect')
        ax.set_xlabel(f'True {pname}')
        ax.set_ylabel(f'Predicted {pname}')
        ax.set_title(f'Parameter: {pname}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Predicted vs Actual Parameters', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pred_vs_actual.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll plots saved to: {save_dir}")


# ============================================================================
# 7. MAIN
# ============================================================================

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    XLSX_PATH = os.path.join(SCRIPT_DIR, "200x200.xlsx")
    SAVE_DIR = os.path.join(SCRIPT_DIR, "comparison_results")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    NUM_SAMPLES = 5000
    EPOCHS = 500
    NOISE_FRAC = 0.05
    
    # ----- Load EDRM -----
    print("=" * 70)
    print("LOADING EDRM MATRIX FROM 200x200.xlsx")
    print("=" * 70)
    new_EDRM, xdata, x_ran = load_edrm(XLSX_PATH)
    print(f"  EDRM shape: {new_EDRM.shape}, xdata range: 1 to {x_ran}")
    
    # ----- Generate synthetic data -----
    print("\n" + "=" * 70)
    print(f"GENERATING {NUM_SAMPLES} SYNTHETIC TRAINING SAMPLES")
    print("=" * 70)
    X_all, y_all = generate_dataset(new_EDRM, xdata, num_samples=NUM_SAMPLES, noise_frac=NOISE_FRAC)
    print(f"  X shape: {X_all.shape}  (measurement vectors)")
    print(f"  y shape: {y_all.shape}  (6 PFF parameters)")
    
    # ----- Normalize -----
    # MinMaxScaler is more robust here since a1 is stored as log10 and other
    # params have different scales but known bounded ranges.
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_all)
    y_scaled = scaler_y.fit_transform(y_all)
    
    # ----- Split data -----
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # ----- Train all 3 models -----
    print("\n" + "=" * 70)
    print("TRAINING MODEL 1: MULTILAYER PERCEPTRON (MLP)")
    print("=" * 70)
    t0 = time.time()
    mlp_model, mlp_hist = train_mlp(X_train, y_train, X_val, y_val, epochs=EPOCHS)
    mlp_time = time.time() - t0
    print(f"  MLP Training Time: {mlp_time:.1f}s")
    
    print("\n" + "=" * 70)
    print("TRAINING MODEL 2: RANDOM FOREST")
    print("=" * 70)
    t0 = time.time()
    rf_model = train_rf(X_train, y_train)
    rf_time = time.time() - t0
    print(f"  Random Forest Training Time: {rf_time:.1f}s")
    
    print("\n" + "=" * 70)
    print("TRAINING MODEL 3: 1D CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("=" * 70)
    t0 = time.time()
    cnn_model, cnn_hist = train_cnn(X_train, y_train, X_val, y_val, epochs=EPOCHS)
    cnn_time = time.time() - t0
    print(f"  CNN Training Time: {cnn_time:.1f}s")
    
    # ----- Evaluate all models -----
    print("\n" + "=" * 70)
    print("EVALUATING ALL MODELS ON TEST SET")
    print("=" * 70)
    
    results = [
        evaluate_model('MLP', mlp_model, X_test, y_test, is_torch=True),
        evaluate_model('Random Forest', rf_model, X_test, y_test, is_torch=False),
        evaluate_model('1D CNN', cnn_model, X_test, y_test, is_torch=True),
    ]
    
    # ----- Print Results Table -----
    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'MSE':>12} {'MAE':>12} {'R²':>12} {'Time (s)':>12}")
    print("-" * 70)
    times = [mlp_time, rf_time, cnn_time]
    for r, t in zip(results, times):
        print(f"{r['name']:<20} {r['mse']:>12.6f} {r['mae']:>12.6f} {r['r2']:>12.4f} {t:>12.1f}")
    print("-" * 70)
    
    print("\nPer-Parameter R² Breakdown:")
    print(f"{'Model':<20}", end="")
    for p in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']:
        print(f"  {p:>8}", end="")
    print()
    print("-" * 74)
    for r in results:
        print(f"{r['name']:<20}", end="")
        for p in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']:
            print(f"  {r['per_param_r2'][p]:>8.4f}", end="")
        print()
    
    # ----- Generate Plots -----
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    histories = {'MLP': mlp_hist, '1D CNN': cnn_hist}
    plot_comparison(results, y_test, histories, SAVE_DIR)
    
    print("\n[DONE] Comparison complete! Check the 'comparison_results' folder for plots.")
