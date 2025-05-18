import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch.nn as nn

def load_model_and_components(weights_path='model_weights.pth', components_path='model_components.joblib'):
    # Load model weights
    model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    
    # Load components
    components = joblib.load(components_path)
    
    return model_state_dict, components

def analyze_model_performance():
    # Load model and components
    model_state_dict, components = load_model_and_components()
    
    # Load original data
    df = pd.read_csv('datafinal.csv')

    # Filter out 'single' type entries
    df = df[df['Type'] != 'single']
    
    # Prepare features as in final_model.py
    target = ['Arcuate_sweep_total']
    y = df[target]
    x = df['Treated_astig'].to_numpy()
    
    other_features = [
        'Age', 'Steep_axis_term', 'MeanK_IOLMaster', 'WTW_IOLMaster', 'Treated_astig'
    ]
    
    # Handle NaN values
    wtw_median = df['WTW_IOLMaster'].median()
    meank_median = df['MeanK_IOLMaster'].median()
    df['WTW_IOLMaster'] = df['WTW_IOLMaster'].fillna(wtw_median)
    df['MeanK_IOLMaster'] = df['MeanK_IOLMaster'].fillna(meank_median)
    
    X_other = df[other_features].copy()
    other_scaler = components['other_scaler']
    target_scaler = components['target_scaler']
    X_other_scaled = pd.DataFrame(
        other_scaler.transform(X_other),
        columns=X_other.columns
    )
    
    # Define the model architecture
    class ArcuateSweepPredictor(nn.Module):
        def __init__(self, other_input_dim):
            super().__init__()
            self.unconstrained_path = nn.Sequential(
                nn.Linear(other_input_dim, 48),
                nn.LeakyReLU(0.1),
                nn.Linear(48, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            )
        def forward(self, x_other):
            prediction = self.unconstrained_path(x_other)
            return prediction
    
    # Initialize and load model
    model = ArcuateSweepPredictor(len(other_features))
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        x_other_tensor = torch.FloatTensor(X_other_scaled.values)
        predictions_scaled = model(x_other_tensor)
        predictions = target_scaler.inverse_transform(predictions_scaled.numpy())
        predictions = np.maximum(0.0, predictions)  # Ensure non-negative
    
    # Calculate metrics
    y_true = y.values
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Actual vs Predicted
    plt.subplot(131)
    plt.scatter(y_true, predictions, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Arcuate Sweep')
    plt.ylabel('Predicted Arcuate Sweep')
    plt.title('Actual vs Predicted')
    
    # Residuals
    plt.subplot(132)
    residuals = predictions - y_true
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Arcuate Sweep')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Distribution of Residuals
    plt.subplot(133)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.savefig('model_analysis.png')
    plt.close()
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return rmse, mae, r2

def plot_treated_astig_vs_sweep():
    """
    Plot Treated_astig against Arcuate_sweep_total
    """
    # Load data
    df = pd.read_csv('datafinal.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Treated_astig'], df['Arcuate_sweep_total'], alpha=0.5)
    plt.xlabel('Treated Astigmatism')
    plt.ylabel('Arcuate Sweep Total')
    plt.title('Treated Astigmatism vs Arcuate Sweep Total')
    z = np.polyfit(df['Treated_astig'], df['Arcuate_sweep_total'], 1)
    p = np.poly1d(z)
    plt.plot(df['Treated_astig'], p(df['Treated_astig']), "r--", alpha=0.8)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.savefig('treated_astig_vs_sweep.png')
    plt.close()

def main():
    # Analyze performance on the full dataset
    analyze_model_performance()

    # Create treated_astig vs sweep plot
    plot_treated_astig_vs_sweep()

if __name__ == "__main__":
    main()