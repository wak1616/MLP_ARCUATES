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
    df = pd.read_csv('datacombo.csv')

    # Filter out 'single' type entries
    df = df[df['type'] != 'single']
    
    # Prepare features as in final_model.py
    target = ['Arcuate_Sweep']
    y = df[target]
    x = df['treated_astig'].to_numpy()
    
    other_features = [
        'Age', 'Steep_axis_term', 'MeanK_IOLMaster', 'Residual_Astigmatism', 'WTW_IOLMaster'
    ]
    
    # Handle NaN values
    wtw_median = df['WTW_IOLMaster'].median()
    meank_median = df['MeanK_IOLMaster'].median()
    df['WTW_IOLMaster'] = df['WTW_IOLMaster'].fillna(wtw_median)
    df['MeanK_IOLMaster'] = df['MeanK_IOLMaster'].fillna(meank_median)
    
    # Create features
    monotonic_features_dict = {
        'constant': np.ones_like(x),
        'linear': x,
        'logistic_shift_left_1': 1 / (1 + np.exp(-(x+1))),
        'logistic_shift_left_0.5': 1 / (1 + np.exp(-(x+0.5))),
        'logistic_center': 1 / (1 + np.exp(-x)),
        'logarithmic': np.log(x - x.min() + 1),
        'logistic_shift_right_0.5': 1 / (1 + np.exp(-(x-0.5))),
        'logistic_shift_right_1': 1 / (1 + np.exp(-(x-1))),
        'logistic_shift_right_1.5': 1 / (1 + np.exp(-(x-1.5))),
        'logistic_shift_left_1.5': 1 / (1 + np.exp(-(x+1.5)))
    }
    
    X_monotonic = pd.DataFrame(monotonic_features_dict)
    X_other = df[other_features].copy()
    
    # Get scalers and encoder from components
    other_scaler = components['other_scaler']
    monotonic_scaler = components['monotonic_scaler']
    target_scaler = components['target_scaler']
    # label_encoder = components['label_encoder']
    
    # Transform and scale features
    # X_other['type'] = label_encoder.transform(X_other['type'])
    X_other_scaled = pd.DataFrame(
        other_scaler.transform(X_other),
        columns=X_other.columns
    )
    X_monotonic_scaled = pd.DataFrame(
        monotonic_scaler.transform(X_monotonic),
        columns=X_monotonic.columns
    )
    
    # Initialize and load model
    model = SimpleMonotonicNN(len(other_features))
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        x_other_tensor = torch.FloatTensor(X_other_scaled.values)
        x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled.values)
        predictions_scaled = model(x_other_tensor, x_monotonic_tensor)
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
    Plot treated_astig against Arcuate_Sweep
    """
    # Load data
    df = pd.read_csv('datacombo.csv')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['treated_astig'], df['Arcuate_Sweep'], alpha=0.5)
    plt.xlabel('Treated Astigmatism')
    plt.ylabel('Arcuate Sweep')
    plt.title('Treated Astigmatism vs Arcuate Sweep')
    
    # Add trend line
    z = np.polyfit(df['treated_astig'], df['Arcuate_Sweep'], 1)
    p = np.poly1d(z)
    plt.plot(df['treated_astig'], p(df['treated_astig']), "r--", alpha=0.8)
    
    # Set both axes to start at 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('treated_astig_vs_sweep.png')
    plt.close()

def main():
    # Analyze performance on the full dataset
    analyze_model_performance()

    # Create treated_astig vs sweep plot
    plot_treated_astig_vs_sweep()

if __name__ == "__main__":
    main()