import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    
    # Prepare features as in final_model.py
    target = ['Arcuate_Sweep_Total']
    y = df[target]
    x = df['treated_astig'].to_numpy()
    
    other_features = [
        'Age', 'Steep_axis_term', 'type', 'MeanK_IOLMaster', 
        'Treatment_astigmatism', 'WTW_IOLMaster'
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
        'quadratic': x**2,
        'cubic': x**3,
        'quartic': x**4,
        'logarithmic': np.log(x - x.min() + 1),
        'exponential': np.exp(x)
    }
    
    X_monotonic = pd.DataFrame(monotonic_features_dict)
    X_other = df[other_features].copy()
    
    # Get scalers and encoder from components
    other_scaler = components['other_scaler']
    monotonic_scaler = components['monotonic_scaler']
    target_scaler = components['target_scaler']
    label_encoder = components['label_encoder']
    
    # Transform and scale features
    X_other['type'] = label_encoder.transform(X_other['type'])
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

if __name__ == "__main__":
    # Add SimpleMonotonicNN class definition
    class SimpleMonotonicNN(torch.nn.Module):
        def __init__(self, other_input_dim):
            super().__init__()
            self.unconstrained_path = torch.nn.Sequential(
                torch.nn.Linear(other_input_dim, 24),
                torch.nn.ReLU(),
                torch.nn.Linear(24, 7),
                torch.nn.ReLU()
            )
            
            # Initialize weights
            for m in self.unconstrained_path.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        
        def forward(self, x_other, x_monotonic):
            weights = self.unconstrained_path(x_other)
            weighted_features = weights * x_monotonic
            return weighted_features.sum(dim=1, keepdim=True)
    
    analyze_model_performance() 