import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class SimpleMonotonicNN(torch.nn.Module):
    def __init__(self, other_input_dim):
        super().__init__()
        self.unconstrained_path = torch.nn.Sequential(
            torch.nn.Linear(other_input_dim, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, 8),  # 8 features including logistic
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
    target = ['Arcuate_Sweep_Half']
    y = df[target]
    x = df['treated_astig_half'].to_numpy()
    
    other_features = [
        'Age', 'Steep_axis_term', 'MeanK_IOLMaster', 'Treatment_astigmatism_half', 'WTW_IOLMaster'
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
        'logistic1': 1 / (1 + np.exp(-(x+1))),
        'logistic2': 1 / (1 + np.exp(-(x+0.5))),
        'logistic3': 1 / (1 + np.exp(-x)),
        'logarithmic': np.log(x - x.min() + 1),
        'logistic4': 1 / (1 + np.exp(-(x-0.5))),
        'logistic5': 1 / (1 + np.exp(-(x-1)))
    }
    
    X_monotonic = pd.DataFrame(monotonic_features_dict)
    X_other = df[other_features].copy()
    
    # Get scalers and encoder from components
    other_scaler = components['other_scaler']
    monotonic_scaler = components['monotonic_scaler']
    target_scaler = components['target_scaler']
    label_encoder = components['label_encoder']
    
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

def analyze_feature_importance(model, data_sample):
    """
    Analyze the importance of each feature by examining weights and transformations
    """
    print("\nFeature Analysis:")
    print("-----------------")
    
    # Feature names for reference - updated to match monotonic_features_dict
    monotonic_features = ['Constant', 'Linear', 'Logistic1', 'Logistic2', 
                         'Logistic3', 'Logarithmic', 'Logistic4', 'Logistic5']
    
    # Get weights from the model for this sample
    with torch.no_grad():
        weights = model.unconstrained_path(data_sample['x_other'])
    
    # Get monotonic features for this sample
    x_monotonic = data_sample['x_monotonic']
    
    # Calculate contribution of each feature
    contributions = weights * x_monotonic
    contributions = contributions.numpy()[0]
    
    # Print analysis
    print("\nFeature Contributions:")
    for i, (feature, contribution) in enumerate(zip(monotonic_features, contributions)):
        print(f"{feature:12} Weight: {weights[0][i]:8.4f}  "
              f"Value: {x_monotonic[0][i]:8.4f}  "
              f"Contribution: {contribution:8.4f}")
    
    # Calculate percentage contributions
    total_contribution = abs(contributions).sum()
    print("\nRelative Importance:")
    for feature, contribution in zip(monotonic_features, contributions):
        percentage = (abs(contribution) / total_contribution) * 100
        print(f"{feature:12} {percentage:6.2f}%")

def plot_treated_astig_vs_sweep():
    """
    Plot treated_astig against Arcuate_Sweep
    """
    # Load data
    df = pd.read_csv('datacombo.csv')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['treated_astig_half'], df['Arcuate_Sweep_Half'], alpha=0.5)
    plt.xlabel('Treated Astigmatism')
    plt.ylabel('Arcuate Sweep')
    plt.title('Treated Astigmatism vs Arcuate Sweep')
    
    # Add trend line
    z = np.polyfit(df['treated_astig_half'], df['Arcuate_Sweep_Half'], 1)
    p = np.poly1d(z)
    plt.plot(df['treated_astig_half'], p(df['treated_astig_half']), "r--", alpha=0.8)
    
    # Set both axes to start at 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('treated_astig_vs_sweep.png')
    plt.close()

def plot_feature_importance(model, data_sample):
    """
    Create a bar plot showing the relative importance of monotonic features
    """
    # Feature names
    monotonic_features = ['Constant', 'Linear', 'Logistic1', 'Logistic2', 
                         'Logistic3', 'Logarithmic', 'Logistic4', 'Logistic5']
    
    # Get weights and contributions
    with torch.no_grad():
        weights = model.unconstrained_path(data_sample['x_other'])
    x_monotonic = data_sample['x_monotonic']
    contributions = weights * x_monotonic
    contributions = contributions.numpy()[0]
    
    # Calculate absolute percentage contributions
    total_contribution = abs(contributions).sum()
    percentages = [(abs(contribution) / total_contribution) * 100 for contribution in contributions]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(monotonic_features, percentages)
    
    # Customize plot
    plt.title('Relative Importance of Monotonic Features')
    plt.xlabel('Features')
    plt.ylabel('Relative Importance (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Add grid and adjust layout
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    # First run analyze_model_performance which creates the model
    rmse, mae, r2 = analyze_model_performance()
    
    # Define other_features list
    other_features = [
        'Age', 'Steep_axis_term', 'MeanK_IOLMaster', 
        'Treatment_astigmatism_half', 'WTW_IOLMaster'
    ]
    
    # Load model and components
    model_state_dict = torch.load('model_weights.pth', map_location=torch.device('cpu'), weights_only=True)
    components = joblib.load('model_components.joblib')
    
    # Initialize model
    model = SimpleMonotonicNN(len(other_features))
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Get scalers from components
    other_scaler = components['other_scaler']
    monotonic_scaler = components['monotonic_scaler']
    target_scaler = components['target_scaler']
    
    # Add feature analysis
    print("\nAnalyzing sample prediction...")
    sample_data = {
        'x_other': torch.FloatTensor(other_scaler.transform([[65, 0.5, 44.0, 1.0, 12.0]])),
        'x_monotonic': torch.FloatTensor(monotonic_scaler.transform([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Include logistic
        ]))
    }
    analyze_feature_importance(model, sample_data) 
    
    # Create treated_astig vs sweep plot
    plot_treated_astig_vs_sweep()
    
    # Create feature importance plot
    plot_feature_importance(model, sample_data)