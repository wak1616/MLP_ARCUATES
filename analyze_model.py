import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# Just use basic seaborn styling
sns.set_theme(style="whitegrid")

class SimpleMonotonicNN(nn.Module):
    def __init__(self, regular_input_dim):
        super().__init__()
        self.regular_path = nn.Sequential(
            nn.Linear(regular_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Base monotonic weight
        self.monotonic_weight = nn.Parameter(torch.tensor([0.1]))
    
    def forward(self, x_regular, x_monotonic):
        regular_out = self.regular_path(x_regular)
        monotonic_out = F.softplus(self.monotonic_weight) * x_monotonic
        return regular_out + monotonic_out

# Load the trained model and preprocessing components
model = SimpleMonotonicNN(regular_input_dim=5)
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

regular_scaler = joblib.load('regular_scaler.pkl')
monotonic_scaler = joblib.load('monotonic_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Load and preprocess data
df = pd.read_csv('datacombo.csv')
regular_features = [
    'Age', 'Steep_axis_term', 'type', 'Residual_Astigmatism', 'treated_astig'
]
X_regular = df[regular_features].copy()
X_regular['type'] = le.transform(X_regular['type'])
y = df[['Arcuate_Sweep_Total']]
X_monotonic = df[['treated_astig']]

# Scale the data
X_regular_scaled = regular_scaler.transform(X_regular)
X_monotonic_scaled = monotonic_scaler.transform(X_monotonic)

# Get predictions
with torch.no_grad():
    x_regular_tensor = torch.FloatTensor(X_regular_scaled)
    x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled)
    predictions_scaled = model(x_regular_tensor, x_monotonic_tensor)
    predictions = target_scaler.inverse_transform(predictions_scaled.numpy())
    
    # Apply the threshold rule: set prediction to 0.00 where treated_astig < -0.25
    mask = (X_monotonic['treated_astig'] < 0.25).values
    predictions[mask] = 0.00

# 1. Predictions vs Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Arcuate Sweep (degrees)')
plt.ylabel('Predicted Arcuate Sweep (degrees)')
plt.title('Predicted vs Actual Arcuate Sweep Values')
r2 = r2_score(y, predictions)
plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
plt.savefig('predicted_vs_actual.png')
plt.close()

# 2. Error Distribution
errors = predictions - y.values
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel('Prediction Error (degrees)')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.axvline(x=0, color='r', linestyle='--')
plt.text(0.05, 0.95, f'Mean Error: {np.mean(errors):.2f}°\nStd Dev: {np.std(errors):.2f}°', 
         transform=plt.gca().transAxes)
plt.savefig('error_distribution.png')
plt.close()

# 3. Error vs Astigmatism
plt.figure(figsize=(10, 6))
plt.scatter(df['treated_astig'], np.abs(errors), alpha=0.5)
plt.xlabel('Ideal Tx Astigmatism')
plt.ylabel('Absolute Error (degrees)')
plt.title('Prediction Error vs Astigmatism')
plt.savefig('error_vs_astigmatism.png')
plt.close()

# 4. Error vs Age
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], np.abs(errors), alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Absolute Error (degrees)')
plt.title('Prediction Error vs Age')
plt.savefig('error_vs_age.png')
plt.close()

# 5. Box Plot by Type
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['type'], y=errors.flatten())
plt.xlabel('Type (Paired vs Single)')
plt.ylabel('Error (degrees)')
plt.title('Error Distribution by Type')
plt.savefig('error_by_type.png')
plt.close()

# 6. Heatmap of Errors
plt.figure(figsize=(12, 8))
error_pivot = pd.DataFrame({
    'Age_Group': pd.qcut(df['Age'], q=5, labels=['Very Young', 'Young', 'Middle', 'Old', 'Very Old']),
    'Astig_Group': pd.qcut(df['treated_astig'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']),
    'Error': np.abs(errors.flatten())
}).pivot_table(values='Error', index='Age_Group', columns='Astig_Group', aggfunc='mean', observed=False)

sns.heatmap(error_pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Average Absolute Error by Age and Astigmatism Groups')
plt.savefig('error_heatmap.png')
plt.close()

# 7. Prediction vs Actual by Type
plt.figure(figsize=(12, 6))
for type_name in df['type'].unique():
    mask = df['type'] == type_name
    plt.scatter(y[mask], predictions[mask], alpha=0.5, label=type_name)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Arcuate Sweep (degrees)')
plt.ylabel('Predicted Arcuate Sweep (degrees)')
plt.title('Predicted vs Actual Values by Type')
plt.legend()
plt.savefig('predicted_vs_actual_by_type.png')
plt.close()

# 8. Residuals Plot
plt.figure(figsize=(10, 6))
plt.scatter(predictions, errors, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values (degrees)')
plt.ylabel('Residuals (degrees)')
plt.title('Residuals vs Predicted Values')
plt.savefig('residuals_plot.png')
plt.close()

# Add correlation analysis
print("\nCorrelation with Absolute Error:")
correlations = pd.DataFrame({
    'Age': df['Age'],
    'Astigmatism': df['treated_astig'],
    'Abs_Error': np.abs(errors.flatten())
}).corr()['Abs_Error'].drop('Abs_Error')
print(correlations)

# Add error quartile analysis
print("\nError Quartile Analysis:")
error_quartiles = pd.qcut(np.abs(errors.flatten()), q=4, labels=['Best', 'Good', 'Fair', 'Worst'])
quartile_analysis = pd.DataFrame({
    'Error_Quartile': error_quartiles,
    'Age': df['Age'],
    'Astigmatism': df['treated_astig'],
    'Type': df['type']
}).groupby('Error_Quartile').agg({
    'Age': ['mean', 'std'],
    'Astigmatism': ['mean', 'std'],
    'Type': lambda x: (x == 'Single').mean()
})
print(quartile_analysis)

# 4. Performance Analysis
error_df = pd.DataFrame({
    'Actual': y['Arcuate_Sweep_Total'],
    'Predicted': predictions.flatten(),
    'Error': errors.flatten(),
    'Abs_Error': np.abs(errors.flatten()),
    'Age': df['Age'],
    'Type': df['type'],
    'Astigmatism': df['treated_astig']
})

# Print performance analysis
print("\nPerformance Analysis:")
print(f"Overall R² Score: {r2:.4f}")
print(f"\nError Statistics:")
print(f"Mean Error: {np.mean(errors):.2f}°")
print(f"Std Dev: {np.std(errors):.2f}°")
print(f"Mean Absolute Error: {np.mean(np.abs(errors)):.2f}°")
print(f"Median Absolute Error: {np.median(np.abs(errors)):.2f}°")
print(f"95th Percentile Error: {np.percentile(np.abs(errors), 95):.2f}°")

# 5. Worst Cases Analysis
print("\nWorst 10 Predictions:")
worst_cases = error_df.nlargest(10, 'Abs_Error')
print(worst_cases[['Actual', 'Predicted', 'Error', 'Age', 'Type', 'Astigmatism']])

# 6. Performance by Type
print("\nPerformance by Type:")
for type_val in error_df['Type'].unique():
    mask = error_df['Type'] == type_val
    type_r2 = r2_score(error_df[mask]['Actual'], error_df[mask]['Predicted'])
    type_mae = np.mean(error_df[mask]['Abs_Error'])
    print(f"\nType {type_val}:")
    print(f"R² Score: {type_r2:.4f}")
    print(f"Mean Absolute Error: {type_mae:.2f}°")

# Analyze the dual effect of treated_astig
print("\nAnalyzing treated_astig effect:")
print(f"Direct Monotonic Weight: {F.softplus(model.monotonic_weight).item():.4f}")
print("\nNote: This is only the direct monotonic effect.")
print("The total effect includes additional contributions through the regular path.")

# We could analyze the total marginal effect
with torch.no_grad():
    # Create two versions of the input with small difference in treated_astig
    delta = 0.1
    x_regular_base = x_regular_tensor.clone()
    x_regular_delta = x_regular_tensor.clone()
    x_regular_delta[:, 4] += delta  # Assuming treated_astig is the 5th feature
    
    x_monotonic_base = x_monotonic_tensor.clone()
    x_monotonic_delta = x_monotonic_tensor.clone()
    x_monotonic_delta += delta
    
    # Get predictions
    pred_base = model(x_regular_base, x_monotonic_base)
    pred_delta = model(x_regular_delta, x_monotonic_delta)
    
    # Calculate average marginal effect
    marginal_effect = ((pred_delta - pred_base) / delta).mean().item()
    
print(f"\nAverage Marginal Effect of treated_astig: {marginal_effect:.4f}")
print("(This represents the average total effect including both paths)") 