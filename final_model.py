import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Initialize encoders and scalers
le = LabelEncoder()
other_scaler = StandardScaler()
monotonic_scaler = StandardScaler()
target_scaler = StandardScaler()

class SimpleMonotonicNN(nn.Module):
    def __init__(self, other_input_dim):
        super().__init__()
        self.unconstrained_path = nn.Sequential(
            nn.Linear(other_input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 8),
            nn.ReLU()
        )
        
        # Initialize weights with smaller values
        for m in self.unconstrained_path.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_other, x_monotonic):
        weights = self.unconstrained_path(x_other)
        weighted_features = weights * x_monotonic
        return weighted_features.sum(dim=1, keepdim=True)

# Load dataset
df = pd.read_csv('datacombo.csv')

# Setting up features and target
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

# Create monotonic features
monotonic_features_dict = {
    'constant': np.ones_like(x),
    'linear': x,
    'quadratic': x**2,
    'cubic': x**3,
    'quartic': x**4,
    'logarithmic': np.log(x - x.min() + 1),
    'exponential': np.exp(x),
    'logistic': 1 / (1 + np.exp(-(x-1)))
}

# Convert to DataFrame and keep as DataFrame
X_monotonic = pd.DataFrame(monotonic_features_dict)
X_other = df[other_features].copy()
X_other['type'] = le.fit_transform(X_other['type'])

# Scale while maintaining DataFrame structure
X_other_scaled = pd.DataFrame(
    other_scaler.fit_transform(X_other),
    columns=X_other.columns,
    index=X_other.index
)

X_monotonic_scaled = pd.DataFrame(
    monotonic_scaler.fit_transform(X_monotonic),
    columns=X_monotonic.columns,
    index=X_monotonic.index
)

y_scaled = pd.DataFrame(
    target_scaler.fit_transform(y.values.reshape(-1, 1)),
    columns=['target'],
    index=y.index
)

# Convert to tensors
x_other_tensor = torch.FloatTensor(X_other_scaled.values)
x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled.values)
y_tensor = torch.FloatTensor(y_scaled.values)

# Initialize model
model = SimpleMonotonicNN(len(other_features))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Training loop
print("\nTraining model...")
num_epochs = 1000
best_loss = float('inf')
patience = 30
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    outputs = model(x_other_tensor, x_monotonic_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Early stopping check
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
        # Save model state separately from other components
        torch.save(model.state_dict(), 'model_weights.pth')
        # Save other components using pickle or joblib
        joblib.dump({
            'other_scaler': other_scaler,
            'monotonic_scaler': monotonic_scaler,
            'target_scaler': target_scaler,
            'label_encoder': le
        }, 'model_components.joblib')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
        
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Final evaluation
print("\nFINAL MODEL PERFORMANCE:")
model.eval()
with torch.no_grad():
    final_outputs = model(x_other_tensor, x_monotonic_tensor)
    predictions = target_scaler.inverse_transform(final_outputs.numpy())
    predictions = np.maximum(0.0, predictions)  # Ensure non-negative
    y_original = target_scaler.inverse_transform(y_tensor.numpy())
    
    rmse = np.sqrt(mean_squared_error(y_original, predictions))
    mae = mean_absolute_error(y_original, predictions)
    r2 = r2_score(y_original, predictions)
    
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R² Score: {r2:.4f}')
   

