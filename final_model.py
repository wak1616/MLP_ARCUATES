import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# F.softplus(x) = log(1 + exp(x))

class SimpleMonotonicNN(nn.Module):
    def __init__(self, regular_input_dim):
        super().__init__()
        
        # Regular path (Unconstrained Path) - learns complex patterns
        self.regular_path = nn.Sequential(
            nn.Linear(regular_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Monotonic weight (Constrained, Monotonic Path)
        # Softplus ensures positive weight, maintaining monotonicity
        self.monotonic_weight = nn.Parameter(torch.tensor([0.1]))
        
        for m in self.regular_path.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_regular, x_monotonic):
        regular_out = self.regular_path(x_regular)
        monotonic_out = F.softplus(self.monotonic_weight) * x_monotonic
        return regular_out + monotonic_out

# Load and prepare data
df = pd.read_csv('datacombo.csv')

# Define features and target
target = ['Arcuate_Sweep_Total']
regular_features = [
    'Age', 'Steep_axis_term', 'type', 'Residual_Astigmatism', 'ideal_tx_astig'
]

# Prepare data
X_regular = df[regular_features].copy()
le = LabelEncoder()
X_regular['type'] = le.fit_transform(X_regular['type'])
y = df[target]
X_monotonic = df[['ideal_tx_astig']]

# Scale the features
regular_scaler = StandardScaler()
monotonic_scaler = StandardScaler()
target_scaler = StandardScaler()

# Create pandas DataFrames with named columns for scalers
X_regular_scaled = pd.DataFrame(
    regular_scaler.fit_transform(X_regular),
    columns=regular_features
)
X_monotonic_scaled = pd.DataFrame(
    monotonic_scaler.fit_transform(X_monotonic),
    columns=['ideal_tx_astig']
)
y_scaled = target_scaler.fit_transform(y)

# Convert to tensors
x_regular_tensor = torch.FloatTensor(X_regular_scaled.values)
x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled.values)
y_tensor = torch.FloatTensor(y_scaled)

# Initialize model
model = SimpleMonotonicNN(
    regular_input_dim=len(regular_features)
)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# Training loop
print("\nTraining final model...")
num_epochs = 1000
best_loss = float('inf')
patience = 50
patience_counter = 0
scheduler_patience = 20

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(x_regular_tensor, x_monotonic_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        torch.save(model.state_dict(), 'model_weights.pth', _use_new_zipfile_serialization=True)
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
        
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Load best model weights
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

# Save scalers and encoder
joblib.dump(regular_scaler, 'regular_scaler.pkl')
joblib.dump(monotonic_scaler, 'monotonic_scaler.pkl')
joblib.dump(target_scaler, 'target_scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Evaluate final model performance
print("\nFINAL MODEL PERFORMANCE:")
with torch.no_grad():
    all_outputs = model(x_regular_tensor, x_monotonic_tensor)
    predictions = target_scaler.inverse_transform(all_outputs.numpy())
    y_original = target_scaler.inverse_transform(y_tensor.numpy())
    
    rmse = np.sqrt(mean_squared_error(y_original, predictions))
    r2 = r2_score(y_original, predictions)
    
    print(f"Final Monotonic Weight: {F.softplus(model.monotonic_weight).item():.4f}") 
    print(f'RMSE: {rmse:.2f}°')
    print(f'R² Score: {r2:.4f}')
   

