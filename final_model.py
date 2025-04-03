import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from torch.utils.data import TensorDataset, DataLoader

# Initialize encoders and scalers
le = LabelEncoder()
other_scaler = StandardScaler()
target_scaler = StandardScaler()

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

# Load dataset
df = pd.read_csv('datacombo.csv')

# Filter out 'single' type entries
df = df[df['type'] != 'single']

print(f"\nTotal number of entries in dataset: {len(df)}")
# Setting up features and target
target = ['Arcuate_Sweep']
y = df[target]
x = df['treated_astig'].to_numpy()

other_features = [
    'Age', 'Steep_axis_term', 'MeanK_IOLMaster', 'Residual_Astigmatism', 'WTW_IOLMaster', 'treated_astig'
]

# Handle NaN values
wtw_median = df['WTW_IOLMaster'].median()
meank_median = df['MeanK_IOLMaster'].median()
df['WTW_IOLMaster'] = df['WTW_IOLMaster'].fillna(wtw_median)
df['MeanK_IOLMaster'] = df['MeanK_IOLMaster'].fillna(meank_median)


# Convert to DataFrame and keep as DataFrame
X_other = df[other_features].copy()

# Scale while maintaining DataFrame structure
X_other_scaled = pd.DataFrame(
    other_scaler.fit_transform(X_other),
    columns=X_other.columns,
    index=X_other.index
)



y_scaled = pd.DataFrame(
    target_scaler.fit_transform(y.values.reshape(-1, 1)),
    columns=['target'],
    index=y.index
)

# Convert to tensors
x_other_tensor = torch.FloatTensor(X_other_scaled.values)
# x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled.values)
y_tensor = torch.FloatTensor(y_scaled.values)

# Create dataset and dataloader
batch_size = 32  # You can adjust this value
dataset = TensorDataset(x_other_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = ArcuateSweepPredictor(len(other_features))
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
    epoch_loss = 0
    batch_count = 0
    
    for batch_other, batch_y in dataloader:
        outputs = model(batch_other)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
    
    avg_epoch_loss = epoch_loss / batch_count
    
    # Early stopping check
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
        # Save model state separately from other components
        torch.save(model.state_dict(), 'model_weights.pth')
        # Save other components using pickle or joblib
        joblib.dump({
            'other_scaler': other_scaler,
            'target_scaler': target_scaler,
            'label_encoder': le
        }, 'model_components.joblib')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
        
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}')

# Final evaluation
print("\nFINAL MODEL PERFORMANCE:")
model.eval()
with torch.no_grad():
    final_outputs = model(x_other_tensor)
    predictions = target_scaler.inverse_transform(final_outputs.numpy())
    predictions = np.maximum(0.0, predictions)  # Ensure non-negative
    y_original = target_scaler.inverse_transform(y_tensor.numpy())
    
    rmse = np.sqrt(mean_squared_error(y_original, predictions))
    mae = mean_absolute_error(y_original, predictions)
    r2 = r2_score(y_original, predictions)
    
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R² Score: {r2:.4f}')
   

