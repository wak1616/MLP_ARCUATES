import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW


# F.softplus(x) = log(1 + exp(x))

class SimpleMonotonicNN(nn.Module):
    def __init__(self, regular_input_dim):
        super().__init__()
        # Simple network for regular features
        self.regular_path = nn.Sequential(
            nn.Linear(regular_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Initialize monotonic weight
        self.monotonic_weight = nn.Parameter(torch.tensor([0.1]))
        
        # Initialize weights properly
        for m in self.regular_path.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_regular, x_monotonic):
        regular_out = self.regular_path(x_regular)
        # Use softplus for smoother positive constraint
        monotonic_out = F.softplus(self.monotonic_weight) * x_monotonic
        return regular_out + monotonic_out
    
    def get_monotonic_coefficient(self):
        """Returns the actual monotonic coefficient being used"""
        return F.softplus(self.monotonic_weight).item()

# load dataset
df = pd.read_csv('datacombo.csv')

# Setting up features and target
target = ['Arcuate_Sweep_Total']
regular_features = [
    'Age', 'Steep_axis_term', 'type', 'MeanK_IOLMaster', 'Treatment_astigmatism', 'WTW_IOLMaster'
]

# Create DataFrames and encode categorical data
X_regular = df[regular_features].copy()
le = LabelEncoder()
X_regular['type'] = le.fit_transform(X_regular['type'])
y = df[target]
X_monotonic = df[['treated_astig']]

# Scale the features and target separately
regular_scaler = StandardScaler()
monotonic_scaler = StandardScaler()
target_scaler = StandardScaler()

X_regular_scaled = regular_scaler.fit_transform(X_regular)
X_monotonic_scaled = monotonic_scaler.fit_transform(X_monotonic)
y_scaled = target_scaler.fit_transform(y)

# Convert to tensors
x_regular_tensor = torch.FloatTensor(X_regular_scaled)
x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled)
y_tensor = torch.FloatTensor(y_scaled)

# Convert data to numpy for splitting
X_regular_np = x_regular_tensor.numpy()
X_monotonic_np = x_monotonic_tensor.numpy()
y_np = y_tensor.numpy()

# Set training parameters
num_epochs = 500

# Set up K-fold cross validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store metrics for each fold
fold_metrics = {
    'train_losses': [],
    'val_losses': [],
    'rmse_scores': [],
    'mae_scores': [],
    'r2_scores': [],
    'monotonic_weights': []
}

print(f"\nStarting {n_folds}-fold cross-validation")

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X_regular_np)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    # Split data for this fold
    x_regular_train = torch.FloatTensor(X_regular_np[train_idx])
    x_regular_val = torch.FloatTensor(X_regular_np[val_idx])
    x_monotonic_train = torch.FloatTensor(X_monotonic_np[train_idx])
    x_monotonic_val = torch.FloatTensor(X_monotonic_np[val_idx])
    y_train = torch.FloatTensor(y_np[train_idx])
    y_val = torch.FloatTensor(y_np[val_idx])

    # Initialize model and optimizer for this fold
    model = SimpleMonotonicNN(
        regular_input_dim=len(regular_features)
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-6
    )

    # Training loop
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_regular_train, x_monotonic_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_regular_val, x_monotonic_val)
            val_loss = criterion(val_outputs, y_val)
            
            scheduler.step(val_loss)
            
            if scheduler._last_lr[0] != optimizer.param_groups[0]['lr']:
                print(f'Learning rate adjusted to: {scheduler._last_lr[0]:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            if (epoch + 1) % 50 == 0:  # Reduced printing frequency
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {loss.item():.6f}, '
                      f'Val Loss: {val_loss.item():.6f}')
        
    # Final evaluation for this fold
    model.eval()
    with torch.no_grad():
        final_val_outputs = model(x_regular_val, x_monotonic_val)
        final_val_loss = criterion(final_val_outputs, y_val)
        
        # Convert predictions back to original scale
        final_predictions = target_scaler.inverse_transform(final_val_outputs.numpy())
        y_val_original = target_scaler.inverse_transform(y_val.numpy())
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val_original, final_predictions))
        mae = mean_absolute_error(y_val_original, final_predictions)
        r2 = r2_score(y_val_original, final_predictions)
        
        # Store metrics for this fold
        fold_metrics['train_losses'].append(loss.item())
        fold_metrics['val_losses'].append(final_val_loss.item())
        fold_metrics['rmse_scores'].append(rmse)
        fold_metrics['mae_scores'].append(mae)
        fold_metrics['r2_scores'].append(r2)
        fold_metrics['monotonic_weights'].append(model.get_monotonic_coefficient())
        
        print(f'\nFold {fold + 1} Results:')
        print(f'RMSE: {rmse:.2f}')
        print(f'MAE: {mae:.2f}')
        print(f'R² Score: {r2:.4f}')
        print(f'Monotonic Weight: {model.get_monotonic_coefficient():.4f}')

# Print average metrics across all folds
print('\nAverage Results Across All Folds:')
print(f'RMSE: {np.mean(fold_metrics["rmse_scores"]):.2f} ± {np.std(fold_metrics["rmse_scores"]):.2f}')
print(f'MAE: {np.mean(fold_metrics["mae_scores"]):.2f} ± {np.std(fold_metrics["mae_scores"]):.2f}')
print(f'R² Score: {np.mean(fold_metrics["r2_scores"]):.4f} ± {np.std(fold_metrics["r2_scores"]):.4f}')
print(f'Monotonic Weight: {np.mean(fold_metrics["monotonic_weights"]):.4f} ± {np.std(fold_metrics["monotonic_weights"]):.4f}')
