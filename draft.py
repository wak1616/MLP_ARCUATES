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
            nn.Linear(24, 7),
            nn.ReLU()
        )
        
        # Initialize weights with smaller values
        for m in self.unconstrained_path.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_other, x_monotonic):
        # Get the weights from unconstrained path
        weights = self.unconstrained_path(x_other)  # Shape: [batch_size, 7]
        
        # Element-wise multiplication with the monotonic features
        weighted_features = weights * x_monotonic  # Shape: [batch_size, 7]
        
        # Sum up all 7 weighted features for each entry in the batch
        # This reduces from [batch_size, 7] to [batch_size, 1]
        return weighted_features.sum(dim=1, keepdim=True)

# load dataset
df = pd.read_csv('datacombo.csv')

# Setting up features and target
target = ['Arcuate_Sweep_Total']
y = df[target]  # Define y from the target column
x = df['treated_astig'].to_numpy()

other_features = [
    'Age', 'Steep_axis_term', 'type', 'MeanK_IOLMaster', 'Treatment_astigmatism', 'WTW_IOLMaster'
]

# Handle NaN values
wtw_median = df['WTW_IOLMaster'].median()
meank_median = df['MeanK_IOLMaster'].median()
df['WTW_IOLMaster'] = df['WTW_IOLMaster'].fillna(wtw_median)
df['MeanK_IOLMaster'] = df['MeanK_IOLMaster'].fillna(meank_median)

# Create the monotonic feature transformations
monotonic_features_dict = {
    'constant': np.ones_like(x),
    'linear': x,
    'quadratic': x**2,
    'cubic': x**3,
    'quartic': x**4,
    'logarithmic': np.log(x - x.min() + 1),
    'exponential': np.exp(x)
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

# Convert to tensors only when needed for the model
x_other_tensor = torch.FloatTensor(X_other_scaled.values)
x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled.values)
y_tensor = torch.FloatTensor(y_scaled.values)

# Convert data to numpy for splitting
X_other_np = x_other_tensor.numpy()
X_monotonic_np = x_monotonic_tensor.numpy()
y_np = y_tensor.numpy()

# Set training parameters
num_epochs = 1000

# Set up K-fold cross validation with a different seed
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)  # Changed to 42

# Store metrics for each fold
fold_metrics = {
    'train_losses': [],
    'val_losses': [],
    'rmse_scores': [],
    'mae_scores': [],
    'r2_scores': []
}

print(f"\nStarting {n_folds}-fold cross-validation")

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X_other_np)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    # Split data for this fold
    x_other_train = torch.FloatTensor(X_other_np[train_idx])
    x_other_val = torch.FloatTensor(X_other_np[val_idx])
    x_monotonic_train = torch.FloatTensor(X_monotonic_np[train_idx])
    x_monotonic_val = torch.FloatTensor(X_monotonic_np[val_idx])
    y_train = torch.FloatTensor(y_np[train_idx])
    y_val = torch.FloatTensor(y_np[val_idx])

    # Initialize model and optimizer for this fold
    model = SimpleMonotonicNN(
        other_input_dim=len(other_features)
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    # Training loop
    print("\nTraining model...")
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        outputs = model(x_other_train, x_monotonic_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping to prevent exploding gradients
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_other_val, x_monotonic_val)
            val_outputs_unscaled = target_scaler.inverse_transform(val_outputs.numpy())
            val_outputs_unscaled = np.maximum(0.0, val_outputs_unscaled)  # Ensure non-negative
            val_outputs = torch.FloatTensor(target_scaler.transform(val_outputs_unscaled))
            val_loss = criterion(val_outputs, y_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
        
    # Final evaluation for this fold
    model.eval()
    with torch.no_grad():
        final_val_outputs = model(x_other_val, x_monotonic_val)
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
        
        print(f'\nFold {fold + 1} Results:')
        print(f'RMSE: {rmse:.2f}')
        print(f'MAE: {mae:.2f}')
        print(f'R² Score: {r2:.4f}')

# Print average metrics across all folds
print('\nAverage Results Across All Folds:')
print(f'RMSE: {np.mean(fold_metrics["rmse_scores"]):.2f} ± {np.std(fold_metrics["rmse_scores"]):.2f}')
print(f'MAE: {np.mean(fold_metrics["mae_scores"]):.2f} ± {np.std(fold_metrics["mae_scores"]):.2f}')
print(f'R² Score: {np.mean(fold_metrics["r2_scores"]):.4f} ± {np.std(fold_metrics["r2_scores"]):.4f}')

