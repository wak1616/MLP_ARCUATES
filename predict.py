import torch
import pandas as pd
import numpy as np
import joblib

def predict_arcuate_sweep(age, steep_axis_term, type_val, meank_iolmaster, 
                         treatment_astigmatism, wtw_iolmaster, treated_astig, 
                         weights_path='model_weights.pth',
                         components_path='model_components.joblib'):
    
    # Load model weights safely
    model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    
    # Load other components
    components = joblib.load(components_path)
    
    other_scaler = components['other_scaler']
    monotonic_scaler = components['monotonic_scaler']
    target_scaler = components['target_scaler']
    label_encoder = components['label_encoder']
    
    # Verify all components are loaded
    if not all([model_state_dict, other_scaler, monotonic_scaler, target_scaler, label_encoder]):
        raise ValueError("Missing components in the model checkpoint")
    
    # Create DataFrames for features
    other_data = pd.DataFrame({
        'Age': [age],
        'Steep_axis_term': [steep_axis_term],
        'type': [type_val],
        'MeanK_IOLMaster': [meank_iolmaster],
        'Treatment_astigmatism': [treatment_astigmatism],
        'WTW_IOLMaster': [wtw_iolmaster]
    })
    
    # Create monotonic features
    monotonic_features = {
        'constant': np.ones(1),
        'linear': np.array([treated_astig]),
        'quadratic': np.array([treated_astig**2]),
        'cubic': np.array([treated_astig**3]),
        'quartic': np.array([treated_astig**4]),
        'logarithmic': np.log(np.array([treated_astig - min(treated_astig, 0) + 1])),
        'exponential': np.exp(np.array([treated_astig]))
    }
    monotonic_data = pd.DataFrame(monotonic_features)
    
    # Transform type using label encoder
    other_data['type'] = label_encoder.transform([other_data['type'].iloc[0]])
    
    # Scale the features while maintaining DataFrame structure
    other_scaled = pd.DataFrame(
        other_scaler.transform(other_data),
        columns=other_data.columns,
        index=other_data.index
    )
    
    monotonic_scaled = pd.DataFrame(
        monotonic_scaler.transform(monotonic_data),
        columns=monotonic_data.columns,
        index=monotonic_data.index
    )
    
    # Convert to tensors
    x_other = torch.FloatTensor(other_scaled.values)
    x_monotonic = torch.FloatTensor(monotonic_scaled.values)
    
    # Load model architecture and weights
    model = SimpleMonotonicNN(len(other_data.columns))
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(x_other, x_monotonic)
        prediction = target_scaler.inverse_transform(prediction_scaled.numpy())
        prediction = max(0.0, float(prediction.item()))  # Ensure non-negative
        return prediction

# Example usage
if __name__ == "__main__":
    class SimpleMonotonicNN(torch.nn.Module):
        def __init__(self, other_input_dim):
            super().__init__()
            self.unconstrained_path = torch.nn.Sequential(
                torch.nn.Linear(other_input_dim, 24),
                torch.nn.ReLU(),
                torch.nn.Linear(24, 7),
                torch.nn.ReLU()
            )
            
            # Initialize weights with smaller values
            for m in self.unconstrained_path.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
            
        def forward(self, x_other, x_monotonic):
            weights = self.unconstrained_path(x_other)
            weighted_features = weights * x_monotonic
            return weighted_features.sum(dim=1, keepdim=True)
    
    # Example prediction
    try:
        prediction = predict_arcuate_sweep(
            age=65,
            steep_axis_term=0.5,
            type_val='paired',
            meank_iolmaster=44.0,
            treatment_astigmatism=1.0,
            wtw_iolmaster=12.0,
            treated_astig=1.0
        )
        print(f"Predicted Arcuate Sweep: {prediction:.2f}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}") 