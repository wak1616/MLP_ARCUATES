import torch
import pandas as pd
import numpy as np
import joblib

def predict_arcuate_sweep(age, steep_axis_term, meank_iolmaster, 
                         treatment_astigmatism_half, wtw_iolmaster, treated_astig_half, 
                         weights_path='model_weights.pth',
                         components_path='model_components.joblib'):
    
    # Load model weights safely
    model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    
    # Load other components
    components = joblib.load(components_path)
    print(components)
    other_scaler = components['other_scaler']
    monotonic_scaler = components['monotonic_scaler']
    target_scaler = components['target_scaler']
    label_encoder = components['label_encoder']
    
    # Verify all components are loaded
    if not all([model_state_dict, other_scaler, monotonic_scaler, target_scaler, label_encoder]):
        raise ValueError("Missing components in the model checkpoint")
    
    # Create DataFrames for features with column names
    other_data = pd.DataFrame({
        'Age': [age],
        'Steep_axis_term': [steep_axis_term],
        'MeanK_IOLMaster': [meank_iolmaster],
        'Treatment_astigmatism_half': [treatment_astigmatism_half],
        'WTW_IOLMaster': [wtw_iolmaster]
    })
    
    # Create monotonic features
    x_monotonic = np.array([[
            1.0,  # constant
            treated_astig_half,  # lin  ear
        1 / (1 + np.exp(-(treated_astig_half+1))),  # logistic1
        1 / (1 + np.exp(-(treated_astig_half+0.5))),  # logistic2
        1 / (1 + np.exp(-treated_astig_half)),  # logistic3
        np.log(treated_astig_half - min(treated_astig_half, 0) + 1),  # logarithmic
        1 / (1 + np.exp(-(treated_astig_half-0.5))),  # logistic4
        1 / (1 + np.exp(-(treated_astig_half-1)))  # logistic5
    ]])
    
    # Transform type using label encoder
    # other_data['type'] = label_encoder.transform([other_data['type'].iloc[0]])
    
    # Scale the features while maintaining DataFrame structure
    other_scaled = pd.DataFrame(
        other_scaler.transform(other_data),
        columns=other_data.columns,
        index=other_data.index
    )
    
    monotonic_scaled = pd.DataFrame(
        monotonic_scaler.transform(x_monotonic),
        columns=['constant', 'linear', 'logistic1', 'logistic2', 'logistic3', 
                'logarithmic', 'logistic4', 'logistic5'],
        index=[0]
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
                torch.nn.Linear(24, 8),
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
    
    # Example prediction
    try:
        # Get treated_astig value first
        treated_astig_val = 0.5  # default value
        prediction = predict_arcuate_sweep(
            age=65,
            steep_axis_term=0.5,
            meank_iolmaster=44.0,
            treatment_astigmatism_half=treated_astig_val,  # automatically matches treated_astig
            wtw_iolmaster=12.0,
            treated_astig_half=treated_astig_val
        )
        print(f"Predicted Arcuate Sweep: {prediction:.2f}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}") 