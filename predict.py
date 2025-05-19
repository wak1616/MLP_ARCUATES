import torch
import pandas as pd
import numpy as np
import joblib
import torch.nn as nn

def predict_arcuate_sweep(age, steep_axis_term, meank_iolmaster, wtw_iolmaster, treated_astig, weights_path='model_weights.pth', components_path='model_components.joblib'):
    
    # Load model weights safely
    model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    
    # Load other components
    components = joblib.load(components_path)
    other_scaler = components['other_scaler']
    target_scaler = components['target_scaler']
    label_encoder = components['label_encoder']
    
    # Verify all components are loaded
    if not all([model_state_dict, other_scaler, target_scaler, label_encoder]):
        raise ValueError("Missing components in the model checkpoint")
    
    # Create DataFrames for features with column names
    other_data = pd.DataFrame({
        'Age': [age],
        'Steep_axis_term': [steep_axis_term],
        'MeanK_IOLMaster': [meank_iolmaster],
        'WTW_IOLMaster': [wtw_iolmaster],
        'Treated_astig': [treated_astig]
    })
    
    # Scale the features while maintaining DataFrame structure
    other_scaled = pd.DataFrame(
        other_scaler.transform(other_data),
        columns=other_data.columns,
        index=other_data.index
    )
    
    # Convert to tensors
    x_other = torch.FloatTensor(other_scaled.values)
    
    # Define the model architecture locally
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

    # Load model architecture and weights
    model = ArcuateSweepPredictor(len(other_data.columns))
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(x_other)
        prediction = target_scaler.inverse_transform(prediction_scaled.numpy())
        prediction = max(0.0, float(prediction.item()))  # Ensure non-negative
        return prediction

# Example usage
if __name__ == "__main__":
    # Define the model architecture locally (MUST match the one used in prediction function)
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
    
    # Example prediction
    try:
        # Get treated_astig value first
        treated_astig_total = .5  # default value
        prediction = predict_arcuate_sweep(
            age=65,
            steep_axis_term=1,
            meank_iolmaster=45,
            wtw_iolmaster=12.0,
            treated_astig=treated_astig_total/2
        )
        print(f"Predicted Arcuate Sweep: {prediction:.2f}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}") 