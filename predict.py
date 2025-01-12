import torch
import pandas as pd
import numpy as np
import joblib

def predict_arcuate_sweep(age, steep_axis_term, meank_iolmaster, 
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
    
    # Create DataFrames for features with column names
    other_data = pd.DataFrame({
        'Age': [age],
        'Steep_axis_term': [steep_axis_term],
        'MeanK_IOLMaster': [meank_iolmaster],
        'Treatment_astigmatism': [treatment_astigmatism],
        'WTW_IOLMaster': [wtw_iolmaster]
    })
    
    # Create monotonic features using DataFrame
    monotonic_features_dict = {
        'constant': [1.0],
        'linear': [treated_astig],
        'logistic_shift_left_1': [1 / (1 + np.exp(-(treated_astig+1)))],
        'logistic_shift_left_0.5': [1 / (1 + np.exp(-(treated_astig+0.5)))],
        'logistic_center': [1 / (1 + np.exp(-treated_astig))],
        'logarithmic': [np.log(treated_astig - min(treated_astig, 0) + 1)],
        'logistic_shift_right_0.5': [1 / (1 + np.exp(-(treated_astig-0.5)))],
        'logistic_shift_right_1': [1 / (1 + np.exp(-(treated_astig-1)))],
        'logistic_shift_right_1.5': [1 / (1 + np.exp(-(treated_astig-1.5)))],
        'logistic_shift_left_1.5': [1 / (1 + np.exp(-(treated_astig+1.5)))]
    }
    x_monotonic = pd.DataFrame(monotonic_features_dict)
    
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
        columns=x_monotonic.columns,
        index=x_monotonic.index
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
                torch.nn.Linear(other_input_dim, 48),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Linear(48, 10),
                torch.nn.ReLU()
            )
            
        def forward(self, x_other, x_monotonic):
            coefficients = self.unconstrained_path(x_other)
            monotonic_feature_contributions = coefficients * x_monotonic
            return monotonic_feature_contributions.sum(dim=1, keepdim=True)
    
    # Example prediction
    try:
        # Get treated_astig value first
        treated_astig_total = 0.4  # default value
        prediction = predict_arcuate_sweep(
            age=65,
            steep_axis_term=1,
            meank_iolmaster= 45,
            treatment_astigmatism=treated_astig_total/2,  # automatically matches treated_astig
            wtw_iolmaster=12.0,
            treated_astig=treated_astig_total/2
        )
        print(f"Predicted Arcuate Sweep: {prediction:.2f}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}") 