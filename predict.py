import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import joblib
import numpy as np

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

def predict_arcuate_sweep(age, steep_axis_term, type_str, residual_astig, ideal_tx_astig):
    """
    Make predictions with the trained model.
    
    Parameters:
    - age: patient age
    - steep_axis_term: steep axis term
    - type_str: type (string, either 'paired' [typemapped as 0] or 'single' [typemapped as 1])
    - residual_astig: residual astigmatism
    - ideal_tx_astig: ideal tx astigmatism
    
    Returns:
    - predicted arcuate sweep
    """
    # Check if ideal_tx_astig is below threshold
    if ideal_tx_astig < 0.25:
        return 0.00
        
    # Load encoders and scalers
    le = joblib.load('label_encoder.pkl')
    regular_scaler = joblib.load('regular_scaler.pkl')
    monotonic_scaler = joblib.load('monotonic_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    
    # Initialize and load model
    model = SimpleMonotonicNN(regular_input_dim=5)
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    model.eval()
    
    # Prepare input data
    regular_features = ['Age', 'Steep_axis_term', 'type', 'Residual_Astigmatism', 'ideal_tx_astig']
    type_encoded = le.transform([type_str])[0]
    regular_data = pd.DataFrame([[age, steep_axis_term, type_encoded, residual_astig, ideal_tx_astig]], 
                              columns=regular_features)
    monotonic_data = pd.DataFrame([[ideal_tx_astig]], 
                                columns=['ideal_tx_astig'])
    
    # Scale inputs
    regular_scaled = regular_scaler.transform(regular_data)
    monotonic_scaled = monotonic_scaler.transform(monotonic_data)
    
    # Convert to tensors
    x_regular = torch.FloatTensor(regular_scaled)
    x_monotonic = torch.FloatTensor(monotonic_scaled)
    
    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(x_regular, x_monotonic)
        prediction = target_scaler.inverse_transform(prediction_scaled.numpy())
    
    return float(prediction[0][0])

if __name__ == "__main__":
    # Example usage
    print("\nExample predictions:")
    examples = [
        {
            'age': 65,
            'steep_axis_term': 0,
            'type_str': 'paired',
            'residual_astig': 0,
            'ideal_tx_astig': 0.4
        },
        {
            'age': 70,
            'steep_axis_term': 0,
            'type_str': 'single',
            'residual_astig': 0,
            'ideal_tx_astig': 0.4
        }
    ]
    
    for i, example in enumerate(examples, 1):
        prediction = predict_arcuate_sweep(
            example['age'],
            example['steep_axis_term'],
            example['type_str'],
            example['residual_astig'],
            example['ideal_tx_astig']
        )
        print(f"\nExample {i}:")
        print(f"Inputs: {example}")
        print(f"Total Arcuate Sweep: {prediction:.2f}°") 