# Monotonic

A Python project for implementing monotonic constraints in a neural network (multi layer perceptron) that is trained to predict laser arcuate incision length used in cataract surgery.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Linux/Mac
   # OR
   .venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

This project relies on the following main libraries:
- PyTorch (>=2.0.0)
- pandas (>=2.0.0)
- scikit-learn (>=1.0.2)
- NumPy (>=1.24.0)
- joblib (>=1.3.0)
- matplotlib (>=3.7.0)
- seaborn (>=0.12.0) 