# Arcuate Sweep Predictor MLP

A Python project implementing a Multi-Layer Perceptron (MLP) neural network trained to predict laser arcuate incision sweep (degrees) used in cataract surgery based on patient and treatment data.

## Project Structure

*   `datafinal.csv`: The dataset containing patient and treatment information.
*   `draft.py`: Script for experimenting with the model architecture and training using K-Fold cross-validation. Includes detailed printouts and metrics for each fold.
*   `final_model.py`: Script for training the final model on the entire dataset and saving the trained model weights (`model_weights.pth`) and necessary preprocessing components (`model_components.joblib`).
*   `predict.py`: Script to load the saved model and components to make predictions on new, single-instance data.
*   `analyze_model.py`: Script to load the saved model and components, evaluate performance on the dataset, and generate analysis plots (e.g., Actual vs. Predicted, Residuals).
*   `visualize_architecture.py`: Generates a diagram (`network_architecture.png`) of the MLP architecture using Graphviz.
*   `requirements.txt`: Lists the required Python packages.
*   `README.md`: This file.

## Setup

1.  **Clone the repository (if applicable).**
2.  **Create and activate a Python virtual environment:**
    ```bash
    # Use python3 or python depending on your system alias
    python3 -m venv venv 
    source venv/bin/activate  # On Linux/Mac
    # venv\Scripts\activate    # On Windows
    ```
3.  **Install Graphviz (System Dependency):**
    *   **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y graphviz`
    *   **MacOS (using Homebrew):** `brew install graphviz`
    *   **Windows:** Download from the official [Graphviz website](https://graphviz.org/download/) and ensure the `bin` directory is added to your system's PATH.
4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Model Architecture

The model is a feedforward neural network (MLP) with the following structure:

*   **Input Layer:** Takes 5 features:
    *   `Age`
    *   `Steep_axis_term`
    *   `MeanK_IOLMaster`
    *   `WTW_IOLMaster`
    *   `Treated_astig`
*   **Hidden Layer 1:** Linear(5 -> 48) + LeakyReLU(0.1)
*   **Hidden Layer 2:** Linear(48 -> 10) + ReLU()
*   **Output Layer:** Linear(10 -> 1) (Predicts `Arcuate_sweep_total`)

## Usage

1.  **Experimentation (Optional):**
    ```bash
    python draft.py
    ```
    This script trains the model using 5-fold cross-validation and prints detailed performance metrics for each fold and the average across folds.

2.  **Train Final Model:**
    ```bash
    python final_model.py
    ```
    This script trains the model on the full dataset and saves `model_weights.pth` and `model_components.joblib`.

3.  **Make Predictions:**
    Modify the example values in the `if __name__ == "__main__":` block of `predict.py` and run:
    ```bash
    python predict.py
    ```

4.  **Analyze Model Performance:**
    ```bash
    python analyze_model.py
    ```
    This loads the saved model, evaluates it on the dataset, prints metrics (RMSE, MAE, R²), and saves analysis plots (`model_analysis.png`, `treated_astig_vs_sweep.png`).

5.  **Visualize Architecture:**
    ```bash
    python visualize_architecture.py
    ```
    This generates `network_architecture.png` showing the model layers.

## Dependencies

Key Python libraries used:

*   PyTorch
*   pandas
*   scikit-learn
*   NumPy
*   joblib
*   matplotlib
*   seaborn
*   graphviz (Python library)

*(Note: The Graphviz system software is also required for `visualize_architecture.py`)*