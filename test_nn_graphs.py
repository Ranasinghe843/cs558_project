import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class NeuralNetwork(nn.Module):
    def __init__(self, obsv_dim, cost_dim):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obsv_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, cost_dim),
            nn.Softplus() 
        )

    def forward(self, x):
        return self.net(x)

def run_visualizations(actuals, predictions):
    """Generates plots to visually assess model performance."""
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    plt.figure(figsize=(12, 5))

    # Plot 1: Predicted vs Actual Scatter
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.5, color='royalblue', edgecolors='k')
    
    # Diagonal line representing perfect prediction
    line_range = [min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())]
    plt.plot(line_range, line_range, 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Cost-to-Go')
    plt.ylabel('Predicted Cost-to-Go')
    plt.title('Predicted vs. Actual')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Error Distribution (Residuals)
    plt.subplot(1, 2, 2)
    errors = predictions - actuals
    plt.hist(errors, bins=30, color='seagreen', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel('Prediction Error (Pred - Actual)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.show()

def test(args):
    # 1. Load Data
    print(f"Loading data from: {args.data_file}...")
    data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)
    np.random.shuffle(data)
    
    # Use a subset for testing performance
    totest = data[0:1000] if len(data) > 1000 else data

    # 2. Instantiate and Load Model
    model = NeuralNetwork(obsv_dim=4, cost_dim=1)
    weight_path = args.model_path + 'cost2go_weights.pth'
    
    try:
        model.load_state_dict(torch.load(weight_path))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find weights at {weight_path}")
        return

    model.eval()

    all_preds = []
    all_actuals = []

    print(f"Running inference on {len(totest)} samples...")
    
    for raw_test in totest:
        # Since no normalization was used in training, we pass raw data
        raw_input = torch.tensor((raw_test[0:4])).float()

        with torch.no_grad(): 
            prediction = model(raw_input)

        all_preds.append(prediction.item())
        all_actuals.append(raw_test[4])

    # 3. Calculate Metrics
    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    r2 = r2_score(all_actuals, all_preds)

    # 4. Report Results
    print("\n" + "="*30)
    print("      PERFORMANCE METRICS")
    print("="*30)
    print(f"Mean Absolute Error (MAE):  {mae:.4f}")
    print(f"Root Mean Sq. Error (RMSE): {rmse:.4f}")
    print(f"R² Score:                   {r2:.4f}")
    print("="*30)
    
    if r2 > 0.9:
        print("Excellent fit! The model explains most of the variance.")
    elif r2 > 0.7:
        print("Good fit, but there might be some specific cases it struggles with.")
    else:
        print("Model performance is low. Consider normalization or more training data.")

    # 5. Visualize
    run_visualizations(all_actuals, all_preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/', help='path to the data folder')
    parser.add_argument('--model-path', type=str, default='./models/', help='path to the model file')
    parser.add_argument('--data-file', type=str, default='cost2go_6000_1.csv', help='path to the data file')

    args = parser.parse_args()
    test(args)