import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from nn import NeuralNetwork
import yaml

def run_visualizations(actual_c2g, prediction_c2g):
    actual_c2g = np.array(actual_c2g)
    prediction_c2g = np.array(prediction_c2g)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(actual_c2g, prediction_c2g, alpha=0.5, color='royalblue', edgecolors='k')
    
    line_range = [min(actual_c2g.min(), prediction_c2g.min()), max(actual_c2g.max(), prediction_c2g.max())]
    plt.plot(line_range, line_range, 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Cost-to-Go')
    plt.ylabel('Predicted Cost-to-Go')
    plt.title('Predicted vs. Actual')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    errors = prediction_c2g - actual_c2g
    plt.hist(errors, bins=30, color='seagreen', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel('Prediction Error (Pred - Actual)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.show()

def test(config):

    num_samples = config['num_samples']
    data_version = config['version']
    world = config['world']
    data_path = f"{config['data_folder']}/{world}_cost2go_{num_samples}_{data_version}.csv"
    epochs = config['epochs']
    optimizer_choice = config['optimizer']
    dr = config['dropout_rate']
    
    print(f"Loading data from: {data_path}")
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    np.random.shuffle(data)
    
    totest = data[0:1000] if len(data) > 1000 else data

    model = NeuralNetwork(obsv_dim=4, cost_dim=1, dr=dr)
    if optimizer_choice == "AdamW":
        if config['nn_version'] != 3:
            weight_path = f"{config['nn_folder']}/{world}/nn{config['nn_version']}_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
        else:
            weight_path = f"{config['nn_folder']}/{world}/nn_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
    else:
        weight_path = f"{config['nn_folder']}/{world}/nn_dr{round(dr*10)}_{optimizer_choice}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"

    print("Loading model weights from:", weight_path)
    model.load_state_dict(torch.load(weight_path))
    
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
        raw_input = torch.tensor((raw_test[0:4])).float()

        with torch.no_grad(): 
            prediction = model(raw_input)

        all_preds.append(prediction.item())
        all_actuals.append(raw_test[4])

    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    r2 = r2_score(all_actuals, all_preds)

    print("\n" + "="*30)
    print("      PERFORMANCE METRICS")
    print("="*30)
    print(f"Mean Absolute Error (MAE):  {mae:.4f}")
    print(f"Root Mean Sq. Error (RMSE): {rmse:.4f}")
    print(f"R² Score:                   {r2:.4f}")
    print("="*30)
    
    run_visualizations(all_actuals, all_preds)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    test(config)