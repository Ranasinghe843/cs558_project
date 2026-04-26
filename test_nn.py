import torch
import torch.nn as nn
import argparse
import numpy as np
import pickle
from nn import NeuralNetwork
import yaml
from shlex import split

def test(config):

    num_samples = config['num_samples']
    data_version = config['version']
    data_path = f"{config['data_folder']}/cost2go_{num_samples}_{data_version}.csv"
    epochs = config['epochs']
    optimizer_choice = config['optimizer']
    dr = config['dropout_rate']

    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    training_data = data[:-10000, :]
    testing_data = data[-10000:, :]
    np.random.shuffle(testing_data)
    totest = testing_data

    model = NeuralNetwork(obsv_dim=4, cost_dim=1, dr=dr)
    if optimizer_choice == "AdamW":
        if config['nn_version'] != 3:
            weight_path = f"{config['nn_folder']}/nn{config['nn_version']}_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
        else:
            weight_path = f"{config['nn_folder']}/nn_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
    else:
        weight_path = f"{config['nn_folder']}/nn_dr{round(dr*10)}_{optimizer_choice}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"

    print("Loading model weights from:", weight_path)
    model.load_state_dict(torch.load(weight_path))

    model.eval()

    print("-" * 50)
    avg_error = 0.0
    error_store = []

    for raw_test in totest:
        raw_input = torch.tensor((raw_test[0:4])).float()

        with torch.no_grad(): 
            prediction = model(raw_input)

        actual = raw_test[4]
        pred_val = prediction.item()
        error = 100.0 * np.abs(pred_val - actual)/(1e-3 + np.abs(actual))
        error_store.append(error)
        # print(f" Error: {error:.4f}")
        avg_error += error

    avg_error /= len(totest)
    print(f"Average Error: {avg_error:.4f} % over {len(totest)} samples")
    print(f"Max Error : {np.max(error_store):.4f} % over {len(totest)} samples")
    print(f"Standard Deviation of Error: {np.std(error_store):.4f} % over {len(totest)} samples")

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    test(config)