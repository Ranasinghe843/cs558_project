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
    data_path = f"{config['data_folder']}/'cost2go_{num_samples}_{data_version}.csv"
    epochs = config['epochs']

    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    np.random.shuffle(data)
    totest = data[0:500] 

    model = NeuralNetwork(obsv_dim=4, cost_dim=1)
    weight_path = f"{config['nn_folder']}/nn_{epochs}_{config['learning_rate']}_{num_samples}_{data_version}.pth"
    model.load_state_dict(torch.load(weight_path))

    model.eval()

    print("-" * 50)
    avg_error = 0.0

    for raw_test in totest:
        raw_input = torch.tensor((raw_test[0:4])).float()

        with torch.no_grad(): 
            prediction = model(raw_input)

        actual = raw_test[4]
        pred_val = prediction.item()
        error = abs(pred_val - actual)

        avg_error += error

    avg_error /= len(totest)
    print(f"Average Error: {avg_error:.4f}")

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    test(config)