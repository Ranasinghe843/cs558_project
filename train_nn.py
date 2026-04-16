from shlex import split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
from nn import NeuralNetwork
import yaml
    
def train(config):

    num_samples = config['num_samples']
    data_version = config['version']
    data_path = f"{config['data_folder']}/'cost2go_{num_samples}_{data_version}.csv"
    epochs = config['epochs']
    lr = 1/config['learning_rate']
    
    raw_data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    np.random.shuffle(raw_data)

    inputs_raw = raw_data[:, 0:4]

    states = torch.from_numpy(inputs_raw).float()
    costs = torch.from_numpy(raw_data[:, 4:5]).float()

    dataset = torch.utils.data.TensorDataset(states, costs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    model = NeuralNetwork(4, 1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_states, batch_costs in loader:
            optimizer.zero_grad()
            preds = model(batch_states)
            loss = criterion(preds, batch_costs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f'Epoch {epoch}, Avg Loss: {avg_loss:.6f}')
    
    torch.save(model.state_dict(), f"{config['nn_folder']}/nn_{epochs}_{config['learning_rate']}_{num_samples}_{data_version}.pth")

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    train(config)