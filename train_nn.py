#####################################################################
# Script to train a feedforward neural network to predict cost-to-go values for a given world and dataset.
#####################################################################

from shlex import split
import torch
import torch.nn as nn
import numpy as np
from nn import NeuralNetwork
import yaml
import matplotlib.pyplot as plt

# Training function
def train(config):

    # load config parameters
    world = config['world']
    num_samples = config['num_samples']
    data_version = config['version']
    data_path = f"{config['data_folder']}/{world}_cost2go_{num_samples}_{data_version}.csv"
    epochs = config['epochs']
    lr = 1/config['learning_rate']
    dr = config['dropout_rate']
    optimizer_choice = config['optimizer']  
    weight_decay = 1/config['weight_decay']
    raw_data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    print(f"Loaded data from {data_path}, shape: {raw_data.shape}")
    np.random.shuffle(raw_data)     # randomize data

    # reserve last 10k samples for testing, use the rest for training
    inputs_raw = raw_data[:-10000, 0:4]
    outputs_raw = raw_data[:-10000, 4:5]
    print(f"Training data: inputs shape {inputs_raw.shape}, outputs shape {outputs_raw.shape}")

    states = torch.from_numpy(inputs_raw).float()
    costs = torch.from_numpy(outputs_raw).float()

    dataset = torch.utils.data.TensorDataset(states, costs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    # initialize model
    model = NeuralNetwork(4, 1, dr=dr)
    
    # initialize optimizer and loss function
    if optimizer_choice == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    elif optimizer_choice == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    elif optimizer_choice == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    criterion = nn.MSELoss()

    # store average loss for plotting
    avg_loss_list = []

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

        if epoch % 50 == 0:
            avg_loss = epoch_loss / len(loader)
            avg_loss_list.append(avg_loss)
            print(f'Epoch {epoch}, Avg Loss: {avg_loss:.6f}')
    
    # torch.save(model.state_dict(), f"{config['nn_folder']}/nn_{round(dr*10)}_{optimizer_choice}_{epochs}_{config['learning_rate']}_{num_samples}_{data_version}.pth")
    # torch.save(model.state_dict(), f"{config['nn_folder']}/data_PRMstar/nn_{epochs}_{config['learning_rate']}.pth")
        
            # Save the model every 50 epochs
            # if optimizer_choice == "AdamW":
            #     if config['nn_version'] != 3:
            #         torch.save(model.state_dict(), f"{config['nn_folder']}/{world}/nn{config['nn_version']}_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epoch}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth")
            #     else:
            #         torch.save(model.state_dict(), f"{config['nn_folder']}/{world}/nn_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epoch}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth")
            # elif optimizer_choice == "Adam":
            #     torch.save(model.state_dict(), f"{config['nn_folder']}/{world}/nn_dr{round(dr*10)}_{optimizer_choice}_epochs{epoch}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth")
            # elif optimizer_choice == "SGD":
            #     torch.save(model.state_dict(), f"{config['nn_folder']}/{world}/nn_dr{round(dr*10)}_{optimizer_choice}_epochs{epoch}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth")

    # plot average loss over epochs
    plt.figure()
    plt.plot(np.arange(0, epochs, 50), avg_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.show()

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    train(config)