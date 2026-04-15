import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle  # ADD THIS IMPORT AT THE TOP
import argparse

class NeuralNetwork(nn.Module):
    def __init__(self, obsv_dim, cost_dim):
        super(NeuralNetwork, self).__init__()
        # Network to predict cost-to-go between two states
        self.net = nn.Sequential(
            nn.Linear(obsv_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, cost_dim),
        )

    def forward(self, x):
        return self.net(x)
    
def train(args):
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    WEIGHT_DECAY = args.weight_decay

    # 1. Load data
    data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)

    # --- NEW: CALCULATE AND SAVE NORMALIZATION STATS ---
    inputs = data[:, 0:4]  # start_x, start_y, goal_x, goal_y
    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0)

    # Save the stats immediately so you don't lose them
    stats = {'mean': mean, 'std': std}
    with open(args.model_path + 'normalization_params.pkl', 'wb') as f:
        pickle.dump(stats, f)
    print(f"Stats saved. Mean: {mean}, Std: {std}")
    # ----------------------------------------------------

    obsv_dim = 4 
    cost_dim = 1 

    model = NeuralNetwork(obsv_dim, cost_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Move tensor conversion OUTSIDE the loop for efficiency
    # and APPLY the normalization here
    states_raw = torch.from_numpy(data[:, 0:4]).float()
    costs = torch.from_numpy(data[:, 4:5]).float()

    # --- NEW: APPLY NORMALIZATION ---
    mean_tensor = torch.from_numpy(mean).float()
    std_tensor = torch.from_numpy(std).float()
    states = (states_raw - mean_tensor) / (std_tensor + 1e-8) # 1e-8 prevents div by zero
    # --------------------------------

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Forward pass (now using normalized states)
        predicted_costs = model(states)

        # Compute loss
        loss = criterion(predicted_costs, costs)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), args.model_path + 'cost2go_weights.pth')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='./data/', help='path to the data folder')
    parser.add_argument('--model-path', type=str, default='./models/', help='path to the model file')
    parser.add_argument('--data-file', type=str, default='cost2go_10000_2026-03-23_16-33-55.csv', help='path to the data file')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')

    args = parser.parse_args()
    print(args)
    train(args)