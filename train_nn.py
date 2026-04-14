import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
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

    data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)  # shape (N, 5) where columns are [s1, s2, s3, s4, cost-to-go]

    obsv_dim = 4 # 8 for Reacher
    cost_dim = 1 # Predicting scalar cost-to-go

    model = NeuralNetwork(obsv_dim, cost_dim)  # initialize neural network
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss() # Mean Squared Error loss for regression

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Convert data to tensors
        states = torch.from_numpy(data[:,0:4]).float()  # shape (N, obsv_dim)
        costs = torch.from_numpy(data[:,4:5]).float()    # shape (N, cost_dim)

        # Forward pass
        predicted_costs = model(states)  # shape (N, cost_dim)

        # Compute loss
        loss = criterion(predicted_costs, costs)

        # Backward pass and optimization step
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