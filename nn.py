import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse

class NeuralNetwork(nn.Module):
    # NN : 1
    def __init__(self, obsv_dim, cost_dim):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obsv_dim, 64),
            nn.ReLU(),        
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, cost_dim),
            nn.Softplus(beta=5.0)
        )

    # NN : 2
    # def __init__(self, obsv_dim, cost_dim):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 512),
    #         nn.ReLU(),        
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )
    
    # def __init__(self, obsv_dim, cost_dim):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 512),
    #         nn.ReLU(),   
    #         nn.Dropout(p=0.2),     
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.2),   
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.2),   
    #         nn.Linear(128, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    def forward(self, x):
        return self.net(x)