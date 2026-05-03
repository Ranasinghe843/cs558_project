import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse

class NeuralNetwork(nn.Module):
    # NN : 1
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 64),
    #         nn.ReLU(),        
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # NN : 2
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 128),
    #         nn.ReLU(),        
    #         nn.Dropout(p=dr),
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),
    #         nn.Linear(32, 16),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),
    #         nn.Linear(16, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )
    
    # NN : 3 (with Dropout) 
    # BEST: nn_dr0_AdamW100_epochs100_lr1000_dataset100000_1.pth
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 256),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(64, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # NN : 4 
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 128),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(32, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # NN : 5
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 256),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(32, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # # NN : 6
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 512),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(32, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # # NN : 7
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 512),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(64, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # NN : 8
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 512),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(512, 128),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(128, 32),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(32, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # # NN : 9 (same # of layers as NN 7 but wider layers)
    def __init__(self, obsv_dim, cost_dim, dr):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obsv_dim, 1024),
            nn.ReLU(),   
            nn.Dropout(p=dr),     
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dr),   
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dr),   
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dr),   
            nn.Linear(128, cost_dim),
            nn.Softplus(beta=5.0)
        )

    # NN : 10 (one more layer than NN 9)
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 1024),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(1024, 512),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),
    #         nn.Linear(64, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # # NN : 11 (one less layer than NN 9)
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 1024),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(1024, 512),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(256, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    # # # NN : 11 (same # of layers as NN 9 but wider layers)
    # def __init__(self, obsv_dim, cost_dim, dr):
    #     super(NeuralNetwork, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obsv_dim, 2048),
    #         nn.ReLU(),   
    #         nn.Dropout(p=dr),     
    #         nn.Linear(2048, 1024),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(1024, 512),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(p=dr),   
    #         nn.Linear(256, cost_dim),
    #         nn.Softplus(beta=5.0)
    #     )

    def forward(self, x):
        return self.net(x)