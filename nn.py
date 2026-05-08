##########################################################
# Neural Network Class
# inputs: start_x, start_y, goal_x, goal_y
# outputs: cost-to-go
##########################################################

import torch.nn as nn

class NeuralNetwork(nn.Module):
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

    def forward(self, x):
        return self.net(x)