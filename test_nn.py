import torch
import torch.nn as nn
import argparse
import numpy as np
import pickle

class NeuralNetwork(nn.Module):
    def __init__(self, obsv_dim, cost_dim):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obsv_dim, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, cost_dim),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)

def test(args):
    # 1. Load Data
    data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)
    np.random.shuffle(data)
    totest = data[0:500] 

    stats_path = args.model_path + 'norm_params.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    mean = torch.from_numpy(stats['mean']).float()
    std = torch.from_numpy(stats['std']).float()

    model = NeuralNetwork(obsv_dim=4, cost_dim=1)
    weight_path = args.model_path + 'cost2go_weights.pth'
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='./data/', help='path to the data folder')
    parser.add_argument('--model-path', type=str, default='./models/', help='path to the model file')
    parser.add_argument('--data-file', type=str, default='cost2go_6000_1.csv', help='path to the data file')

    args = parser.parse_args()
    print(args)
    test(args)