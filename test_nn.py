import torch
import torch.nn as nn
import argparse
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, obsv_dim, cost_dim):
        super(NeuralNetwork, self).__init__()
        # Network to predict cost-to-go between two states
        self.net = nn.Sequential(
            # Layer 1: Initial feature extraction
            nn.Linear(obsv_dim, 64),
            # nn.Tanh(),
            nn.ReLU(),
            # nn.Dropout(p=0.1), # Keep dropout light for regression
            
            # Layer 2: The "Funnel" (Compressing to 32)
            nn.Linear(64, 32), 
            # nn.Tanh(),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            
            # Layer 3: Output (Must match the 32 from above)
            nn.Linear(32, cost_dim),
            nn.Softplus() # Ensures smooth, positive cost-to-go
        )

    def forward(self, x):
        return self.net(x)
    
# def test(args):

#     data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)
#     indices = np.random.permutation(len(data))
#     data = data[indices]
#     totest = data[0:10] # Just test on the first 5 rows for now

#     # 2. Instantiate and Load
#     model = NeuralNetwork(obsv_dim=4, cost_dim=1)
#     model.load_state_dict(torch.load(args.model_path + 'cost2go_weights' + args.data_file.split('.')[0] + '.pth'))

#     # 3. SET TO EVALUATION MODE (Crucial!)
#     model.eval()

#     # Example raw input from your dataset: 2.4466, -2.5257, 2.4295, -2.3264
    
#     for raw_test in totest:
#         test_input = torch.tensor((raw_test[0:4])).float()

#         # 2. Run Inference
#         with torch.no_grad(): # Disables gradient calculation to save memory/speed
#             prediction = model(test_input)

#         print(f"Predicted Cost: {prediction.item():.4f} | Actual Cost : {raw_test[4]} | ")


import pickle # Need this to load your mean/std

def test(args):
    # 1. Load Data
    data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)
    np.random.shuffle(data)
    totest = data[0:500] 

    # --- NEW: LOAD NORMALIZATION PARAMS ---
    # Make sure this path matches where your training script saved the .pkl
    stats_path = args.model_path + 'norm_params.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    mean = torch.from_numpy(stats['mean']).float()
    std = torch.from_numpy(stats['std']).float()
    # --------------------------------------

    # 2. Instantiate and Load
    model = NeuralNetwork(obsv_dim=4, cost_dim=1)
    # Check naming: matches training save?
    weight_path = args.model_path + 'cost2go_weights.pth'
    model.load_state_dict(torch.load(weight_path))

    # 3. SET TO EVALUATION MODE (Crucial!)
    model.eval()

    # print(f"{'PREDICTED':<15} | {'ACTUAL':<15} | {'ERROR':<15}")
    print("-" * 50)
    avg_error = 0.0

    for raw_test in totest:
        raw_input = torch.tensor((raw_test[0:4])).float()

        # # --- NEW: APPLY NORMALIZATION BEFORE INFERENCE ---
        # normalized_input = (raw_input - mean) / (std + 1e-8)
        # # -------------------------------------------------

        with torch.no_grad(): 
            # Use the NORMALIZED input for the model
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