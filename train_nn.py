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
    
# def train(args):
#     LEARNING_RATE = args.learning_rate
#     EPOCHS = args.epochs
#     WEIGHT_DECAY = args.weight_decay

#     # 1. Load data
#     data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)
#     indices = np.random.permutation(len(data))
#     data = data[indices]

#     # --- NEW: CALCULATE AND SAVE NORMALIZATION STATS ---
#     inputs = data[:, 0:4]  # start_x, start_y, goal_x, goal_y
#     mean = np.mean(inputs, axis=0)
#     std = np.std(inputs, axis=0)

#     # Save the stats immediately so you don't lose them
#     stats = {'mean': mean, 'std': std}
#     with open(args.model_path + 'normalization_params' + args.data_file.split('.')[0] + '.pkl', 'wb') as f:
#         pickle.dump(stats, f)
#     print(f"Stats saved. Mean: {mean}, Std: {std}")
#     # ----------------------------------------------------

#     obsv_dim = 4 
#     cost_dim = 1 

#     model = NeuralNetwork(obsv_dim, cost_dim)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6, nesterov=True)
#     criterion = nn.MSELoss()

#     # Move tensor conversion OUTSIDE the loop for efficiency
#     # and APPLY the normalization here
#     states = torch.from_numpy(data[:, 0:4]).float()
#     costs = torch.from_numpy(data[:, 4:5]).float()

#     # # --- NEW: APPLY NORMALIZATION ---
#     # mean_tensor = torch.from_numpy(mean).float()
#     # std_tensor = torch.from_numpy(std).float()
#     # states = (states - mean_tensor) / (std_tensor + 1e-8) # 1e-8 prevents div by zero
#     # # --------------------------------

#     for epoch in range(EPOCHS):
#         model.train()
#         optimizer.zero_grad()

#         # Forward pass (now using normalized states)
#         predicted_costs = model(states)

#         # Compute loss
#         loss = criterion(predicted_costs, costs)

#         # Backward pass
#         loss.backward()
#         optimizer.step()

#         if epoch % 100 == 0:
#             # print(f"shape of states: {states.shape}, shape of costs: {costs.shape}")
#             print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
#     torch.save(model.state_dict(), args.model_path + 'cost2go_weights' + args.data_file.split('.')[0] + '.pth')
    
def train(args):
    # ... (Keep imports and Network class as you have them) ...

    # 1. Load and Shuffle Data
    raw_data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)
    np.random.shuffle(raw_data) # Simpler way to shuffle in-place

    # 2. Normalization Stats
    inputs_raw = raw_data[:, 0:4]
    mean = np.mean(inputs_raw, axis=0)
    std = np.std(inputs_raw, axis=0) + 1e-8 # Prevent div by zero
    
    # Save stats for MPC script
    stats = {'mean': mean, 'std': std}
    with open(args.model_path + 'norm_params.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # 3. Apply Normalization to Tensors
    # states = (torch.from_numpy(inputs_raw).float() - torch.from_numpy(mean).float()) / torch.from_numpy(std).float()
    states = torch.from_numpy(inputs_raw).float() # Keep raw for now, normalize in MPC
    costs = torch.from_numpy(raw_data[:, 4:5]).float()

    # 4. Dataset and Loader (Crucial for 6,000 samples)
    dataset = torch.utils.data.TensorDataset(states, costs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    model = NeuralNetwork(4, 1)
    # Increased momentum to 0.9 for better "velocity"
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    criterion = nn.MSELoss()

    # 5. Training Loop
    for epoch in range(args.epochs):
        epoch_loss = 0
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
    
    torch.save(model.state_dict(), args.model_path + 'cost2go_weights.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='./data/', help='path to the data folder')
    parser.add_argument('--model-path', type=str, default='./models/', help='path to the model file')
    parser.add_argument('--data-file', type=str, default='cost2go_6000_1.csv', help='path to the data file')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    args = parser.parse_args()
    print(args)
    train(args)