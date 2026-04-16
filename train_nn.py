import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse

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
    
def train(args):
    
    raw_data = np.loadtxt(args.data_path + args.data_file, delimiter=',', skiprows=1)
    np.random.shuffle(raw_data)

    inputs_raw = raw_data[:, 0:4]
    mean = np.mean(inputs_raw, axis=0)
    std = np.std(inputs_raw, axis=0) + 1e-8
    
    stats = {'mean': mean, 'std': std}
    with open(args.model_path + 'norm_params.pkl', 'wb') as f:
        pickle.dump(stats, f)

    states = torch.from_numpy(inputs_raw).float()
    costs = torch.from_numpy(raw_data[:, 4:5]).float()

    dataset = torch.utils.data.TensorDataset(states, costs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    model = NeuralNetwork(4, 1)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    criterion = nn.MSELoss()

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
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    args = parser.parse_args()
    print(args)
    train(args)