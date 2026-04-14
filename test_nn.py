import torch
import torch.nn as nn
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
    
def test(args):
    # 2. Instantiate and Load
    model = NeuralNetwork(obsv_dim=4, cost_dim=1)
    model.load_state_dict(torch.load(args.model_path + 'cost2go_weights.pth'))

    # 3. SET TO EVALUATION MODE (Crucial!)
    model.eval()

    # Example raw input from your dataset: 2.4466, -2.5257, 2.4295, -2.3264
    test_input = torch.tensor([2.4295,-2.3264,2.2913,-1.7794]).float()

    # 2. Run Inference
    with torch.no_grad(): # Disables gradient calculation to save memory/speed
        prediction = model(test_input)

    print(f"Predicted Cost: {prediction.item():.4f}")
    print(f"Actual Cost from Dataset: 0.6024")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, default='./models/', help='path to the model file')

    args = parser.parse_args()
    print(args)
    test(args)