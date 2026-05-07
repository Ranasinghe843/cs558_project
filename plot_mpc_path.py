import matplotlib.pyplot as plt
import numpy as np
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_nn = np.loadtxt("data/mpc_path/data_nn_1.csv", delimiter=',')
# data_heuristic = np.loadtxt('my_data.txt', delimiter=',')

obs_list = config[config['world']]['obstacles']
obs_positions = np.array(obs_list)[:, :2]
obs_half_extents = np.array(obs_list)[:, 2:]

resolution = 0.1
x_range = np.arange(-5, 5, resolution)
y_range = np.arange(-5, 5, resolution)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

# 2. Plot the Landscape
plt.figure(figsize=(5, 5))
# 4. Plot Goal and Obstacles
plt.plot(data_nn[:, 0], data_nn[:, 1], 'b-', linewidth=4, label='NN-MPC')
# Draw Obstacle Boxes
for pos, ext in zip(obs_positions, obs_half_extents):
    rect = plt.Rectangle((pos[0]-ext[0], pos[1]-ext[1]), ext[0]*2, ext[1]*2, color='red', alpha=0.5)
    plt.gca().add_patch(rect)

plt.title("MPC Horizon over Potential Field")
plt.legend()
plt.show()