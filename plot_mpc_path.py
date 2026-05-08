##########################################################
# Script to plot NN-MPC and Heuristic-MPC paths for same start-goal.
##########################################################

import matplotlib.pyplot as plt
import numpy as np
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# load robot trajectories from MPC runs
case = config['case']
fname_nn = f"{config['data_mpc_path']}/data_nn_case{case}.csv"
data_nn = np.loadtxt(fname_nn, delimiter=',')
fname_heuristic = f"{config['data_mpc_path']}/data_heuristic_case{case}.csv"
data_heuristic = np.loadtxt(fname_heuristic, delimiter=',')
start = config['start']
goal = config['goal']

# load obstacle info from config
obs_list = config[config['world']]['obstacles']
obs_positions = np.array(obs_list)[:, :2]
obs_half_extents = np.array(obs_list)[:, 2:]

# figure setup
resolution = 0.1
x_range = np.arange(-5, 5, resolution)
y_range = np.arange(-5, 5, resolution)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

plt.figure(figsize=(5, 5))
plt.plot(start[0], start[1], 'rs', markersize=10)
plt.plot(goal[0], goal[1], 'gs', markersize=10)
plt.plot(data_nn[:, 0], data_nn[:, 1], 'b-', linewidth=4, label='NN-MPC')
plt.plot(data_heuristic[:, 0], data_heuristic[:, 1], 'm-', linewidth=4, label='Heuristic-MPC')
# Draw Obstacle Boxes
for pos, ext in zip(obs_positions, obs_half_extents):
    rect = plt.Rectangle((pos[0]-ext[0], pos[1]-ext[1]), ext[0]*2, ext[1]*2, color='black', alpha=0.5)
    plt.gca().add_patch(rect)
plt.grid()
plt.legend()
plt.show()