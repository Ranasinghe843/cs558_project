######################################################################
# Script for MPC Implementation for Differential Drive Robot in PyBullet
######################################################################

import pybullet as p
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import torch
from env_setup import SimulationEnv
from nn import NeuralNetwork
import yaml
import numpy as np

# Class to hold MPC configuration and methods
class MPCConfig:
    def __init__(self, robot_id, obs_list, terminal_model, terminal_cost_type, horizon_length, start, goal, Q, R, W, T):
        self.horizon_length = horizon_length
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.robot_id = robot_id
        self.robot_radius = 0.105
        self.dt = 0.1
        self.terminal_cost_model = terminal_model 
        self.obs_positions = np.array(obs_list)[:, :2]
        self.obs_half_extents = np.array(obs_list)[:, 2:]
        self.terminal_cost_type = terminal_cost_type 
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.W = np.array(W)
        self.T = np.array(T)
        self.norm_dist = np.linalg.norm(self.start - self.goal)

        # which terminal cost function to use based on config
        if self.terminal_cost_type == "nn":
            print("Using Neural Network for Terminal Cost")
            self.cost_to_go = self.nn_cost_to_go
        elif self.terminal_cost_type == "heuristic":
            print("Using Heuristic for Terminal Cost")
            self.cost_to_go = self.heuristic_cost_to_go

    # Get the current state of the robot (x, y, theta)
    def get_robot_state(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(ori)
        self.state = np.array([pos[0], pos[1], euler[2]])
        return self.state
    
    # Predict the next state using a kinematic model
    def motion_model(self, state, v, omega):
        x, y, theta = state
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        new_theta = theta + omega * self.dt
        return np.array([new_x, new_y, new_theta])
    
    # heuristic cost to go based on Euclidean distance to the goal
    def heuristic_cost_to_go(self, state):
        return np.linalg.norm(state[:2] - self.goal)
    
    # neural network cost to go prediction
    def nn_cost_to_go(self, state):
        raw_input = torch.tensor([state[0], state[1], self.goal[0], self.goal[1]]).float()
        with torch.no_grad():
            prediction = self.terminal_cost_model(raw_input)
        return prediction.item()

    # Objective function for optimization
    def objective_function(self, u_flattened):
        u = u_flattened.reshape(self.horizon_length, 2)
        total_cost = 0
        temp_state = self.state

        HARD_MARGIN = 0.15
        
        # total stage over predicted horizon
        for i in range(self.horizon_length):
            v, omega = u[i]
            temp_state = self.motion_model(temp_state, v, omega)
            x, y, theta = temp_state

            # goal cost
            dist_to_goal = np.linalg.norm(temp_state[:2] - self.goal)
            goal_cost = (self.Q[0] /self.norm_dist) * dist_to_goal**2
            total_cost += goal_cost

            # control cost
            control_cost = (self.R[0] * v**2) + (self.R[1] * omega**2)
            total_cost += control_cost

            # yaw error cost
            dx = self.goal[0] - x
            dy = self.goal[1] - y
            desired_yaw = np.arctan2(dy, dx)

            yaw_error = np.arctan2(np.sin(desired_yaw - theta), np.cos(desired_yaw - theta))
            total_cost += self.Q[1] * (yaw_error**2)

            # obstacle cost
            delta = np.abs(temp_state[:2] - self.obs_positions) - self.obs_half_extents

            outside_dist = np.linalg.norm(np.maximum(delta, 0), axis=1)

            inside_dist = np.minimum(np.max(delta, axis=1), 0)

            signed_dist = outside_dist + inside_dist

            clearance = signed_dist - self.robot_radius
            violation = np.maximum(0, HARD_MARGIN - clearance)
            
            if violation.size > 0:
                max_violation = np.max(violation)
                obs_cost = self.W * (max_violation**2) * (1 - np.exp(-5 * dist_to_goal))
                total_cost += obs_cost
            else:
                obs_cost = 0

        remaining_cost = (self.T/self.norm_dist) * ((self.cost_to_go(temp_state))**2)

        total_cost += remaining_cost

        return total_cost

# to apply control commands to the robot in the simulation
def apply_control(robot_id, v, omega):
    WHEEL_BASE = 0.16
    WHEEL_RADIUS = 0.033
    left_v = (v - (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    right_v = (v + (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=left_v)
    p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=right_v)

# Main MPC loop
def mpc(config):
    # Load configuration parameters
    world = config['world']
    num_samples = config['num_samples']
    data_version = config['version']
    epochs = config['epochs']
    optimizer_choice = config['optimizer']
    dr = config['dropout_rate']
    case = config['case']
    H = config['horizon_length']       
    GOAL = np.array(config['goal'])
    START =np.array(config['start'])


    bounds = [(-0.22, 0.22), (-2.84, 2.84)] * H         # on robot commands (v, omega) 
    # setup env and visualization
    env = SimulationEnv(render=True, start_pos_2d=START[0:2], start_angle=START[2], inflation_radius=0.0)
    pt_start = [START[0], START[1], 0.05]
    pt_goal  = [GOAL[0], GOAL[1], 0.05]
    p.addUserDebugPoints([pt_start], pointColorsRGB=[[1, 0, 0]], pointSize=15.0)
    p.addUserDebugPoints([pt_goal], pointColorsRGB=[[0, 1, 0]], pointSize=15.0)
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])

    # Load the trained neural network model for terminal cost prediction
    trained_model = NeuralNetwork(obsv_dim=4, cost_dim=1, dr=dr)
    if optimizer_choice == "AdamW":
        if config['nn_version'] != 3:
            weight_path = f"{config['nn_folder']}/{world}/nn{config['nn_version']}_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
        else:
            weight_path = f"{config['nn_folder']}/{world}/nn_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
    else:
        weight_path = f"{config['nn_folder']}/{world}/nn_dr{round(dr*10)}_{optimizer_choice}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
    print("Loading model weights from:", weight_path)
    trained_model.load_state_dict(torch.load(weight_path))
    trained_model.eval()
    
    # initialize MPC class
    MPC = MPCConfig(
        robot_id=env.robot_id,
        obs_list = config[config['world']]['obstacles'],
        terminal_model=trained_model,
        terminal_cost_type=config['terminal_cost_type'],
        horizon_length=H,
        start=START[0:2],
        goal=GOAL,
        Q=config['Q'],
        R=config['R'],
        W=config['W'],
        T=config['T']
    )
    
    u_guess = np.zeros(H * 2)   # Initial guess for optimization
    u_guess[0::2] = 0.001
    store_state = []            # to store the robot's trajectory 

    print("MPC Started. Heading to:", GOAL)

    try:
        while True:
            # get current state of the robot
            state = MPC.get_robot_state()
            store_state.append(state[0:2])
            
            # Check if goal reached
            if np.linalg.norm(state[:2] - GOAL) < 0.1:
                print("Goal Reached!")
                apply_control(env.robot_id , 0, 0) # Stop the robot
                break

            # Solve MPC Optimization
            res = minimize(
                MPC.objective_function, 
                u_guess, 
                method='SLSQP',
                bounds=bounds,
            )
            
            # apply the first control command from the optimized sequence
            best_v, best_omega = res.x[0], res.x[1]

            u_guess = np.concatenate([res.x[2:], res.x[-2:]])   # warm start for next optimization

            steps_per_mpc = int(MPC.dt * 240)
            for _ in range(steps_per_mpc):
                apply_control(env.robot_id, best_v, best_omega)
                p.stepSimulation()
                time.sleep(1./240.)

        # save robot trajectory to txt
        if config['terminal_cost_type'] == "nn":
            fname = f"{config['data_mpc_path']}/data_nn_case{case}.csv"
        elif config['terminal_cost_type'] == "heuristic":
            fname = f"{config['data_mpc_path']}/data_heuristic_case{case}.csv"

        np.savetxt(fname, store_state, delimiter=',', comments='')

    except KeyboardInterrupt:
        # save robot trajectory to txt
        if config['terminal_cost_type'] == "nn":
            fname = f"{config['data_mpc_path']}/data_nn_case{case}.csv"
        elif config['terminal_cost_type'] == "heuristic":
            fname = f"{config['data_mpc_path']}/data_heuristic_case{case}.csv"

        np.savetxt(fname, store_state, delimiter=',', comments='')
        p.disconnect()

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    mpc(config)