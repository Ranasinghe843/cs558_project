import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.optimize import minimize
import torch
import pickle
import torch.nn as nn
import argparse
from env_setup import SimulationEnv
from nn import NeuralNetwork
import yaml
from shlex import split

class MPCConfig:
    def __init__(self, robot_id, obs_ids, terminal_model, terminal_cost_type, horizon_length, goal, Q, R, W, T):
        self.horizon_length = horizon_length
        self.goal = np.array(goal)
        self.robot_id = robot_id
        self.robot_radius = 0.105
        self.dt = 0.1
        self.terminal_cost_model = terminal_model 
        self.obs_ids = obs_ids
        self.terminal_cost_type = terminal_cost_type 
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.W = np.array(W)
        self.T = np.array(T)

        # 2. Assign the FUNCTION REFERENCE (No parentheses here!)
        if self.terminal_cost_type == "nn":
            print("Using Neural Network for Terminal Cost")
            self.cost_to_go = self.nn_cost_to_go
        elif self.terminal_cost_type == "heuristic":
            print("Using Heuristic for Terminal Cost")
            self.cost_to_go = self.heuristic_cost_to_go

        # Initialize containers
        pos_list = []
        extents_list = []
        
        for obs_id in self.obs_ids:
            # 1. Get Position (x, y)
            pos, _ = p.getBasePositionAndOrientation(obs_id)
            pos_list.append([pos[0], pos[1]])
            
            # 2. Get Geometry (Half-Extents)
            # getCollisionShapeData returns a list of shapes; we take the first one [0]
            # Index 3 is the 'half-extents' for boxes: [width/2, length/2, height/2]
            shape_data = p.getCollisionShapeData(obs_id, -1)
            if shape_data:
                half_extents = shape_data[0][3] 
                extents_list.append([half_extents[0], half_extents[1]])
            else:
                # Fallback for unexpected shapes
                extents_list.append([0.1, 0.1]) 

        # Convert to NumPy arrays for vectorized math
        self.obs_positions = np.array(pos_list)      # Shape: (num_obs, 2)
        self.obs_half_extents = np.array(extents_list) # Shape: (num_obs, 2)

    def get_robot_state(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(ori)
        self.state = np.array([pos[0], pos[1], euler[2]])
        return self.state
    
    def motion_model(self, state, v, omega):
        x, y, theta = state
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        new_theta = theta + omega * self.dt
        return np.array([new_x, new_y, new_theta])
    
    def heuristic_cost_to_go(self, state):
        return np.linalg.norm(state[:2] - self.goal)
    
    def nn_cost_to_go(self, state):
        raw_input = torch.tensor([state[0], state[1], self.goal[0], self.goal[1]]).float()
        
        with torch.no_grad():
            prediction = self.terminal_cost_model(raw_input)
        return prediction.item()

    def objective_function(self, u_flattened):
        u = u_flattened.reshape(self.horizon_length, 2)
        total_cost = 0
        temp_state = self.state  
        
        for i in range(self.horizon_length):
            v, omega = u[i]
            temp_state = self.motion_model(temp_state, v, omega)
            x, y, theta = temp_state

            # goal cost
            dist_to_goal = np.linalg.norm(temp_state[:2] - self.goal)
            goal_cost = self.Q[0] * dist_to_goal**2
            total_cost += goal_cost

            # control cost
            control_cost = (self.R[0] * v**2) + (self.R[1] * omega**2)
            total_cost += control_cost
            
            # UNCOMMENT TO ACCOUNT FOR YAW ERROR IN COST
            # if dist_to_goal > 0.02:
            #     dx = self.goal[0] - x
            #     dy = self.goal[1] - y
            #     desired_yaw = np.arctan2(dy, dx)

            #     yaw_error = desired_yaw - theta
            #     yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            
            #     total_cost += self.Q[1] * yaw_error**2 

            # HARD_MARGIN = self.robot_radius + 0.05 

            # for obs_id in self.obs_ids:
            #     obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
            #     obs_xy = np.array(obs_pos[:2])
                
            #     dist_obs = np.linalg.norm(temp_state[:2] - obs_xy)
                
            #     if dist_obs < HARD_MARGIN:
            #         total_cost += self.W * (1.0 / (dist_obs + 1e-3))

            # obstacle cost
            SAFETY_MARGIN = 0.05
            delta = np.abs(temp_state[:2] - self.obs_positions) - self.obs_half_extents
                
            dist_to_edges = np.linalg.norm(np.maximum(delta, 0), axis=1)

            true_clearances = (dist_to_edges - self.robot_radius)**2

            mask = true_clearances < SAFETY_MARGIN
            if np.any(mask):
                obs_cost = np.sum(self.W * (1.0 / (true_clearances[mask] + 1e-3)))
                # obs_cost = 0
                total_cost += obs_cost
            else:
                obs_cost = 0

        # remaining cost to go from final predicted state to goal
        remaining_cost = self.T * (self.cost_to_go(temp_state)**2)
        total_cost += remaining_cost

        print(f"Goal Cost: {goal_cost:.4f}, Control Cost: {control_cost:.4f}, Obstacle Cost: {obs_cost:.4f}, Remaining Cost: {remaining_cost:.4f}")

        return total_cost


def apply_control(robot_id, v, omega):
    WHEEL_BASE = 0.16
    WHEEL_RADIUS = 0.033
    left_v = (v - (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    right_v = (v + (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=left_v)
    p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=right_v)

def mpc(config):
    world = config['world']
    num_samples = config['num_samples']
    data_version = config['version']
    epochs = config['epochs']
    optimizer_choice = config['optimizer']
    dr = config['dropout_rate']

    H = config['horizon_length']             
    GOAL = np.array(config['goal'])
    START = np.concatenate([np.array(config['start']), [0]])
    bounds = [(-0.22, 0.22), (-2.84, 2.84)] * H # on robot commands (v, omega) for each step in horizon

    env = SimulationEnv(render=True, start_pos_2d=config['start'], inflation_radius=0.0)
    pt_start = [START[0], START[1], 0.05]
    pt_goal  = [GOAL[0], GOAL[1], 0.05]

    p.addUserDebugPoints([pt_start], pointColorsRGB=[[1, 0, 0]], pointSize=15.0)
    p.addUserDebugPoints([pt_goal], pointColorsRGB=[[0, 1, 0]], pointSize=15.0)
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])


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
    
    MPC = MPCConfig(
        robot_id=env.robot_id,
        obs_ids=env.obstacles,
        terminal_model=trained_model,
        terminal_cost_type=config['terminal_cost_type'],
        horizon_length=H,
        goal=GOAL,
        Q=config['Q'],
        R=config['R'],
        W=config['W'],
        T=config['T']
    )
    u_guess = np.zeros(H * 2) 

    print("MPC Started. Heading to:", GOAL)

    try:
        while True:
            state = MPC.get_robot_state()
            
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
                options={'ftol': 1e-3, 'maxiter': 20}
            )
            
            best_v, best_omega = res.x[0], res.x[1]
            # print(f"Optimal Command: v={best_v:.3f}, omega={best_omega:.3f} | Cost: {res.fun:.4f}")
            
            #apply_control(env.robot_id, best_v, best_omega)
            
            #u_guess = res.x
            u_guess = np.concatenate([res.x[2:], res.x[-2:]])
            
            #p.stepSimulation()
            #time.sleep(1./240.)

            steps_per_mpc = int(MPC.dt * 240)
            for _ in range(steps_per_mpc):
                apply_control(env.robot_id, best_v, best_omega)
                p.stepSimulation()
                time.sleep(1./240.)

    except KeyboardInterrupt:
        p.disconnect()

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    mpc(config)