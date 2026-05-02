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

            dist_to_goal = np.linalg.norm(temp_state[:2] - self.goal)
            total_cost += self.Q[0] * dist_to_goal**2
            
            # UNCOMMENT TO ACCOUNT FOR YAW ERROR IN COST
            # if dist_to_goal > 0.1:
            #     dx = self.goal[0] - x
            #     dy = self.goal[1] - y
            #     desired_yaw = np.arctan2(dy, dx)

            #     yaw_error = desired_yaw - theta
            #     yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            
            #     total_cost += self.Q[2] * yaw_error**2 

            # HARD_MARGIN = self.robot_radius + 0.05 

            # for obs_id in self.obs_ids:
            #     obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
            #     obs_xy = np.array(obs_pos[:2])
                
            #     dist_obs = np.linalg.norm(temp_state[:2] - obs_xy)
                
            #     if dist_obs < HARD_MARGIN:
            #         total_cost += self.W * (1.0 / (dist_obs + 1e-3))

            for obs_id in self.obs_ids:
                obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
                shape_data = p.getCollisionShapeData(obs_id, -1)[0]
                half_extents = shape_data[3]
                
                dx = abs(temp_state[0] - obs_pos[0]) - half_extents[0]
                dy = abs(temp_state[1] - obs_pos[1]) - half_extents[1]

                dist_to_edge = np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)

                true_clearance = dist_to_edge - self.robot_radius

                SAFETY_MARGIN = 0.1
                if true_clearance < SAFETY_MARGIN:
                    print(shape_data)
                    total_cost += self.W * (1.0 / (max(true_clearance, 1e-3)))
                        
            total_cost += (self.R[0] * v**2) + (self.R[1] * omega**2)

        if self.terminal_cost_type == "heuristic":
            total_cost += self.T * self.cost_to_go(temp_state)
        elif self.terminal_cost_type == "nn":
            total_cost += self.T * self.cost_to_go(temp_state)
        return total_cost


def apply_control(robot_id, v, omega):
    WHEEL_BASE = 0.16
    WHEEL_RADIUS = 0.033
    left_v = (v - (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    right_v = (v + (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=left_v)
    p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=right_v)

def mpc(config):
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

    trained_model = NeuralNetwork(obsv_dim=4, cost_dim=1, dr=dr)
    if optimizer_choice == "AdamW":
        if config['nn_version'] != 3:
            weight_path = f"{config['nn_folder']}/nn{config['nn_version']}_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
        else:
            weight_path = f"{config['nn_folder']}/nn_dr{round(dr*10)}_{optimizer_choice}{config['weight_decay']}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
    else:
        weight_path = f"{config['nn_folder']}/nn_dr{round(dr*10)}_{optimizer_choice}_epochs{epochs}_lr{config['learning_rate']}_dataset{num_samples}_{data_version}.pth"
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
                bounds=bounds
            )
            
            best_v, best_omega = res.x[0], res.x[1]
            # print(f"Optimal Command: v={best_v:.3f}, omega={best_omega:.3f} | Cost: {res.fun:.4f}")
            
            apply_control(env.robot_id, best_v, best_omega)
            
            u_guess = res.x 
            
            p.stepSimulation()
            time.sleep(1./240.)

    except KeyboardInterrupt:
        p.disconnect()

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    mpc(config)