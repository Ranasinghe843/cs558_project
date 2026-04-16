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
    
class MPCConfig:
    def __init__(self, robot_id, obs_ids, terminal_model, norm_mean, norm_std, terminal_cost="nn", horizon_length=10, goal=[0.0, 0.0]):
        self.horizon_length = horizon_length
        self.goal = np.array(goal)
        self.robot_id = robot_id
        self.robot_radius = 0.105
        self.safe_dist = 1.0 
        self.dt = 0.1
        self.terminal_cost_model = terminal_model 
        self.obs_ids = obs_ids
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.terminal_cost_type = terminal_cost 
        self.Q = np.array([0.5, 0.00]) 
        self.R = np.array([0.01, 0.01]) 
        self.W = 1.0

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
            
            if dist_to_goal > 0.1:
                dx = self.goal[0] - x
                dy = self.goal[1] - y
                desired_yaw = np.arctan2(dy, dx)

                yaw_error = desired_yaw - theta
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            
                total_cost += self.Q[1] * yaw_error**2 

            for obs_id in self.obs_ids:
                obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
                dist_obs = np.linalg.norm(temp_state[:2] - np.array(obs_pos[:2])) - self.robot_radius
                if dist_obs < self.safe_dist:
                    total_cost += self.W * (1.0 / (dist_obs + 1e-3))
            
            total_cost += (self.R[0] * v**2) + (self.R[0] * omega**2)
            
        cost2go = self.cost_to_go(temp_state) 
        total_cost += cost2go

        return total_cost


def apply_control(robot_id, v, omega):
    WHEEL_BASE = 0.16
    WHEEL_RADIUS = 0.033
    left_v = (v - (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    right_v = (v + (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=left_v)
    p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=right_v)

def mpc(args):

    with open(args.model_path + 'norm_params.pkl', 'rb') as f:
        stats = pickle.load(f)
        norm_mean = torch.tensor(stats['mean']).float()
        norm_std = torch.tensor(stats['std']).float()

    H = args.horizon_length              
    GOAL = np.array(args.goal)
    START = np.concatenate([np.array(args.start), [0]])
    bounds = [(-0.22, 0.22), (-2.84, 2.84)] * H 

    env = SimulationEnv(render=True, start_pos_2d=args.start)
    pt_start = [START[0], START[1], 0.05]
    pt_goal  = [GOAL[0], GOAL[1], 0.05]

    p.addUserDebugPoints([pt_start], pointColorsRGB=[[1, 0, 0]], pointSize=15.0)
    p.addUserDebugPoints([pt_goal], pointColorsRGB=[[0, 1, 0]], pointSize=15.0)

    trained_model = NeuralNetwork(obsv_dim=4, cost_dim=1)
    trained_model.load_state_dict(torch.load(args.model_path + 'cost2go_weights.pth'))
    trained_model.eval()
    
    MPC = MPCConfig(
        robot_id=env.robot_id, 
        obs_ids=env.obstacles, 
        terminal_model=trained_model, 
        norm_mean=norm_mean,
        norm_std=norm_std,
        terminal_cost=args.terminal_cost,
        horizon_length=H, 
        goal=GOAL
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
            # time.sleep(1./240.)

    except KeyboardInterrupt:
        p.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, default='./models/', help='path to the model file')
    parser.add_argument('--horizon-length', type=int, default=10, help='length of MPC horizon (number of steps to look ahead)')
    parser.add_argument('--terminal-cost', type=str, default='heuristic', help='type of terminal cost to use ("nn" or "heuristic")')
    parser.add_argument('--start', type=float, nargs=2, default=[-2.0, 1.0], help='Start position [x, y]')
    parser.add_argument('--goal', type=float, nargs=2, default=[1.0, 0.0], help='Goal position [x, y]')

    args = parser.parse_args()
    print(args)
    mpc(args)