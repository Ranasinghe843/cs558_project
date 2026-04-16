import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.optimize import minimize
import torch
import pickle
import torch.nn as nn
import argparse

class NeuralNetwork(nn.Module):
    def __init__(self, obsv_dim, cost_dim):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obsv_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, cost_dim),
        )

    def forward(self, x):
        return self.net(x)
    
class MPCConfig:
    def __init__(self, robot_id, obs_ids, terminal_model, norm_mean, norm_std, terminal_cost="nn", horizon_length=10, goal=[0.0, 0.0]):
        self.horizon_length = horizon_length
        self.goal = np.array(goal)
        self.robot_id = robot_id
        self.robot_radius = 0.105
        self.safe_dist = self.robot_radius + 0.10 # Minimum safe distance to obstacles
        self.dt = 0.1
        self.terminal_cost_model = terminal_model # Pass the TRAINED model here
        self.obs_ids = obs_ids
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        # 1. Store the choice string
        self.terminal_cost_type = terminal_cost 

        # 2. Assign the FUNCTION REFERENCE (No parentheses here!)
        if self.terminal_cost_type == "nn":
            print("Using Neural Network for Terminal Cost")
            self.cost_to_go = self.nn_cost_to_go
        elif self.terminal_cost_type == "heuristic":
            print("Using Heuristic for Terminal Cost")
            self.cost_to_go = self.heuristic_cost_to_go
            self.Q = np.array([2.0, 0.5]) 
            self.R = np.array([0.2, 0.2]) # Control effort weights for [v, omega] 
            self.W = 2.4 

    def get_robot_state(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(ori)
        self.state = np.array([pos[0], pos[1], euler[2]])
        return self.state
    
    def motion_model(self, state, v, omega):
        """Predicts next state based on the PROVIDED state, not current robot position"""
        x, y, theta = state
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        new_theta = theta + omega * self.dt
        return np.array([new_x, new_y, new_theta])
    
    def heuristic_cost_to_go(self, state):
        """Calculates distance from the predicted FUTURE state to goal"""
        return ((state[:2] - self.goal)**2).sum()
    
    def nn_cost_to_go(self, state):
        """Uses the Neural Network on the predicted FUTURE state"""
        raw_input = torch.tensor([state[0], state[1], self.goal[0], self.goal[1]]).float()
        # YOU MUST DO THIS if you did it during training:
        normalized_input = (raw_input - self.norm_mean) / self.norm_std 
        
        with torch.no_grad():
            prediction = self.terminal_cost_model(normalized_input)
        return prediction.item()

    def objective_function(self, u_flattened):
        u = u_flattened.reshape(self.horizon_length, 2)
        total_cost = 0
        temp_state = self.state  
        
        for i in range(self.horizon_length):
            v, omega = u[i]
            temp_state = self.motion_model(temp_state, v, omega)
            
            # Current Position and Heading
            x, y, theta = temp_state
            
            # 1. Stage Cost: Distance to Goal
            dist_to_goal = np.linalg.norm(temp_state[:2] - self.goal)
            total_cost += self.Q[0] * dist_to_goal**2
            
            # --- NEW: HEADING ERROR COST ---
            # Calculate angle to goal from current position
            if dist_to_goal > 0.1:
                dx = self.goal[0] - x
                dy = self.goal[1] - y
                desired_yaw = np.arctan2(dy, dx)
                
                # Shortest angular distance (Normalizes to -pi to pi)
                yaw_error = desired_yaw - theta
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            
            # Add heading cost (Only if we aren't already at the goal)
                total_cost += self.Q[1] * yaw_error**2 # Weight of 2.0 as an example
            # -------------------------------

            # 2. Stage Cost: Obstacle Avoidance
            for obs_id in self.obs_ids:
                obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
                dist_obs = np.linalg.norm(temp_state[:2] - np.array(obs_pos[:2])) - self.robot_radius
                # total_cost += self.W * (1.0 / (dist_obs + 1e-3))
                total_cost += self.W * np.maximum(0, self.safe_dist - dist_obs)**2
            
            # 3. Stage Cost: Smoothness
            # Note: Added v and omega penalty separately as per your snippet
            total_cost += (self.R[0] * v**2) + (self.R[0] * omega**2)
            
        # 4. Terminal Cost
        total_cost += self.cost_to_go(temp_state)
        
        return total_cost

# Obstacle creation from your original script
def create_box(pos):
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25])
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25], rgbaColor=[0.6, 0.6, 0.6, 1])
    return p.createMultiBody(0, col_id, vis_id, pos)


def apply_control(robot_id, v, omega):
    """Converts [v, w] to wheel angular velocities for Burger"""
    WHEEL_BASE = 0.16
    WHEEL_RADIUS = 0.033
    left_v = (v - (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    right_v = (v + (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=left_v)
    p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=right_v)

def mpc(args):
    # --- MAIN CONTROL LOOP ---
    terminalCost = NeuralNetwork(obsv_dim=4, cost_dim=1)
    terminalCost.load_state_dict(torch.load(args.model_path + 'cost2go_weights.pth'))
    terminalCost.eval()

    with open(args.model_path + 'normalization_params.pkl', 'rb') as f:
        stats = pickle.load(f)
        # Ensure these are tensors for fast math
        norm_mean = torch.tensor(stats['mean']).float()
        norm_std = torch.tensor(stats['std']).float()

    H = args.horizon_length              # Prediction Horizon (look 1.0s ahead)
    GOAL = np.array(args.goal) # Target location
    START = np.concatenate([np.array(args.start), [0]])
    bounds = [(-0.22, 0.22), (-2.84, 2.84)] * H # Burger hardware limits

    # --- PHYSICS SETUP ---
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf")

    trained_model = NeuralNetwork(obsv_dim=4, cost_dim=1)
    trained_model.load_state_dict(torch.load(args.model_path + 'cost2go_weights.pth'))
    trained_model.eval()

    robot_id = p.loadURDF("turtlebot3_burger.urdf", START, p.getQuaternionFromEuler([0, 0, 0]))
    obs_ids = [create_box([1, 1, 0.25]), create_box([-1, 1, 0.25])]
    MPC = MPCConfig(
        robot_id=robot_id, 
        obs_ids=obs_ids, 
        terminal_model=trained_model, # Fixed
        norm_mean=norm_mean,
        norm_std=norm_std,
        terminal_cost=args.terminal_cost, # "nn" or "heuristic"
        horizon_length=H, 
        goal=GOAL
    )
    u_guess = np.zeros(H * 2) # Initial guess for velocities

    print("MPC Started. Heading to:", GOAL)

    try:
        while True:
            state = MPC.get_robot_state()
            
            # Check if goal reached
            if np.linalg.norm(state[:2] - GOAL) < 0.1:
                print("Goal Reached!")
                apply_control(robot_id , 0, 0) # Stop the robot
                break

            # Solve MPC Optimization
            res = minimize(
                MPC.objective_function, 
                u_guess, 
                method='SLSQP', 
                bounds=bounds
            )
            
            # Extract first optimal command
            best_v, best_omega = res.x[0], res.x[1]
            # print(f"Optimal Command: v={best_v:.3f}, omega={best_omega:.3f} | Cost: {res.fun:.4f}")
            
            # Apply to robot
            apply_control(robot_id, best_v, best_omega)
            
            # Warm start for next loop
            u_guess = res.x 
            
            p.stepSimulation()
            time.sleep(1./240.)

    except KeyboardInterrupt:
        p.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--data-path', type=str, default='./data/', help='path to the data folder')
    parser.add_argument('--model-path', type=str, default='./models/', help='path to the model file')
    # parser.add_argument('--data-file', type=str, default='cost2go_10000_2026-03-23_16-33-55.csv', help='path to the data file')
    # parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--weight-decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--horizon-length', type=int, default=10, help='length of MPC horizon (number of steps to look ahead)')
    parser.add_argument('--terminal-cost', type=str, default='heuristic', help='type of terminal cost to use ("nn" or "heuristic")')
    # Specifically requires exactly 2 values
    parser.add_argument('--start', type=float, nargs=2, default=[2.0, 2.0], help='Start position [x, y]')
    parser.add_argument('--goal', type=float, nargs=2, default=[0.0, 0.0], help='Goal position [x, y]')

    args = parser.parse_args()
    # print(args)
    # train(args)
    mpc(args)