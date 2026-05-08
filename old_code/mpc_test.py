import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.optimize import minimize
import torch
import yaml
from env_setup import SimulationEnv
from nn import NeuralNetwork

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
        self.Q = np.array(Q)  # Expected: [Dist_Weight, Yaw_Weight]
        self.R = np.array(R)  # Expected: [V_Weight, Omega_Weight]
        self.W = W            # Obstacle Weight
        self.T = T            # Terminal Weight
        
        self.cached_obs_pos = []
        self.state = None

        if self.terminal_cost_type == "nn":
            self.cost_to_go = self.nn_cost_to_go
        else:
            self.cost_to_go = self.heuristic_cost_to_go

    def get_robot_state(self):
        # Fetch robot state
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(ori)
        self.state = np.array([pos[0], pos[1], euler[2]])
        
        # SPEED OPTIMIZATION: Cache obstacle positions once per control step
        # This prevents the optimizer from calling PyBullet thousands of times
        self.cached_obs_pos = []
        for obs_id in self.obs_ids:
            obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
            self.cached_obs_pos.append(np.array(obs_pos[:2]))
        self.cached_obs_pos = np.array(self.cached_obs_pos)
        
        return self.state

    def motion_model(self, state, v, omega):
        # Kinematic unicycle model (Forward Euler)
        new_x = state[0] + v * np.cos(state[2]) * self.dt
        new_y = state[1] + v * np.sin(state[2]) * self.dt
        new_theta = state[2] + omega * self.dt
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
        curr_state = self.state  
        
        # Increase these in your config.yaml for better results:
        # W (Obstacle) should be significantly higher than Q (Goal)
        
        for i in range(self.horizon_length):
            v, omega = u[i]
            curr_state = self.motion_model(curr_state, v, omega)
            x, y, theta = curr_state

            # 1. Position Cost
            dist_to_goal = np.linalg.norm(curr_state[:2] - self.goal)
            total_cost += self.Q[0] * (dist_to_goal**2)
            
            # 2. Heading Cost (Crucial for not getting stuck)
            desired_yaw = np.arctan2(self.goal[1] - y, self.goal[0] - x)
            yaw_error = np.arctan2(np.sin(desired_yaw - theta), np.cos(desired_yaw - theta))
            total_cost += self.Q[1] * (yaw_error**2)

            # 3. REFINED OBSTACLE COST (Reciprocal Barrier)
            if len(self.cached_obs_pos) > 0:
                dists = np.linalg.norm(self.cached_obs_pos - curr_state[:2], axis=1)
                
                # The 'safe_dist' is the radius where the robot MUST start turning.
                # Increase this if the robot is too brave.
                safe_dist = self.robot_radius + 0.25 
                
                for d in dists:
                    if d < safe_dist:
                        # Quadratic reciprocal cost: very aggressive as d gets small
                        # We use (safe_dist - d) so the cost is 0 at the edge and inf at d=0
                        total_cost += self.W * (1.0 / (d - self.robot_radius + 1e-4)**2)
                    else:
                        # Optional: small 'repulsion' even outside the safe zone
                        total_cost += self.W * 0.1 * np.exp(-(d - safe_dist))

            # 4. Control Effort
            total_cost += (self.R[0] * v**2) + (self.R[1] * omega**2)

        # 5. Terminal Cost
        total_cost += self.T * (self.cost_to_go(curr_state)**2)
        return total_cost

def apply_control(robot_id, v, omega):
    WHEEL_BASE = 0.16
    WHEEL_RADIUS = 0.033
    left_v = (v - (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    right_v = (v + (omega * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=left_v)
    p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=right_v)

def mpc_run(config):
    # Setup parameters
    H = config['horizon_length']             
    GOAL = np.array(config['goal'])
    START = np.array(config['start'])
    
    # Differential drive constraints: [v_min, v_max], [omega_min, omega_max]
    bounds = [(-0.22, 0.22), (-2.84, 2.84)] * H 

    # Initialize PyBullet Env
    env = SimulationEnv(render=True, start_pos_2d=START, inflation_radius=0.0)
    p.addUserDebugPoints([[GOAL[0], GOAL[1], 0.05]], pointColorsRGB=[[0, 1, 0]], pointSize=15.0)

    # Load Neural Network if used
    trained_model = None
    if config['terminal_cost_type'] == "nn":
        trained_model = NeuralNetwork(obsv_dim=4, cost_dim=1, dr=config['dropout_rate'])
        # Simplified path construction for the example
        weight_path = f"{config['nn_folder']}/nn_model.pth" 
        trained_model.load_state_dict(torch.load(weight_path))
        trained_model.eval()
    
    # Initialize MPC Class
    MPC = MPCConfig(
        robot_id=env.robot_id, 
        obs_ids=env.obstacles, 
        terminal_model=trained_model, 
        terminal_cost_type=config['terminal_cost_type'],
        horizon_length=H, 
        goal=GOAL,
        Q=config['Q'], R=config['R'], W=config['W'], T=config['T']
    )

    u_guess = np.zeros(H * 2) 

    print(f"MPC active. Target: {GOAL}")

    try:
        while True:
            # 1. Update State and Cache Obstacles
            state = MPC.get_robot_state()
            
            # 2. Check Success
            if np.linalg.norm(state[:2] - GOAL) < 0.05:
                print("Goal Reached!")
                apply_control(env.robot_id, 0, 0)
                break

            # 3. Solve MPC Optimization
            # method='SLSQP' is best for bounded non-linear problems
            res = minimize(
                MPC.objective_function, 
                u_guess, 
                method='SLSQP', 
                bounds=bounds,
                options={'ftol': 1e-3, 'maxiter': 40} # Speed up by limiting iterations
            )
            
            # 4. Extract first control action
            best_v, best_omega = res.x[0], res.x[1]
            apply_control(env.robot_id, best_v, best_omega)
            
            # 5. Warm Start: Shift the guess for the next iteration
            # Take the current solution, move it forward one step, and append a zero
            u_reshaped = res.x.reshape(H, 2)
            u_guess = np.roll(u_reshaped, -1, axis=0)
            u_guess[-1] = np.array([0, 0])
            u_guess = u_guess.flatten()
            
            # 6. Step Simulation
            p.stepSimulation()
            time.sleep(1./240.)

    except KeyboardInterrupt:
        p.disconnect()

if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    mpc_run(config)