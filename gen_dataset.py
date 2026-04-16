import pybullet as p
import pybullet_data
import csv
import math
import random
import datetime
from RRT import RRTStar
from pathlib import Path
from env_setup import SimulationEnv

env = SimulationEnv(render=False)

NUM_SAMPLES = 10000
Path("data").mkdir(exist_ok=True)
counter = 1
while Path(f"data/cost2go_{NUM_SAMPLES}_{counter}.csv").exists():
    counter += 1
DATASET_FILENAME = f"data/cost2go_{NUM_SAMPLES}_{counter}.csv"
BOUNDS = [-3.0, 3.0, -3.0, 3.0]

def is_point_valid(x, y, plane_id, robot_id):
    ray_start = [x, y, 1.0]
    ray_end = [x, y, 0.0]
    result = p.rayTest(ray_start, ray_end)[0]
    if result[0] in [-1, plane_id, robot_id]:
        return True
    return False

def generate_random_valid_point():
    while True:
        x = random.uniform(BOUNDS[0], BOUNDS[1])
        y = random.uniform(BOUNDS[2], BOUNDS[3])
        if is_point_valid(x, y, env.plane_id, env.robot_id):
            return [x, y]

print(f"Starting dataset generation: Target {NUM_SAMPLES} data points...")

with open(DATASET_FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['start_x', 'start_y', 'goal_x', 'goal_y', 'path_length'])
    
    data_points_collected = 0
    paths_generated = 0
    
    while data_points_collected < NUM_SAMPLES:
        start_pt = generate_random_valid_point()
        goal_pt = generate_random_valid_point()
        
        if math.dist(start_pt, goal_pt) < 0.5:
            continue
            
        rrt = RRTStar(
            start=start_pt,
            goal=goal_pt,
            bounds=BOUNDS,
            obstacles=env.obstacles,
            plane_id=env.plane_id,
            robot_id=env.robot_id,
            step_size=0.1,
            search_radius=7.5,
            max_iter=5000
        )
        
        result = rrt.plan()
        
        if result is not None:
            path_coords, _ = result
            
            path_coords.reverse() 
            
            cum_dist = [0.0]
            for i in range(1, len(path_coords)):
                d = math.dist(path_coords[i-1], path_coords[i])
                cum_dist.append(cum_dist[-1] + d)
            
            total_path_length = cum_dist[-1]
            start_node = path_coords[0]
            goal_node = path_coords[-1]
            
            for i in range(len(path_coords)):
                current_pt = path_coords[i]
                
                writer.writerow([
                    round(start_node[0], 4), round(start_node[1], 4), 
                    round(current_pt[0], 4), round(current_pt[1], 4), 
                    round(cum_dist[i], 4)
                ])
                
                writer.writerow([
                    round(current_pt[0], 4), round(current_pt[1], 4), 
                    round(goal_node[0], 4), round(goal_node[1], 4), 
                    round(total_path_length - cum_dist[i], 4)
                ])
                data_points_collected += 2
            
            paths_generated += 1
            print(f"Progress: {data_points_collected}/{NUM_SAMPLES} data points from {paths_generated} full paths.")

print(f"\nDataset generation complete! Saved to {DATASET_FILENAME}")
p.disconnect()