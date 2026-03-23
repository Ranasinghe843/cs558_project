import pybullet as p
import pybullet_data
import csv
import math
import random
import datetime

from RRT import RRTStar

physicsClient = p.connect(p.DIRECT) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 0.01]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("turtlebot3_burger.urdf", startPos, startOrientation)

def create_box(pos, inflation_radius=0.15):
    
    base_half_extents = [0.25, 0.25, 0.25]
    
    inflated_half_extents = [
        base_half_extents[0] + inflation_radius,
        base_half_extents[1] + inflation_radius,
        base_half_extents[2] + inflation_radius
    ]
    
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=inflated_half_extents)
    
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half_extents, rgbaColor=[0.6, 0.6, 0.6, 1])
    
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id, 
        baseVisualShapeIndex=vis_id, 
        basePosition=pos
    )
    return box_id

obstacles = [create_box([1, 1, 0.25]), create_box([-1, 1, 0.25])]

NUM_SAMPLES = 10000
DATASET_FILENAME = f'cost_to_go_{NUM_SAMPLES}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
BOUNDS = [-3.0, 3.0, -3.0, 3.0]

def is_point_valid(x, y, obstacles_list, plane_id, robot_id):
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
        if is_point_valid(x, y, obstacles, planeId, robot_id):
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
            obstacles=obstacles,
            plane_id=planeId,
            robot_id=robot_id,
            step_size=0.2,
            search_radius=0.5,
            max_iter=3000
        )
        
        result = rrt.plan()
        
        if result is not None:
            path_coords, _ = result
            
            path_coords.reverse()
            
            cum_dist = [0.0]
            for i in range(1, len(path_coords)):
                d = math.dist(path_coords[i-1], path_coords[i])
                cum_dist.append(cum_dist[-1] + d)
            
            for i in range(len(path_coords)):
                for j in range(i + 1, len(path_coords)):
                    sub_start = path_coords[i]
                    sub_goal = path_coords[j]
                    sub_cost = cum_dist[j] - cum_dist[i]
                    
                    writer.writerow([
                        round(sub_start[0], 4), 
                        round(sub_start[1], 4), 
                        round(sub_goal[0], 4), 
                        round(sub_goal[1], 4), 
                        round(sub_cost, 4)
                    ])
                    
                    data_points_collected += 1
                    
                    if data_points_collected >= NUM_SAMPLES:
                        break
                if data_points_collected >= NUM_SAMPLES:
                    break
            
            paths_generated += 1
            print(f"Progress: {data_points_collected}/{NUM_SAMPLES} data points from {paths_generated} full paths.")

print(f"\nDataset generation complete! Saved to {DATASET_FILENAME}")
p.disconnect()