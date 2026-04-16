import time
import pybullet as p
from RRT import RRTStar
from env_setup import SimulationEnv

start_pos_2d = [0.0, -2.0]
goal_pos_2d = [1.5, 2.0]

env = SimulationEnv(render=True, start_pos_2d=start_pos_2d)

print("Planning path with RRT*...")

rrt = RRTStar(
    start=start_pos_2d,
    goal=goal_pos_2d,
    bounds=[-3.0, 3.0, -3.0, 3.0],
    obstacles=env.obstacles,
    plane_id=env.plane_id,
    robot_id=env.robot_id,
    step_size=0.1,
    search_radius=7.5,
    max_iter=5000
)

result = rrt.plan()

if result is not None:
    path_coords, total_cost = result
    print(f"Path found! Total path length: {total_cost:.4f}")
    
    p.addUserDebugText("START", [start_pos_2d[0], start_pos_2d[1], 0.3], textColorRGB=[0, 1, 0], textSize=1.0)
    p.addUserDebugText("GOAL", [goal_pos_2d[0], goal_pos_2d[1], 0.3], textColorRGB=[1, 0, 0], textSize=1.0)

    for i in range(len(path_coords) - 1):
        pt1 = [path_coords[i][0], path_coords[i][1], 0.05]
        pt2 = [path_coords[i+1][0], path_coords[i+1][1], 0.05]
        p.addUserDebugLine(pt1, pt2, lineColorRGB=[1, 0, 0], lineWidth=4.0)
        
else:
    print("RRT* failed to find a path")

try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    env.disconnect()