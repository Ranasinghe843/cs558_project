import pybullet as p
import pybullet_data
import time
from RRT import RRTStar

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
planeId = p.loadURDF("plane.urdf")

start_pos_2d = [0.0, -2.0]
goal_pos_2d = [1.5, 2.0]

startPos = [start_pos_2d[0], start_pos_2d[1], 0.01]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("turtlebot3_burger.urdf", startPos, startOrientation)

def create_box(pos, inflation_radius=0.1):
    base_half_extents = [0.25, 0.25, 0.25]
    inflated_half_extents = [
        base_half_extents[0] + inflation_radius,
        base_half_extents[1] + inflation_radius,
        base_half_extents[2] + inflation_radius
    ]
    
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=inflated_half_extents)
    
    ghost_vis_id = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=inflated_half_extents, 
        rgbaColor=[0.2, 0.6, 1, 0.2]
    )
    
    core_vis_id = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=base_half_extents, 
        rgbaColor=[0.4, 0.4, 0.4, 1] # Solid Grey
    )
    
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id, 
        baseVisualShapeIndex=ghost_vis_id, 
        basePosition=pos
    )
    
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=core_vis_id,
        basePosition=pos
    )
    
    return box_id

obstacles = [
    create_box([1, 1, 0.25]), 
    create_box([-1, 1, 0.25]), 
    create_box([0, 0, 0.25])
]

print("Planning path with RRT*...")
rrt = RRTStar(
    start=start_pos_2d,
    goal=goal_pos_2d,
    bounds=[-3.0, 3.0, -3.0, 3.0],
    obstacles=obstacles,
    plane_id=planeId,
    robot_id=robot_id,
    step_size=0.1,
    search_radius=1.5, 
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
    p.disconnect()