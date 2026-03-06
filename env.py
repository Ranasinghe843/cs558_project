import pybullet as p
import pybullet_data
import time
import numpy as np

# 1. Setup Simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0) # We want to step manually for control loops

# 2. Load Environment
planeId = p.loadURDF("plane.urdf")

# Load Turtlebot (Burger). Note: Ensure you have the urdf in your path 
# or use the built-in 'turtlebot.urdf' if available in your pybullet_data
startPos = [0, 0, 0.01]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("turtlebot3_burger.urdf", startPos, startOrientation)

# 3. Define Obstacles (For RRT* to avoid)
def create_box(pos):
    return p.loadURDF("cube.urdf", pos, globalScaling=0.5)

obstacles = [create_box([1, 1, 0.25]), create_box([-1, 1, 0.25])]

# 4. Control Logic Placeholder
def apply_wheel_velocities(left_v, right_v):
    # Turtlebot Burger joint indices for wheels are typically 0 and 1
    # Check your specific URDF using p.getJointInfo(robot_id, i)
    p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=left_v)
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=right_v)

# --- SIMULATION LOOP ---
print("Environment Ready. Press Ctrl+C to stop.")
try:
    while True:
        # This is where you would:
        # 1. Run MPC to get wheel velocities
        # 2. Apply velocities
        apply_wheel_velocities(0.5, 0.5) # Example: Simple forward move
        
        p.stepSimulation()
        time.sleep(1./240.) # 240Hz simulation frequency
except KeyboardInterrupt:
    p.disconnect()