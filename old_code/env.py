import pybullet as p
import pybullet_data
import time
import numpy as np

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

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

def apply_wheel_velocities(left_v, right_v):
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=left_v)
    p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=right_v)

print("Environment Ready. Press Ctrl+C to stop.")

try:
    while True:
        apply_wheel_velocities(0.5, 0.5)
        
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()