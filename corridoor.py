import pybullet as p
import pybullet_data
import time

# 1. Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. Load Ground Plane
p.loadURDF("plane.urdf")

def create_wall(position, orientation, length, width, height):
    """Helper to create a static box wall."""
    # halfExtents defines the distance from the center to the face (half the total size)
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2], 
                                 rgbaColor=[0.8, 0.8, 0.8, 1]) # Grey color
    
    # baseMass=0 makes the object static
    wall_id = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=col_id,
                                baseVisualShapeIndex=vis_id,
                                basePosition=position,
                                baseOrientation=orientation)
    return wall_id

# 3. Build the Corridor
# Define corridor dimensions
length = 10
width = 0.2
height = 1.0
gap = 2.0  # The distance between the walls

# Left Wall
create_wall(position=[0, gap/2, height/2], 
            orientation=[0, 0, 0, 1], 
            length=length, width=width, height=height)

# Right Wall
create_wall(position=[0, -gap/2, height/2], 
            orientation=[0, 0, 0, 1], 
            length=length, width=width, height=height)

# 4. Keep simulation running
while True:
    p.stepSimulation()
    time.sleep(1./240.)