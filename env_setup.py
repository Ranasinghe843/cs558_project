import pybullet as p
import pybullet_data
import yaml
import time
import math

CONFIG_FILE = "config.yaml"

class SimulationEnv:
    def __init__(self, render=True, start_pos_2d=[0, 0], start_angle=180,inflation_radius=0.11):
        self.mode = p.GUI if render else p.DIRECT
        self.client = p.connect(self.mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        self.start_pos = [start_pos_2d[0], start_pos_2d[1], 0.01]
        self.robot_id = p.loadURDF("turtlebot3_burger.urdf",
                                   self.start_pos,
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, math.radians(start_angle)]))

        with open(CONFIG_FILE, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.bounds = self.config[self.config['world']]['bounds']

        self.inflation_radius = inflation_radius

        self.obstacles = self._setup_obstacles()

        self._draw_boarder()

    def _setup_obstacles(self):
        obs_ids = []
        for values in self.config[self.config['world']]['obstacles']:
            pos = [values[0], values[1], 0.25]
            obs_ids.append(self.create_inflated_box(pos, values[2], values[3]))
        return obs_ids

    def create_inflated_box(self, pos, hx, hy):
        base_half_extents = [hx, hy, 0.25]
        
        inflated_half_extents = [
            base_half_extents[0] + self.inflation_radius,
            base_half_extents[1] + self.inflation_radius,
            base_half_extents[2]
        ]
        
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=inflated_half_extents)
        
        ghost_vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=inflated_half_extents, rgbaColor=[0.2, 0.6, 1, 0.2]
        )
        
        core_vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=base_half_extents, rgbaColor=[0.4, 0.4, 0.4, 1]
        )
        
        box_id = p.createMultiBody(0, col_id, ghost_vis_id, pos)
        p.createMultiBody(0, -1, core_vis_id, pos)
        return box_id
    
    def _draw_boarder(self):
        p.addUserDebugLine([self.bounds[0], self.bounds[2], 0.05], [self.bounds[0], self.bounds[3], 0.05], lineColorRGB=[0, 0, 0], lineWidth=2.0)
        p.addUserDebugLine([self.bounds[1], self.bounds[2], 0.05], [self.bounds[1], self.bounds[3], 0.05], lineColorRGB=[0, 0, 0], lineWidth=2.0)
        p.addUserDebugLine([self.bounds[0], self.bounds[2], 0.05], [self.bounds[1], self.bounds[2], 0.05], lineColorRGB=[0, 0, 0], lineWidth=2.0)
        p.addUserDebugLine([self.bounds[0], self.bounds[3], 0.05], [self.bounds[1], self.bounds[3], 0.05], lineColorRGB=[0, 0, 0], lineWidth=2.0)

    def disconnect(self):
        p.disconnect(self.client)

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    env = SimulationEnv(render=True, start_pos_2d=[0.2,0.2])

    def apply_wheel_velocities(left_v, right_v):
        p.setJointMotorControl2(env.robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=left_v)
        p.setJointMotorControl2(env.robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=right_v)

    try:
        while True:
            apply_wheel_velocities(0.5, 0.5)
            
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        env.disconnect()