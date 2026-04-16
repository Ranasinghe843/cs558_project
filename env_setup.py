import pybullet as p
import pybullet_data

class SimulationEnv:
    def __init__(self, render=True, start_pos_2d=[0, 0]):
        self.mode = p.GUI if render else p.DIRECT
        self.client = p.connect(self.mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        self.start_pos = [start_pos_2d[0], start_pos_2d[1], 0.01]
        self.robot_id = p.loadURDF("turtlebot3_burger.urdf", self.start_pos)
        
        self.bounds = [-3.0, 3.0, -3.0, 3.0]
        
        self.obstacles = self._setup_obstacles()

    def _setup_obstacles(self):
        box_positions = [
            [1, 1, 0.25],
            [-1, 1, 0.30],
            [0, 0, 0.35]
        ]
        box_half_extents = [0.25, 0.30, 0.35]
        obs_ids = []
        for i in range(len(box_positions)):
            obs_ids.append(self.create_inflated_box(box_positions[i], box_half_extents[i]))
        return obs_ids

    def create_inflated_box(self, pos, half_extent, inflation_radius=0.11):
        base_half_extents = [half_extent, half_extent, half_extent]
        inflated_half_extents = [
            base_half_extents[0] + inflation_radius,
            base_half_extents[1] + inflation_radius,
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

    def disconnect(self):
        p.disconnect(self.client)