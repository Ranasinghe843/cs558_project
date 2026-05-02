import math
import random
import pickle
import time
import yaml
import pybullet as p
import numpy as np
from env_setup import SimulationEnv

class PRMStar:
    def __init__(self, bounds, plane_id, robot_id, z_offset=0.15):
        self.bounds = bounds  
        self.plane_id = plane_id
        self.robot_id = robot_id
        self.z_offset = z_offset
        
        self.nodes = [] 
        self.adj = {}   
        self.debug_line_ids = []
        self.gamma = 6.0 

    def is_point_free(self, pos):
        """Checks if a point is free by shooting a ray from the sky."""
        ray_start = [pos[0], pos[1], 2.0]
        ray_end = [pos[0], pos[1], 0.01]
        results = p.rayTest(ray_start, ray_end)
        if not results: return True
        hit_id = results[0][0]
        return hit_id == -1 or hit_id in [self.plane_id, self.robot_id]

    def check_collision(self, start_pos, end_pos):
        """Checks if the line between two nodes is clear at test_z height."""
        test_z = 0.05 
        ray_start = [start_pos[0], start_pos[1], test_z]
        ray_end = [end_pos[0], end_pos[1], test_z]
        results = p.rayTest(ray_start, ray_end)
        if not results or results[0][0] == -1: return True
        return results[0][0] in [self.plane_id, self.robot_id]

    def sample_node(self):
        while True:
            x = random.uniform(self.bounds[0], self.bounds[1])
            y = random.uniform(self.bounds[2], self.bounds[3])
            if self.is_point_free([x, y]):
                return [x, y]

    def grow(self, num_new_nodes):
        start_idx = len(self.nodes)
        for _ in range(num_new_nodes):
            self.nodes.append(self.sample_node())
            self.adj[len(self.nodes) - 1] = []

        n = len(self.nodes)
        radius = self.gamma * math.sqrt(math.log(n) / n)
        
        for i in range(start_idx, n):
            for j in range(n):
                if i == j: continue
                if math.dist(self.nodes[i], self.nodes[j]) <= radius:
                    if self.check_collision(self.nodes[i], self.nodes[j]):
                        if j not in self.adj[i]: self.adj[i].append(j)
                        if i not in self.adj[j]: self.adj[j].append(i)

    def visualize(self):
        """Draws debug lines in the GUI. These are fast but might not show in getCameraImage."""
        # Clear old lines
        for line_id in self.debug_line_ids:
            p.removeUserDebugItem(line_id)
        self.debug_line_ids = []

        # Draw current roadmap
        for i, neighbors in self.adj.items():
            start = [self.nodes[i][0], self.nodes[i][1], self.z_offset]
            for n_idx in neighbors:
                if n_idx > i:
                    end = [self.nodes[n_idx][0], self.nodes[n_idx][1], self.z_offset]
                    # Bright green lines
                    line_id = p.addUserDebugLine(start, end, [0, 1, 0], lineWidth=0.05)
                    self.debug_line_ids.append(line_id)
        
        print(f"Roadmap: {len(self.nodes)} nodes, {len(self.debug_line_ids)} edges visualized.")

def main():
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    env = SimulationEnv(render=True)
    
    # Force GUI Camera to Top-Down
    # cameraDistance: how far back, pitch: -90 is straight down
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])

    prm = PRMStar(
        bounds=env.bounds, 
        plane_id=env.plane_id,
        robot_id=env.robot_id
    )

    print("--- PRM* Incremental Builder ---")
    node_increment = 100
    folder = config['prm_folder']

    try:
        while len(prm.nodes) < config['nodes']:
            current_n = len(prm.nodes)
            print(f"\nGenerating nodes {current_n} to {current_n + node_increment}...")
            
            prm.grow(node_increment)
            prm.visualize()
            
            total = len(prm.nodes)
            with open(f"{folder}/{config['world']}/prm_{total}.pkl", "wb") as f:
                pickle.dump({"nodes": prm.nodes, "adj": prm.adj}, f)
            
            print(f"Iteration complete. Graph has {total} nodes.")
            input(">>> Press Enter to add 100 more nodes (or Ctrl+C to exit)...")
                
    except KeyboardInterrupt:
        pass
    finally:
        env.disconnect()

if __name__ == "__main__":
    main()