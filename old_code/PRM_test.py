import math
import random
import pickle
import time
import yaml
import pybullet as p
import numpy as np
from env_setup import SimulationEnv

class PRMStar:
    def __init__(self, bounds, plane_id, robot_id, inflation_radius=0.11, z_offset=0.15):
        self.bounds = bounds  
        self.plane_id = plane_id
        self.robot_id = robot_id
        self.inflation_radius = inflation_radius
        self.z_offset = z_offset
        
        self.nodes = [] 
        self.adj = {}   
        self.debug_line_ids = []
        self.gamma = 6.0 

    def is_point_free(self, pos):
        """Checks if a point is free."""
        ray_start = [pos[0], pos[1], 2.0]
        ray_end = [pos[0], pos[1], 0.01]
        results = p.rayTest(ray_start, ray_end)
        if not results: return True
        hit_id = results[0][0]
        return hit_id == -1 or hit_id in [self.plane_id, self.robot_id]

    def check_collision(self, start_pos, end_pos):
        """Checks if the line between two nodes is clear."""
        test_z = 0.05 
        ray_start = [start_pos[0], start_pos[1], test_z]
        ray_end = [end_pos[0], end_pos[1], test_z]
        results = p.rayTest(ray_start, ray_end)
        if not results or results[0][0] == -1: return True
        return results[0][0] in [self.plane_id, self.robot_id]

    def seed_corners(self, world_config):
        """Injects corners and immediately connects them to form a visibility backbone."""
        obstacles = world_config['obstacles']
        eps = 0.01 
        start_idx = len(self.nodes)
        
        # 1. Add corner nodes
        for obs in obstacles:
            x, y, hx, hy = obs
            mx = hx + self.inflation_radius + eps
            my = hy + self.inflation_radius + eps
            
            potential_corners = [[x-mx, y-my], [x-mx, y+my], [x+mx, y-my], [x+mx, y+my]]
            
            for pt in potential_corners:
                if (self.bounds[0] < pt[0] < self.bounds[1] and self.bounds[2] < pt[1] < self.bounds[3]):
                    if self.is_point_free(pt):
                        self.nodes.append(pt)
                        self.adj[len(self.nodes) - 1] = []

        # 2. Backbone Connection: Connect corners to each other if they have line-of-sight
        # This ensures the 'optimal' paths are available from the start
        end_idx = len(self.nodes)
        for i in range(start_idx, end_idx):
            for j in range(i + 1, end_idx):
                if self.check_collision(self.nodes[i], self.nodes[j]):
                    self.adj[i].append(j)
                    self.adj[j].append(i)
        print(f"Backbone built with {end_idx - start_idx} corner nodes.")

    def sample_node(self):
        """Mixed Sampler: 50% Uniform, 50% Bridge Sampling to find narrow passages."""
        while True:
            if random.random() < 0.5:
                # Standard Uniform
                x = random.uniform(self.bounds[0], self.bounds[1])
                y = random.uniform(self.bounds[2], self.bounds[3])
                if self.is_point_free([x, y]):
                    return [x, y]
            else:
                # Bridge Sampler
                p1 = [random.uniform(self.bounds[0], self.bounds[1]), 
                      random.uniform(self.bounds[2], self.bounds[3])]
                if not self.is_point_free(p1):
                    # p1 is in collision, pick a p2 nearby
                    sigma = 0.2
                    p2 = [p1[0] + random.gauss(0, sigma), p1[1] + random.gauss(0, sigma)]
                    if not self.is_point_free(p2):
                        # p2 is also in collision, check if midpoint is free
                        mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
                        if self.is_point_free(mid):
                            return mid

    def grow(self, num_new_nodes):
        """Adds nodes and connects using PRM* radius + K-Nearest fallback for mazes."""
        start_idx = len(self.nodes)
        for _ in range(num_new_nodes):
            self.nodes.append(self.sample_node())
            self.adj[len(self.nodes) - 1] = []

        n = len(self.nodes)
        radius = self.gamma * math.sqrt(math.log(n) / n)
        
        # In mazes, the PRM* radius can get too small to bridge narrow gaps.
        # We ensure each new node checks at least its 10 nearest neighbors.
        K_NEAREST = 10 

        for i in range(start_idx, n):
            # Calculate distances to all other nodes
            dists = []
            for j in range(n):
                if i == j: continue
                dists.append((math.dist(self.nodes[i], self.nodes[j]), j))
            
            dists.sort() # Sort by distance
            
            for d, j in dists:
                # Connect if within PRM* radius OR if it's one of the K nearest neighbors
                # (Connecting to K nearest prevents the graph from fragmenting in narrow halls)
                if d <= radius or dists.index((d, j)) < K_NEAREST:
                    if self.check_collision(self.nodes[i], self.nodes[j]):
                        if j not in self.adj[i]: self.adj[i].append(j)
                        if i not in self.adj[j]: self.adj[j].append(i)

    def visualize(self):
        """Draws debug lines in the GUI."""
        for line_id in self.debug_line_ids:
            p.removeUserDebugItem(line_id)
        self.debug_line_ids = []

        for i, neighbors in self.adj.items():
            start = [self.nodes[i][0], self.nodes[i][1], self.z_offset]
            for n_idx in neighbors:
                if n_idx > i:
                    end = [self.nodes[n_idx][0], self.nodes[n_idx][1], self.z_offset]
                    line_id = p.addUserDebugLine(start, end, [0, 1, 0], lineWidth=0.05)
                    self.debug_line_ids.append(line_id)
        
        print(f"Roadmap: {len(self.nodes)} nodes, {len(self.debug_line_ids)} edges visualized.")

def main():
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    env = SimulationEnv(render=True)
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])

    prm = PRMStar(
        bounds=env.bounds, 
        plane_id=env.plane_id,
        robot_id=env.robot_id,
        inflation_radius=env.inflation_radius
    )

    prm.seed_corners(config[config['world']])
    prm.visualize()

    print("--- PRM* Maze-Optimized Incremental Builder ---")
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