######################################################################
# Script for Generating a PRM* graph through a Pybullet environment
######################################################################

import math
import random
import pickle
import time
import yaml
import pybullet as p
import numpy as np
import argparse
import os
from env_setup import SimulationEnv
from dataset_gen import a_star_search, smooth_path

class PRMStar:
    def __init__(self, bounds, plane_id, robot_id, z_offset=0.15, gamma=6.0):
        self.bounds = bounds  
        self.plane_id = plane_id
        self.robot_id = robot_id
        self.z_offset = z_offset
        
        self.nodes = [] 
        self.adj = {}   
        self.debug_line_ids = []
        self.gamma = gamma

    def is_point_free(self, pos): # checks if the point is in an obstacle
        ray_start = [pos[0], pos[1], 2.0]
        ray_end = [pos[0], pos[1], 0.01]
        results = p.rayTest(ray_start, ray_end)
        if not results or results[0][0] == -1: return True
        return results[0][0] in [self.plane_id, self.robot_id]

    def check_collision(self, start_pos, end_pos): # checks if the path between 2 points are obstacle free
        test_z = 0.05 
        ray_start = [start_pos[0], start_pos[1], test_z]
        ray_end = [end_pos[0], end_pos[1], test_z]
        results = p.rayTest(ray_start, ray_end)
        if not results or results[0][0] == -1: return True
        return results[0][0] in [self.plane_id, self.robot_id]

    def sample_node(self): # samples a node within the given bounds
        while True:
            x = random.uniform(self.bounds[0], self.bounds[1])
            y = random.uniform(self.bounds[2], self.bounds[3])
            if self.is_point_free([x, y]):
                return [x, y]

    def grow(self, num_new_nodes): # grows the graph bt the given number of nodes and connects using PRM* formula
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

    def get_nearest_valid_node(self, point, k=15):
        dists = sorted(enumerate(self.nodes), key=lambda x: math.dist(point, x[1]))
        
        for idx, node_pos in dists[:k]:
            if self.check_collision(point, node_pos):
                return idx
        return None

    def visualize_query_paths(self, start_goal_pairs):
        path_color = [1, 0, 0] # Red
        
        for start_pt, goal_pt in start_goal_pairs:
            s_idx = self.get_nearest_valid_node(start_pt)
            g_idx = self.get_nearest_valid_node(goal_pt)

            if s_idx is not None and g_idx is not None:
                path_nodes, _ = a_star_search(s_idx, g_idx, self.nodes, self.adj)
                
                if path_nodes:
                    full_path = [start_pt] + path_nodes + [goal_pt]
                    final_path = smooth_path(full_path, self.plane_id, self.robot_id)
                    
                    for i in range(len(final_path) - 1):
                        p.addUserDebugLine(
                            [final_path[i][0], final_path[i][1], self.z_offset + 0.01], 
                            [final_path[i+1][0], final_path[i+1][1], self.z_offset + 0.01], 
                            path_color, 
                            lineWidth=2.0
                        )
            else:
                print(f"Could not find valid graph entry/exit for query: {start_pt} to {goal_pt}")

    def visualize(self):
        for line_id in self.debug_line_ids:
            p.removeUserDebugItem(line_id)
        self.debug_line_ids = []

        for i, neighbors in self.adj.items():
            start = [self.nodes[i][0], self.nodes[i][1], self.z_offset]
            for n_idx in neighbors:
                if n_idx > i:
                    end = [self.nodes[n_idx][0], self.nodes[n_idx][1], self.z_offset]
                    line_id = p.addUserDebugLine(start, end, [0, 1, 0, ], lineWidth=0.001)
                    self.debug_line_ids.append(line_id)
        
        print(f"{len(self.nodes)} nodes, {len(self.debug_line_ids)} edges.")

def main():
    parser = argparse.ArgumentParser(description="Generate or Visualize PRM* graphs.")
    parser.add_argument('--load', type=str, help="Path to a .pkl PRM file to visualize.")
    args = parser.parse_args()

    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    env = SimulationEnv(render=True)
    
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])

    prm = PRMStar(
        bounds=env.bounds, 
        plane_id=env.plane_id,
        robot_id=env.robot_id
    )

    if args.load:
        if os.path.exists(args.load):
            print(f"Loading PRM from {args.load}...")
            with open(args.load, 'rb') as f:
                data = pickle.load(f)
                prm.nodes = data['nodes']
                prm.adj = data['adj']

            # prm.visualize()

            prm.visualize_query_paths([
                [[-1.7, -0.1], [-1.2, -1.3]],
                [[1.95, 1.95], [-1.2, -1.3]],
                [[-0.3, 1.0], [-1.2, -1.3]]
            ])
            input("Press Enter in terminal or Ctrl+C to close.")
        else:
            print(f"Error: File {args.load} not found.")
    
    else:

        node_increment = 100
        folder = config['prm_folder']

        try:
            while len(prm.nodes) < config['nodes']:
                current_n = len(prm.nodes)
                print(f"\nGenerating {current_n + node_increment} nodes")
                
                prm.grow(node_increment)
                prm.visualize()
                
                total = len(prm.nodes)
                with open(f"{folder}/{config['world']}/prm_{total}.pkl", "wb") as f:
                    pickle.dump({"nodes": prm.nodes, "adj": prm.adj}, f)
                
                print(f"{total} nodes.")
                input(">>> 100 more nodes.")
                    
        except KeyboardInterrupt:
            pass
        finally:
            env.disconnect()

if __name__ == "__main__":
    main()