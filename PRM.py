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
from env_setup import SimulationEnv
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

    def visualize(self):
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
        
        print(f"{len(self.nodes)} nodes, {len(self.debug_line_ids)} edges.")

def main():
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    env = SimulationEnv(render=True)
    
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])

    prm = PRMStar(
        bounds=env.bounds, 
        plane_id=env.plane_id,
        robot_id=env.robot_id
    )

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