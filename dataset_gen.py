import math
import random
import pickle
import yaml
import pybullet as p
import numpy as np
import csv
import heapq
from env_setup import SimulationEnv

def a_star_search(start_idx, goal_idx, nodes, adj):
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    came_from = {}
    g_score = {i: float('inf') for i in range(len(nodes))}
    g_score[start_idx] = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_idx:
            path = []
            while current in came_from:
                path.append(nodes[current])
                current = came_from[current]
            path.append(nodes[start_idx])
            return path[::-1], g_score[goal_idx]

        for neighbor in adj.get(current, []):
            dist = math.dist(nodes[current], nodes[neighbor])
            tentative_g = g_score[current] + dist
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + math.dist(nodes[neighbor], nodes[goal_idx])
                heapq.heappush(open_set, (f_score, neighbor))
    return None, float('inf')

def smooth_path(path, plane_id, robot_id):
    if len(path) <= 2: return path
    
    smoothed = [path[0]]
    curr = 0
    
    while curr < len(path) - 1:
        found_shortcut = False
        
        for next_idx in range(len(path) - 1, curr, -1):
            res = p.rayTest([smoothed[-1][0], smoothed[-1][1], 0.05], 
                            [path[next_idx][0], path[next_idx][1], 0.05])
            
            if not res or res[0][0] in [-1, plane_id, robot_id]:
                smoothed.append(path[next_idx])
                curr = next_idx
                found_shortcut = True
                break
        
        if not found_shortcut:
            curr += 1
            smoothed.append(path[curr])
            
    return smoothed
class DatasetGenerator:
    def __init__(self, config_path, prm_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env = SimulationEnv(render=False)
        
        with open(prm_path, 'rb') as f:
            data = pickle.load(f)
            self.nodes = data['nodes']
            self.adj = data['adj']

    def is_obstructed(self, start, goal):
        res = p.rayTest([start[0], start[1], 0.05], [goal[0], goal[1], 0.05])
        if not res: return False
        return res[0][0] not in [-1, self.env.plane_id, self.env.robot_id]

    def get_nearest_valid_node(self, point):
        dists = sorted(enumerate(self.nodes), key=lambda x: math.dist(point, x[1]))
        for idx, node_pos in dists[:15]: # Check 15 nearest
            res = p.rayTest([point[0], point[1], 0.05], [node_pos[0], node_pos[1], 0.05])
            if not res or res[0][0] in [-1, self.env.plane_id, self.env.robot_id]:
                return idx
        return None

    def run(self, num_samples=2000):
        output_file=f"cost_to_go_{self.config['world']}.csv"
        dataset = []
        obstructed_count = 0
        target_obstructed_ratio = 0.7

        print(f"Generating {num_samples} samples...")
        
        while len(dataset) < num_samples:
            s_ptr = [random.uniform(-2, 2), random.uniform(-2, 2)]
            g_ptr = [random.uniform(-2, 2), random.uniform(-2, 2)]
            
            res_s = p.rayTest([s_ptr[0], s_ptr[1], 2.0], [s_ptr[0], s_ptr[1], 0.01])
            res_g = p.rayTest([g_ptr[0], g_ptr[1], 2.0], [g_ptr[0], g_ptr[1], 0.01])
            if (res_s and res_s[0][0] not in [-1, self.env.plane_id, self.env.robot_id]) or \
               (res_g and res_g[0][0] not in [-1, self.env.plane_id, self.env.robot_id]):
                continue

            obstructed = self.is_obstructed(s_ptr, g_ptr)
            
            if not obstructed and (len(dataset) - obstructed_count) > (num_samples * (1 - target_obstructed_ratio)):
                continue

            s_idx = self.get_nearest_valid_node(s_ptr)
            g_idx = self.get_nearest_valid_node(g_ptr)

            if s_idx is not None and g_idx is not None:
                path_pts, _ = a_star_search(s_idx, g_idx, self.nodes, self.adj)
                
                if path_pts:
                    full_path = [s_ptr] + path_pts + [g_ptr]
                    final_path = smooth_path(full_path, self.env.plane_id, self.env.robot_id)
                    
                    cost = sum(math.dist(final_path[i], final_path[i+1]) for i in range(len(final_path)-1))
                    
                    dataset.append([s_ptr[0], s_ptr[1], g_ptr[0], g_ptr[1], cost])
                    if obstructed: obstructed_count += 1
                    
                    if len(dataset) % 100 == 0:
                        print(f"Collected {len(dataset)} samples ({obstructed_count} obstructed)")

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["start_x","start_y","goal_x","goal_y","path_length"])
            writer.writerows(dataset)
        
        print(f"Done! Saved to {output_file}")
        self.env.disconnect()

if __name__ == "__main__":
    gen = DatasetGenerator(config_path="config.yaml", prm_path="prm/world2/prm_1528.pkl")
    gen.run(num_samples=100000)