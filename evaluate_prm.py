import math
import pickle
import yaml
import os
import heapq
import pybullet as p
import numpy as np
import random
from env_setup import SimulationEnv

class PRMEvaluator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env = SimulationEnv(render=False)
        self.bounds = self.env.bounds
        self.inflation = self.env.inflation_radius
        
        print("Building Visibility Graph Oracle...")
        self.vg_vertices, self.vg_static_adj = self._build_static_visibility_graph()

    def _is_collision_free(self, p1, p2):
        res = p.rayTest([p1[0], p1[1], 0.05], [p2[0], p2[1], 0.05])
        if not res or res[0][0] in [-1, self.env.plane_id, self.env.robot_id]:
            return True
        return False

    def _build_static_visibility_graph(self):
        vertices = []
        
        for obs in self.env.config[ self.env.config['world']]['obstacles']:
            x, y, hx, hy = obs
            
            mx = hx + self.inflation
            my = hy + self.inflation
            
            vertices.extend([
                [x - mx, y - my],
                [x - mx, y + my],
                [x + mx, y - my],
                [x + mx, y + my]
            ])

        adj = {i: [] for i in range(len(vertices))}
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if self._is_collision_free(vertices[i], vertices[j]):
                    adj[i].append(j)
                    adj[j].append(i)
                    
        return vertices, adj

    def get_oracle_dist(self, start, goal):
        nodes = [start, goal] + self.vg_vertices
        adj = {i: [] for i in range(len(nodes))}
        for i, neighbors in self.vg_static_adj.items():
            for n in neighbors: adj[i+2].append(n+2)
        
        for i in [0, 1]:
            for j in range(len(nodes)):
                if i != j and self._is_collision_free(nodes[i], nodes[j]):
                    adj[i].append(j); adj[j].append(i)
                    
        return self.a_star_length(0, 1, nodes, adj)

    def a_star_length(self, start_idx, goal_idx, nodes, adj):
        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        g_score = {i: float('inf') for i in range(len(nodes))}
        g_score[start_idx] = 0
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path = []
                temp = current
                while temp in came_from:
                    path.append(nodes[temp])
                    temp = came_from[temp]
                path.append(nodes[start_idx])
                return g_score[current], path[::-1]

            for neighbor in adj.get(current, []):
                d = math.dist(nodes[current], nodes[neighbor])
                tentative_g = g_score[current] + d
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + math.dist(nodes[neighbor], nodes[goal_idx])
                    heapq.heappush(open_set, (f_score, neighbor))
        return float('inf'), []

    def smooth_path(self, path):
        if len(path) <= 2: return path
        smoothed = [path[0]]
        curr = 0
        while curr < len(path) - 1:
            found_shortcut = False
            for next_idx in range(len(path) - 1, curr, -1):
                if self._is_collision_free(smoothed[-1], path[next_idx]):
                    smoothed.append(path[next_idx])
                    curr = next_idx
                    found_shortcut = True
                    break
            if not found_shortcut:
                curr += 1
                smoothed.append(path[curr])
        return smoothed

    def sample_valid_point(self):
        while True:
            pt = [random.uniform(self.bounds[0], self.bounds[1]), 
                  random.uniform(self.bounds[2], self.bounds[3])]
            res = p.rayTest([pt[0], pt[1], 2.0], [pt[0], pt[1], 0.01])
            if not res or res[0][0] in [-1, self.env.plane_id, self.env.robot_id]:
                return pt

    def evaluate(self, num_queries=1000):
        prm_folder = self.config['prm_folder'] + '/' + self.config['world']
        files = sorted([f for f in os.listdir(prm_folder) if f.endswith(".pkl")],
                       key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

        print(f"\n--- Benchmark: {num_queries} Samples (Error Thresh: 10%) ---")
        
        for filename in files:
            with open(os.path.join(prm_folder, filename), 'rb') as f:
                prm_data = pickle.load(f)
            
            prm_nodes = prm_data['nodes']
            prm_adj = prm_data['adj']
            errors = []
            high_error_count = 0
            success_count = 0

            for _ in range(num_queries):
                start = self.sample_valid_point()
                goal = self.sample_valid_point()
                
                l_true, _ = self.get_oracle_dist(start, goal)
                if l_true == float('inf') or l_true < 0.2: continue 

                dists = sorted(enumerate(prm_nodes), key=lambda x: math.dist(start, x[1]))
                s_idx = next((i for i, pos in dists[:50] if self._is_collision_free(start, pos)), None)
                
                dists_g = sorted(enumerate(prm_nodes), key=lambda x: math.dist(goal, x[1]))
                g_idx = next((i for i, pos in dists_g[:15] if self._is_collision_free(goal, pos)), None)

                if s_idx is not None and g_idx is not None:
                    _, path_nodes = self.a_star_length(s_idx, g_idx, prm_nodes, prm_adj)
                    if path_nodes:
                        raw_path = [start] + path_nodes + [goal]
                        refined_path = self.smooth_path(raw_path)
                        l_prm = sum(math.dist(refined_path[i], refined_path[i+1]) for i in range(len(refined_path)-1))
                        
                        error_pct = (l_prm - l_true) / l_true * 100
                        errors.append(error_pct)
                        if error_pct > 10.0:
                            high_error_count += 1
                        success_count += 1

            if errors:
                avg_err = sum(errors) / len(errors)
                high_err_pct = (high_error_count / success_count) * 100
                print(f"File: {filename:15} | Success: {success_count/num_queries*100:5.1f}% | Avg: {avg_err:5.2f}% | High Err (>10%): {high_err_pct:5.2f}% | Max: {max(errors):6.2f}%")

        self.env.disconnect()

if __name__ == "__main__":
    evaluator = PRMEvaluator("config.yaml")
    evaluator.evaluate(num_queries=1000)