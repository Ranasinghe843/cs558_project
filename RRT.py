import math
import random
import pybullet as p

def check_collision(node_a, node_b, plane_id, robot_id, z_offset=0.2):
    ray_start = [node_a[0], node_a[1], z_offset]
    ray_end = [node_b[0], node_b[1], z_offset]

    hit_object_id = p.rayTest(ray_start, ray_end)[0][0]

    return hit_object_id in [-1, plane_id, robot_id]

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, bounds, obstacles, plane_id, robot_id, 
                 step_size=0.5, search_radius=1.0, max_iter=5000):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.bounds = bounds
        self.obstacles = obstacles
        self.plane_id = plane_id
        self.robot_id = robot_id
        
        self.step_size = step_size
        self.search_radius = search_radius
        self.max_iter = max_iter
        
        self.node_list = [self.start]

    def plan(self):
        best_goal_node = None

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(self.node_list, rnd_node)
            new_node = self.steer(nearest_node, rnd_node, self.step_size)
            
            if check_collision([nearest_node.x, nearest_node.y], [new_node.x, new_node.y], 
                               self.plane_id, self.robot_id):
                
                near_nodes = self.find_near_nodes(new_node)
                new_node = self.choose_best_parent(new_node, near_nodes)
                
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_nodes)
                    
                    if self.calc_dist(new_node, self.goal) <= self.step_size:
                        final_node = self.steer(new_node, self.goal, self.step_size)
                        if check_collision([new_node.x, new_node.y], [final_node.x, final_node.y], 
                                           self.plane_id, self.robot_id):
                            
                            if best_goal_node is None or final_node.cost < best_goal_node.cost:
                                best_goal_node = final_node

        if best_goal_node:
            return self.generate_final_course(best_goal_node)
            
        return None

    def get_random_node(self):
        if random.randint(0, 100) > 5:
            x = random.uniform(self.bounds[0], self.bounds[1])
            y = random.uniform(self.bounds[2], self.bounds[3])
        else:
            x, y = self.goal.x, self.goal.y
        return Node(x, y)

    def get_nearest_node(self, node_list, rnd_node):
        distances = [self.calc_dist(node, rnd_node) for node in node_list]
        min_index = distances.index(min(distances))
        return node_list[min_index]

    def steer(self, from_node, to_node, extend_length):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_dist_and_angle(new_node, to_node)
        
        actual_step = min(extend_length, d)
        
        new_node.x += actual_step * math.cos(theta)
        new_node.y += actual_step * math.sin(theta)
        new_node.parent = from_node
        new_node.cost = from_node.cost + actual_step
        return new_node

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = min(self.search_radius * math.sqrt((math.log(nnode) / nnode)), self.step_size * 5)
        distances = [self.calc_dist(node, new_node) for node in self.node_list]
        near_indices = [distances.index(d) for d in distances if d <= r]
        return [self.node_list[i] for i in near_indices]

    def choose_best_parent(self, new_node, near_nodes):
        if not near_nodes: return new_node
        costs = []
        for near_node in near_nodes:
            if check_collision([near_node.x, near_node.y], [new_node.x, new_node.y], 
                               self.plane_id, self.robot_id):
                costs.append(near_node.cost + self.calc_dist(near_node, new_node))
            else:
                costs.append(float("inf"))
                
        min_cost = min(costs)
        if min_cost == float("inf"):
            return None
            
        min_index = costs.index(min_cost)
        new_node.parent = near_nodes[min_index]
        new_node.cost = min_cost
        return new_node

    def propagate_cost_to_children(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost + self.calc_dist(parent_node, node)
                self.propagate_cost_to_children(node)
    
    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            edge_node_cost = new_node.cost + self.calc_dist(new_node, near_node)
            if edge_node_cost < near_node.cost:
                if check_collision([new_node.x, new_node.y], [near_node.x, near_node.y], 
                                self.plane_id, self.robot_id):
                    near_node.parent = new_node
                    near_node.cost = edge_node_cost
                    self.propagate_cost_to_children(near_node)

    def generate_final_course(self, goal_node):
        path = [[self.goal.x, self.goal.y]]
        node = goal_node
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path, goal_node.cost
    
    def generate_final_course(self, goal_node):
        path = [[self.goal.x, self.goal.y]]
        node = goal_node
        while node.parent is not None:
            if math.dist([node.x, node.y], path[-1]) > 1e-6:
                path.append([node.x, node.y])
            node = node.parent
            
        if math.dist([self.start.x, self.start.y], path[-1]) > 1e-6:
            path.append([self.start.x, self.start.y])
            
        return path, goal_node.cost

    def calc_dist(self, from_node, to_node):
        return math.dist([from_node.x, from_node.y], [to_node.x, to_node.y])

    def calc_dist_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta