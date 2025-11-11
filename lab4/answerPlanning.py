import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 1
TARGET_THREHOLD = 0.25
MAX_ITER = 18000  # 最大迭代次数
GOAL_SAMPLE_RATE = 0.2  # 目标点采样率
MINV=0.1
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.path = None
        self.current_target_index = 0
        self.remaining_steps = 5  # 每个路径点停留的步数
        
        walls_array = np.array(walls)
        self.min_x = np.min(walls_array[:,0]) - 0.75
        self.max_x = np.max(walls_array[:,0]) + 0.75
        self.min_y = np.min(walls_array[:,1]) - 0.75
        self.max_y = np.max(walls_array[:,1]) + 0.75
        ### 你的代码 ###
        
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        self.path = self.build_tree(current_position, next_food)
        self.current_target_index = 1
        self.remaining_steps = 5
        ### 你的代码 ###
       
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        ### 你的代码 ###
        if self.current_target_index>=len(self.path)-1:
            return self.path[-1]
        if self.remaining_steps==5:
            self.remaining_steps=0
            self.current_target_index+=1
            target_pose=self.path[self.current_target_index]
        else:
            self.remaining_steps+=1
            target_pose=self.path[self.current_target_index]
        if np.linalg.norm(current_velocity)<=MINV:
            self.path=self.build_tree(current_position, self.path[-1])
            self.current_target_index = 1
            self.remaining_steps = 5
            target_pose=self.path[1]
        ### 你的代码 ###
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        graph: List[TreeNode] = []
        graph.append(TreeNode(-1, start[0], start[1]))
        ### 你的代码 ###
        for _ in range(MAX_ITER):
            if np.random.random() < GOAL_SAMPLE_RATE:
                sample_point = goal
            else:
                sample_point = np.array([
                    np.random.uniform(self.min_x, self.max_x),
                    np.random.uniform(self.min_y, self.max_y)
                ])
            
            nearest_idx, _ = self.find_nearest_point(sample_point, graph)
            nearest_node = graph[nearest_idx]
            
            valid, new_point = self.connect_a_to_b(nearest_node.pos, sample_point)
            
            if valid:
                if self.map.checkoccupy(new_point):
                    continue
                
                graph.append(TreeNode(nearest_idx, new_point[0], new_point[1]))
                

                if np.linalg.norm(new_point - goal) < TARGET_THREHOLD:
                    node_idx = len(graph) - 1
                    while node_idx != -1:
                        node = graph[node_idx]
                        path.append(node.pos.copy())
                        node_idx = node.parent_idx
                    path.reverse()
                    return path
        
        return [start,goal]
        
        ### 你的代码 ###
        

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = float("inf")
        ### 你的代码 ###
        for i, node in enumerate(graph):
            dist = np.linalg.norm(point - node.pos)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_idx = i
                
        return nearest_idx, nearest_distance
        ### 你的代码 ###
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        ### 你的代码 ###
        direction = point_b - point_a
        dist = np.linalg.norm(direction)
        
        towards = direction / dist
        new_point = point_a + towards * min(STEP_DISTANCE, dist)
        
        strike, _ = self.map.checkline(point_a, new_point)
        
        return not strike, new_point
        ### 你的代码 ###
