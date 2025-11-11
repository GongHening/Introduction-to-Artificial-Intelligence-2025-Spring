from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
k = 0.5
POSITION_NOISE=0.06
THETA_NOISE=0.3
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    xmin = np.min(walls[:, 0]) +0.75
    xmax = np.max(walls[:, 0]) -0.75
    ymin = np.min(walls[:, 1]) +0.75
    ymax = np.max(walls[:, 1]) -0.75

    wall_boxes = [((x-0.75,x+0.75), (y-0.75,y+0.75)) for x, y in walls]

    def is_in_wall(pt):
        x, y = pt
        for (xmin_, xmax_), (ymin_, ymax_) in wall_boxes:
            if xmin_ <= x <= xmax_ and ymin_ <= y <= ymax_:
                return True
        return False
    np.random.seed(1)
    cnt=0
    while cnt < N:  
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        if not is_in_wall((x, y)):
            theta = np.random.uniform(0, 2 * np.pi)
            theta=theta%(2*np.pi)
            all_particles.append(Particle(x, y, theta, 1.0 / N))
            cnt += 1
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    ### 你的代码 ###
    weight=np.exp(-k * np.linalg.norm(estimated - gt))
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    N = len(particles)
    num=[]
    tot=0
    for i in particles:
        a=int(i.weight*N)
        num.append(a)
        tot=tot+a
    ot=N-tot
    for i,part in enumerate(particles):
        for j in range(num[i]):
            # 添加位置和方向噪声
            x = part.position[0] + np.random.normal(0, POSITION_NOISE)
            y = part.position[1] + np.random.normal(0, POSITION_NOISE)
            theta = (part.theta + np.random.normal(0, THETA_NOISE))% (2 * np.pi)
            
            wall_boxes = [((x-0.75,x+0.75), (y-0.75,y+0.75)) for x, y in walls]

            def is_in_wall(pt):
                x, y = pt
                for (xmin_, xmax_), (ymin_, ymax_) in wall_boxes:
                    if xmin_ <= x <= xmax_ and ymin_ <= y <= ymax_:
                        return True
                return False
            
            if is_in_wall((x, y)):
                new_p = Particle(
                    part.position[0], 
                    part.position[1],
                    part.theta,
                    1.0/N
                )
            else:
                new_p = Particle(x, y, theta, 1.0/N)
            
            resampled_particles.append(new_p)
    resampled_particles.extend(generate_uniform_particles(walls, ot))
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta = (p.theta+dtheta)%(2*np.pi)
    dx = traveled_distance * np.cos(p.theta)
    dy = traveled_distance * np.sin(p.theta)
    p.position += np.array([dx, dy])
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    ### 你的代码 ###
    maxn=0
    for p in particles:
        if p.weight>maxn:
            maxn=p.weight
            avg_x=p.position[0]
            avg_y=p.position[1]
            avg_theta=p.theta
    ### 你的代码 ###
    final_result = Particle(avg_x,avg_y,avg_theta,1.0)
    return final_result