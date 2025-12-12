# 场景：实现值迭代算法（Value Iteration），在一个存在障碍物的二维网格中寻找从任意点到终点的最短路径。

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 环境设置
GRID_SIZE = 5
GOAL = (4, 4)
OBSTACLES = [(1, 1), (1, 3), (3, 1), (3, 3)]  # 对应文中的 '#'

# 动作空间: 上, 下, 左, 右
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 初始化值函数 (V表)，终点为0，其余无穷大
V = np.full((GRID_SIZE, GRID_SIZE), np.inf)
V[GOAL] = 0


# 2. 值迭代算法 (Bellman Update)
def is_valid(r, c):
    if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
        return False
    if (r, c) in OBSTACLES:
        return False
    return True


print("开始值迭代...")
for iteration in range(20):  # 迭代足够次数以收敛
    V_new = V.copy()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) == GOAL or (r, c) in OBSTACLES:
                continue

            # 寻找所有动作中成本最小的
            costs = []
            for dr, dc in ACTIONS:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc):
                    costs.append(1 + V[nr, nc])  # 移动成本=1
                else:
                    costs.append(1 + V[r, c])  # 撞墙/障碍保持原地，成本+1

            V_new[r, c] = min(costs)

    # 检查收敛
    if np.sum(np.abs(V - V_new)) < 1e-4:
        print(f"在第 {iteration} 次迭代收敛。")
        break
    V = V_new

# 3. 可视化值函数热力图
plt.figure(figsize=(6, 5))
plt.imshow(V, cmap='viridis_r', interpolation='nearest')
plt.colorbar(label='Min Cost to Goal')

# 标注数值
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        text = 'G' if (r, c) == GOAL else ('#' if (r, c) in OBSTACLES else f'{V[r, c]:.0f}')
        plt.text(c, r, text, ha='center', va='center', color='black' if V[r, c] > 5 else 'white')

plt.title("Dynamic Programming Value Function Map")
plt.show()