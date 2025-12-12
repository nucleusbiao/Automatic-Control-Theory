# 场景：利用 scipy.optimize 求解一个简单的双积分系统（类似于卫星姿态调整）的能量最小化问题。
# 我们将连续控制问题离散化为非线性规划问题。

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 1. 定义系统动力学 (双积分系统: pos'' = u)
def dynamics(state, u, dt):
    pos, vel = state
    # 简单的欧拉积分
    vel_new = vel + u * dt
    pos_new = pos + vel * dt + 0.5 * u * dt ** 2
    return np.array([pos_new, vel_new])


# 2. 定义目标函数 (能量最小化: sum(u^2))
def objective(u_seq):
    return np.sum(u_seq ** 2)


# 3. 定义约束条件 (必须到达目标状态)
def constraint_final_state(u_seq):
    dt = 0.1
    state = np.array([0.0, 0.0])  # 初始状态 [位置, 速度]
    target = np.array([10.0, 0.0])  # 目标状态 [位置, 速度]

    # 前向仿真
    for u in u_seq:
        state = dynamics(state, u, dt)

    # 返回与目标的距离 (优化器会尝试使其为0)
    return state - target


# 4. 设置优化问题
T = 2.0  # 总时间
dt = 0.1
N = int(T / dt)  # 时间步数
u_init = np.zeros(N)  # 初始猜测控制量

# 定义约束字典
cons = ({'type': 'eq', 'fun': constraint_final_state})

# 5. 求解
print("正在优化控制序列...")
res = minimize(objective, u_init, constraints=cons, method='SLSQP', options={'disp': True})

# 6. 可视化结果
u_opt = res.x
time = np.linspace(0, T, N)
states = [[0, 0]]
curr = np.array([0.0, 0.0])
for u in u_opt:
    curr = dynamics(curr, u, dt)
    states.append(curr)
states = np.array(states)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time, u_opt, 'r-o')
plt.title("Optimal Control Input u(t)")
plt.xlabel("Time (s)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, T, N + 1), states[:, 0], 'b-', label='Position')
plt.plot(np.linspace(0, T, N + 1), states[:, 1], 'g--', label='Velocity')
plt.legend()
plt.title("State Trajectory")
plt.grid(True)
plt.show()