# 场景：模拟双积分系统的最短时间控制。
# 根据理论推导，控制器利用开关曲线（Switching Curve）在 $u=+1$ 和 $u=-1$ 之间切换。

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义系统动态 (双积分)
def system(state, t):
    x1, x2 = state  # x1: 位置, x2: 速度

    # --- Bang-Bang 控制逻辑 ---
    # 开关函数 sigma = -x1 - 0.5 * x2 * |x2| (基于相平面分析)
    # 如果 sigma > 0, u = +1; 如果 sigma < 0, u = -1
    # 这里的开关曲线方程对应文中 3.2 节推导的结果
    sigma = -x1 - 0.5 * x2 * np.abs(x2)

    u = 1.0 if sigma > 0 else -1.0

    # 简单的防止在原点震荡的死区逻辑
    if abs(x1) < 0.01 and abs(x2) < 0.01:
        u = 0

    dx1dt = x2
    dx2dt = u
    return [dx1dt, dx2dt]


# 2. 仿真设置
t = np.linspace(0, 10, 1000)
initial_states = [
    [-2, 2],  # 初始状态 1
    [3, 1],  # 初始状态 2
    [-4, -2],  # 初始状态 3
    [2, -3]  # 初始状态 4
]

# 3. 绘制相平面图
plt.figure(figsize=(8, 8))

# 绘制开关曲线 (x1 = -0.5 * x2 * |x2|)
x2_range = np.linspace(-4, 4, 100)
x1_switch = -0.5 * x2_range * np.abs(x2_range)
plt.plot(x1_switch, x2_range, 'k--', linewidth=2, label='Switching Curve')

# 仿真并绘制轨迹
for x0 in initial_states:
    sol = odeint(system, x0, t)
    plt.plot(sol[:, 0], sol[:, 1], linewidth=1.5, label=f'Start {x0}')
    plt.plot(x0[0], x0[1], 'go')  # 起点

plt.plot(0, 0, 'rx', markersize=10, label='Target (0,0)')
plt.title("Phase Portrait of Time-Optimal Control (Bang-Bang)")
plt.xlabel("Position ($x_1$)")
plt.ylabel("Velocity ($x_2$)")
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()