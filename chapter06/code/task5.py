import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义系统矩阵 (文中倒立摆参数)
# x = [p, v, theta, omega]
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -0.981, 0],
    [0, 0, 0, 1],
    [0, 0, 21.582, 0]
])

B = np.array([
    [0],
    [1],
    [0],
    [-2]
])

# 2. 定义权重矩阵
Q = np.diag([100, 10, 100, 10]) # 对应文中的Q
R = np.array([[0.01]])          # 对应文中的R

# 3. 求解黎卡提方程 (P) 并计算增益 (K)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("计算得到的反馈增益矩阵 K:")
print(np.round(K, 2))

# 4. 定义闭环系统
def closed_loop_dynamics(x, t):
    # u = -K * x
    u = -np.dot(K, x)
    # dx/dt = Ax + Bu
    dxdt = np.dot(A, x) + np.dot(B, u).flatten()
    return dxdt

# 5. 仿真
t = np.linspace(0, 5, 500)
x0 = [0, 0, 0.2, 0] # 初始状态：小车原位，杆偏离 0.2 rad (约11.5度)

sol = odeint(closed_loop_dynamics, x0, t)

# 6. 绘图
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, sol[:, 0], label='Cart Position (m)')
plt.plot(t, sol[:, 2], label='Pole Angle (rad)', linewidth=2)
plt.title("LQR Control Response (Inverted Pendulum)")
plt.ylabel("State")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
# 计算控制输入 u(t) = -Kx(t) 用于绘图
u_trace = [-np.dot(K, x) for x in sol]
plt.plot(t, np.array(u_trace).flatten(), 'r', label='Control Force (N)')
plt.xlabel("Time (s)")
plt.ylabel("Control Input")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()