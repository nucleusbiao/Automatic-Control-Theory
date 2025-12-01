import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义原系统
A = np.array([[0, 1], [1, 0]])  # 不稳定原系统
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = 0
n = A.shape[0]

# 2. 设计状态反馈控制器（极点配置）
desired_ctrl_poles = [-2, -3]
ctrl_result = place_poles(A, B, desired_ctrl_poles)
K = ctrl_result.gain_matrix
print("控制器增益 K：", K)

# 3. 设计龙伯格观测器
desired_obs_poles = [-5, -6]
obs_result = place_poles(A.T, C.T, desired_obs_poles)
L = obs_result.gain_matrix.T
print("观测器增益 L：\n", L)

# 4. 仿真闭环系统（控制器+观测器）
t = np.linspace(0, 5, 1000)
u = np.ones_like(t)  # 单位阶跃输入
x_true = np.zeros((n, len(t)))  # 真实状态
x_est = np.zeros((n, len(t)))  # 估计状态
x_est[:, 0] = [1, 0]  # 初始估计误差
y_history = []

dt = t[1] - t[0]
for i in range(1, len(t)):
    # 计算控制输入：u_ctrl = -K * x_est
    u_ctrl = -np.dot(K, x_est[:, i - 1])[0] + u[i - 1]  # 含参考输入u

    # 真实状态更新
    x_true[:, i] = x_true[:, i - 1] + dt * (np.dot(A, x_true[:, i - 1]) + np.dot(B, [u_ctrl]).flatten())
    y_true = np.dot(C, x_true[:, i - 1])[0]
    y_history.append(y_true)

    # 观测器状态更新
    y_est = np.dot(C, x_est[:, i - 1])[0]
    x_est[:, i] = x_est[:, i - 1] + dt * (
                np.dot(A, x_est[:, i - 1]) + np.dot(B, [u_ctrl]).flatten() + L.flatten() * (y_true - y_est))

# 绘图展示闭环系统性能
plt.figure(figsize=(8, 4))
plt.plot(t[:-1], y_history, label='系统输出 y(t)')
plt.xlabel('时间 t (s)')
plt.ylabel('输出')
plt.title('控制器与观测器结合的闭环系统响应')
plt.legend()
plt.grid(True)
plt.show()