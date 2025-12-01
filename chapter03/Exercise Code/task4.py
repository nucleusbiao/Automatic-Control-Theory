import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 原系统矩阵（PPT观测器设计示例）
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = 0
n = A.shape[0]

# 期望观测器极点（比系统极点更快收敛）
observer_poles = np.array([-5, -6])

# 计算观测器增益L（基于能观性，极点配置）
# 观测器闭环矩阵：A_obs = A - L*C
result = place_poles(A.T, C.T, observer_poles)
L = result.gain_matrix.T  # 转置得到L
print("观测器增益 L：\n", L)

# 仿真状态估计过程
t = np.linspace(0, 5, 1000)
u = np.ones_like(t)  # 单位阶跃输入
x_true = np.zeros((n, len(t)))  # 真实状态
x_est = np.zeros((n, len(t)))   # 估计状态
x_est[:, 0] = [1, 0]  # 观测器初始估计误差

# 数值积分求解真实状态和估计状态
dt = t[1] - t[0]
for i in range(1, len(t)):
    # 真实状态更新：dx/dt = A*x + B*u
    x_true[:, i] = x_true[:, i-1] + dt * (np.dot(A, x_true[:, i-1]) + np.dot(B, [u[i-1]]).flatten())
    # 观测器更新：dx_est/dt = A*x_est + B*u + L*(y - C*x_est)
    y_true = np.dot(C, x_true[:, i-1])[0]
    y_est = np.dot(C, x_est[:, i-1])[0]
    x_est[:, i] = x_est[:, i-1] + dt * (np.dot(A, x_est[:, i-1]) + np.dot(B, [u[i-1]]).flatten() + L.flatten()*(y_true - y_est))

# 绘图对比真实状态与估计状态
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x_true[0, :], label='真实x1(t)')
plt.plot(t, x_est[0, :], '--', label='估计x1(t)')
plt.legend()
plt.title('龙伯格观测器状态估计结果')

plt.subplot(2, 1, 2)
plt.plot(t, x_true[1, :], label='真实x2(t)')
plt.plot(t, x_est[1, :], '--', label='估计x2(t)')
plt.xlabel('时间 t (s)')
plt.legend()
plt.tight_layout()
plt.show()