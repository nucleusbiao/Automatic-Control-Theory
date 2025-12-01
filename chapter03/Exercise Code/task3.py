import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles, StateSpace, step
from scipy.integrate import solve_ivp

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 原系统矩阵
A = np.array([[0, 1], [1, 0]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = 0

# 期望极点
desired_poles = np.array([-2, -3])

# 极点配置计算反馈增益K
result = place_poles(A, B, desired_poles)
K = result.gain_matrix
print("状态反馈增益 K：", K)

# 闭环系统矩阵：A_cl = A - B*K
A_cl = A - np.dot(B, K)
sys_cl = StateSpace(A_cl, B, C, D)

# 仿真闭环系统阶跃响应
t, y_cl = step(sys_cl)

# 手动计算状态响应
def closed_loop_system(t, x):
    return A_cl @ x + B.flatten()  # 阶跃输入 u=1

x0 = [0, 0]  # 初始状态
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)
sol = solve_ivp(closed_loop_system, t_span, x0, t_eval=t_eval)
x_cl = sol.y.T

# 绘图
plt.figure(figsize=(8, 4))
plt.plot(t, y_cl, label='闭环系统输出')
plt.xlabel('时间 t (s)')
plt.ylabel('输出 y(t)')
plt.title('线性状态反馈控制后的阶跃响应')
plt.legend()
plt.grid(True)
plt.show()