import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, step
from scipy.integrate import solve_ivp

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 定义系统矩阵
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# 构建状态空间模型
sys = StateSpace(A, B, C, D)

# 计算单位阶跃响应
t, y = step(sys)

# 手动计算状态响应
def system_dynamics(t, x, A, B):
    return A @ x + B.flatten()  # B需要展平以适应solve_ivp

# 初始条件
x0 = [0, 0]
# 时间点
t_span = (0, t[-1])
t_eval = t

# 求解状态方程
solution = solve_ivp(system_dynamics, t_span, x0, t_eval=t_eval, args=(A, B))
x = solution.y.T  # 转置以匹配step函数的格式

# 绘图展示
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x[:, 0], label='x1(t)')
plt.plot(t, x[:, 1], label='x2(t)')
plt.xlabel('时间 t (s)')
plt.ylabel('状态变量')
plt.legend()
plt.title('线性定常系统阶跃响应-状态轨迹')

plt.subplot(2, 1, 2)
plt.plot(t, y, label='输出 y(t)')
plt.xlabel('时间 t (s)')
plt.ylabel('输出')
plt.legend()
plt.tight_layout()
plt.show()