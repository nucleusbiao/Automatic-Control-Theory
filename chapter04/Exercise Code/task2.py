import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def linear_system_response(A, x0, t_span=[0, 10], t_eval=None):
    # 定义微分方程：dx/dt = A@x
    def dxdt(t, x):
        return A @ x

    # 求解微分方程
    sol = solve_ivp(dxdt, t_span, x0, t_eval=t_eval)

    # 计算特征值
    eigenvalues, _ = np.linalg.eig(A)
    print("系统特征值：", eigenvalues)
    print("特征值实部：", [np.real(e) for e in eigenvalues])
    if all(np.real(e) < 0 for e in eigenvalues):
        print("系统渐近稳定")
    else:
        print("系统不稳定或临界稳定")

    # 绘制响应曲线
    plt.figure(figsize=(8, 4))
    plt.plot(sol.t, sol.y[0], label='$x_1(t)$')
    plt.plot(sol.t, sol.y[1], label='$x_2(t)$')
    plt.xlabel('时间 t (s)')
    plt.ylabel('状态变量')
    plt.title('线性系统零输入响应')
    plt.legend()
    plt.grid(True)
    plt.show()


# 验证PPT例4-1：A = [[0,6],[1,-1]]（特征值2和-3，不稳定）
A = np.array([[0, 6], [1, -1]])
linear_system_response(A, x0=[1, 0], t_eval=np.linspace(0, 10, 100))