import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 符号计算雅可比矩阵
x1, x2, mu = sp.symbols('x1 x2 mu')
f1 = x2  # 非线性系统第一个方程：dx1/dt = x2
f2 = mu * (1 - x1**2) * x2 - x1  # 第二个方程：dx2/dt = mu(1-x1²)x2 -x1
f = sp.Matrix([f1, f2])
J = f.jacobian(sp.Matrix([x1, x2]))  # 雅可比矩阵
print("雅可比矩阵 J：")
print(J)

# 2. 代入平衡状态 x_e=[0,0]，得到线性化矩阵 A
A_sym = J.subs({x1:0, x2:0})
print("平衡状态 x_e=[0,0] 处的线性化矩阵 A：")
print(A_sym)

# 3. 代入参数 mu，分析稳定性
mu_value = -0.5  # PPT中要求 mu<0 且 -1<α<1（此处α=√(1+μ)）
A = np.array(A_sym.subs(mu, mu_value), dtype=float)
eigenvalues, _ = np.linalg.eig(A)
print(f"mu={mu_value} 时，A的特征值：", eigenvalues)
if all(np.real(e) < 0 for e in eigenvalues):
    print("原非线性系统在 x_e 处渐近稳定")
else:
    print("原非线性系统在 x_e 处不稳定")

# 4. 仿真非线性系统与线性化系统响应
def nonlinear_dxdt(t, x):
    return [x[1], mu_value*(1-x[0]**2)*x[1] - x[0]]

def linear_dxdt(t, x):
    return A @ x

t_span = [0, 20]
t_eval = np.linspace(0, 20, 200)
x0 = [0.1, 0]  # 小初始扰动

# 求解并绘图
sol_nonlinear = solve_ivp(nonlinear_dxdt, t_span, x0, t_eval=t_eval)
sol_linear = solve_ivp(linear_dxdt, t_span, x0, t_eval=t_eval)

plt.figure(figsize=(8, 4))
plt.plot(sol_nonlinear.t, sol_nonlinear.y[0], label='非线性系统 $x_1(t)$')
plt.plot(sol_linear.t, sol_linear.y[0], '--', label='线性化系统 $x_1(t)$')
plt.xlabel('时间 t (s)')
plt.ylabel('$x_1(t)$')
plt.title(f'Vanderpol方程（mu={mu_value}）小扰动响应')
plt.legend()
plt.grid(True)
plt.show()