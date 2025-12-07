import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义非线性系统
def nonlinear_system(t, x):
    x1, x2 = x
    dx1 = x2 - x1 * (x1**2 + x2**2)
    dx2 = -x1 - x2 * (x1**2 + x2**2)
    return [dx1, dx2]

# 2. 构造Lyapunov函数 V(x) = x1² + x2²，并计算其导数 \dot{V}(x)
def V(x):
    return x[0]**2 + x[1]**2

def dot_V(x):
    x1, x2 = x
    dx1 = x2 - x1 * (x1**2 + x2**2)
    dx2 = -x1 - x2 * (x1**2 + x2**2)
    return 2*x1*dx1 + 2*x2*dx2  # 链式法则求导

# 3. 验证 V(x) 和 \dot{V}(x) 的符号性质（修正：用 \\d 转义，或用原始字符串 r""）
test_points = [[1, 0], [0, 1], [2, 3], [-1, 2]]  # 多个测试点
print("验证 V(x) 和 \\dot{V}(x) 的符号：")  # 方案1：反斜杠转义
# 或用原始字符串：print(r"验证 V(x) 和 \dot{V}(x) 的符号：")

for x in test_points:
    v = V(x)
    dv = dot_V(x)
    print(f"x={x}: V(x)={v:.4f}（{'正定' if v>0 else '非正定'}）, "
          f"\\dot{V}(x)={dv:.4f}（{'负定' if dv<0 else '非负定'}）")  # 同样转义 \dot

# 4. 验证径向无界性（||x||→∞时 V(x)→∞）
print("\n径向无界性验证：")
for r in [10, 100, 1000]:
    x = [r, 0]
    v = V(x)
    print(f"||x||={r}: V(x)={v:.2f}（随||x||增大而增大）")

# 5. 仿真系统相轨迹（不同初始条件）
t_span = [0, 10]
t_eval = np.linspace(0, 10, 200)
initial_conditions = [[1, 0], [0, 2], [-1, -1], [2, -1]]  # 多个初始点

plt.figure(figsize=(8, 8))
for x0 in initial_conditions:
    sol = solve_ivp(nonlinear_system, t_span, x0, t_eval=t_eval)
    plt.plot(sol.y[0], sol.y[1], label=f'初始条件 {x0}')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('非线性系统相轨迹（收敛到原点，大范围渐近稳定）')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()