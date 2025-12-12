# 场景：使用 SymPy 符号计算库，自动推导最速降线问题（Brachistochrone）的欧拉-拉格朗日方程。
#  这展示了如何用计算机辅助完成繁琐的数学推导。

import sympy as sp

# 1. 定义符号
x = sp.Symbol('x')
y = sp.Function('y')(x)
y_prime = y.diff(x)
g = sp.Symbol('g', positive=True)

# 2. 定义被积函数 F (最速降线的时间泛函)
# ds / v = sqrt(1 + y'^2) / sqrt(2gy)
# 注意：SymPy处理时通常不需要积分符号，只需处理被积函数 F
F = sp.sqrt(1 + y_prime**2) / sp.sqrt(2 * g * y)

print("被积函数 F:")
sp.pprint(F)
print("-" * 30)

# 3. 计算欧拉-拉格朗日方程的各项
# 方程: d/dx (∂F/∂y') - ∂F/∂y = 0

# ∂F/∂y
dF_dy = sp.diff(F, y)

# ∂F/∂y'
dF_dy_prime = sp.diff(F, y_prime)

# d/dx (∂F/∂y')
ddx_dF_dy_prime = sp.diff(dF_dy_prime, x)

# 4. 组装方程
EL_eq = ddx_dF_dy_prime - dF_dy

print("欧拉-拉格朗日方程 (未简化):")
# 简化表达式
simplified_eq = sp.simplify(EL_eq)
sp.pprint(simplified_eq)

print("\n解释：")
print("如果上述结果为0，即通过求解该微分方程可得最优曲线（摆线）。")