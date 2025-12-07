import numpy as np
from scipy.linalg import solve_continuous_lyapunov, det


def lyapunov_quadratic(A, Q):
    # 求解Lyapunov方程：A^T P + P A = -Q
    P = solve_continuous_lyapunov(A.T, -Q)
    print("Lyapunov方程的解 P：")
    print(P)

    # 验证P的正定性（主子式全正）
    def is_positive_definite(mat):
        n = mat.shape[0]
        for i in range(1, n + 1):
            minor = mat[:i, :i]
            if det(minor) <= 0:
                return False, i
        return True, n

    is_pd, _ = is_positive_definite(P)
    if is_pd:
        print("P是正定矩阵，系统渐近稳定")
    else:
        print("P非正定，系统不稳定或临界稳定")

    # 验证 A^T P + P A = -Q
    check = A.T @ P + P @ A
    print("验证 A^T P + P A = -Q：")
    print("计算结果：", check)
    print("期望结果（-Q）：", -Q)
    return P


# 定义稳定的线性系统矩阵 A
A = np.array([[-1, -2], [1, -1]])
# 取 Q 为单位矩阵
Q = np.eye(2)

# 求解并验证
P = lyapunov_quadratic(A, Q)

# 验证 V(x) 和 \dot{V}(x) 的符号（以x=[1,1]为例）
x = np.array([1, 1]).reshape(-1, 1)
V = x.T @ P @ x
dot_V = x.T @ (A.T @ P + P @ A) @ x
print(f"\n对于x=[1,1]：")
print(f"V(x) = {V[0, 0]:.4f}（正定）")
print(f"\\dot{V}(x) = {dot_V[0, 0]:.4f}（负定）")