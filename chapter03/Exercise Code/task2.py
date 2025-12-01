import numpy as np

def controllability_matrix(A, B):
    """计算能控矩阵 M = [B, AB, A²B, ..., A^(n-1)B]"""
    n = A.shape[0]
    M = B
    for i in range(1, n):
        M = np.hstack((M, np.dot(A**i, B)))
    return M

# 示例1：单小车系统（PPT能控性举例）
A1 = np.array([[0, 1], [0, -0.5]])
B1 = np.array([[0], [1]])
M1 = controllability_matrix(A1, B1)
rank1 = np.linalg.matrix_rank(M1)
print("单小车系统能控矩阵：\n", M1)
print(f"能控矩阵秩：{rank1}，系统维度：{A1.shape[0]}")
print("系统能控性：", "能控" if rank1 == A1.shape[0] else "不可控\n")

# 示例2：两小车系统（PPT能控性分析）
A2 = np.array([[0, 1, 0, 0],
               [-2, -1, 2, 0],
               [0, 0, 0, 1],
               [1, 0, -1, 0]])
B2 = np.array([[0], [1], [0], [0]])
M2 = controllability_matrix(A2, B2)
rank2 = np.linalg.matrix_rank(M2)
print("两小车系统能控矩阵：\n", M2)
print(f"能控矩阵秩：{rank2}，系统维度：{A2.shape[0]}")
print("系统能控性：", "能控" if rank2 == A2.shape[0] else "不可控")