# 案例 2：UKF —— 强非线性的一维系统
# 目标：演示 UKF 如何利用 Sigma点 处理非线性，避免计算雅可比矩阵。使用一个强非线性函数 $y = x^2$ 来说明。
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from scipy.linalg import cholesky


def case4_simple_ukf():
    # 系统: x(k+1) = x(k) + sin(k), z(k) = x(k)^2 + noise
    # 这里的观测函数 h(x) = x^2 具有极强的非线性，EKF在x=0附近会失效(导数为0)

    def f(x, k):
        return x + np.sin(k)  # 状态转移

    def h(x):
        return x ** 2  # 观测函数

    # UKF 参数
    alpha, beta, kappa = 1e-3, 2, 0
    n = 1  # 状态维度
    lam = alpha ** 2 * (n + kappa) - n

    # 权重生成
    wm = np.full(2 * n + 1, 0.5 / (n + lam))
    wc = np.full(2 * n + 1, 0.5 / (n + lam))
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)

    x_est = np.array([1.5])  # 初始估计
    P = np.eye(1) * 1.0
    Q = np.eye(1) * 0.1
    R = np.eye(1) * 0.5

    true_vals = []
    est_vals = []

    x_true = np.array([1.5])

    for k in range(50):
        # 模拟真实世界
        x_true = f(x_true, k) + np.random.normal(0, np.sqrt(Q[0, 0]))
        z = h(x_true) + np.random.normal(0, np.sqrt(R[0, 0]))

        # --- UKF 预测 ---
        # 1. 生成 Sigma 点
        sigmas = np.zeros((2 * n + 1, n))
        U = cholesky((n + lam) * P)  # 矩阵平方根
        sigmas[0] = x_est
        for i in range(n):
            sigmas[i + 1] = x_est + U[i]
            sigmas[n + i + 1] = x_est - U[i]

        # 2. 传播 Sigma 点 (通过 f)
        sigmas_f = np.array([f(s, k) for s in sigmas])

        # 3. 恢复均值和方差
        x_pred = np.dot(wm, sigmas_f)
        P_pred = Q.copy()
        for i in range(2 * n + 1):
            y = sigmas_f[i] - x_pred
            P_pred += wc[i] * np.outer(y, y)

        # --- UKF 更新 ---
        # 1. 再次生成预测分布的 Sigma 点 (或复用)
        sigmas_pred = np.zeros((2 * n + 1, n))
        U = cholesky((n + lam) * P_pred)
        sigmas_pred[0] = x_pred
        for i in range(n):
            sigmas_pred[i + 1] = x_pred + U[i]
            sigmas_pred[n + i + 1] = x_pred - U[i]

        # 2. 观测传播
        Z_sigmas = np.array([h(s) for s in sigmas_pred])
        z_pred = np.dot(wm, Z_sigmas)

        # 3. 观测协方差
        S = R.copy()
        Pxz = np.zeros((n, 1))
        for i in range(2 * n + 1):
            y_z = Z_sigmas[i] - z_pred
            y_x = sigmas_pred[i] - x_pred
            S += wc[i] * np.outer(y_z, y_z)
            Pxz += wc[i] * np.outer(y_x, y_z)

        # 4. 卡尔曼增益
        K = np.dot(Pxz, np.linalg.inv(S))
        x_est = x_pred + np.dot(K, (z - z_pred))
        P = P_pred - np.dot(K, np.dot(S, K.T))

        true_vals.append(x_true[0])
        est_vals.append(x_est[0])

    plt.figure()
    plt.plot(true_vals, label='True')
    plt.plot(est_vals, '--', label='UKF Estimate')
    plt.title('案例4: UKF 处理非线性观测 y=x^2')
    plt.legend()
    plt.show()


case4_simple_ukf()