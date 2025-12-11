# 案例 2：KF进阶 —— 2D车辆轨迹跟踪
# 目标：演示车辆跟踪。系统是线性的（匀速模型 CV），观测也是线性的（GPS）。
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def case2_tracking_kf():
    dt = 0.1
    t = np.arange(0, 10, dt)
    n = len(t)

    # 状态转移矩阵 F (4x4)
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # 观测矩阵 H (2x4) - 只能观测位置 x, y
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    Q = np.eye(4) * 0.01  # 过程噪声
    R = np.eye(2) * 2.0  # 测量噪声 (GPS误差较大)

    # 初始状态
    x_true = np.array([0, 0, 2, 1])  # x=0, y=0, vx=2, vy=1
    x_est = np.array([0, 0, 0, 0])  # 初始猜测
    P = np.eye(4) * 10

    log_true = []
    log_meas = []
    log_est = []

    for i in range(n):
        # 生成真实数据
        w = np.random.multivariate_normal([0] * 4, Q)
        x_true = F @ x_true + w

        # 生成观测数据
        v = np.random.multivariate_normal([0] * 2, R)
        z = H @ x_true + v

        # --- KF 预测 ---
        x_pred = F @ x_est
        P_pred = F @ P @ F.T + Q

        # --- KF 更新 ---
        y = z - H @ x_pred  # 新息
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_est = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred

        log_true.append(x_true)
        log_meas.append(z)
        log_est.append(x_est)

    log_true = np.array(log_true)
    log_meas = np.array(log_meas)
    log_est = np.array(log_est)

    plt.figure(figsize=(8, 6))
    plt.plot(log_true[:, 0], log_true[:, 1], 'g-', label='真实轨迹')
    plt.scatter(log_meas[:, 0], log_meas[:, 1], c='k', s=10, marker='x', label='GPS测量')
    plt.plot(log_est[:, 0], log_est[:, 1], 'b-', linewidth=2, label='KF估计轨迹')
    plt.title('案例2: 2D车辆轨迹跟踪 (线性系统)')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()


case2_tracking_kf()