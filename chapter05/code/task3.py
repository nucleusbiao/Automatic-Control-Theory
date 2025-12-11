# 案例 3：EKF —— 雷达目标跟踪 (非线性观测)目标：演示处理非线性测量。雷达测量的是距离 $r$ 和角度 $\theta$，但我们需要在笛卡尔坐标系 $(x, y, v_x, v_y)$ 下跟踪。
# 需要计算雅可比矩阵 $H_j$。对

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def case3_radar_ekf():
    dt = 0.1
    # 状态: [x, y, vx, vy]
    # 真实运动是线性的，但观测是非线性的
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    Q = np.eye(4) * 0.01
    R = np.diag([0.5, 0.05])  # 距离误差0.5m, 角度误差0.05rad

    x_est = np.array([10, 10, 1, 0.5])  # 初始位置离雷达原点(0,0)有一定距离
    P = np.eye(4) * 1

    x_true = np.array([10, 10, 1, 0.5])

    est_path = []
    true_path = []

    for i in range(100):
        # 模拟运动
        x_true = F @ x_true + np.random.multivariate_normal([0] * 4, Q)

        # 模拟非线性雷达测量 h(x)
        r = np.sqrt(x_true[0] ** 2 + x_true[1] ** 2)
        theta = np.arctan2(x_true[1], x_true[0])
        z = np.array([r, theta]) + np.random.multivariate_normal([0] * 2, R)

        # --- EKF 预测 ---
        x_pred = F @ x_est
        P_pred = F @ P @ F.T + Q

        # --- EKF 更新 ---
        # 1. 计算预测观测值 h(x_pred)
        px, py = x_pred[0], x_pred[1]
        r_pred = np.sqrt(px ** 2 + py ** 2)
        theta_pred = np.arctan2(py, px)
        z_pred = np.array([r_pred, theta_pred])

        # 2. 计算雅可比矩阵 H (线性化)
        # H = d(meas)/d(state)
        r2 = px ** 2 + py ** 2
        r_val = np.sqrt(r2)
        H_jac = np.array([
            [px / r_val, py / r_val, 0, 0],
            [-py / r2, px / r2, 0, 0]
        ])

        y = z - z_pred
        # 角度归一化处理 (-pi 到 pi)
        while y[1] > np.pi: y[1] -= 2 * np.pi
        while y[1] < -np.pi: y[1] += 2 * np.pi

        S = H_jac @ P_pred @ H_jac.T + R
        K = P_pred @ H_jac.T @ np.linalg.inv(S)

        x_est = x_pred + K @ y
        P = (np.eye(4) - K @ H_jac) @ P_pred

        est_path.append(x_est)
        true_path.append(x_true)

    est_path = np.array(est_path)
    true_path = np.array(true_path)

    plt.figure()
    plt.plot(true_path[:, 0], true_path[:, 1], 'g--', label='真实路径')
    plt.plot(est_path[:, 0], est_path[:, 1], 'b-', label='EKF估计')
    plt.scatter(0, 0, c='r', marker='o', label='雷达位置')  # 雷达在原点
    plt.title('案例3: EKF 雷达跟踪 (非线性观测线性化)')
    plt.legend()
    plt.grid()
    plt.show()


case3_radar_ekf()