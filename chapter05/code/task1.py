# 案例 1：KF基础 —— 一维恒定电压测量
# 目标：演示卡尔曼滤波最基础的“预测-更新”循环，观察滤波器如何从嘈杂的数据中收敛到真实值。
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def case1_simple_kf():
    # 1. 初始化
    n_iter = 50
    sz = (n_iter,)
    x_true = -0.37727  # 真实电压值 (我们希望估计的值)

    # 模拟测量数据 (真实值 + 噪声)
    z = np.random.normal(x_true, 0.1, size=sz)

    Q = 1e-5  # 过程噪声方差 (假设真实电压几乎不变，所以很小)
    R = 0.1 ** 2  # 测量噪声方差 (来自传感器的精度)

    # 初始估计
    x_est = 0.0
    P = 1.0

    x_est_history = []
    P_history = []

    # 2. 滤波循环
    for k in range(n_iter):
        # --- 预测步骤 (Time Update) ---
        # 状态预测: x(k|k-1) = x(k-1|k-1) (恒定模型)
        x_pred = x_est
        # 协方差预测: P(k|k-1) = P(k-1|k-1) + Q
        P_pred = P + Q

        # --- 更新步骤 (Measurement Update) ---
        # 计算卡尔曼增益: K = P / (P + R)
        K = P_pred / (P_pred + R)

        # 状态更新: x = x_pred + K * (z - x_pred)
        x_est = x_pred + K * (z[k] - x_pred)

        # 协方差更新: P = (1 - K) * P_pred
        P = (1 - K) * P_pred

        x_est_history.append(x_est)
        P_history.append(P)

    # 3. 可视化
    plt.figure(figsize=(10, 4))
    plt.plot(z, 'k+', label='嘈杂测量值')
    plt.plot(x_est_history, 'b-', linewidth=2, label='KF估计值')
    plt.axhline(x_true, color='g', linestyle='--', label='真实值')
    plt.title('案例1: 一维电压估计 (KF收敛过程)')
    plt.legend()
    plt.grid(True)
    plt.show()


# 运行案例
case1_simple_kf()