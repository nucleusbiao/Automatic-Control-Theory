# 案例 2：2D 移动机器人全局定位
# 场景描述：
# 一个机器人在 $100m \times 100m$ 的场地中移动。场地四个角落有 4 个已知位置的信标（地标）。
# 机器人最初完全不知道自己在哪里（粒子均匀撒满全图）。随着机器人的移动和观测，粒子群应当收敛到机器人的真实位置。

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def case5_2d_global_localization():
    # --- 1. 环境设置 ---
    # 地图范围 (100m x 100m)
    map_size = 100

    # 地标位置 (Landmarks): 设置在四个角落 [x, y]
    landmarks = np.array([
        [10.0, 10.0],
        [90.0, 10.0],
        [90.0, 90.0],
        [10.0, 90.0]
    ])

    # 粒子数量 (2D空间需要更多粒子来覆盖)
    N = 1000

    # --- 2. 初始化 ---
    # 真实机器人初始状态 [x, y, theta]
    # 让机器人从中心开始绕圈跑
    x_true = np.array([50.0, 50.0, 0.0])

    # 粒子初始化：均匀分布在整个地图上 (全局定位 - 绑架问题)
    particles = np.empty((N, 3))
    particles[:, 0] = np.random.uniform(0, map_size, N)  # x
    particles[:, 1] = np.random.uniform(0, map_size, N)  # y
    particles[:, 2] = np.random.uniform(0, 2 * np.pi, N)  # theta

    weights = np.ones(N) / N  # 初始权重相等

    # 运动控制输入 (恒定速度和角速度)
    v = 2.0  # 线速度
    omega = 0.1  # 角速度 (绕圈)
    dt = 1.0

    # 噪声参数
    sigma_v = 0.5  # 速度噪声
    sigma_omega = 0.05  # 角速度噪声
    sigma_z = 3.0  # 测量距离噪声 (传感器精度)

    # --- 3. 辅助函数 ---

    # 测量函数：计算到所有地标的距离
    def get_measurements(state, landmarks):
        # state: [x, y, theta]
        # 返回: [d1, d2, d3, d4]
        dx = state[0] - landmarks[:, 0]
        dy = state[1] - landmarks[:, 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        return dist

    # 归一化角度到 [0, 2pi]
    def normalize_angle(angles):
        return np.mod(angles, 2 * np.pi)

    # --- 4. 仿真循环 ---
    steps = 40
    plt.figure(figsize=(15, 10))
    plot_idx = 1

    # 我们选择几个关键帧来绘图
    plot_frames = [0, 5, 20, 39]

    for step in range(steps):
        # ==========================
        # A. 真实世界 (Real World)
        # ==========================
        # 1. 机器人移动 (运动学模型)
        x_true[0] += v * np.cos(x_true[2]) * dt
        x_true[1] += v * np.sin(x_true[2]) * dt
        x_true[2] += omega * dt
        x_true[2] = normalize_angle(x_true[2])

        # 2. 机器人观测 (获取到4个地标的真实距离 + 噪声)
        z_real = get_measurements(x_true, landmarks) + np.random.normal(0, sigma_z, len(landmarks))

        # ==========================
        # B. 粒子滤波器 (Particle Filter)
        # ==========================

        # 1. 预测 (Prediction) - 移动所有粒子
        # 给控制量加噪声，模拟不确定的运动
        v_noisy = v + np.random.normal(0, sigma_v, N)
        omega_noisy = omega + np.random.normal(0, sigma_omega, N)

        particles[:, 0] += v_noisy * np.cos(particles[:, 2]) * dt
        particles[:, 1] += v_noisy * np.sin(particles[:, 2]) * dt
        particles[:, 2] += omega_noisy * dt
        particles[:, 2] = normalize_angle(particles[:, 2])

        # 边界处理 (如果粒子跑出地图，把它拉回来或者由重采样淘汰)
        particles[:, 0] = np.clip(particles[:, 0], 0, map_size)
        particles[:, 1] = np.clip(particles[:, 1], 0, map_size)

        # 2. 权重更新 (Weighting) - 核心步骤
        # 计算每个粒子对地标的预测距离
        # 这里的计算逻辑是：如果粒子的位置很好，它算出来的地标距离应该和机器人测到的 z_real 很像

        # 为了性能，这里使用矩阵运算计算所有粒子到所有地标的距离
        # (这是一个稍微复杂的numpy广播操作，为了代码可读性，我们简化为循环或简单处理)

        new_weights = np.ones(N)
        for i, landmark in enumerate(landmarks):
            # 计算所有粒子到当前地标 i 的距离
            dx = particles[:, 0] - landmark[0]
            dy = particles[:, 1] - landmark[1]
            pred_dist = np.sqrt(dx ** 2 + dy ** 2)

            # 高斯似然函数: 观测值 z_real[i] vs 预测值 pred_dist
            # 概率 = exp( - (diff)^2 / (2 * sigma^2) )
            prob = np.exp(-((z_real[i] - pred_dist) ** 2) / (2 * sigma_z ** 2))

            # 累乘所有地标的概率 (假设测量独立)
            new_weights *= prob

        weights = new_weights
        weights += 1.e-300  # 避免除0
        weights /= np.sum(weights)  # 归一化

        # 3. 状态估计 (Estimation)
        # 使用加权平均
        est_x = np.sum(particles[:, 0] * weights)
        est_y = np.sum(particles[:, 1] * weights)

        # 4. 重采样 (Resampling)
        # 只有当有效粒子数太少时才重采样 (这里为了演示，每次都做)
        indices = np.random.choice(N, size=N, p=weights)
        particles = particles[indices]

        # ==========================
        # C. 可视化
        # ==========================
        if step in plot_frames:
            ax = plt.subplot(2, 2, plot_idx)

            # 画地标
            ax.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='s', s=100, label='Landmarks')

            # 画粒子 (绿色点，透明度表示密集程度)
            ax.scatter(particles[:, 0], particles[:, 1], c='g', s=5, alpha=0.3, label='Particles')

            # 画真实机器人 (蓝色大圆点)
            ax.scatter(x_true[0], x_true[1], c='b', s=100, marker='o', edgecolors='k', label='Robot True')

            # 画估计位置 (橙色叉)
            ax.scatter(est_x, est_y, c='orange', s=150, marker='x', linewidth=3, label='PF Est')

            ax.set_xlim(0, map_size)
            ax.set_ylim(0, map_size)
            ax.set_title(f'Step {step}: Localization Process')
            ax.grid(True)
            if plot_idx == 1:
                ax.legend(loc='upper right')
            plot_idx += 1

    plt.tight_layout()
    plt.show()


# 运行演示
case5_2d_global_localization()