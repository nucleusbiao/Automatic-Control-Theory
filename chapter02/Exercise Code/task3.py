import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


def second_order_performance(omega_n, xi):
    """
    计算二阶欠阻尼系统（0<xi<1）的性能指标
    参数：
        omega_n: float，无阻尼固有频率（rad/s）
        xi: float，阻尼比（0<xi<1，如0.5）
    返回：
        sigma: float，最大超调量（%，保留2位小数）
        t_p: float，峰值时间（s，保留3位小数）
        t_s: float，调节时间（s，Δ=0.05，保留3位小数）
    """
    # 校验阻尼比范围（确保欠阻尼）
    if not (0 < xi < 1):
        raise ValueError("阻尼比xi必须满足0<xi<1（欠阻尼系统）")

    # 1. 最大超调量：σ% = e^(-xiπ/√(1-xi²)) × 100%
    sigma = np.exp(-xi * np.pi / np.sqrt(1 - xi ** 2)) * 100

    # 2. 峰值时间：t_p = π/(omega_n × √(1-xi²))
    omega_d = omega_n * np.sqrt(1 - xi ** 2)  # 有阻尼固有频率
    t_p = np.pi / omega_d

    # 3. 调节时间（Δ=0.05）：t_s ≈ 3/(xi×omega_n)
    t_s = 3 / (xi * omega_n)

    # 保留小数位
    sigma_rounded = round(sigma, 2)
    t_p_rounded = round(t_p, 3)
    t_s_rounded = round(t_s, 3)

    return sigma_rounded, t_p_rounded, t_s_rounded


def second_order_response(t, omega_n, xi, y0=0, v0=0, F=1, m=1):
    """
    计算二阶系统的单位阶跃响应
    """
    omega_d = omega_n * np.sqrt(1 - xi ** 2)

    if xi < 1:  # 欠阻尼
        A = F / (m * omega_n ** 2)
        y = A * (1 - np.exp(-xi * omega_n * t) *
                 (np.cos(omega_d * t) + (xi * omega_n / omega_d) * np.sin(omega_d * t)))
    elif xi == 1:  # 临界阻尼
        A = F / (m * omega_n ** 2)
        y = A * (1 - (1 + omega_n * t) * np.exp(-omega_n * t))
    else:  # 过阻尼
        s1 = -xi * omega_n + omega_n * np.sqrt(xi ** 2 - 1)
        s2 = -xi * omega_n - omega_n * np.sqrt(xi ** 2 - 1)
        A = F / (m * omega_n ** 2)
        y = A * (1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1))

    return y


def plot_system_response(omega_n, xi):
    """
    绘制二阶系统的响应曲线
    """
    # 计算性能指标
    sigma, t_p, t_s = second_order_performance(omega_n, xi)

    # 生成时间序列
    t = np.linspace(0, 3 * t_s, 1000)

    # 计算系统响应
    y = second_order_response(t, omega_n, xi)

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 绘制响应曲线
    ax1.plot(t, y, 'b-', linewidth=2, label=f'响应 (ω_n={omega_n}, ξ={xi})')
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='稳态值')

    # 标记关键点
    peak_response = second_order_response(t_p, omega_n, xi)
    ax1.plot(t_p, peak_response, 'ro', markersize=8, label=f'峰值点 (t={t_p}s)')
    ax1.axvline(x=t_s, color='g', linestyle='--', alpha=0.7,
                label=f'调节时间 (t={t_s}s)')

    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('位移 (m)')
    ax1.set_title(f'二阶系统阶跃响应\n超调量: {sigma}%, 峰值时间: {t_p}s, 调节时间: {t_s}s')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 绘制不同阻尼比的比较
    damping_ratios = [0.2, 0.5, 0.7, 1.0]
    colors = ['red', 'blue', 'green', 'purple']

    for xi_comp, color in zip(damping_ratios, colors):
        y_comp = second_order_response(t, omega_n, xi_comp)
        ax2.plot(t, y_comp, color=color, linewidth=2,
                 label=f'ξ={xi_comp}')

    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('位移 (m)')
    ax2.set_title('不同阻尼比的响应比较')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return sigma, t_p, t_s


def draw_improved_spring_damper_diagram(omega_n, xi, m=1):
    """
    绘制改进的弹簧-质量-阻尼器系统示意图
    """
    # 物理参数计算
    k = m * omega_n ** 2  # 弹簧刚度
    c = 2 * xi * m * omega_n  # 阻尼系数

    # 创建图形
    fig = plt.figure(figsize=(14, 10))

    # 系统示意图 - 更大的子图
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    # 响应曲线
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    # 参数显示
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    # 设置系统示意图区域
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 4)
    ax1.set_aspect('equal')
    ax1.set_title('弹簧-质量-阻尼器系统示意图', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 绘制更清晰的墙壁
    wall_height = 0.8
    wall_width = 0.3
    wall_x = -2.5
    wall_y = 2.0

    # 墙壁
    wall = patches.Rectangle((wall_x, wall_y), wall_width, wall_height,
                             facecolor='darkgray', edgecolor='black', linewidth=2)
    ax1.add_patch(wall)

    # 墙壁支撑
    support_y = np.linspace(wall_y, wall_y + wall_height, 10)
    for i in range(5):
        support_x = wall_x + wall_width + i * 0.05
        ax1.plot([support_x, support_x], [support_y[0], support_y[-1]],
                 'k-', linewidth=1, alpha=0.7)

    # 质量块参数
    mass_width = 1.2
    mass_height = 0.6
    mass_x = -mass_width / 2  # 初始位置
    mass_y = 1.0

    # 绘制质量块
    mass = patches.Rectangle((mass_x, mass_y), mass_width, mass_height,
                             facecolor='lightblue', edgecolor='blue', linewidth=2,
                             alpha=0.8)
    ax1.add_patch(mass)

    # 质量块标注
    ax1.text(mass_x + mass_width / 2, mass_y + mass_height / 2, '质量 m',
             ha='center', va='center', fontsize=12, fontweight='bold')

    # 绘制更清晰的弹簧
    spring_start_x = wall_x + wall_width
    spring_end_x = mass_x
    spring_y = 2.4

    # 弹簧线圈
    spring_coils = 8
    spring_length = spring_end_x - spring_start_x
    coil_spacing = spring_length / (spring_coils * 2)

    spring_x = [spring_start_x]
    spring_y_points = [spring_y]

    for i in range(spring_coils):
        # 前半线圈
        spring_x.append(spring_start_x + (2 * i + 1) * coil_spacing)
        spring_y_points.append(spring_y + 0.15)
        # 后半线圈
        spring_x.append(spring_start_x + (2 * i + 2) * coil_spacing)
        spring_y_points.append(spring_y - 0.15)

    spring_x.append(spring_end_x)
    spring_y_points.append(spring_y)

    ax1.plot(spring_x, spring_y_points, 'k-', linewidth=3, label='弹簧')

    # 弹簧标注
    ax1.text((spring_start_x + spring_end_x) / 2, spring_y + 0.3, f'刚度 k = {k:.1f} N/m',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # 绘制更清晰的阻尼器
    damper_start_x = wall_x + wall_width
    damper_end_x = mass_x
    damper_y = 1.8

    # 阻尼器外壳
    damper_length = damper_end_x - damper_start_x
    damper_outer_length = damper_length * 0.7
    damper_outer_x = damper_start_x + (damper_length - damper_outer_length) / 2

    ax1.add_patch(patches.Rectangle((damper_outer_x, damper_y - 0.1),
                                    damper_outer_length, 0.2,
                                    facecolor='lightcoral', edgecolor='red', linewidth=2))

    # 阻尼器活塞杆
    piston_start_x = damper_outer_x
    piston_end_x = mass_x
    ax1.plot([piston_start_x, piston_end_x], [damper_y, damper_y],
             'red', linewidth=3, linestyle='-')

    # 阻尼器活塞头
    piston_head_size = 0.15
    ax1.add_patch(patches.Circle((damper_outer_x + damper_outer_length, damper_y),
                                 piston_head_size, facecolor='red', edgecolor='darkred'))

    # 阻尼器标注
    ax1.text((damper_start_x + damper_end_x) / 2, damper_y - 0.4, f'阻尼 c = {c:.1f} N·s/m',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    # 绘制导轨
    rail_y = mass_y - 0.2
    ax1.plot([-2.8, 2.8], [rail_y, rail_y], 'gray', linewidth=3)
    ax1.plot([-2.8, 2.8], [rail_y - 0.05, rail_y - 0.05], 'gray', linewidth=1)

    # 绘制响应曲线
    t_max = 8
    t = np.linspace(0, t_max, 300)
    response = second_order_response(t, omega_n, xi)

    ax2.plot(t, response, 'b-', linewidth=3, label='位移响应')
    ax2.fill_between(t, 0, response, alpha=0.3, color='blue')
    ax2.set_xlabel('时间 (s)', fontsize=12)
    ax2.set_ylabel('位移 (m)', fontsize=12)
    ax2.set_title('质量块位移响应曲线', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 在参数显示区域添加系统信息
    ax3.axis('off')  # 关闭坐标轴
    info_text = f"""
系统参数详情

物理参数:
• 质量 m = {m} kg
• 弹簧刚度 k = {k:.1f} N/m
• 阻尼系数 c = {c:.1f} N·s/m

系统特性:
• 固有频率 ω_n = {omega_n:.2f} rad/s
• 阻尼比 ξ = {xi:.2f}
"""

    if 0 < xi < 1:
        sigma, t_p, t_s = second_order_performance(omega_n, xi)
        info_text += f"""
性能指标:
• 最大超调量: {sigma}%
• 峰值时间: {t_p} s
• 调节时间: {t_s} s

系统类型: 欠阻尼
"""
    elif xi == 1:
        info_text += "\n系统类型: 临界阻尼"
    elif xi > 1:
        info_text += "\n系统类型: 过阻尼"
    else:
        info_text += "\n系统类型: 无阻尼"

    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=1.0", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.show()


def analyze_physical_system(m, k, c):
    """
    分析实际物理系统
    """
    omega_n = np.sqrt(k / m)  # 固有频率
    xi = c / (2 * np.sqrt(m * k))  # 阻尼比

    print("=== 弹簧-质量-阻尼器系统分析 ===")
    print(f"系统参数:")
    print(f"  质量 m = {m} kg")
    print(f"  弹簧刚度 k = {k} N/m")
    print(f"  阻尼系数 c = {c} N·s/m")
    print(f"计算得到的:")
    print(f"  固有频率 ω_n = {omega_n:.2f} rad/s")
    print(f"  阻尼比 ξ = {xi:.2f}")

    if 0 < xi < 1:
        sigma, t_p, t_s = second_order_performance(omega_n, xi)
        print(f"性能指标:")
        print(f"  最大超调量: {sigma}%")
        print(f"  峰值时间: {t_p} s")
        print(f"  调节时间: {t_s} s")

        # 绘制响应
        plot_system_response(omega_n, xi)

        # 绘制改进的系统示意图
        print("生成改进的系统示意图...")
        draw_improved_spring_damper_diagram(omega_n, xi, m)

    elif xi == 0:
        print("无阻尼系统 - 持续振荡")
    elif xi == 1:
        print("临界阻尼系统 - 最快无超调响应")
    else:
        print("过阻尼系统 - 缓慢响应")


# ---------------------- 示例调用 ----------------------
if __name__ == "__main__":
    # 方法1: 直接使用固有频率和阻尼比
    print("方法1: 直接参数分析")
    omega_n = 5
    xi = 0.5

    sigma, t_p, t_s = second_order_performance(omega_n, xi)
    print(f"性能指标: 超调量={sigma}%, 峰值时间={t_p}s, 调节时间={t_s}s")

    # 绘制响应曲线
    plot_system_response(omega_n, xi)

    # 绘制改进的系统示意图
    print("生成改进的系统示意图...")
    draw_improved_spring_damper_diagram(omega_n, xi)

    # 方法2: 通过物理参数分析实际系统
    print("\n方法2: 物理系统分析")
    # 示例: 机械振动系统
    analyze_physical_system(m=10, k=1000, c=50)

    # 示例: 更小系统，便于观察
    print("\n示例: 小型振动系统")
    analyze_physical_system(m=1, k=25, c=2)