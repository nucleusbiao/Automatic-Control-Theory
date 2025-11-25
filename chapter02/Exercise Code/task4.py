import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def rlc_tf_model(R, L, C):
    """
    建立RLC串联电路的传递函数（输入u_i，输出u_0）
    参数：
        R: float，电阻（Ω）
        L: float，电感（H）
        C: float，电容（F）
    返回：
        num: list，传递函数分子系数
        den: list，传递函数分母系数
        omega_n: float，无阻尼固有频率（rad/s）
        xi: float，阻尼比
        system_type: str，系统类型（过阻尼/临界阻尼/欠阻尼/无阻尼）
    """
    # 1. 传递函数形式：G(s) = 1/(LC s² + RC s + 1)
    num = [1]  # 分子系数（常数项1）
    den = [L * C, R * C, 1]  # 分母系数（LC s² + RC s + 1）

    # 2. 计算无阻尼固有频率和阻尼比
    omega_n = 1 / np.sqrt(L * C)  # ωₙ = 1/√(LC)
    xi = (R * np.sqrt(C)) / (2 * np.sqrt(L))  # ξ = R√C/(2√L)

    # 3. 判断系统类型
    if xi > 1:
        system_type = "过阻尼系统 (Overdamped)"
    elif xi == 1:
        system_type = "临界阻尼系统 (Critically damped)"
    elif 0 < xi < 1:
        system_type = "欠阻尼系统 (Underdamped)"
    else:
        system_type = "无阻尼系统 (Undamped)"

    # 4. 计算其他重要参数
    omega_d = omega_n * np.sqrt(1 - xi ** 2) if xi < 1 else 0  # 阻尼固有频率
    sigma = xi * omega_n  # 衰减系数

    # 计算极点
    poles = np.roots(den)

    # 保留3位小数
    omega_n_rounded = round(omega_n, 3)
    xi_rounded = round(xi, 3)
    omega_d_rounded = round(omega_d, 3) if omega_d > 0 else 0
    sigma_rounded = round(sigma, 3)

    return num, den, omega_n_rounded, xi_rounded, system_type, poles, omega_d_rounded, sigma_rounded


def analyze_rlc_response(R, L, C, plot=True):
    """
    分析RLC电路的时域和频域响应
    """
    # 获取传递函数参数
    num, den, omega_n, xi, system_type, poles, omega_d, sigma = rlc_tf_model(R, L, C)

    # 创建传递函数系统
    sys = signal.TransferFunction(num, den)

    print("=" * 60)
    print("RLC串联电路详细分析报告")
    print("=" * 60)
    print(f"电路参数：R = {R}Ω, L = {L}H, C = {C}F")
    print(f"\n传递函数：G(s) = 1 / ({den[0]:.4f}s² + {den[1]:.4f}s + {den[2]:.4f})")
    print(f"\n系统特征：")
    print(f"  - 系统类型: {system_type}")
    print(f"  - 无阻尼固有频率 ω_n = {omega_n} rad/s")
    print(f"  - 阻尼比 ξ = {xi}")
    if xi < 1:
        print(f"  - 阻尼固有频率 ω_d = {omega_d:.3f} rad/s")
    print(f"  - 衰减系数 σ = {sigma:.3f}")
    print(f"  - 系统极点: {poles[0]:.3f}, {poles[1]:.3f}")

    # 计算时域性能指标（阶跃响应）
    if xi < 1 and xi > 0:  # 欠阻尼系统
        # 上升时间
        tr = (np.pi - np.arccos(xi)) / (omega_n * np.sqrt(1 - xi ** 2))
        # 峰值时间
        tp = np.pi / (omega_n * np.sqrt(1 - xi ** 2))
        # 超调量
        Mp = np.exp(-xi * np.pi / np.sqrt(1 - xi ** 2)) * 100
        # 调节时间（2%准则）
        ts = 4 / (xi * omega_n)

        print(f"\n时域性能指标：")
        print(f"  - 上升时间 t_r = {tr:.3f} s")
        print(f"  - 峰值时间 t_p = {tp:.3f} s")
        print(f"  - 超调量 M_p = {Mp:.2f}%")
        print(f"  - 调节时间 t_s = {ts:.3f} s")

    # 计算频域特性
    w_resonant = omega_n * np.sqrt(1 - 2 * xi ** 2) if xi < 0.707 else 0
    if 0 < xi < 0.707:
        print(f"\n频域特性：")
        print(f"  - 谐振频率 ω_r = {w_resonant:.3f} rad/s")

    # 绘制图形
    if plot:
        plot_rlc_response(sys, R, L, C, omega_n, xi, system_type)


def plot_rlc_response(sys, R, L, C, omega_n, xi, system_type):
    """
    绘制RLC电路的响应曲线
    """
    # 创建时间向量（根据阻尼比调整时间范围）
    time_scale = max(0.1, xi * omega_n) if omega_n > 0 else 1
    t = np.linspace(0, 10 / time_scale, 1000)

    # 计算阶跃响应
    t_step, y_step = signal.step(sys, T=t)

    # 计算频率响应
    w, mag, phase = signal.bode(sys, np.logspace(-1, 3, 1000))

    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # 时域响应
    ax1.plot(t_step, y_step, 'b-', linewidth=2, label=f'阶跃响应 (ξ={xi})')
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='稳态值')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅度')
    ax1.set_title(f'RLC电路阶跃响应\nR={R}Ω, L={L}H, C={C}F - {system_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 幅频特性
    ax2.semilogx(w, mag, 'g-', linewidth=2)
    ax2.set_xlabel('频率 (rad/s)')
    ax2.set_ylabel('幅度 (dB)')
    ax2.set_title('幅频特性')
    ax2.axvline(x=omega_n, color='r', linestyle='--', alpha=0.7, label=f'ω_n={omega_n} rad/s')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 相频特性
    ax3.semilogx(w, phase, 'm-', linewidth=2)
    ax3.set_xlabel('频率 (rad/s)')
    ax3.set_ylabel('相位 (度)')
    ax3.set_title('相频特性')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_different_damping():
    """
    比较不同阻尼比下的系统响应
    """
    L, C = 1, 0.01  # 固定电感和电容
    omega_n = 1 / np.sqrt(L * C)  # 固定固有频率

    # 不同阻尼比对应的电阻值
    damping_cases = [
        (0.2, "欠阻尼 (ξ=0.2)", 'b-'),
        (0.7, "欠阻尼 (ξ=0.7)", 'g-'),
        (1.0, "临界阻尼 (ξ=1.0)", 'r-'),
        (2.0, "过阻尼 (ξ=2.0)", 'm-')
    ]

    plt.figure(figsize=(12, 8))

    for xi, label, linestyle in damping_cases:
        R = 2 * xi * np.sqrt(L / C)  # 计算对应的电阻
        num, den = [1], [L * C, R * C, 1]
        sys = signal.TransferFunction(num, den)

        t = np.linspace(0, 2, 1000)
        t_step, y_step = signal.step(sys, T=t)

        plt.plot(t_step, y_step, linestyle, linewidth=2, label=label)

    plt.xlabel('时间 (s)')
    plt.ylabel('幅度')
    plt.title('不同阻尼比下的RLC电路阶跃响应比较 (ω_n=10 rad/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def rlc_design_specifications(omega_n=None, xi=None, settling_time=None, overshoot=None):
    """
    根据性能指标设计RLC电路参数
    """
    print("=" * 60)
    print("RLC电路参数设计")
    print("=" * 60)

    if omega_n and xi:
        # 根据固有频率和阻尼比设计
        L = 0.1  # 选择合理的电感值
        C = 1 / (L * omega_n ** 2)
        R = 2 * xi * np.sqrt(L / C)

        print(f"设计规格：ω_n = {omega_n} rad/s, ξ = {xi}")
        print(f"推荐电路参数：")
        print(f"  - 电感 L = {L:.4f} H")
        print(f"  - 电容 C = {C:.4f} F")
        print(f"  - 电阻 R = {R:.4f} Ω")

        return R, L, C

    elif settling_time and overshoot:
        # 根据调节时间和超调量设计
        xi = -np.log(overshoot / 100) / np.sqrt(np.pi ** 2 + np.log(overshoot / 100) ** 2)
        omega_n = 4 / (xi * settling_time)

        L = 0.1
        C = 1 / (L * omega_n ** 2)
        R = 2 * xi * np.sqrt(L / C)

        print(f"设计规格：调节时间 t_s = {settling_time}s, 超调量 M_p = {overshoot}%")
        print(f"计算得到：ω_n = {omega_n:.3f} rad/s, ξ = {xi:.3f}")
        print(f"推荐电路参数：")
        print(f"  - 电感 L = {L:.4f} H")
        print(f"  - 电容 C = {C:.4f} F")
        print(f"  - 电阻 R = {R:.4f} Ω")

        return R, L, C


# ---------------------- 示例调用 ----------------------
if __name__ == "__main__":
    # 示例1：基本验证
    print("示例1：基本参数验证")
    R, L, C = 10, 1, 0.01
    analyze_rlc_response(R, L, C, plot=True)

    # 示例2：不同阻尼情况比较
    print("\n" + "=" * 60)
    print("示例2：不同阻尼比系统")
    print("=" * 60)

    # 欠阻尼系统
    print("\n1. 欠阻尼系统 (ξ=0.3):")
    R_under = 2 * 0.3 * np.sqrt(1 / 0.01)  # R = 6Ω
    analyze_rlc_response(R_under, 1, 0.01, plot=False)

    # 临界阻尼系统
    print("\n2. 临界阻尼系统 (ξ=1.0):")
    R_critical = 2 * 1.0 * np.sqrt(1 / 0.01)  # R = 20Ω
    analyze_rlc_response(R_critical, 1, 0.01, plot=False)

    # 过阻尼系统
    print("\n3. 过阻尼系统 (ξ=2.0):")
    R_over = 2 * 2.0 * np.sqrt(1 / 0.01)  # R = 40Ω
    analyze_rlc_response(R_over, 1, 0.01, plot=False)

    # 比较不同阻尼比的响应
    compare_different_damping()

    # 示例3：谐振电路分析
    print("\n" + "=" * 60)
    print("示例3：谐振电路分析")
    print("=" * 60)
    # 选择较小的电阻以获得明显的谐振峰值
    R_resonant = 2  # 小电阻
    analyze_rlc_response(R_resonant, 0.1, 0.001, plot=True)

    # 示例4：电路参数设计
    print("\n" + "=" * 60)
    print("示例4：根据性能指标设计电路")
    print("=" * 60)

    # 设计1：指定固有频率和阻尼比
    rlc_design_specifications(omega_n=15, xi=0.7)

    # 设计2：指定调节时间和超调量
    rlc_design_specifications(settling_time=0.5, overshoot=10)