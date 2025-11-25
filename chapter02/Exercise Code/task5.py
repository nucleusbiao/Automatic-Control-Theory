import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


class SecondOrderSystemAnalyzer:
    """
    二阶系统综合分析器
    提供脉冲响应、阶跃响应、频率响应等多种分析功能
    """

    def __init__(self, omega_n, xi_list, t_max=8, freq_max=10):
        """
        初始化分析器
        参数：
            omega_n: float，固定无阻尼固有频率（rad/s）
            xi_list: list，待对比的阻尼比
            t_max: float，时域仿真总时长
            freq_max: float，频域分析最大频率
        """
        self.omega_n = omega_n
        self.xi_list = xi_list
        self.t_max = t_max
        self.freq_max = freq_max
        self.t = np.arange(0, t_max, 0.01)
        self.colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB', '#9B59B6']

    def calculate_impulse_response(self, xi, t):
        """计算单位脉冲响应"""
        if 0 < xi < 1:
            # 欠阻尼
            omega_d = self.omega_n * np.sqrt(1 - xi ** 2)
            w = (self.omega_n / omega_d) * np.exp(-xi * self.omega_n * t) * np.sin(omega_d * t)
            damping_type = '欠阻尼'

        elif xi == 1:
            # 临界阻尼
            w = (self.omega_n ** 2) * t * np.exp(-self.omega_n * t)
            damping_type = '临界阻尼'

        elif xi > 1:
            # 过阻尼
            sqrt_xi2_1 = np.sqrt(xi ** 2 - 1)
            term1 = np.exp(-(xi - sqrt_xi2_1) * self.omega_n * t)
            term2 = np.exp(-(xi + sqrt_xi2_1) * self.omega_n * t)
            w = (self.omega_n / (2 * sqrt_xi2_1)) * (term1 - term2)
            damping_type = '过阻尼'

        else:
            raise ValueError(f"阻尼比ξ={xi}无效（需ξ≥0）")

        return w, damping_type

    def calculate_step_response(self, xi, t):
        """计算单位阶跃响应"""
        if 0 < xi < 1:
            # 欠阻尼
            omega_d = self.omega_n * np.sqrt(1 - xi ** 2)
            phi = np.arccos(xi)
            w = 1 - (np.exp(-xi * self.omega_n * t) / np.sqrt(1 - xi ** 2)) * np.sin(omega_d * t + phi)

        elif xi == 1:
            # 临界阻尼
            w = 1 - (1 + self.omega_n * t) * np.exp(-self.omega_n * t)

        elif xi > 1:
            # 过阻尼
            sqrt_xi2_1 = np.sqrt(xi ** 2 - 1)
            s1 = -self.omega_n * (xi - sqrt_xi2_1)
            s2 = -self.omega_n * (xi + sqrt_xi2_1)
            w = 1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1)

        return w

    def calculate_frequency_response(self, xi, omega):
        """计算频率响应"""
        if xi == 0:
            # 无阻尼特殊情况
            H = 1 / (1 - (omega / self.omega_n) ** 2 + 0j)
        else:
            H = 1 / (1 - (omega / self.omega_n) ** 2 + 2j * xi * (omega / self.omega_n))

        magnitude = np.abs(H)
        phase = np.angle(H, deg=True)
        return magnitude, phase

    def plot_comprehensive_comparison(self):
        """绘制综合对比图"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)

        # 1. 脉冲响应对比
        ax1 = fig.add_subplot(gs[0, 0])
        # 2. 阶跃响应对比
        ax2 = fig.add_subplot(gs[0, 1])
        # 3. 频率响应幅值对比
        ax3 = fig.add_subplot(gs[1, 0])
        # 4. 频率响应相位对比
        ax4 = fig.add_subplot(gs[1, 1])
        # 5. 3D响应曲面
        ax5 = fig.add_subplot(gs[2, :], projection='3d')

        # 生成频率范围
        omega = np.logspace(-1, np.log10(self.freq_max), 500)

        for i, xi in enumerate(self.xi_list):
            color = self.colors[i % len(self.colors)]

            # 计算并绘制脉冲响应
            w_impulse, damping_type = self.calculate_impulse_response(xi, self.t)
            ax1.plot(self.t, w_impulse, color=color, linewidth=2.5,
                     label=f'{damping_type} (ξ={xi})')

            # 计算并绘制阶跃响应
            w_step = self.calculate_step_response(xi, self.t)
            ax2.plot(self.t, w_step, color=color, linewidth=2.5,
                     label=f'{damping_type} (ξ={xi})')

            # 计算并绘制频率响应
            magnitude, phase = self.calculate_frequency_response(xi, omega)
            ax3.semilogx(omega, 20 * np.log10(magnitude), color=color, linewidth=2.5,
                         label=f'ξ={xi}')
            ax4.semilogx(omega, phase, color=color, linewidth=2.5, label=f'ξ={xi}')

        # 美化脉冲响应图
        ax1.set_xlabel('时间 t (s)')
        ax1.set_ylabel('单位脉冲响应 w(t)')
        ax1.set_title('二阶系统单位脉冲响应对比', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 美化阶跃响应图
        ax2.set_xlabel('时间 t (s)')
        ax2.set_ylabel('单位阶跃响应 h(t)')
        ax2.set_title('二阶系统单位阶跃响应对比', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)

        # 美化频率响应幅值图
        ax3.set_xlabel('频率 ω (rad/s)')
        ax3.set_ylabel('幅值 (dB)')
        ax3.set_title('频率响应 - 幅频特性', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=self.omega_n, color='red', linestyle='--', alpha=0.7, label=f'ωₙ={self.omega_n}')

        # 美化频率响应相位图
        ax4.set_xlabel('频率 ω (rad/s)')
        ax4.set_ylabel('相位 (度)')
        ax4.set_title('频率响应 - 相频特性', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=self.omega_n, color='red', linestyle='--', alpha=0.7, label=f'ωₙ={self.omega_n}')

        # 绘制3D响应曲面
        self._plot_3d_response(ax5)

        plt.tight_layout()
        plt.show()

    def _plot_3d_response(self, ax):
        """绘制3D响应曲面"""
        # 创建阻尼比网格
        xi_range = np.linspace(0.1, 2.0, 50)
        t_range = np.linspace(0, self.t_max, 100)

        Xi, T = np.meshgrid(xi_range, t_range)
        Z = np.zeros_like(Xi)

        # 计算每个点的响应值
        for i in range(len(xi_range)):
            for j in range(len(t_range)):
                w, _ = self.calculate_impulse_response(xi_range[i], t_range[j])
                Z[j, i] = w

        # 绘制3D曲面
        surf = ax.plot_surface(Xi, T, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('阻尼比 ξ')
        ax.set_ylabel('时间 t (s)')
        ax.set_zlabel('脉冲响应 w(t)')
        ax.set_title('脉冲响应随阻尼比和时间的变化', fontweight='bold')

        # 添加颜色条
        fig = ax.get_figure()
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)

    def plot_performance_metrics(self):
        """绘制性能指标对比"""
        metrics = {
            '阻尼比': [],
            '峰值时间': [],
            '超调量': [],
            '调节时间(2%)': [],
            '上升时间': []
        }

        for xi in self.xi_list:
            if xi >= 1:
                # 临界阻尼和过阻尼没有超调
                metrics['阻尼比'].append(xi)
                metrics['峰值时间'].append(np.nan)
                metrics['超调量'].append(0)

                # 计算调节时间（近似）
                w_step = self.calculate_step_response(xi, self.t)
                settling_idx = np.where(np.abs(w_step - 1) <= 0.02)[0]
                if len(settling_idx) > 0:
                    metrics['调节时间(2%)'].append(self.t[settling_idx[0]])
                else:
                    metrics['调节时间(2%)'].append(np.nan)

                # 计算上升时间（10%到90%）
                rise_start_idx = np.where(w_step >= 0.1)[0][0]
                rise_end_idx = np.where(w_step >= 0.9)[0][0]
                metrics['上升时间'].append(self.t[rise_end_idx] - self.t[rise_start_idx])

            else:
                # 欠阻尼系统
                metrics['阻尼比'].append(xi)

                # 峰值时间
                omega_d = self.omega_n * np.sqrt(1 - xi ** 2)
                metrics['峰值时间'].append(np.pi / omega_d)

                # 超调量
                metrics['超调量'].append(np.exp(-xi * np.pi / np.sqrt(1 - xi ** 2)) * 100)

                # 调节时间（2%准则）
                metrics['调节时间(2%)'].append(4 / (xi * self.omega_n))

                # 上升时间（近似）
                metrics['上升时间'].append((np.pi - np.arccos(xi)) / omega_d)

        # 绘制性能指标图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('二阶系统性能指标对比', fontsize=16, fontweight='bold')

        # 峰值时间
        axes[0, 0].bar(range(len(metrics['阻尼比'])), metrics['峰值时间'],
                       color=self.colors[:len(metrics['阻尼比'])])
        axes[0, 0].set_title('峰值时间')
        axes[0, 0].set_xticks(range(len(metrics['阻尼比'])))
        axes[0, 0].set_xticklabels([f'ξ={xi}' for xi in metrics['阻尼比']])

        # 超调量
        axes[0, 1].bar(range(len(metrics['阻尼比'])), metrics['超调量'],
                       color=self.colors[:len(metrics['阻尼比'])])
        axes[0, 1].set_title('超调量 (%)')
        axes[0, 1].set_xticks(range(len(metrics['阻尼比'])))
        axes[0, 1].set_xticklabels([f'ξ={xi}' for xi in metrics['阻尼比']])

        # 调节时间
        axes[1, 0].bar(range(len(metrics['阻尼比'])), metrics['调节时间(2%)'],
                       color=self.colors[:len(metrics['阻尼比'])])
        axes[1, 0].set_title('调节时间 (2%)')
        axes[1, 0].set_xticks(range(len(metrics['阻尼比'])))
        axes[1, 0].set_xticklabels([f'ξ={xi}' for xi in metrics['阻尼比']])

        # 上升时间
        axes[1, 1].bar(range(len(metrics['阻尼比'])), metrics['上升时间'],
                       color=self.colors[:len(metrics['阻尼比'])])
        axes[1, 1].set_title('上升时间')
        axes[1, 1].set_xticks(range(len(metrics['阻尼比'])))
        axes[1, 1].set_xticklabels([f'ξ={xi}' for xi in metrics['阻尼比']])

        plt.tight_layout()
        plt.show()


def second_order_impulse_response(omega_n, xi_list, t_max=5):
    """
    原函数的增强版本 - 保持向后兼容
    """
    analyzer = SecondOrderSystemAnalyzer(omega_n, xi_list, t_max)
    analyzer.plot_comprehensive_comparison()


# ---------------------- 示例调用 ----------------------
if __name__ == "__main__":
    # 系统参数
    omega_n = 4
    xi_list = [0.2, 0.7, 1.0, 1.5, 2.0]  # 扩展阻尼比范围

    # 创建分析器
    analyzer = SecondOrderSystemAnalyzer(omega_n, xi_list)

    print("=" * 60)
    print("二阶系统综合分析演示")
    print(f"固有频率 ωₙ = {omega_n} rad/s")
    print(f"阻尼比范围: {xi_list}")
    print("=" * 60)

    # 1. 显示综合对比图
    print("生成综合对比图...")
    analyzer.plot_comprehensive_comparison()

    # 2. 显示性能指标
    print("生成性能指标对比...")
    analyzer.plot_performance_metrics()

    # 3. 使用原函数（兼容模式）
    print("使用原函数生成脉冲响应对比...")
    second_order_impulse_response(omega_n, [0.2, 1.0, 2.0])