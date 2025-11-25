import numpy as np
import matplotlib.pyplot as plt  # 用于绘图

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def first_order_step_response(T, t_max):
    """
    计算并绘制一阶系统单位阶跃响应
    参数：
        T: float，时间常数（如0.5秒）
        t_max: float，仿真总时长（如5秒）
    """
    # 1. 生成时间序列：从0到t_max，步长0.01秒（保证曲线平滑）
    t = np.arange(0, t_max, 0.01)

    # 2. 计算阶跃响应：x0(t) = 1 - e^(-t/T)
    x0 = 1 - np.exp(-t / T)

    # 3. 计算关键时刻（0.5T、T、2T、3T、4T）的响应值
    key_times = [0.5 * T, T, 2 * T, 3 * T, 4 * T]
    # 遍历关键时刻，找到最接近的时间点（因t是离散序列）
    key_responses = []
    for kt in key_times:
        # 找到t中最接近kt的索引
        idx = np.argmin(np.abs(t - kt))
        key_responses.append(round(x0[idx], 3))

    # 4. 输出关键时刻结果
    print(f"一阶系统单位阶跃响应关键值（时间常数T={T}）：")
    for i in range(len(key_times)):
        print(f"t={key_times[i]:.2f}s（{key_times[i] / T}T）: x0={key_responses[i]}")

    # 5. 绘制响应曲线
    plt.figure(figsize=(10, 6))  # 设置图大小
    plt.plot(t, x0, color='#2E86AB', linewidth=2, label=f'T={T}s')  # 响应曲线
    plt.scatter(key_times, key_responses, color='#A23B72', s=50, zorder=5)  # 标记关键 points

    # 添加图表标签和格式
    plt.xlabel('时间 t (s)', fontsize=12)
    plt.ylabel('阶跃响应 x0(t)', fontsize=12)
    plt.title('一阶系统单位阶跃响应', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)  # 显示网格（透明度0.3）
    plt.show()


# ---------------------- 示例调用 ----------------------
if __name__ == "__main__":
    # 输入：时间常数T=1秒，仿真时长t_max=5秒
    first_order_step_response(T=1, t_max=5)
    # 预期关键值：t=0.5s(0.5T):~0.393, t=1s(1T):~0.632, t=2s(2T):~0.865