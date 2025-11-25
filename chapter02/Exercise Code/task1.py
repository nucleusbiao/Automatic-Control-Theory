import numpy as np  # 用于多项式根计算


def calc_tf_zeros_poles(num_coeff, den_coeff):
    """
    计算传递函数的零点和极点
    参数：
        num_coeff: list，分子多项式系数（按s降幂排列，如[2,5]对应2s+5）
        den_coeff: list，分母多项式系数（按s降幂排列，如[1,3,2]对应s²+3s+2）
    返回：
        zeros: list，传递函数零点（保留2位小数）
        poles: list，传递函数极点（保留2位小数）
    """
    # 1. 计算零点：分子多项式=0的解（numpy.roots求解多项式根）
    zeros = np.roots(num_coeff)
    # 2. 计算极点：分母多项式=0的解
    poles = np.roots(den_coeff)

    # 3. 保留2位小数（处理复数情况时，实部和虚部分别保留）
    zeros_rounded = [round(z.real, 2) + round(z.imag, 2) * 1j if z.imag != 0
                     else round(z.real, 2) for z in zeros]
    poles_rounded = [round(p.real, 2) + round(p.imag, 2) * 1j if p.imag != 0
                     else round(p.real, 2) for p in poles]

    return zeros_rounded, poles_rounded


# ---------------------- 示例调用 ----------------------
if __name__ == "__main__":
    # 示例输入：分子2s+5（系数[2,5]），分母s²+3s+2（系数[1,3,2]）
    num = [2, 5]
    den = [1, 3, 2]

    # 计算零点和极点
    tf_zeros, tf_poles = calc_tf_zeros_poles(num, den)

    # 输出结果
    print("传递函数零点：", tf_zeros)
    print("传递函数极点：", tf_poles)
    # 预期输出：零点：[-2.5]，极点：[-1.0, -2.0]