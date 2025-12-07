import numpy as np


def routh_stability(coefficients):
    # 必要条件检查：系数非零且同号
    if any(c == 0 for c in coefficients):
        print("特征方程缺项，系统不稳定")
        return
    if any(np.sign(c) != np.sign(coefficients[0]) for c in coefficients):
        print("系数符号不同，系统不稳定")
        return

    n = len(coefficients) - 1  # 特征方程阶数
    routh_table = np.zeros((n + 1, n + 1))
    routh_table[0, :n + 1] = coefficients  # 第一行

    # 构造Routh表
    for i in range(1, n + 1):
        for j in range(n - i + 1):
            if routh_table[i - 1, 0] == 0:
                routh_table[i - 1, 0] = 1e-6  # ε替换
            # 按Routh规则计算元素
            routh_table[i, j] = (routh_table[i - 1, 0] * routh_table[i - 2, j + 1] -
                                 routh_table[i - 2, 0] * routh_table[i - 1, j + 1]) / routh_table[i - 1, 0]

    # 统计第一列符号变化次数
    sign_changes = 0
    for i in range(1, n + 1):
        if np.sign(routh_table[i, 0]) != np.sign(routh_table[i - 1, 0]):
            sign_changes += 1

    print("Routh表：")
    print(routh_table[:, :n + 1 - np.count_nonzero(routh_table[-1, :])])  # 裁剪全零列
    if sign_changes == 0:
        print("系统稳定，无正实部极点")
    else:
        print(f"系统不稳定，正实部极点个数：{sign_changes}")


# 验证PPT例1：D(s)=s^4+s^3-28s^2+20s+48=0
routh_stability([1, 1, -28, 20, 48])