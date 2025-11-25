# 自动控制理论

《自动控制理论》研究生课程资源库，包含完整的课件、练习代码和相关学习资料。

## 📚 课程简介

本课程系统介绍自动控制理论的核心内容，从经典控制理论到现代控制方法，再到前沿的智能控制技术。课程内容涵盖控制系统建模、分析、设计和优化的各个方面，为研究生阶段深入学习控制理论及相关研究奠定坚实基础。

## 🗂 课程结构

### 第1章 序论
- 控制系统基本概念
- 控制理论发展历程
- 控制系统分类与性能指标
- 典型控制系统实例分析

### 第2章 基于传递函数的经典控制理论
- 拉普拉斯变换与传递函数
- 系统时域分析
- 根轨迹法
- 频率响应法（Bode图、Nyquist图）
- PID控制器设计与整定

### 第3章 基于状态空间的现代控制理论
- 状态空间表达式
- 系统能控性与能观性
- 状态反馈与极点配置
- 观测器设计
- 多变量系统分析

### 第4章 控制系统稳定性分析
- Lyapunov稳定性理论
- 输入输出稳定性
- 鲁棒稳定性分析
- 频域稳定性判据

### 第5章 控制系统状态估计
- Kalman滤波器
- 扩展Kalman滤波器
- 粒子滤波器
- 状态估计在控制系统中的应用

### 第6章 最优控制理论
- 变分法与最优控制
- 极大值原理
- 动态规划
- LQR控制器设计
- H∞控制理论

### 第7章 基于深度强化学习的控制
- 强化学习基础
- 深度Q网络（DQN）
- 策略梯度方法
- 演员-评论家算法
- 深度强化学习在控制中的应用

## 💻 环境要求

### 基本环境
- Python 3.8+
- MATLAB R2020a+ (可选)

### Python依赖包
```bash
pip install numpy scipy matplotlib control
pip install torch tensorflow gym
pip install jupyter notebook
```

### 主要工具包
- `python-control`: 经典控制理论分析
- `scipy.signal`: 信号处理与系统分析
- `gym`: 强化学习环境
- `PyTorch/TensorFlow`: 深度学习框架

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/automatic-control-theory.git
cd automatic-control-theory
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行示例
```python
# 示例：二阶系统分析
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# 定义传递函数
sys = signal.TransferFunction([1], [1, 0.5, 1])
t, y = signal.step(sys)

# 绘制阶跃响应
plt.plot(t, y)
plt.title('二阶系统阶跃响应')
plt.xlabel('时间 (s)')
plt.ylabel('幅值')
plt.grid(True)
plt.show()
```

## 📖 内容说明

### 课件文件
- `lectures/`: 各章节课件（PDF/PPT格式）
- `notes/`: 补充笔记和参考资料

### 代码示例
- `examples/`: 各章节配套示例代码
- `exercises/`: 练习题和作业代码
- `projects/`: 课程项目示例

### 数据集
- `data/`: 实验数据和模型文件
- `models/`: 预训练模型

## 🛠 实用工具

### 控制系统设计工具
- 根轨迹绘制器
- Bode图分析工具
- 状态空间分析工具
- LQR控制器设计工具

### 强化学习训练框架
- DQN实现
- 策略梯度算法
- 环境接口封装

## 📝 学习建议

1. **理论学习**：结合课件理解控制理论的基本概念和数学基础
2. **代码实践**：通过示例代码验证理论结果，加深理解
3. **项目应用**：完成课程项目，将理论知识应用于实际问题
4. **扩展阅读**：参考推荐文献，深入了解前沿研究

## 🤝 贡献指南

欢迎提交Issue和Pull Request来完善本资源库：
- 报告错误或提出改进建议
- 提交新的示例代码或练习
- 完善文档和注释
- 分享有趣的应用案例

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：your-email@example.com

## 🌟 致谢

感谢所有为本课程资源库做出贡献的老师和同学们！

---

*最后更新: 2024年1月*
