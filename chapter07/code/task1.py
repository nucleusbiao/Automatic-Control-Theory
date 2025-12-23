import numpy as np
import gymnasium as gym

# 环境设置
env = gym.make('FrozenLake-v1', is_slippery=False)
Q = np.zeros([env.observation_space.n, env.action_space.n])  # 初始化Q表

# 超参数
lr = 0.8
gamma = 0.95
epsilon = 0.1
num_episodes = 2000

for i in range(num_episodes):
    s, _ = env.reset()
    done = False
    while not done:
        # Epsilon-Greedy 策略
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s, :])

        # 与环境交互
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        # Q-Learning 更新公式
        # Q(s,a) = Q(s,a) + lr * (r + gamma * max(Q(s', :)) - Q(s,a))
        Q[s, a] = Q[s, a] + lr * (r + gamma * np.max(Q[s_next, :]) - Q[s, a])

        s = s_next