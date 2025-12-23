import torch
import torch.nn as nn
import torch.optim as optim
import random
import gymnasium as gym
import numpy as np
from collections import deque
from PIL import Image  # 用于保存GIF


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# 动画保存函数
def save_animation(model, env_name, filename='dqn_result.gif'):
    # 创建专门用于渲染的环境 (render_mode='rgb_array' 是新版gym的标准)
    try:
        env = gym.make(env_name, render_mode='rgb_array')
    except:
        env = gym.make(env_name)  # 兼容旧版

    state, _ = env.reset()
    frames = []
    done = False

    print(f"正在录制动画: {filename} ...")
    while not done:
        # 获取画面帧
        try:
            frame = env.render()
        except:
            frame = env.render(mode='rgb_array')  # 兼容旧版
        frames.append(Image.fromarray(frame))

        # 模型推理
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        # 环境步进
        step_result = env.step(action)
        if len(step_result) == 5:  # 新版API
            state, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:  # 旧版API
            state, _, done, _ = step_result

    # 保存为GIF
    if len(frames) > 0:
        frames[0].save(filename, save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
        print(f"动画已保存至: {filename}")
    env.close()


# --- 主训练流程 ---
env = gym.make('CartPole-v1')
q_net = QNet(4, 2)
target_net = QNet(4, 2)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
buffer = deque(maxlen=10000)

# 简单训练循环
print("开始训练 DQN...")
for episode in range(200):  # 仅演示，实际收敛需要更多回合
    state, _ = env.reset()
    done = False
    score = 0
    while not done:
        if random.random() < 0.1:  # Epsilon greedy
            action = env.action_space.sample()
        else:
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action = q_net(state_t).argmax().item()

        # 兼容不同gym版本
        res = env.step(action)
        next_state, reward, term, trunc, _ = res if len(res) == 5 else (*res, {})
        done = term or trunc

        buffer.append((state, action, reward, next_state, done))
        state = next_state
        score += 1

        # 更新网络
        if len(buffer) > 64:
            batch = random.sample(buffer, 64)
            s, a, r, ns, d = zip(*batch)
            s = torch.tensor(s, dtype=torch.float)
            a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
            r = torch.tensor(r, dtype=torch.float).unsqueeze(1)
            ns = torch.tensor(ns, dtype=torch.float)
            d = torch.tensor(d, dtype=torch.float).unsqueeze(1)

            q_val = q_net(s).gather(1, a)
            with torch.no_grad():
                target_val = r + 0.99 * target_net(ns).max(1)[0].unsqueeze(1) * (1 - d)
            loss = loss_fn(q_val, target_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if (episode + 1) % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())
        print(f"Episode {episode + 1}, Score: {score}")

env.close()
# 训练结束后保存动画
save_animation(q_net, 'CartPole-v1', 'dqn_cartpole_demo.gif')