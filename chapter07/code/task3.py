import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from PIL import Image
import random
from collections import deque
import copy


# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.max_action * self.net(x)


# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# Ornstein-Uhlenbeck噪声
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# DDPG智能体
class DDPG:
    def __init__(self, state_dim, action_dim, max_action=2.0, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-3):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

        # 网络初始化
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 噪声
        self.noise = OUNoise(action_dim)

        # 经验回放
        self.buffer = ReplayBuffer(100000)

    def select_action(self, state, add_noise=True):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return

        # 采样批次
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # 更新Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()


# 动画保存函数
def save_ddpg_animation(actor, env_name, filename='ddpg_result.gif'):
    try:
        env = gym.make(env_name, render_mode='rgb_array')
    except:
        env = gym.make(env_name)

    state, _ = env.reset()
    frames = []

    print(f"正在录制动画: {filename} ...")
    for _ in range(200):
        try:
            frame = env.render()
        except:
            frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))

        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            action = actor(state_t).numpy()[0]

        res = env.step(action)
        state = res[0]
        done = res[2] or res[3] if len(res) == 5 else res[2]
        if done:
            break

    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=40, loop=0)
    print(f"动画已保存至: {filename}")
    env.close()


# 训练函数
def train_ddpg(env_name='Pendulum-v1', episodes=200, max_steps=200):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)

    rewards_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        update_count = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.buffer.push(state, action, reward, next_state, done)

            # 更新网络
            if len(agent.buffer) > 1000:
                critic_loss, actor_loss = agent.update()
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                update_count += 1

            state = next_state
            episode_reward += reward

            if done:
                break

        # 计算平均损失
        avg_critic_loss = episode_critic_loss / update_count if update_count > 0 else 0
        avg_actor_loss = episode_actor_loss / update_count if update_count > 0 else 0

        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)

        if episode % 10 == 0:
            print(f"Episode: {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Critic Loss: {avg_critic_loss:7.4f} | "
                  f"Actor Loss: {avg_actor_loss:7.4f}")

    env.close()
    return agent, rewards_history


# 主程序
if __name__ == "__main__":
    # 训练DDPG智能体
    print("开始训练DDPG...")
    agent, rewards_history = train_ddpg('Pendulum-v1', episodes=100, max_steps=200)

    # 录制动画
    print("\nDDPG训练完成，录制演示动画...")
    save_ddpg_animation(agent.actor, 'Pendulum-v1', 'ddpg_pendulum_demo.gif')

    # 可选：绘制训练曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DDPG Training on Pendulum-v1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ddpg_training_curve.png')
    plt.show()
    print("训练曲线已保存至: ddpg_training_curve.png")