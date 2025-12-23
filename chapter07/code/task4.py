import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from PIL import Image
from collections import deque
import matplotlib.pyplot as plt


class PPOActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, a_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == 1:  # 最后一层critic
                nn.init.uniform_(module.weight, -3e-3, 3e-3)
            else:
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        feat = self.base(x)
        return self.actor(feat), self.critic(feat)

    def get_action(self, state, action=None):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


class PPOBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.returns = []
        self.advantages = []
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, last_value, done):
        # 将列表转换为numpy数组
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [done])

        gae = 0
        advantages = []
        returns = []

        # 反向计算GAE和returns
        for t in reversed(range(len(self.rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, advantages[0] + values[t])

        # 标准化advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = advantages
        self.returns = np.array(returns)

        # 转换为张量
        self.states = torch.tensor(np.array(self.states), dtype=torch.float32)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.int64)
        self.log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32)
        self.values = torch.tensor(np.array(self.values), dtype=torch.float32)
        self.advantages = torch.tensor(self.advantages, dtype=torch.float32)
        self.returns = torch.tensor(self.returns, dtype=torch.float32)

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return (self.states, self.actions, self.log_probs,
                    self.values, self.advantages, self.returns)

        # 随机采样
        indices = torch.randperm(len(self.states))[:batch_size]
        return (self.states[indices], self.actions[indices],
                self.log_probs[indices], self.values[indices],
                self.advantages[indices], self.returns[indices])

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.returns = []
        self.advantages = []


class PPO:
    def __init__(self, env, clip_ratio=0.2, lr=3e-4, update_epochs=10,
                 batch_size=64, entropy_coef=0.01, value_coef=0.5):
        self.env = env
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # 模型和优化器
        self.model = PPOActorCritic(
            env.observation_space.shape[0],
            env.action_space.n
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 缓冲区
        self.buffer = PPOBuffer()

        # 训练历史
        self.rewards_history = []
        self.loss_history = []

    def collect_trajectories(self, max_steps=1000):
        state, _ = self.env.reset()
        episode_reward = 0

        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action(state_tensor)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            self.buffer.store(state, action.item(), log_prob.item(),
                              value.item(), reward, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 计算最后一个状态的价值
        if not done:
            with torch.no_grad():
                last_value = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[1].item()
        else:
            last_value = 0

        # 计算GAE和returns
        self.buffer.compute_gae(last_value, done)

        return episode_reward, step + 1

    def update(self):
        # 获取完整批次数据
        states, actions, old_log_probs, old_values, advantages, returns = self.buffer.get_batch()

        # 多轮更新
        for _ in range(self.update_epochs):
            # 随机洗牌数据
            indices = torch.randperm(len(states))

            # 小批次更新
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_old_values = old_values[idx]

                # 前向传播
                logits, values = self.model(batch_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # 计算新log probs和熵
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO损失函数
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic损失
                values = values.squeeze(-1)
                critic_loss = nn.MSELoss()(values, batch_returns)

                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                # 记录损失
                self.loss_history.append({
                    'total': loss.item(),
                    'actor': actor_loss.item(),
                    'critic': critic_loss.item(),
                    'entropy': entropy.item()
                })

        # 清空缓冲区
        self.buffer.clear()

    def train(self, episodes=500, max_steps=1000, save_freq=50):
        print("开始PPO训练...")

        for episode in range(episodes):
            # 收集轨迹
            episode_reward, episode_length = self.collect_trajectories(max_steps)
            self.rewards_history.append(episode_reward)

            # 更新策略
            self.update()

            # 打印进度
            avg_reward = np.mean(self.rewards_history[-20:]) if len(self.rewards_history) >= 20 else np.mean(
                self.rewards_history)

            if episode % 10 == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Length: {episode_length:4d} | "
                      f"Avg Reward: {avg_reward:7.2f}")

            # 保存检查点
            if save_freq > 0 and episode % save_freq == 0 and episode > 0:
                torch.save(self.model.state_dict(), f'ppo_model_ep{episode}.pth')

        print("训练完成!")
        return self.model, self.rewards_history, self.loss_history


def save_ppo_animation(model, env_name, filename='ppo_result.gif', max_steps=1000):
    try:
        env = gym.make(env_name, render_mode='rgb_array')
    except:
        print(f"无法创建环境 {env_name}，尝试CartPole-v1")
        env = gym.make('CartPole-v1', render_mode='rgb_array')

    state, _ = env.reset()
    frames = []
    print(f"正在录制动画: {filename} ...")

    total_reward = 0
    for step in range(max_steps):
        try:
            frame = env.render()
        except:
            frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))

        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state_t)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        done = terminated or truncated

        if done:
            break
        if len(frames) > 500:
            break

    print(f"测试奖励: {total_reward:.2f}")

    if len(frames) > 0:
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=20, loop=0)
        print(f"动画已保存至: {filename}")
    env.close()


def plot_training_results(rewards_history, loss_history=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 奖励曲线
    axes[0, 0].plot(rewards_history)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].grid(True)

    # 移动平均奖励
    if len(rewards_history) >= 20:
        moving_avg = np.convolve(rewards_history, np.ones(20) / 20, mode='valid')
        axes[0, 1].plot(range(19, len(rewards_history)), moving_avg)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Moving Avg Reward')
        axes[0, 1].set_title('Moving Average Reward (window=20)')
        axes[0, 1].grid(True)

    # 损失曲线
    if loss_history:
        total_loss = [l['total'] for l in loss_history]
        actor_loss = [l['actor'] for l in loss_history]
        critic_loss = [l['critic'] for l in loss_history]

        axes[1, 0].plot(total_loss, label='Total Loss', alpha=0.7)
        axes[1, 0].plot(actor_loss, label='Actor Loss', alpha=0.7)
        axes[1, 0].plot(critic_loss, label='Critic Loss', alpha=0.7)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 熵曲线
        entropy = [l['entropy'] for l in loss_history]
        axes[1, 1].plot(entropy, color='purple')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('ppo_training_results.png', dpi=100)
    plt.show()
    print("训练结果图已保存至: ppo_training_results.png")


# 主程序
if __name__ == "__main__":
    # 选择环境
    # env_name = 'CartPole-v1'  # 简单环境，适合快速训练
    env_name = 'LunarLander-v2'  # 需要安装box2d: pip install gymnasium[box2d]

    try:
        env = gym.make(env_name)
        print(f"成功创建环境: {env_name}")
    except:
        print(f"无法创建环境 {env_name}，使用CartPole-v1")
        env = gym.make('CartPole-v1')
        env_name = 'CartPole-v1'

    # 创建并训练PPO
    ppo_agent = PPO(
        env,
        clip_ratio=0.2,
        lr=3e-4,
        update_epochs=10,
        batch_size=64,
        entropy_coef=0.01,
        value_coef=0.5
    )

    # 训练
    model, rewards_history, loss_history = ppo_agent.train(
        episodes=300,
        max_steps=1000,
        save_freq=100
    )

    # 保存最终模型
    torch.save(model.state_dict(), 'ppo_final_model.pth')
    print("最终模型已保存至: ppo_final_model.pth")

    # 绘制训练结果
    plot_training_results(rewards_history, loss_history)

    # 录制动画
    save_ppo_animation(model, env_name, 'ppo_animation.gif')