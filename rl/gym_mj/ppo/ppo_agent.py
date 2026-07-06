# ppo_agent.py — PPO 算法核心
# =============================
# 包含 GAE 优势估计、PPO Clipped 目标、Actor-Critic 更新逻辑。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .ppo_networks import ActorNetwork, CriticNetwork
from .ppo_buffer import PPOBuffer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体。
    包含 Actor（策略）和 Critic（价值）网络，以及训练更新逻辑。
    """

    def __init__(self, obs_dim, act_dim, device="cpu"):
        """
        参数:
            obs_dim: 观测空间维度
            act_dim: 动作空间维度
            device: 计算设备
        """
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # 构建网络
        self.actor = ActorNetwork(obs_dim, act_dim, hidden_sizes=config.ACTOR_HIDDEN).to(device)
        self.critic = CriticNetwork(obs_dim, hidden_sizes=config.CRITIC_HIDDEN).to(device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE)

        # 经验缓冲区
        self.buffer = PPOBuffer(obs_dim, act_dim, max_size=config.STEPS_PER_EPOCH, device=device)

        # print(self.actor)
        # print(self.critic)
        # print(self.buffer)

    def select_action(self, state, deterministic=False):
        """
        根据当前状态选择动作。

        参数:
            state: numpy 数组 [obs_dim]
            deterministic: 是否使用确定性策略

        返回:
            action: numpy 数组 [act_dim]（已裁剪到 [-1,1]）
            log_prob: float
            value: float
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # actor网络初步获取action, critic网络评估当前状态state获取value
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)

        return (
            action.cpu().numpy().flatten(),
            log_prob.cpu().item(),
            value.cpu().item(),
        )

    def store_transition(self, state, action, reward, value, log_prob, done):
        """存入一条经验到缓冲区。"""
        self.buffer.store(state, action, reward, value, log_prob, done)

    def compute_gae(self, rewards, values, dones, last_value):
        """
        计算 GAE (Generalized Advantage Estimation) 优势和回报。

        参数:
            rewards: 奖励序列 [T]
            values: 价值序列 [T]
            dones: 终止标志 [T]
            last_value: 最后状态的价值（用于 bootstrap）float

        返回:
            returns: 折扣回报 [T]
            advantages: GAE 优势 [T]
        """
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        advantages = torch.zeros(T, device=self.device)

        gae = 0.0
        next_return = last_value

        # 从后往前计算 GAE
        for t in reversed(range(T)):
            # TD 误差: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = (rewards[t] + config.GAMMA * next_return * (1.0 - dones[t]) - values[t])

            # GAE: A_t = δ_t + γλ * A_{t+1} * (1 - done)
            gae = delta + config.GAMMA * config.LAMBDA * gae * (1.0 - dones[t])

            advantages[t] = gae
            returns[t] = advantages[t] + values[t] # 没看懂???? td_target?
            next_return = values[t]

        return returns, advantages

    def update(self):
        """
        执行一次 PPO 更新（多个 epoch 遍历收集的数据）。

        返回:
            dict: 包含各项损失和指标的字典
        """
        # 从缓冲区取出所有数据
        data = self.buffer.get_all()
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        values = data["values"] # vlaues是在select_action中通过Critic网络预测得到的
        old_log_probs = data["log_probs"]
        dones = data["dones"]

        # 计算最后一个状态的价值（用于 bootstrap）
        # 如果最后一步是 done，则 last_value = 0
        last_value = 0.0 if dones[-1] > 0.5 else values[-1]

        # 计算 GAE 优势和回报
        returns, advantages = self.compute_gae(rewards, values, dones, last_value)

        # 优势归一化（稳定训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = len(states)
        indices = np.arange(total_steps)

        # 统计信息
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        # PPO 多轮更新
        for _ in range(config.PPO_EPOCHS):
            np.random.shuffle(indices) # 打乱顺序

            # 小批量训练 
            # 起始0， 终止total_steps， 步长 = 批次大小config.BATCH_SIZE = 64
            for start in range(0, total_steps, config.BATCH_SIZE):
                batch_idx = indices[start : start + config.BATCH_SIZE]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # --- Critic 更新 ---
                pred_values = self.critic(batch_states).squeeze(-1)
                # 在train_policy(A2C_GAE).py中，计算loss是value_pred和td_target
                # 在compute_gae函数中，delta = td_target - value
                critic_loss = F.mse_loss(pred_values, batch_returns)  

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
                self.critic_optimizer.step()

                # --- Actor 更新 (PPO Clipped) ---
                # 需要还原"原始动作"用于 evaluate（因为存储的动作是 tanh 后的）
                # 用 atanh 反推原始动作
                batch_actions_original = torch.atanh(torch.clamp(batch_actions, -0.999999, 0.999999))

                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions_original)

                # 概率比: r_t(θ) = π_new / π_old
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO Clipped 目标
                surr1 = ratio * batch_advantages
                surr2 = (torch.clamp(ratio, 1.0 - config.CLIP_EPSILON, 1.0 + config.CLIP_EPSILON) * batch_advantages)
                actor_loss = (-torch.min(surr1, surr2).mean() - config.ENTROPY_COEF * entropy.mean())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
                self.actor_optimizer.step()

                # 累计统计
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1

        return {
            "actor_loss": total_actor_loss / max(update_count, 1),
            "critic_loss": total_critic_loss / max(update_count, 1),
            "entropy": total_entropy / max(update_count, 1),
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

    def save(self, path):
        """保存模型权重。"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
            },
            path,
        )
        print(f"[保存] 模型已保存至: {path}")

    def load(self, path):
        """加载模型权重。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        print(f"[加载] 模型已从 {path} 加载")

agent = PPOAgent(11, 3)