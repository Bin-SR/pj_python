# ppo_buffer.py — PPO 经验缓冲区
# ===============================
# 存储 on-policy 的轨迹数据，供 PPO 更新使用。
# 每轮收集完毕后一次性取出，然后清空。

import torch
import numpy as np


class PPOBuffer:
    """
    PPO 经验回放缓冲区。
    收集 states, actions, rewards, values, log_probs, dones，
    用于计算 GAE 优势和 PPO 更新。
    """

    def __init__(self, obs_dim, act_dim, max_size, device="cpu"):
        """
        参数:
            obs_dim: 观测维度
            act_dim: 动作维度
            max_size: 最大存储步数
            device: 计算设备 (cpu / cuda)
        """
        self.device = device
        self.max_size = max_size

        # 预分配存储空间
        self.states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

        self.ptr = 0     # 当前写入位置
        self.full = False

    def store(self, state, action, reward, value, log_prob, done):
        """
        存入一条经验。

        参数:
            state: 观测 [obs_dim]
            action: 动作 [act_dim]
            reward: 奖励 float
            value: 状态价值 float
            log_prob: 对数概率 float
            done: 是否终止 bool
        """
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = float(done)

        self.ptr += 1
        if self.ptr >= self.max_size:
            self.full = True
            self.ptr = 0

    def get_all(self):
        """
        取出所有已存储的数据并清空缓冲区。

        返回:
            dict: 包含所有数据的字典，均为 torch.Tensor
        """
        count = self.max_size if self.full else self.ptr

        data = {
            "states": torch.FloatTensor(self.states[:count]).to(self.device),
            "actions": torch.FloatTensor(self.actions[:count]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[:count]).to(self.device),
            "values": torch.FloatTensor(self.values[:count]).to(self.device),
            "log_probs": torch.FloatTensor(self.log_probs[:count]).to(self.device),
            "dones": torch.FloatTensor(self.dones[:count]).to(self.device),
        }

        # 重置缓冲区
        self.ptr = 0
        self.full = False

        return data

    def __len__(self):
        return self.max_size if self.full else self.ptr
