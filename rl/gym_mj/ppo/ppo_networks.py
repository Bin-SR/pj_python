# ppo_networks.py — Actor-Critic 网络结构
# ========================================
# Actor: 输出动作均值（连续动作空间用高斯分布）
# Critic: 输出状态价值 V(s)

import torch
import torch.nn as nn
import numpy as np


def init_weights(layer, gain=np.sqrt(2)):
    """正交初始化权重，有助于稳定训练。"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


class ActorNetwork(nn.Module):
    """
    Actor（策略网络）：输入状态，输出动作均值。
    连续动作空间使用高斯分布，标准差为可学习参数。
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64]):
        """
        参数:
            obs_dim: 观测维度
            act_dim: 动作维度
            hidden_sizes: 隐藏层大小列表
        """
        super().__init__()

        # 构建隐藏层
        layers = []
        in_size = obs_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.Tanh())
            in_size = h_size
        # 语法解释：*， 表示把列表layers中的每一个元素作为独立的参数，传给函数
        # 如果没有*, 则变成了把整个列表作为一个参数, 即nn.Sequential([A, B]), 我们需要(A, B)
        self.feature_net = nn.Sequential(*layers)

        # 输出层：动作均值
        self.mean_head = nn.Linear(in_size, act_dim)

        # 可学习的 log 标准差（独立于状态，帮助早期探索）
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # 初始化权重
        # 语法解释：lambda的用法，它的作用是临时写一个函数, lambda 参数: 返回值
        # 如f = lambda x: x + 1, 那么f(3) = 4
        self.apply(lambda m: init_weights(m, gain=0.01))
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)

    def forward(self, state):
        """
        前向传播，返回动作均值。

        参数:
            state: 状态张量 [batch, obs_dim]

        返回:
            mean: 动作均值 [batch, act_dim]
        """
        features = self.feature_net(state)
        mean = self.mean_head(features)
        return mean

    def get_action(self, state, deterministic=False):
        """
        根据状态采样动作。

        参数:
            state: 状态张量 [batch, obs_dim]
            deterministic: True 时返回均值（评估用），False 时采样

        返回:
            action: 动作 [batch, act_dim]
            log_prob: 对数概率 [batch]
        """
        # 输入obs = state, 为11维向量，即观测空间维度
        # 输出mean动作均值
        mean = self.forward(state)
        std = torch.exp(self.log_std)

        # 高斯分布
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        # 计算对数概率（对每个维度求和）
        log_prob = dist.log_prob(action).sum(dim=-1)
        # print("log_prob: ", log_prob, dist.log_prob(action))
        
        # 将动作裁剪到 [-1, 1]（tanh 压缩）
        action = torch.tanh(action)
        # print("action: ", action)

        return action, log_prob

    def evaluate(self, state, action):
        """
        评估给定状态-动作对的对数概率和熵（PPO 更新时使用）。

        参数:
            state: 状态张量 [batch, obs_dim]
            action: 动作张量（原始未压缩的动作） [batch, act_dim]

        返回:
            log_prob: 对数概率 [batch]
            entropy: 熵 [batch]
        """
        mean = self.forward(state)
        std = torch.exp(self.log_std)

        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic（价值网络）：输入状态，输出 V(s)。
    """

    def __init__(self, obs_dim, hidden_sizes=[64, 64]):
        """
        参数:
            obs_dim: 观测维度
            hidden_sizes: 隐藏层大小列表
        """
        super().__init__()

        layers = []
        in_size = obs_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.Tanh())
            in_size = h_size
        self.feature_net = nn.Sequential(*layers)

        # 输出层：标量价值
        self.value_head = nn.Linear(in_size, 1)

        # 初始化
        self.apply(lambda m: init_weights(m, gain=1.0))
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state):
        """
        前向传播，返回状态价值。

        参数:
            state: 状态张量 [batch, obs_dim]

        返回:
            value: 状态价值 [batch, 1]
        """
        features = self.feature_net(state)
        value = self.value_head(features)
        return value

TEST = 0
if TEST:
    print("**********ppo_networks.py : TEST*********")
    from hopper.hopper_env import make_hopper_env, get_env_info
    env = make_hopper_env()
    obs, info = env.reset()
    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
    act = ActorNetwork(11, 3)
    act.get_action(state_tensor)
