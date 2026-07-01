from net import PolicyNetwork, ValueNetwork

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

state, info = env.reset()

gamma = 0.99
epsilon = 1.0
epoches = 100
batch_size = 64

policyNet = PolicyNetwork()
opt = optim.Adam(policyNet.parameters(), lr=0.001)


if __name__ == "__main__":
    
    for epoch in range(epoches):
        state, info = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        done = False
        loss = 0
        G = 0

        saved_log_prob = []
        rewards_list = []
        returns_list = []
        while not done:
            probs = policyNet(state_tensor)
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action.item())

            done = terminated or truncated

            # 记录log_prob和reward
            saved_log_prob.append(log_prob)
            rewards_list.append(reward)

        for reward in reversed(rewards_list):
            G = reward + gamma * G
            returns_list.append(G)

        returns_tensor = torch.FloatTensor(returns_list)
        returns_list.reverse()
        # 防止G特别大，给returns进行标准化
        returns_list = torch.tensor(returns_list, dtype=torch.float32)
        returns_list = (returns_list - returns_list.mean()) / (returns_list.std() + 1e-8)

        for log_prob, G in zip(saved_log_prob, returns_list):
            loss += -log_prob * G
        loss = loss.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # loss并不是很重要，loss有正有负是正常现象
        # 在policy训练过程中重点关注每个epoch的reward
        print("epoch: ", epoch, " loss: ", loss.item())
        # print("reward: ", rewards_list)
        # print(len(saved_log_prob))
        # print(len(rewards_list))

        print("sum_reward: ", sum(rewards_list))
    print("return_shape: ", returns_list.shape)
    print("first_returns: ", returns_list[0])
    print("list_returns: ", returns_list[-1])

    