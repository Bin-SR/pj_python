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
lambda_ = 0.95 # GAE的超参数，衰减因子

policyNet = PolicyNetwork()
valueNet = ValueNetwork()
actor_opt = optim.Adam(policyNet.parameters(), lr=0.01)
critic_opt = optim.Adam(valueNet.parameters(), lr=0.01)


if __name__ == "__main__":
    
    for epoch in range(epoches):
        print("epoch: ", epoch)
        state, info = env.reset()

        done = False
        actor_loss = 0
        critic_loss = 0
        G = 0

        saved_log_prob = []
        saved_state = []
        saved_next_state = []
        saved_done = []
        rewards_list = []
        returns_list = []
        while not done:
            state_tensor_tmp = torch.FloatTensor(state).unsqueeze(0)
            probs = policyNet(state_tensor_tmp)
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action.item())

            done = terminated or truncated

            # 记录log_prob和reward
            saved_log_prob.append(log_prob)
            saved_state.append(state)
            saved_done.append(done)
            saved_next_state.append(next_state)
            rewards_list.append(reward)

            state = next_state

        for reward in reversed(rewards_list):
            G = reward + gamma * G
            returns_list.append(G)

        returns_list.reverse()
        # 防止G特别大，给returns进行标准化
        returns_list = torch.tensor(returns_list, dtype=torch.float32)
        # returns_list = (returns_list - returns_list.mean()) / (returns_list.std() + 1e-8)

        state_tensor = torch.FloatTensor(np.array(saved_state))
        done_tensor = torch.FloatTensor(np.array(saved_done))
        next_state_tensor = torch.FloatTensor(np.array(saved_next_state))
        returns_tensor = torch.FloatTensor(returns_list)
        rewards_tensor = torch.FloatTensor(rewards_list)

        # critic: ValueNet
        value_pred = valueNet(state_tensor)
        value_pred = value_pred.squeeze()

        # 计算td_target, 用于替换rewards_tensor
        # critic_loss = F.mse_loss(value_pred, returns_tensor)中的returns_tensor换成计算td_target
        # advantage = returns_tensor - value_pred.detach()中的returns_tensor换成计算td_target
        with torch.no_grad():
            next_value = valueNet(next_state_tensor).squeeze()
        td_target = rewards_tensor + gamma * next_value * (1 - done_tensor)

        critic_loss = F.mse_loss(value_pred, td_target)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        advantage = td_target - value_pred.detach()

        for log_prob, adv in zip(saved_log_prob, advantage):
            actor_loss += -log_prob * adv
        actor_loss = actor_loss.sum()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        print("critic_loss:         ", critic_loss.item())
        print("value_pred.shape:    ", value_pred.shape)
        print("returns_tensor.shape:", returns_tensor.shape)
        print("td_target.shape:     ", td_target.shape)
        print("value_pred:          ", value_pred[:5])
        print("returns_tensor:      ", returns_tensor[:5])
        print("td_target:           ", td_target[:5])
        
        # loss并不是很重要，loss有正有负是正常现象
        # 在policy训练过程中重点关注每个epoch的reward
        # print("epoch: ", epoch, " loss: ", loss.item())
        # print("reward: ", rewards_list)
        # print(len(saved_log_prob))
        # print(len(rewards_list))

        # print("sum_reward: ", sum(rewards_list))
        # print("first_returns:     ", returns_tensor[0])
        # print("last_returns:      ", returns_tensor[-1], "\n")
        print("\n")

    