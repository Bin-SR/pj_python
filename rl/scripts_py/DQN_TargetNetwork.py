# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt

from collections import deque
import random

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


class QNEtwork(nn.Module):
    def __init__(self):
        super(QNEtwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state, info = env.reset()

    buffer = deque(maxlen=10000)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    gamma = 0.99
    epsilon = 1.0
    epoches = 500
    batch_size = 64
    
    tau = 0.005

    net = QNEtwork()
    target_net = QNEtwork()
    target_net.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr = 0.0008)  # lr = 0.0004

    for epoch in range(epoches):
        state, info = env.reset()
        done = False
        epoch_reward = 0
        loss = 0
        loss_list = []
        epoch_reward_list = []

        while not done:
            if np.random.randn() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor_tmp = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_val = net(state_tensor_tmp)
                action = q_val.argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            buffer.append((state, action, reward, next_state, done))

            state = next_state
            epoch_reward += reward

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                state_tensor = torch.FloatTensor(np.array(states))
                next_states_tensor = torch.FloatTensor(np.array(next_states))
                actions_tensor = torch.LongTensor(actions)
                rewards_tensor = torch.FloatTensor(rewards)
                dones_tensor = torch.FloatTensor(dones)
                # print(state_tensor.shape, actions_tensor.shape, rewards_tensor.shape, next_states_tensor.shape, dones_tensor.shape)
                
                # q_val.shape = (batch_size, output_channel)，eg. (64, 2)
                q_val = net(state_tensor)
                # print("q_val max:", q_val.max())
                # print("q_val min:", q_val.min())

                # 这里的gather: 在第一维度，按actions_tensor给出的索引来取值
                # actions_tensor内部都是0或者1，表示left或者right
                q_cur = q_val.gather(1, actions_tensor.unsqueeze(1))
                # print("q_cur max:", q_cur.max())
                # print("q_cur min:", q_cur.min())
            
                with torch.no_grad():
                    # q_next = net(next_states_tensor)

                    # 换成target_net() q_next的预测用target_net完成，这样在计算q_tar时不会立即变化
                    q_next = target_net(next_states_tensor)
                    q_next_max = q_next.max(dim=1)[0]

                q_tar = rewards_tensor + (1 - dones_tensor) * gamma * q_next_max
                # print("q_tar min:", q_tar.max())
                # print("q_tar min:", q_tar.min())
                loss = F.mse_loss(q_cur.squeeze(), q_tar)
                opt.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    10.0
                )
                opt.step()
                loss_list.append(loss.detach().numpy())
                epoch_reward_list.append(epoch_reward)
            print(f"loss: {loss}")
            print(f"epoch: {epoch}, epsilon: {epsilon}, epoch_reward: {epoch_reward}\n")
        
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_((1- tau) * target_param.data + tau * param.data)

        if epoch % 10 == 0:
            target_net.load_state_dict(net.state_dict())
            
        if len(epoch_reward_list) >= 60:
            plt.subplot(1, 2, 1)
            plt.plot(loss_list)
            plt.title("loss")

            plt.subplot(1, 2, 2)
            plt.plot(epoch_reward_list)
            plt.title("reward")
            plt.show()
        epsilon = max(0.01, epsilon * 0.995)

