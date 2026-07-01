import torch
import torch.nn as nn
from torch.distributions import Categorical

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
    
class PolicyNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(4,128),
            nn.ReLU(),

            nn.Linear(128,128),
            nn.ReLU(),

            nn.Linear(128,2)
        )

    def forward(self,x):

        logits = self.net(x)

        probs = torch.softmax(logits, dim=-1)

        return probs

class ValueNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4,128),
            nn.ReLU(),

            nn.Linear(128,128),
            nn.ReLU(),

            nn.Linear(128,1)
        )

    def forward(self,x):
        return self.net(x)

# state = torch.randn(1,4)

# net = ValueNetwork()

# v = net(state)

# print(v)
# print(v.shape)

# QNET输出Q值
# PolicyNetwork输出的是动作的概率prob    
# pnet = PolicyNetwork()
# data = torch.randn(1, 4)
# state = data

# out = pnet(data)
# prob = pnet(state)
# print(out, out.shape, out.sum())

# 按概率抽样
# dist = Categorical(probs=prob)
# for i in range(10):
#     action = dist.sample()
#     print(action)

# 记录log_probs, 它是策略梯度更新的核心训练信号
# 在policy gradient用于计算loss = -log_prob * G
# dist = Categorical(probs=prob)
# action = dist.sample()
# log_prob = dist.log_prob(action)

# print("probs: ", prob)
# print("action:", action)
# print("log_prob:", log_prob)

