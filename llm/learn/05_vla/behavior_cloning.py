# -*- coding: utf-8 -*-
"""
05_vla/behavior_cloning.py - Behavior Cloning for VLA
Trains VLA by imitating expert demonstrations (supervised learning).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from action_head import ActionHead


class DemoDataset(Dataset):
    """Dataset of (observation, action) pairs."""
    def __init__(self, obs, acts):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.float32)
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]


class BCTrainer:
    """Behavior Cloning trainer."""
    def __init__(self, model, device='cpu', lr=3e-4):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
    def train_step(self, obs, acts):
        self.model.train()
        obs, acts = obs.to(self.device), acts.to(self.device)
        loss = F.mse_loss(self.model(obs), acts)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
    def train(self, loader, epochs=10):
        for e in range(epochs):
            total = sum(self.train_step(o, a) for o, a in loader)
            if e % 2 == 0 or e == epochs - 1:
                print(f'  Epoch {e+1}/{epochs}, Loss: {total/len(loader):.6f}')


class SimplePolicy(nn.Module):
    """Simple MLP policy for BC."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    print('=' * 60)
    print('Behavior Cloning Demo')
    print('=' * 60)
    obs_dim, act_dim, N = 10, 3, 500
    X = torch.randn(N, obs_dim)
    Y = torch.sin(X[:,:3]).sum(dim=1, keepdim=True).repeat(1, act_dim)
    ds = DemoDataset(X.numpy(), Y.numpy())
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    policy = SimplePolicy(obs_dim, act_dim)
    trainer = BCTrainer(policy)
    trainer.train(dl, epochs=10)
    with torch.no_grad():
        pred = policy(X[:3])
        print(f'MSE: {F.mse_loss(pred, Y[:3]).item():.6f}')
    print('Done!')