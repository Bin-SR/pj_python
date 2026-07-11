# -*- coding: utf-8 -*-
"""
05_vla/action_head.py - Action Head for VLA Models
Maps VLM hidden states to robot actions.
Supports: continuous regression, discretized bins, action chunks (ACT).
"""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """MLP that maps hidden states to continuous action vectors."""
    def __init__(self, hidden_dim=256, action_dim=7, hidden_layers=None, dropout_p=0.1):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 256]
        layers = []
        in_dim = hidden_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, hidden_states):
        return self.mlp(hidden_states[:, -1, :])


class DiscretizedActionHead(nn.Module):
    """Discretize each action dim into bins (RT-2 style)."""
    def __init__(self, hidden_dim=256, action_dim=7, num_bins=256):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, num_bins) for _ in range(action_dim)])
    def forward(self, hidden_states):
        return torch.stack([h(hidden_states[:,-1,:]) for h in self.heads], dim=1)
    def decode(self, logits, lo=-1.0, hi=1.0):
        idx = torch.argmax(logits, dim=-1).float()
        return lo + (idx + 0.5) * (hi - lo) / self.num_bins


class ActionChunkHead(nn.Module):
    """Predict chunk of future actions (ACT-style)."""
    def __init__(self, hidden_dim=256, action_dim=7, chunk_size=10, hidden_layers=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 512]
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        layers = []
        in_dim = hidden_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, hidden_states):
        return self.mlp(hidden_states[:, -1, :]).view(-1, self.chunk_size, self.action_dim)


if __name__ == '__main__':
    print('=' * 60)
    print('Action Head Demo')
    print('=' * 60)
    h = torch.randn(4, 16, 256)
    head = ActionHead(256, 7)
    a = head(h)
    print(f'Continuous: {h.shape} -> {a.shape}')
    dhead = DiscretizedActionHead(256, 7)
    dl = dhead(h)
    print(f'Discrete: {h.shape} -> {dl.shape}')
    chead = ActionChunkHead(256, 7, 10)
    ca = chead(h)
    print(f'Chunk: {h.shape} -> {ca.shape}')
    print('Done!')