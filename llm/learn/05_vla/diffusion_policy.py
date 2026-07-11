# -*- coding: utf-8 -*-
"""
05_vla/diffusion_policy.py - Simple Diffusion Policy
Generates robot actions by iteratively denoising random noise.
"""

import torch
import torch.nn as nn
import math


class SimpleDiffusionPolicy(nn.Module):
    """Simplified Diffusion Policy for learning."""
    def __init__(self, obs_dim=10, action_dim=7, hidden_dim=256, num_timesteps=100):
        super().__init__()
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.register_buffer('betas', self._cosine_schedule(num_timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))

    def _cosine_schedule(self, T, s=0.008):
        steps = T + 1
        x = torch.linspace(0, T, steps)
        ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        return torch.clip(1 - (ac[1:] / ac[:-1]), 0.0001, 0.02)

    def forward(self, x_noisy, timestep, condition):
        t = timestep.unsqueeze(-1).float() / self.num_timesteps
        return self.net(torch.cat([condition, x_noisy, t], dim=-1))

    @torch.no_grad()
    def sample(self, condition, num_samples=1):
        x = torch.randn(num_samples, self.action_dim, device=condition.device)
        condition = condition.expand(num_samples, -1)
        for t in reversed(range(self.num_timesteps)):
            tb = torch.full((num_samples,), t, device=condition.device)
            noise_pred = self(x, tb, condition)
            alpha = 1 - self.betas[t]
            ac = self.alphas_cumprod[t]
            x = (x - (self.betas[t] / torch.sqrt(1-ac)) * noise_pred) / torch.sqrt(alpha)
            if t > 0:
                x = x + torch.sqrt(self.betas[t]) * torch.randn_like(x)
        return x


if __name__ == '__main__':
    print('=' * 60)
    print('Diffusion Policy Demo')
    print('=' * 60)
    policy = SimpleDiffusionPolicy(obs_dim=10, action_dim=7, num_timesteps=50)
    print(f'Params: {sum(p.numel() for p in policy.parameters()):,}')
    cond = torch.randn(4, 10)
    actions = policy.sample(cond[0:1])
    print(f'Sampled: {cond.shape} -> {actions.shape}')
    print('Done!')