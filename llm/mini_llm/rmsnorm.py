import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.eps = eps

    def forward(self, x):

        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        x = x / rms

        return x * self.weight
    
Test = 1
if Test == 1:
    x = torch.randn(2,8,256)
    norm = RMSNorm(256)

    y = norm(x)

    print(x.shape)
    print(y.shape)