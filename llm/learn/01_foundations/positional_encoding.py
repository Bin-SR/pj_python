# -*- coding: utf-8 -*-
'''
01_foundations/positional_encoding.py - Positional Encoding

Attention has no built-in sense of order, so we inject position info.

Two approaches:
  1. Sinusoidal (Original Transformer): Fixed frequencies, extrapolates well
  2. Learned (GPT-2/3 style): Trainable embedding, more flexible

VLA connection: PE helps model temporal order of actions/observations.
'''

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    '''Sinusoidal PE from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    '''

    def __init__(self, d_model: int = 512, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # x[行, 列]  :表示取所有行,  start : end : step
        pe[:, 1::2] = torch.cos(position * div_term)  # 1::2 表示从第一列开始到最后一列结束，每隔2列取一次, 即1,3,5,7,...
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # 注册为缓冲区, 把张量注册为模型状态的一部分, 但不参与训练
        self.register_buffer('pe', pe)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)'''
        return x + self.pe[:, :x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    '''Learned PE (GPT style). Trainable embedding table.'''

    def __init__(self, d_model: int = 512, max_len: int = 1024):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pe(positions)


if __name__ == '__main__':
    print('=' * 60)
    print('Positional Encoding Demo')
    print('=' * 60)
    batch, seq_len, d_model = 2, 16, 512
    x = torch.randn(batch, seq_len, d_model)

    sin_pe = SinusoidalPositionalEncoding(d_model)
    print(f'Sinusoidal: {x.shape} -> {sin_pe(x).shape}')

    learned_pe = LearnedPositionalEncoding(d_model)
    print(f'Learned:    {x.shape} -> {learned_pe(x).shape}')
    n_params = sum(p.numel() for p in learned_pe.parameters())
    print(f'  Learnable params: {n_params:,}')
    print('Done!')
