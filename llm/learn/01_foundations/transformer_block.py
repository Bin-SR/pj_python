# -*- coding: utf-8 -*-
'''
01_foundations/transformer_block.py - Transformer Block (Pre-Norm)

Structure:
    x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> + -> output

Includes both standard Self-Attention blocks and Cross-Attention blocks
(for VLM/VLA multimodal fusion).
'''

import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):
    '''Position-wise FFN: GELU(x @ W1 + b1) @ W2 + b2. Expansion ratio: 4x.'''

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout_p: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    '''Standard Transformer Block with Pre-Norm and Residual Connections.'''

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout_p=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
        # Self-Attention + Residual
        x = x + self.dropout(self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # FFN + Residual
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    '''Transformer Block with Cross-Attention for multimodal fusion (VLM/VLA).

    Used when text tokens need to attend to image/observation features.
    '''

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout_p=0.1):
        super().__init__()
        # 和TransformerBlock比较，这里实例化了两个MultiHeadAttention
        # self_attention和cross_attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_p)

        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, context, self_mask=None, cross_mask=None):
        # Self-Attention
        x = x + self.dropout(self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), self_mask))

        # Cross-Attention: Q from x, K/V from context (e.g., image features)
        x = x + self.dropout(self.cross_attention(self.norm2(x), context, context, cross_mask))

        # FFN
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


if __name__ == '__main__':
    print('=' * 60)
    print('Transformer Block Demo')
    print('=' * 60)
    batch, seq, d_model = 2, 10, 512

    x = torch.randn(batch, seq, d_model)

    # Standard block
    block = TransformerBlock(d_model=d_model, num_heads=8)
    out = block(x)
    print(f'Standard:   {x.shape} -> {out.shape}')
    print(f'  Params:   {sum(p.numel() for p in block.parameters()):,}') # 计算总参数量

    # Cross-attention block
    cross = CrossAttentionBlock(d_model=d_model, num_heads=8)
    img = torch.randn(batch, 5, d_model)  # 随机生成图像数据
    cout = cross(x, img)
    print(f'Cross-Attn: text {x.shape} + img {img.shape} -> {cout.shape}')

    # Stack 4 layers (MiniGPT style)，堆叠4层TransformerBlock模块
    layers = nn.ModuleList([TransformerBlock(d_model, 8) for _ in range(4)])
    h = x
    for layer in layers:
        h = layer(h)
    print(f'4-L stack:  {h.shape}')
    total = sum(p.numel() for p in layers.parameters())  # 计算总参数量
    print(f'  Params: {total:,} (~{total/1e6:.1f}M)')
    print('Done!')
