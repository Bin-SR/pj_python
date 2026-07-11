# -*- coding: utf-8 -*-
"""
01_foundations/multi_head_attention.py — Multi-Head Attention 实现

Multi-Head Attention 将 Q, K, V 投影到多个"头"，让模型从不同的表示子空间
同时关注信息。公式为：

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

其中每个 head_i = Attention(Q @ W_Qi, K @ W_Ki, V @ W_Vi)

----------
关键设计思想：
    1. 多个头 → 捕捉不同类型的关系（语法、语义、位置等）
    2. 每个头维度更小(d_k = d_model / num_heads) → 总计算量与单头相近
    3. 拼接后线性投影 → 融合多头信息

与具身智能的联系：
    在 VLA 中，Cross-Attention 头可以让模型同时关注图像特征（"看到什么"）
    和文本指令（"要做什么"），从而生成正确的动作。
"""

import torch
import torch.nn as nn
import math
from attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 模块。

    支持两种模式：
    - Self-Attention:  Q, K, V 来自同一输入（Encoder/Decoder 内部使用）
    - Cross-Attention: Q 来自当前层，K, V 来自另一来源（如 VLM 中图像特征）

    架构图：
        Input ──┬──> Linear_Q ──> split heads ──┐
                ├──> Linear_K ──> split heads ──┤──> ScaledDotProductAttention
                └──> Linear_V ──> split heads ──┘          │
                                                           ▼
                                             concat heads ──> Linear_O ──> Output
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout_p: float = 0.1):
        """
        Args:
            d_model:   模型总维度（必须能被 num_heads 整除）
            num_heads: 注意力头的数量
            dropout_p: Dropout 概率
        """
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性投影层：将 d_model 维输入投影到 d_model 维输出
        # 实际计算时会拆分为 num_heads 个 d_k 维的头
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)  # 输出投影

        self.dropout = nn.Dropout(dropout_p)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (batch, seq_len, d_model) 拆分为 (batch, num_heads, seq_len, d_k)

        Args:
            x: shape=(batch, seq_len, d_model)

        Returns:
            shape=(batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        # 重塑: (batch, seq_len, d_model) -> (batch, seq_len, heads, d_k) -> (batch, heads, seq_len, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (batch, num_heads, seq_len, d_k) 合并为 (batch, seq_len, d_model)

        Args:
            x: shape=(batch, num_heads, seq_len, d_k)

        Returns:
            shape=(batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        # 转置+重塑: (batch, heads, seq_len, d_k) -> (batch, seq_len, heads, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            query: shape=(batch, seq_len_q, d_model)
            key:   shape=(batch, seq_len_kv, d_model)
            value: shape=(batch, seq_len_kv, d_model)
            mask:  注意力掩码

        Returns:
            output: shape=(batch, seq_len_q, d_model)
        """
        # 1. 线性投影 + 拆分头
        Q = self._split_heads(self.W_Q(query))  # (batch, heads, seq_len_q, d_k)
        K = self._split_heads(self.W_K(key))    # (batch, heads, seq_len_kv, d_k)
        V = self._split_heads(self.W_V(value))  # (batch, heads, seq_len_kv, d_k)

        # 2. 计算 Scaled Dot-Product Attention
        dropout = self.dropout if self.training else None
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask, dropout)

        # 3. 合并头 + 输出投影
        output = self._merge_heads(attn_output)  # (batch, seq_len_q, d_model)
        output = self.W_O(output)

        return output


# ============================================================
# 演示代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Head Attention 演示")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # 创建模块
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    print(f"\n模型参数:")
    print(f"  d_model={d_model}, num_heads={num_heads}, d_k={d_model // num_heads}")

    # 统计参数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"  总参数量: {total_params:,}")  # 这里的总参数量，就是查看模型的参数量规模

    # --- Self-Attention: Q, K, V 都来自同一输入 ---
    print("\n--- Self-Attention ---")
    x = torch.randn(batch_size, seq_len, d_model)
    output = mha(x, x, x)  # Q=K=V=x
    print(f"  输入: {x.shape} -> 输出: {output.shape}")

    # --- Cross-Attention: Q 与 K, V 来源不同 ---
    print("\n--- Cross-Attention (VLM 风格) ---")
    text_features = torch.randn(batch_size, seq_len, d_model)    # 文本 token
    image_features = torch.randn(batch_size, 5, d_model)          # 5 个图像 token
    # 文本查询图像: Q=text, K=V=img
    cross_output = mha(text_features, image_features, image_features)
    print(f"  文本({text_features.shape}) 关注 图像({image_features.shape})")
    print(f"  输出: {cross_output.shape}")

    # --- Causal Self-Attention (GPT 风格) ---
    print("\n--- Causal Self-Attention (GPT 风格) ---")
    from attention import create_causal_mask
    causal_mask = create_causal_mask(seq_len)
    causal_output = mha(x, x, x, mask=causal_mask)
    print(f"  带 Causal Mask 输出: {causal_output.shape}")

    print("\n√ Multi-Head Attention 工作正常！")
    print("  这是 Transformer、GPT、VLM、VLA 的共同核心组件。")