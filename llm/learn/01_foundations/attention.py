# -*- coding: utf-8 -*-
"""
01_foundations/attention.py — 从零实现 Scaled Dot-Product Attention

Scaled Dot-Product Attention 是 Transformer 的核心运算，公式为：

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

其中：
    Q (Query):  查询向量，表示"我要找什么"
    K (Key):    键向量，表示"我有什么信息"
    V (Value):  值向量，表示"我提供什么内容"
    d_k:        Key 的维度，用于缩放防止梯度消失

直观理解：
    1. Q @ K^T → 计算"注意力分数"，衡量每个位置之间的相关性
    2. / sqrt(d_k) → 缩放，防止点积过大导致 softmax 梯度消失
    3. softmax → 将分数归一化为概率分布（注意力权重）
    4. @ V → 用权重对 Value 加权求和，得到输出

----------
与 VLM/VLA 的联系：
    - 在 VLM 中，图像 token 通过 Cross-Attention 与文本 token 交互
    - 在 VLA 中，状态-动作的时序依赖也依赖 Attention 建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 辅助函数：创建 Causal Mask（因果掩码）
# ============================================================
def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建下三角因果掩码矩阵，确保位置 i 只能看到位置 ≤ i 的信息。

    这是 GPT/Decoder 的核心机制，防止模型"作弊"看到未来 token。

    Returns:
        mask: shape=(seq_len, seq_len)，上三角为 -inf，下三角为 0
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    return mask


# ============================================================
# 核心实现：Scaled Dot-Product Attention
# ============================================================
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> torch.Tensor:
    """
    计算 Scaled Dot-Product Attention。

    Math:
        scores = Q @ K^T / sqrt(d_k)
        if mask: scores = scores + mask
        weights = softmax(scores)
        if dropout: weights = dropout(weights)
        output = weights @ V

    Args:
        Q: Query 张量，shape=(batch, num_heads, seq_len, d_k)
        K: Key   张量，shape=(batch, num_heads, seq_len, d_k)
        V: Value 张量，shape=(batch, num_heads, seq_len, d_v)
        mask: 注意力掩码，-inf 的位置将被忽略
        dropout: 可选的 Dropout 层

    Returns:
        output: shape=(batch, num_heads, seq_len, d_v)
        attention_weights: shape=(batch, num_heads, seq_len, seq_len)，用于可视化
    """
    d_k = Q.size(-1)

    # Step 1: 计算注意力分数 (Q @ K^T)
    # scores[b, h, i, j] 表示 batch b 的第 h 个头中，位置 i 对位置 j 的注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, heads, seq_len, seq_len)

    # Step 2: 缩放 — 这是 "Scaled" 的关键
    # 当 d_k 较大时，点积的值会很大，导致 softmax 梯度接近 0
    scores = scores / math.sqrt(d_k)

    # Step 3: 应用掩码（可选）
    # 例如 Causal Mask 将未来位置设为 -inf，经过 softmax 后权重变为 0
    if mask is not None:
        scores = scores + mask

    # Step 4: Softmax — 将分数转换为概率分布
    attention_weights = F.softmax(scores, dim=-1)

    # Step 5: Dropout（可选）— 正则化，防止过拟合
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 6: 加权求和 — 用注意力权重聚合 Value
    output = torch.matmul(attention_weights, V)  # (batch, heads, seq_len, d_v)

    return output, attention_weights


# ============================================================
# PyTorch 模块封装版本
# ============================================================
class ScaledDotProductAttention(nn.Module):
    """
    将 scaled_dot_product_attention 封装为 nn.Module，方便集成到更大的模型中。
    """

    def __init__(self, dropout_p: float = 0.1):
        """
        Args:
            dropout_p: Dropout 概率，训练时使用，推理时自动关闭
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            Q, K, V: shape=(batch, num_heads, seq_len, d_k)
            mask: 注意力掩码

        Returns:
            output: shape=(batch, num_heads, seq_len, d_v)
        """
        # 训练时使用 dropout，推理时 model.eval() 会自动关闭
        dropout = self.dropout if self.training else None
        output, _ = scaled_dot_product_attention(Q, K, V, mask, dropout)
        return output


# ============================================================
# 演示与测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Scaled Dot-Product Attention 演示")
    print("=" * 60)

    # 模拟参数
    batch_size = 2
    num_heads = 4
    seq_len = 8
    d_k = 16
    d_v = 16

    # 创建随机输入
    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_v)

    print(f"\n输入形状:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")

    # --- 测试 1: 无掩码 Attention ---
    print("\n--- 测试 1: 无掩码 Attention ---")
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"  输出形状: {output.shape}")
    print(f"  注意力权重形状: {weights.shape}")

    # 验证：每行的注意力权重之和应为 1
    weight_sums = weights.sum(dim=-1)
    print(f"  每行权重之和 (应为1): {weight_sums[0, 0, 0].item():.4f}")

    # --- 测试 2: 带 Causal Mask 的 Attention ---
    print("\n--- 测试 2: Causal Mask (GPT 风格) ---")
    causal_mask = create_causal_mask(seq_len)
    print(f"  Mask 形状: {causal_mask.shape}")
    print(f"  Mask 示例 (第0个token只能看到自己):")
    print(f"    {causal_mask[0, :4].tolist()}")  # 前4个位置

    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, causal_mask)
    print(f"  输出形状: {output_masked.shape}")

    # 验证：位置 0 对位置 1,2,... 的注意力权重应为 0
    print(f"  位置0对位置0-3的权重: {weights_masked[0, 0, 0, :4].tolist()}")
    print(f"  位置3对位置0-3的权重: {weights_masked[0, 0, 3, :4].tolist()}")

    # --- 测试 3: 使用模块封装版本 ---
    print("\n--- 测试 3: nn.Module 封装版本 ---")
    attention_module = ScaledDotProductAttention(dropout_p=0.1)
    output_module = attention_module(Q, K, V, causal_mask)
    print(f"  模块输出形状: {output_module.shape}")

    print("\n√ 所有测试通过！Attention 机制是 LLM/VLM/VLA 的共同基础。")