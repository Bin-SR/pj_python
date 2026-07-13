# -*- coding: utf-8 -*-
# =============================================================================
# VLA 策略网络模块
# 融合视觉、语言和本体感知信息，输出机器人动作。
#
# 架构:
#   Image  -> VisualEncoder     -> visual_feature (256)
#   Text   -> SimpleTextEncoder -> language_feature (128)
#   State  -> MLP               -> proprio_feature (32)
#              |
#              +--> Concat -> Fusion MLP -> Action Head -> (8,) action
#
# 总参数量: ~500K, 适合 RTX 3050
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.config import (
    VISION_FEATURE_DIM, LANGUAGE_FEATURE_DIM, PROPRIO_FEATURE_DIM,
    FUSION_HIDDEN_DIM, POLICY_HIDDEN_DIM,
    ACTION_DIM, STATE_DIM,
    VOCAB_SIZE, TEXT_EMBED_DIM,
)
from vla.vision import VisualEncoder
from vla.language import SimpleTextEncoder


# ============================================================
# 本体感知编码器
# ============================================================

class ProprioEncoder(nn.Module):
    """将关节状态编码为固定维度特征。

    输入: 9 维 (7 arm joints + 2 gripper joints)
    输出: proprio_feature_dim 维
    """

    def __init__(self,
                 input_dim: int = STATE_DIM,
                 feature_dim: int = PROPRIO_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 9) 关节状态
        Returns:
            (batch, feature_dim)
        """
        return self.net(x)


# ============================================================
# VLA 策略网络
# ============================================================

class VLAPolicy(nn.Module):
    """Vision-Language-Action 策略网络。

    输入:
        - image:        (batch, 3, H, W) RGB
        - token_ids:    (batch, seq_len) 语言 token IDs
        - proprio:      (batch, 9) 关节状态

    输出:
        - action:       (batch, 8) 归一化到 [-1, 1] 的动作
            - [0:7] 手臂关节目标 (归一化)
            - [7]   夹爪执行器 (归一化)
    """

    def __init__(self,
                 vision_feature_dim: int = VISION_FEATURE_DIM,
                 language_feature_dim: int = LANGUAGE_FEATURE_DIM,
                 proprio_feature_dim: int = PROPRIO_FEATURE_DIM,
                 fusion_hidden_dim: int = FUSION_HIDDEN_DIM,
                 policy_hidden_dim: int = POLICY_HIDDEN_DIM,
                 action_dim: int = ACTION_DIM,
                 state_dim: int = STATE_DIM,
                 vocab_size: int = VOCAB_SIZE,
                 text_embed_dim: int = TEXT_EMBED_DIM):
        super().__init__()

        # ---- 子模块 ----
        self.vision_encoder = VisualEncoder(feature_dim=vision_feature_dim)
        self.language_encoder = SimpleTextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            feature_dim=language_feature_dim,
        )
        self.proprio_encoder = ProprioEncoder(
            input_dim=state_dim,
            feature_dim=proprio_feature_dim,
        )

        # ---- 融合模块 ----
        total_feature_dim = (
            vision_feature_dim + language_feature_dim + proprio_feature_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # ---- 动作头 ----
        self.action_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, policy_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(policy_hidden_dim, policy_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(policy_hidden_dim, action_dim),
            nn.Tanh(),  # 输出归一化到 [-1, 1]
        )

        self.action_dim = action_dim

    def forward(self,
                image: torch.Tensor,
                token_ids: torch.Tensor,
                proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:      (batch, 3, H, W) float [0, 1]
            token_ids:  (batch, seq_len) long
            proprio:    (batch, 9) float
        Returns:
            (batch, 8) 归一化动作
        """
        vis_feat = self.vision_encoder(image)
        lang_feat = self.language_encoder(token_ids)
        prop_feat = self.proprio_encoder(proprio)

        fused = torch.cat([vis_feat, lang_feat, prop_feat], dim=1)
        fused = self.fusion(fused)
        action = self.action_head(fused)
        return action

    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        """单独提取视觉特征 (调试用)。"""
        return self.vision_encoder(image)

    def encode_language(self, token_ids: torch.Tensor) -> torch.Tensor:
        """单独提取语言特征 (调试用)。"""
        return self.language_encoder(token_ids)


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    from vla.config import IMAGE_HEIGHT, IMAGE_WIDTH
    from vla.language import create_default_tokenizer

    tokenizer = create_default_tokenizer()
    policy = VLAPolicy()

    # 构造假输入
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    dummy_text = tokenizer.encode_batch([
        "grasp the red cube",
        "pick up the block",
        "grab the red cube",
        "reach for the cube",
    ])
    dummy_proprio = torch.randn(batch_size, 9)

    # 前向传播
    action = policy(dummy_image, dummy_text, dummy_proprio)
    print(f"策略网络: 输入 (img={dummy_image.shape}, text={dummy_text.shape}, prop={dummy_proprio.shape})")
    print(f"          输出 action={action.shape}, 值域 [{action.min().item():.3f}, {action.max().item():.3f}]")
    print(f"参数量: {sum(p.numel() for p in policy.parameters()):,}")
