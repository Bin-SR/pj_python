# -*- coding: utf-8 -*-
# =============================================================================
# VLA 视觉感知模块
# 包含:
#   1. RedCubeDetector  —— 基于 HSV 颜色空间检测红色方块 (经典 CV)
#   2. VisualEncoder     —— 轻量 CNN 提取视觉特征 (PyTorch)
# =============================================================================

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS,
    CUBE_COLOR_LOWER, CUBE_COLOR_UPPER,
    VISION_FEATURE_DIM,
)


# ============================================================
# 经典 CV：红色方块检测器
# ============================================================

class RedCubeDetector:
    """使用 HSV 颜色阈值检测场景中的红色方块。

    适合在仿真环境中快速定位目标物体，无需训练。
    """

    def __init__(self,
                 lower: tuple = CUBE_COLOR_LOWER,
                 upper: tuple = CUBE_COLOR_UPPER,
                 min_area: int = 50):
        """
        Args:
            lower: HSV 下界 (H, S, V)
            upper: HSV 上界 (H, S, V)
            min_area: 最小轮廓面积 (过滤噪点)
        """
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> tuple:
        """在 RGB 图像中检测红色方块。

        Args:
            image: (H, W, 3) uint8 RGB 图像
        Returns:
            (center_xy, bbox) 或 (None, None)
                - center_xy: (cx, cy) 像素坐标
                - bbox: (x, y, w, h) 边界框
        """
        # RGB → HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 因为红色在 HSV 中跨越 0 度，需要处理两端
        # 下限 0-10, 上限 170-180
        mask1 = cv2.inRange(hsv, self.lower, self.upper)

        lower2 = np.array([170, self.lower[1], self.lower[2]], dtype=np.uint8)
        upper2 = np.array([180, self.upper[1], self.upper[2]], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学去噪
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # 取最大轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return None, None

        x, y, w, h = cv2.boundingRect(largest)
        cx, cy = x + w // 2, y + h // 2
        return (cx, cy), (x, y, w, h)

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """返回红色区域的二值掩码 (用于可视化调试)。"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        lower2 = np.array([170, self.lower[1], self.lower[2]], dtype=np.uint8)
        upper2 = np.array([180, self.upper[1], self.upper[2]], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask1, mask2)


# ============================================================
# 深度学习：轻量视觉编码器 (适配 RTX 3050)
# ============================================================

class VisualEncoder(nn.Module):
    """轻量 CNN 视觉编码器。

    将 RGB 图像编码为固定维度的特征向量。

    架构:
        Conv2d(3, 16, 7, 2) -> Conv2d(16, 32, 5, 2) ->
        Conv2d(32, 64, 3, 2) -> Conv2d(64, 128, 3, 1) ->
        AdaptiveAvgPool2d(4) -> Flatten -> Linear -> output

    参数量: ~150K, 适合 RTX 3050。
    """

    def __init__(self, feature_dim: int = VISION_FEATURE_DIM):
        super().__init__()
        self.feature_dim = feature_dim

        # 卷积层
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # 全局池化 + 全连接
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) float32, 值域 [0, 1]
        Returns:
            (batch, feature_dim) 视觉特征
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# 图像预处理工具
# ============================================================

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """将 numpy 图像转为模型输入 Tensor。

    Args:
        image: (H, W, 3) uint8 RGB
    Returns:
        (1, 3, H, W) float32 Tensor, 值域 [0, 1]
    """
    img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0)


def preprocess_image_batch(images: np.ndarray) -> torch.Tensor:
    """批量图像预处理。

    Args:
        images: (batch, H, W, 3) uint8
    Returns:
        (batch, 3, H, W) float32
    """
    batch = []
    for img in images:
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        batch.append(img)
    return torch.from_numpy(np.stack(batch, axis=0))


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    # 测试 VisualEncoder
    encoder = VisualEncoder()
    dummy = torch.randn(2, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    features = encoder(dummy)
    print(f"VisualEncoder: 输入 {dummy.shape} -> 输出 {features.shape}")
    print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")
