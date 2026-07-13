# -*- coding: utf-8 -*-
# =============================================================================
# VLA 数据集模块
# 将演示数据封装为 PyTorch Dataset，供训练使用。
# =============================================================================

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from vla.config import (
    IMAGE_WIDTH, IMAGE_HEIGHT,
    N_ARM_JOINTS, JOINT_RANGES,
    DEMO_DIR,
)
from vla.language import SimpleTokenizer, create_default_tokenizer


class VLADataset(Dataset):
    """VLA 演示数据集。

    每个样本:
        - image:        (3, H, W) float32 [0, 1]
        - token_ids:    (seq_len,) long
        - proprio:      (9,) float32 关节状态
        - action:       (8,) float32 目标动作 (归一化)
    """

    def __init__(self,
                 samples: list,
                 tokenizer: SimpleTokenizer = None):
        """
        Args:
            samples: 演示样本列表
            tokenizer: 语言分词器
        """
        self.samples = samples
        self.tokenizer = tokenizer or create_default_tokenizer()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ---- 图像 ----
        image = sample["image"]
        image = cv2_resize(image, IMAGE_WIDTH, IMAGE_HEIGHT)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # ---- 语言 ----
        instruction = sample.get("instruction", "grasp the red cube")
        token_ids = self.tokenizer.encode(instruction)

        # ---- 本体感知 (9维: 7 arm + 2 gripper) ----
        arm_qpos = sample["arm_qpos"].astype(np.float32)
        gripper_qpos = sample["gripper_qpos"].astype(np.float32)
        proprio = np.concatenate([arm_qpos, gripper_qpos])

        # ---- 动作 (归一化到 [-1, 1]) ----
        action_raw = sample["action"].astype(np.float32)  # (8,)
        action = np.zeros(8, dtype=np.float32)

        # 手臂关节归一化
        for i in range(N_ARM_JOINTS):
            lo, hi = JOINT_RANGES[i]
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            action[i] = (action_raw[i] - mid) / half

        # 夹爪归一化: [0, 255] -> [-1, 1]
        action[7] = (action_raw[7] / 255.0) * 2.0 - 1.0

        return {
            "image": torch.from_numpy(image),
            "token_ids": token_ids.clone(),
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action),
        }


def cv2_resize(image: np.ndarray, w: int, h: int) -> np.ndarray:
    """OpenCV resize (避免依赖 torchvision)。"""
    import cv2
    return cv2.resize(image, (w, h))


def load_demo_data(filename: str = "demo_data.pkl") -> list:
    """从磁盘加载演示数据。"""
    path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到演示数据: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def create_dataloaders(samples: list,
                       batch_size: int = 32,
                       val_split: float = 0.1,
                       num_workers: int = 0,
                       tokenizer: SimpleTokenizer = None):
    """创建训练和验证 DataLoader。

    Args:
        samples: 所有演示样本
        batch_size: 批大小
        val_split: 验证集比例
        num_workers: 数据加载线程数
        tokenizer: 分词器
    Returns:
        (train_loader, val_loader)
    """
    dataset = VLADataset(samples, tokenizer)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    # 用假数据测试
    dummy_samples = []
    for i in range(100):
        dummy_samples.append({
            "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "arm_qpos": np.zeros(7, dtype=np.float32),
            "gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
            "gripper_ctrl": np.array([0.0], dtype=np.float32),
            "action": np.zeros(8, dtype=np.float32),
            "instruction": "grasp the red cube",
            "cube_pos": np.array([0.55, 0.0, 0.03], dtype=np.float32),
        })

    train_loader, val_loader = create_dataloaders(dummy_samples, batch_size=8)
    batch = next(iter(train_loader))
    print(f"批次键: {list(batch.keys())}")
    print(f"图像: {batch['image'].shape}")
    print(f"token_ids: {batch['token_ids'].shape}")
    print(f"本体感知: {batch['proprio'].shape}")
    print(f"动作: {batch['action'].shape}")
