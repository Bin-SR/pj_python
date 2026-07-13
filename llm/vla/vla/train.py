# -*- coding: utf-8 -*-
# =============================================================================
# VLA 训练模块
# 使用行为克隆 (Behavioral Cloning) 训练 VLA 策略网络。
#
# 训练流程:
#   1. 收集演示数据 (或加载已有数据)
#   2. MSE 损失: 最小化预测动作与演示动作的差异
#   3. 保存最佳模型
# =============================================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from vla.config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, VAL_SPLIT,
    GRAD_CLIP,
    LR_SCHEDULER_STEP, LR_SCHEDULER_GAMMA,
    DEVICE, MODEL_DIR, ensure_dirs,
    ACTION_DIM,
)
from vla.policy import VLAPolicy
from vla.dataset import VLADataset, create_dataloaders, load_demo_data
from vla.language import create_default_tokenizer


# ============================================================
# 训练器
# ============================================================

class VLATrainer:
    """VLA 策略网络训练器。"""

    def __init__(self,
                 policy: VLAPolicy = None,
                 device: str = DEVICE,
                 lr: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY):
        """
        Args:
            policy: VLA 策略网络 (None 则自动创建)
            device: 训练设备
            lr: 学习率
            weight_decay: 权重衰减
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        print(f"[训练器] 使用设备: {self.device}")

        self.policy = (policy or VLAPolicy()).to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = StepLR(
            self.optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA
        )
        self.criterion = nn.MSELoss()

        # 训练历史
        self.train_losses = []
        self.val_losses = []

    def train(self,
              train_loader,
              val_loader=None,
              epochs: int = NUM_EPOCHS,
              save_best: bool = True,
              model_name: str = "vla_policy.pt"):
        """训练策略网络。

        Args:
            train_loader: 训练 DataLoader
            val_loader: 验证 DataLoader (可选)
            epochs: 训练轮数
            save_best: 是否保存最佳模型
            model_name: 模型文件名
        Returns:
            训练历史字典
        """
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # ---- 训练 ----
            self.policy.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                images = batch["image"].to(self.device)
                token_ids = batch["token_ids"].to(self.device)
                proprio = batch["proprio"].to(self.device)
                targets = batch["action"].to(self.device)

                # 前向传播
                predictions = self.policy(images, token_ids, proprio)
                loss = self.criterion(predictions, targets)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP)
                self.optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # ---- 验证 ----
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.val_losses.append(val_loss)

            # 学习率调度
            self.scheduler.step()

            # 日志
            log_str = f"Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.6f}"
            if val_loss is not None:
                log_str += f" | val_loss={val_loss:.6f}"
            log_str += f" | lr={self.scheduler.get_last_lr()[0]:.2e}"
            print(log_str)

            # 保存最佳模型
            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model_name)
                print(f"  -> 保存最佳模型 (val_loss={best_val_loss:.6f})")

        # 如果没有验证集，保存最终模型
        if save_best and val_loader is None:
            self.save_model(model_name)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def _validate(self, val_loader) -> float:
        """验证循环。"""
        self.policy.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                token_ids = batch["token_ids"].to(self.device)
                proprio = batch["proprio"].to(self.device)
                targets = batch["action"].to(self.device)

                predictions = self.policy(images, token_ids, proprio)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_model(self, filename: str):
        """保存模型权重。"""
        ensure_dirs()
        path = os.path.join(MODEL_DIR, filename)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, path)
        print(f"[保存] 模型已保存至: {path}")

    def load_model(self, filename: str):
        """加载模型权重。"""
        path = os.path.join(MODEL_DIR, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        print(f"[加载] 模型已从 {path} 加载")
        return checkpoint


# ============================================================
# 便捷函数
# ============================================================

def train_from_demos(demo_file: str = "demo_data.pkl",
                     epochs: int = NUM_EPOCHS,
                     batch_size: int = BATCH_SIZE,
                     device: str = DEVICE):
    """从演示数据文件训练模型的便捷函数。

    Args:
        demo_file: 演示数据文件名
        epochs: 训练轮数
        batch_size: 批大小
        device: 训练设备
    Returns:
        (trainer, history)
    """
    # 加载数据
    print(f"[训练] 加载演示数据: {demo_file}")
    samples = load_demo_data(demo_file)
    print(f"[训练] 共 {len(samples)} 个样本")

    # 创建分词器和数据加载器
    tokenizer = create_default_tokenizer()
    train_loader, val_loader = create_dataloaders(
        samples, batch_size=batch_size, tokenizer=tokenizer,
    )

    # 训练
    trainer = VLATrainer(device=device)
    history = trainer.train(
        train_loader, val_loader,
        epochs=epochs,
        save_best=True,
    )

    return trainer, history


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    import numpy as np

    # 创建假数据集进行冒烟测试
    tokenizer = create_default_tokenizer()

    dummy_samples = []
    for i in range(200):
        dummy_samples.append({
            "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "arm_qpos": np.zeros(7, dtype=np.float32),
            "gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
            "gripper_ctrl": np.array([0.0], dtype=np.float32),
            "action": np.array([0, 0.3, 0, -1.57, 0, 2.0, -0.785, 255.0], dtype=np.float32),
            "instruction": "grasp the red cube",
            "cube_pos": np.array([0.55, 0.0, 0.03], dtype=np.float32),
        })

    train_loader, val_loader = create_dataloaders(
        dummy_samples, batch_size=8, tokenizer=tokenizer,
    )

    trainer = VLATrainer(device="cpu")
    history = trainer.train(
        train_loader, val_loader, epochs=3, save_best=False,
    )
    print(f"\n训练完成!")
    print(f"最终 train_loss: {history['train_losses'][-1]:.6f}")
    print(f"最终 val_loss: {history['val_losses'][-1]:.6f}")
