# -*- coding: utf-8 -*-
# =============================================================================
# VLA (Vision-Language-Action) 系统配置文件
# 适配 Franka Emika Panda 机械臂 + MuJoCo 仿真环境
# =============================================================================

import os

# ============================================================
# 路径配置
# ============================================================

# MuJoCo 场景文件路径
SCENE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "..", "mj", "model", "franka_emika_panda", "scene2.xml"
)
SCENE_PATH = os.path.normpath(SCENE_PATH)

# 数据保存目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
DEMO_DIR = os.path.join(DATA_DIR, "demonstrations")

# ============================================================
# 相机配置
# ============================================================

CAMERA_NAME = "front_cam"
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3

# ============================================================
# Panda 机械臂关节与执行器配置
# ============================================================

ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7"
]

GRIPPER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]

ACTUATOR_NAMES = [
    "actuator1", "actuator2", "actuator3", "actuator4",
    "actuator5", "actuator6", "actuator7", "actuator8"
]

N_ARM_JOINTS = 7
N_GRIPPER_JOINTS = 2
N_ACTUATORS = 8
ACTION_DIM = 8
STATE_DIM = 9

JOINT_RANGES = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]

GRIPPER_RANGE = (0.0, 0.04)
GRIPPER_CTRL_RANGE = (0.0, 255.0)

DEFAULT_ARM_QPOS = [0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853]
DEFAULT_GRIPPER_QPOS = [0.04, 0.04]

# ============================================================
# 仿真配置
# ============================================================

SIM_TIMESTEP = 0.005
CTRL_DECIMATION = 10
VIEWER_REFRESH_RATE = 0.02

# ============================================================
# 物体 (红色方块) 配置
# ============================================================

CUBE_BODY_NAME = "red_cube"
CUBE_SIZE = 0.03
CUBE_COLOR_LOWER = (0, 100, 100)
CUBE_COLOR_UPPER = (10, 255, 255)

# ============================================================
# VLA 模型超参数
# ============================================================

VISION_FEATURE_DIM = 256
VOCAB_SIZE = 2000
TEXT_EMBED_DIM = 64
LANGUAGE_FEATURE_DIM = 128
PROPRIO_FEATURE_DIM = 32
FUSION_HIDDEN_DIM = 256
POLICY_HIDDEN_DIM = 128

# ============================================================
# 训练配置
# ============================================================

BATCH_SIZE = 70
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
VAL_SPLIT = 0.1
GRAD_CLIP = 1.0
LR_SCHEDULER_STEP = 20
LR_SCHEDULER_GAMMA = 0.5
DEVICE = "cuda"

# ============================================================
# 推理 / 抓取配置
# ============================================================

PREGRASP_HEIGHT_OFFSET = 0.15
GRASP_HEIGHT_OFFSET = 0.01
LIFT_HEIGHT = 0.3
IK_MAX_ITER = 200
IK_TOLERANCE = 0.001
IK_DAMPING = 0.1
TRAJECTORY_STEPS = 50

# ============================================================
# 辅助函数
# ============================================================

def ensure_dirs():
    """确保所有需要的目录都存在。"""
    for d in [DATA_DIR, MODEL_DIR, DEMO_DIR]:
        os.makedirs(d, exist_ok=True)


def normalize_joints(joint_angles, joint_ranges=None):
    """将关节角度归一化到 [-1, 1]."""
    import numpy as np
    if joint_ranges is None:
        joint_ranges = JOINT_RANGES
    joint_angles = np.asarray(joint_angles, dtype=np.float32)
    out = np.zeros_like(joint_angles)
    for i, (lo, hi) in enumerate(joint_ranges):
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        if joint_angles.ndim == 1:
            out[i] = (joint_angles[i] - mid) / half
        else:
            out[:, i] = (joint_angles[:, i] - mid) / half
    return out


def denormalize_joints(normed, joint_ranges=None):
    """将归一化值 [-1, 1] 还原为实际关节角度。"""
    import numpy as np
    if joint_ranges is None:
        joint_ranges = JOINT_RANGES
    normed = np.asarray(normed, dtype=np.float32)
    out = np.zeros_like(normed)
    for i, (lo, hi) in enumerate(joint_ranges):
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        if normed.ndim == 1:
            out[i] = normed[i] * half + mid
        else:
            out[:, i] = normed[:, i] * half + mid
    return out
