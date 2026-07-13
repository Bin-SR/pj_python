# -*- coding: utf-8 -*-
# =============================================================================
# VLA 推理与自主抓取模块
# 加载训练好的模型，根据语言指令和视觉输入自主执行抓取。
#
# 支持两种推理模式:
#   1. 端到端策略推理 (model-based): 使用 VLA 策略网络直接预测动作
#   2. 脚本化抓取 (script-based):  使用 IK + 轨迹规划 (无需模型)
# =============================================================================

import os
import time
import numpy as np
import cv2
import torch

from vla.config import (
    MODEL_DIR, DEVICE,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    N_ARM_JOINTS, JOINT_RANGES,
    GRIPPER_CTRL_RANGE,
    CTRL_DECIMATION, SIM_TIMESTEP,
    PREGRASP_HEIGHT_OFFSET, GRASP_HEIGHT_OFFSET, LIFT_HEIGHT,
    ACTION_DIM, STATE_DIM,
    DEFAULT_ARM_QPOS, DEFAULT_GRIPPER_QPOS,
)
from vla.policy import VLAPolicy
from vla.language import create_default_tokenizer, SimpleTokenizer
from vla.vision import RedCubeDetector, preprocess_image
from vla.env import VLAEnv
from vla.controller import (
    IKSolver, JointInterpolationController,
    create_controller_from_env,
)


class VLAInference:
    """VLA 模型推理与自主抓取。"""

    def __init__(self,
                 env: VLAEnv,
                 model_path: str = "vla_policy.pt",
                 device: str = DEVICE):
        self.env = env
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        # ---- 加载模型 ----
        self.policy = VLAPolicy().to(self.device)
        self.tokenizer = create_default_tokenizer()

        model_full_path = os.path.join(MODEL_DIR, model_path)
        if os.path.exists(model_full_path):
            checkpoint = torch.load(model_full_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.policy.eval()
            print(f"[推理] 模型已加载: {model_full_path}")
            self.model_loaded = True
        else:
            print(f"[推理] 警告: 未找到模型 {model_full_path}，将使用脚本化模式")
            self.model_loaded = False

        # ---- 视觉检测器 ----
        self.detector = RedCubeDetector()

        # ---- 控制器 (使用工厂函数) ----
        self.ik, self.ctrl = create_controller_from_env(env)

    # ============================================================
    # 端到端策略推理
    # ============================================================

    def predict_action(self,
                       image: np.ndarray,
                       instruction: str,
                       arm_qpos: np.ndarray,
                       gripper_qpos: np.ndarray) -> np.ndarray:
        """使用 VLA 策略网络预测下一步动作。"""
        if not self.model_loaded:
            raise RuntimeError("模型未加载，无法预测")

        img_tensor = preprocess_image(image).to(self.device)
        token_ids = self.tokenizer.encode(instruction).unsqueeze(0).to(self.device)
        proprio = np.concatenate([arm_qpos, gripper_qpos])
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            normed_action = self.policy(img_tensor, token_ids, proprio_tensor)
            normed_action = normed_action.cpu().numpy()[0]

        action = np.zeros(8, dtype=np.float32)
        for i in range(N_ARM_JOINTS):
            lo, hi = JOINT_RANGES[i]
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            action[i] = np.clip(normed_action[i] * half + mid, lo, hi)
        action[7] = np.clip((normed_action[7] + 1.0) / 2.0 * 255.0, 0.0, 255.0)

        return action

    # ============================================================
    # 自主抓取主循环
    # ============================================================

    def run_autonomous_grasp(self,
                             instruction: str = "grasp the red cube",
                             max_steps: int = 300,
                             use_model: bool = True,
                             render_detection: bool = True):
        """执行自主抓取。

        Args:
            instruction: 语言指令
            max_steps: 最大步数
            use_model: 是否使用策略网络
            render_detection: 是否显示检测结果
        Returns:
            (success, info_dict)
        """
        print(f"[自主抓取] 指令: ''{instruction}''")
        use_model = use_model and self.model_loaded
        print(f"[自主抓取] 模式: {'模型推理' if use_model else '脚本化IK'}")

        obs = self.env.reset()
        step = 0
        grasp_phase = "approach"
        success = False

        for step in range(max_steps):
            image = obs["image"]
            arm_qpos = obs["arm_qpos"]
            gripper_qpos = obs["gripper_qpos"]
            cube_pos = obs["cube_pos"]
            hand_pos = obs["hand_pos"]

            # ---- 检测红色方块 ----
            center_xy, bbox = self.detector.detect(image)

            # ---- 计算动作 ----
            if use_model:
                action = self.predict_action(image, instruction, arm_qpos, gripper_qpos)
            else:
                action = self._scripted_action(
                    center_xy, obs, grasp_phase
                )

            # ---- 执行动作 ----
            obs, reward, done, info = self.env.step(action)

            # ---- 阶段转换 ----
            distance = np.linalg.norm(obs["cube_pos"] - obs["hand_pos"])
            gripper_closed = obs["gripper_ctrl"][0] > 200
            hand_z = obs["hand_pos"][2]

            if grasp_phase == "approach" and distance < 0.08:
                grasp_phase = "grasp"
                print(f"  [Step {step}] 进入抓取阶段, dist={distance:.3f}m")

            elif grasp_phase == "grasp" and gripper_closed and distance < 0.05:
                grasp_phase = "lift"
                print(f"  [Step {step}] 进入抬升阶段")

            elif grasp_phase == "lift" and hand_z > 0.25:
                grasp_phase = "done"
                success = True
                print(f"  [Step {step}] 抓取完成! hand_z={hand_z:.3f}m")
                break

            # ---- 可视化 ----
            if render_detection:
                self._render_debug(image, center_xy, bbox, obs,
                                   grasp_phase, step)

        return success, {
            "steps": step + 1,
            "final_distance": np.linalg.norm(obs["cube_pos"] - obs["hand_pos"]),
            "cube_pos": obs["cube_pos"],
            "hand_pos": obs["hand_pos"],
            "grasp_phase": grasp_phase,
        }

    def _scripted_action(self,
                         center_xy: tuple,
                         obs: dict,
                         phase: str) -> np.ndarray:
        """生成脚本化动作 (不依赖模型)。

        Args:
            center_xy: 方块像素中心 (可能为 None)
            obs: 当前观测字典
            phase: 当前阶段 "approach" | "grasp" | "lift"
        Returns:
            (8,) 动作 [arm_joints(7), gripper_ctrl(1)]
        """
        action = np.zeros(8, dtype=np.float32)
        action[:7] = obs["arm_qpos"]
        action[7] = obs["gripper_ctrl"][0]

        if center_xy is None:
            # 没检测到方块, 保持原位
            return action

        cube_pos = obs["cube_pos"]
        # 关键修复: 传入当前关节角度作为 IK 初始值
        current_arm_qpos = obs["arm_qpos"].copy()

        try:
            if phase == "approach":
                target = cube_pos.copy()
                target[2] += PREGRASP_HEIGHT_OFFSET
                qpos = self.ik.solve(target, init_qpos=current_arm_qpos)
                action[:7] = qpos
                action[7] = 0.0

            elif phase == "grasp":
                target = cube_pos.copy()
                target[2] += GRASP_HEIGHT_OFFSET + 0.01
                qpos = self.ik.solve(target, init_qpos=current_arm_qpos)
                action[:7] = qpos
                action[7] = 255.0

            elif phase == "lift":
                target = cube_pos.copy()
                target[2] += LIFT_HEIGHT
                qpos = self.ik.solve(target, init_qpos=current_arm_qpos)
                action[:7] = qpos
                action[7] = 255.0

        except Exception as e:
            print(f"  [IK警告] {e}")
            import traceback
            traceback.print_exc()

        return action

    def _render_debug(self, image, center_xy, bbox, obs, phase, step):
        """渲染调试可视化。"""
        display_img = image.copy()
        if center_xy is not None:
            cv2.circle(display_img, center_xy, 5, (0, 255, 0), -1)
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display_img, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
        dist = np.linalg.norm(obs["cube_pos"] - obs["hand_pos"])
        cv2.putText(display_img, f"Phase: {phase}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(display_img, f"Step: {step}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(display_img, f"Dist: {dist:.3f}m",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.imshow("VLA Autonomous Grasp",
                   cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt("用户按 ESC 退出")


# ============================================================
# 便捷函数
# ============================================================

def run_grasp_demo(instruction: str = "grasp the red cube",
                   use_model: bool = True,
                   model_path: str = "vla_policy.pt"):
    """运行抓取演示的便捷函数。"""
    env = VLAEnv(render=True)
    inference = VLAInference(env, model_path=model_path)

    try:
        success, info = inference.run_autonomous_grasp(
            instruction=instruction,
            use_model=use_model,
        )
        print(f"\n{'='*50}")
        print(f"抓取结果: {'成功!' if success else '未完成'}")
        print(f"总步数: {info['steps']}")
        print(f"最终距离: {info['final_distance']:.4f}m")
        print(f"方块位置: {info['cube_pos']}")
        print(f"手爪位置: {info['hand_pos']}")
        print(f"{'='*50}")

        if success:
            print("按任意键退出...")
            cv2.waitKey(0)

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_grasp_demo(use_model=False)
