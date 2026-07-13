# -*- coding: utf-8 -*-
# =============================================================================
# VLA 演示数据收集模块
# 通过脚本化控制器生成抓取演示数据，用于训练策略网络。
# =============================================================================

import os
import time
import pickle
import numpy as np
import cv2

from vla.env import VLAEnv
from vla.controller import (
    IKSolver, JointInterpolationController,
    GraspTrajectoryGenerator, create_controller_from_env,
)
from vla.config import (
    DEMO_DIR, ensure_dirs,
    N_ARM_JOINTS, JOINT_RANGES, GRIPPER_CTRL_RANGE,
    CTRL_DECIMATION, SIM_TIMESTEP,
    TRAJECTORY_STEPS,
    PREGRASP_HEIGHT_OFFSET, GRASP_HEIGHT_OFFSET, LIFT_HEIGHT,
    CUBE_SIZE,
)
from vla.language import DEFAULT_INSTRUCTIONS


class DemonstrationCollector:
    """收集抓取演示数据。"""

    def __init__(self, env: VLAEnv, render: bool = False):
        self.env = env
        self.render_flag = render
        self.ik, self.ctrl = create_controller_from_env(env)
        self.grasp_gen = GraspTrajectoryGenerator(self.ik, self.ctrl)

    def collect_episode(self,
                        cube_pos: np.ndarray,
                        instruction: str,
                        collect_steps: int = 20) -> list:
        """收集单次抓取演示的数据。"""
        if len(cube_pos) == 2:
            cube_pos = np.array([cube_pos[0], cube_pos[1], CUBE_SIZE])

        samples = []
        self.env.reset(cube_pos[:2])
        time.sleep(0.1)

        actual_cube = self.env.get_cube_position()
        qpos_adr = self.env.get_arm_qpos_adr()
        data = self.env.get_data()

        # ---- 阶段 1: 预抓取 ----
        pregrasp_pos = actual_cube.copy()           # 获取当前cube的坐标，作为预抓取的目标，末端到达的位置
        pregrasp_pos[2] += PREGRASP_HEIGHT_OFFSET   # 给z坐标增加一个偏置
        current_qpos = data.qpos[qpos_adr].copy()   # 获取当前关节角度
            # 当前的current_qpos表示机械臂初始的位置，准备向物体上方移动
        pregrasp_qpos = self.ik.solve(pregrasp_pos, init_qpos=current_qpos)  # 经过ik.solve得到的solution即关节角

        start_qpos = data.qpos[qpos_adr].copy()
        for i in range(collect_steps):
            alpha = (i + 1) / collect_steps
            interp_qpos = start_qpos + (pregrasp_qpos - start_qpos) * alpha  # 即缓冲机制，(1 - alpha) * start + alpha * tar
            # 在step内进行了data.ctrl的赋值
            obs, _, _, _ = self.env.step(self._make_action(interp_qpos, 0.0))       # self._make_action的返回值作为action
            samples.append(self._build_sample(obs, interp_qpos, 0.0, instruction))  # self._build_sample的返回值为当前的状态数据

        # ---- 阶段 2: 下降抓取 ----
        grasp_pos = actual_cube.copy()
        grasp_pos[2] += GRASP_HEIGHT_OFFSET + 0.01 # 给z坐标增加一个偏置
        current_qpos = data.qpos[qpos_adr].copy()
            # 当前的current_qpos表示机械臂末端已经到了物体上方，准备下降夹取
        grasp_qpos = self.ik.solve(grasp_pos, init_qpos=current_qpos)   

        for i in range(collect_steps):
            alpha = (i + 1) / collect_steps
            interp_qpos = pregrasp_qpos + (grasp_qpos - pregrasp_qpos) * alpha

            obs, _, _, _ = self.env.step(self._make_action(interp_qpos, 0.0))
            samples.append(self._build_sample(obs, interp_qpos, 0.0, instruction))

        # 闭合夹爪 (逐步)
        close_steps = max(collect_steps // 2, 5)
        for i in range(close_steps):
            alpha = (i + 1) / close_steps
            gripper_val = alpha * 255.0

            obs, _, _, _ = self.env.step(self._make_action(grasp_qpos, gripper_val))
            samples.append(self._build_sample(obs, grasp_qpos, 255.0, instruction))

        # ---- 阶段 3: 抬升 ----
        lift_pos = actual_cube.copy()
        lift_pos[2] += LIFT_HEIGHT
        current_qpos = data.qpos[qpos_adr].copy()
        lift_qpos = self.ik.solve(lift_pos, init_qpos=current_qpos)

        for i in range(collect_steps):
            alpha = (i + 1) / collect_steps
            interp_qpos = grasp_qpos + (lift_qpos - grasp_qpos) * alpha

            obs, _, _, _ = self.env.step(self._make_action(interp_qpos, 255.0))
            samples.append(self._build_sample(obs, interp_qpos, 255.0, instruction))

        return samples

    @staticmethod
    def _make_action(arm_qpos: np.ndarray, gripper_ctrl: float) -> np.ndarray:
        """构造 (8,) 动作数组。"""
        act = np.zeros(8, dtype=np.float64)
        act[:7] = np.asarray(arm_qpos, dtype=np.float64)  # arm
        act[7] = gripper_ctrl                             # gripper 一共8个actuator
        return act

    @staticmethod
    def _build_sample(obs: dict,
                      target_arm_qpos: np.ndarray,
                      target_gripper: float,
                      instruction: str) -> dict:
        """构建单个训练样本。

        Args:
            obs: 环境观测
            target_arm_qpos: (7,) 目标手臂关节角度
            target_gripper : (1,) 目标夹爪执行器值
            instruction: 语言指令
        """
        action = np.zeros(8, dtype=np.float32)
        action[:7] = np.asarray(target_arm_qpos, dtype=np.float32)[:7]
        action[7] = float(target_gripper)

        return {
            "image":        obs["image"].copy(),
            "arm_qpos":     obs["arm_qpos"].astype(np.float32).copy(),
            "gripper_qpos": obs["gripper_qpos"].astype(np.float32).copy(),
            "gripper_ctrl": obs["gripper_ctrl"].astype(np.float32).copy(),
            "action":       action,
            "instruction":  instruction,
            "cube_pos":     obs["cube_pos"].astype(np.float32).copy(),
        }

    def collect_dataset(self,
                        num_episodes: int = 50,
                        cube_x_range: tuple = (0.35, 0.65),
                        cube_y_range: tuple = (-0.2, 0.2),
                        instructions: list = None,
                        save: bool = True,
                        filename: str = "demo_data.pkl"):
        """批量收集演示数据集。"""
        if instructions is None:
            instructions = DEFAULT_INSTRUCTIONS

        all_samples = []
        # 创建独立的随机数生成器
        rng = np.random.RandomState(42)

        for ep in range(num_episodes):
            # 生成cube_x_range区间的随机数， rng.uniform(low, high)
            cx = rng.uniform(*cube_x_range)
            cy = rng.uniform(*cube_y_range)
            cube_pos = np.array([cx, cy])
            inst = instructions[rng.randint(0, len(instructions))]

            print(f"[Demo {ep+1}/{num_episodes}] 方块=({cx:.3f}, {cy:.3f}), "
                  f"指令=''{inst}''")

            try:
                samples = self.collect_episode(cube_pos, inst)
                all_samples.extend(samples)
                print(f"  -> 收集 {len(samples)} 个样本")
            except Exception as e:
                print(f"  -> 失败: {e}")
                import traceback
                traceback.print_exc()
                self.env.reset()
                continue

            if self.render_flag:
                cv2.waitKey(1)

        print(f"\n总计收集 {len(all_samples)} 个样本")

        if save and len(all_samples) > 0:
            ensure_dirs()
            path = os.path.join(DEMO_DIR, filename)
            with open(path, "wb") as f:
                pickle.dump(all_samples, f)
            print(f"数据已保存至: {path}")
        elif len(all_samples) == 0:
            print("警告: 未收集到任何样本，请检查环境配置。")

        return all_samples


if __name__ == "__main__":
    env = VLAEnv(render=True)
    collector = DemonstrationCollector(env, render=False)

    try:
        samples = collector.collect_episode(
            cube_pos=np.array([0.55, 0.0]),
            instruction="grasp the red cube",
        )
        print(f"\n单次演示收集了 {len(samples)} 个样本")
        print(f"样本键: {list(samples[0].keys())}")
    finally:
        env.close()
        cv2.destroyAllWindows()
