# -*- coding: utf-8 -*-
"""
06_embodied/franka_panda.py - Franka Emika Panda 7-DoF Robotic Arm
Provides: FK, IK, position control, pick-and-place, VLA integration.
All control methods now actually step the MuJoCo simulation for live visualization.
"""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mujoco_env import MuJoCoEnv, EnvConfig


class FrankaPanda(MuJoCoEnv):
    """Franka Emika Panda: 7 joints + gripper."""

    JOINT_LIMITS = np.array([
        [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
        [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973],
    ])

    def __init__(self, model_xml_path=None, config=None):
        super().__init__(model_xml_path, config)
        self._gripper_open = 0.04
        self._gripper_closed = 0.0
        self._grasped_object = None
        # 每个 VLA 动作拆分为多少个子步（越大运动越平滑）
        self.steps_per_action = 50
        print('Franka Panda arm ready.')

    # ================================================================
    # Kinematics
    # ================================================================
    def forward_kinematics(self, joint_angles):
        """Compute end-effector position from joint angles (simplified FK)."""
        q = np.asarray(joint_angles)
        l1, l2, l3 = 0.333, 0.316, 0.25
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]) + l3 * np.cos(q[0] + q[1] + q[3])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]) + l3 * np.sin(q[0] + q[1] + q[3])
        z = 0.5 + 0.1 * np.sin(q[2])
        return np.array([x, y, z])

    def inverse_kinematics(self, target_pos, initial_guess=None):
        """Compute joint angles via numerical IK (gradient descent)."""
        if initial_guess is None:
            initial_guess = np.zeros(7)
        q = np.asarray(initial_guess, dtype=np.float64).copy()
        target = np.asarray(target_pos)
        for _ in range(200):
            current = self.forward_kinematics(q)
            error = target - current
            if np.linalg.norm(error) < 0.001:
                break
            eps = 0.001
            J = np.zeros((3, 7))
            for j in range(7):
                q_plus = q.copy()
                q_plus[j] += eps
                J[:, j] = (self.forward_kinematics(q_plus) - current) / eps
            dq = np.linalg.pinv(J) @ error * 0.1
            q += dq
            for j in range(7):
                q[j] = np.clip(q[j], self.JOINT_LIMITS[j, 0], self.JOINT_LIMITS[j, 1])
        return q

    # ================================================================
    # Motion control (actually steps the simulation)
    # ================================================================
    def move_to(self, target_pos, steps=None, step_delay=0.01):
        """
        将末端执行器移动到目标位置，并在 MuJoCo 中逐步仿真。

        Args:
            target_pos:  目标位置 (x, y, z)
            steps:       插值步数（越大越平滑）
            step_delay:  每步之间的延时（秒），用于控制可视化速度

        Returns:
            trajectory:  每步的关节角度列表
        """
        if steps is None:
            steps = self.steps_per_action

        # 从 MuJoCo 读取当前关节角度
        current_q = self._data.qpos[:7].copy()
        target_q = self.inverse_kinematics(np.asarray(target_pos), current_q)

        trajectory = []
        for t in range(steps):
            alpha = (t + 1) / steps
            interp_q = current_q + alpha * (target_q - current_q)
            trajectory.append(interp_q.copy())

            # 实际驱动 MuJoCo 仿真
            self.step(interp_q, n_substeps=5, sync_viewer=True)
            time.sleep(step_delay)

        return trajectory

    def move_joints(self, target_q, steps=None, step_delay=0.01):
        """
        直接控制关节角度运动到目标值。

        Args:
            target_q:    目标关节角度 (7,)
            steps:       插值步数
            step_delay:  每步延时
        """
        if steps is None:
            steps = self.steps_per_action

        current_q = self._data.qpos[:7].copy()
        target_q = np.asarray(target_q)

        for t in range(steps):
            alpha = (t + 1) / steps
            interp_q = current_q + alpha * (target_q - current_q)
            self.step(interp_q, n_substeps=5, sync_viewer=True)
            time.sleep(step_delay)

    # ================================================================
    # Task-level actions
    # ================================================================
    def pick_and_place(self, pick_pos, place_pos):
        """Execute a complete pick-and-place task with visualization."""
        print(f'Pick-and-place: {pick_pos} -> {place_pos}')

        # 1. 接近抓取位置（上方）
        approach = np.asarray(pick_pos) + np.array([0, 0, 0.1])
        print('  Moving to approach position...')
        self.move_to(approach, steps=80)

        # 2. 下降到抓取位置
        print('  Descending to grasp...')
        self.move_to(np.asarray(pick_pos), steps=60)

        # 3. 抓取
        print('  Grasping...')
        self._grasped_object = 'block'
        time.sleep(0.3)

        # 4. 抬起
        print('  Lifting...')
        self.move_to(approach, steps=60)

        # 5. 移动到放置位置上方
        place_above = np.asarray(place_pos) + np.array([0, 0, 0.1])
        print('  Moving to place position...')
        self.move_to(place_above, steps=100)

        # 6. 下降到放置位置
        print('  Descending to place...')
        self.move_to(np.asarray(place_pos), steps=60)

        # 7. 释放
        print('  Releasing...')
        self._grasped_object = None
        time.sleep(0.3)

        # 8. 撤回
        print('  Retracting...')
        self.move_to(place_above, steps=60)

        print('Pick-and-place complete!')

    # ================================================================
    # VLA integration
    # ================================================================
    def execute_vla_action(self, action_vector, step_delay=0.01):
        """
        执行 VLA 模型预测的动作，并驱动 MuJoCo 仿真。

        VLA 动作格式: [dx, dy, dz, roll, pitch, yaw, gripper]
        其中 dx,dy,dz 是末端执行器的位移增量（米）。

        这个方法会将增量转换为目标关节角度，然后逐步插值驱动仿真。

        Args:
            action_vector: VLA 输出的 7 维动作向量
            step_delay:    每子步延时（控制可视化速度）

        Returns:
            trajectory: 关节角度轨迹列表
        """
        dx, dy, dz = action_vector[:3]

        # 从 MuJoCo 读取当前状态
        current_q = self._data.qpos[:7].copy()
        current_pos = self.forward_kinematics(current_q)

        # 计算目标末端位置
        scale = 0.05  # 动作缩放因子
        target_pos = current_pos + np.array([dx, dy, dz]) * scale

        # IK 求解目标关节角度
        target_q = self.inverse_kinematics(target_pos, current_q)

        # 逐步插值驱动仿真
        trajectory = []
        for t in range(self.steps_per_action):
            alpha = (t + 1) / self.steps_per_action
            interp_q = current_q + alpha * (target_q - current_q)
            trajectory.append(interp_q.copy())
            self.step(interp_q, n_substeps=5, sync_viewer=True)
            time.sleep(step_delay)

        return trajectory

    def get_current_ee_pos(self):
        """获取当前末端执行器位置（从 MuJoCo 仿真中读取）。"""
        if self._data is not None:
            q = self._data.qpos[:7].copy()
            return self.forward_kinematics(q)
        return np.zeros(3)


if __name__ == '__main__':
    print('=' * 60)
    print('Franka Panda Demo (with visualization)')
    print('=' * 60)

    panda = FrankaPanda()

    # 启动可视化窗口
    panda.launch_viewer()
    time.sleep(0.5)  # 等窗口打开

    try:
        # FK 测试
        pos = panda.forward_kinematics(np.zeros(7))
        print(f'FK (zero joints): {pos}')

        # IK 测试
        target = np.array([0.5, 0.0, 0.4])
        q = panda.inverse_kinematics(target)
        print(f'IK target: {target}')
        print(f'IK result pos: {panda.forward_kinematics(q)}')
        print(f'IK error: {np.linalg.norm(panda.forward_kinematics(q) - target):.4f}')

        # 移动机械臂（可以看到机械臂运动）
        print('\nMoving to target position...')
        panda.move_to(np.array([0.5, 0.0, 0.6]), steps=80)

        # 做一次完整的 pick-and-place
        print('\nRunning pick-and-place...')
        panda.pick_and_place(
            np.array([0.4, 0.2, 0.2]),
            np.array([0.6, -0.1, 0.3])
        )

    except KeyboardInterrupt:
        print('Interrupted.')
    finally:
        panda.close_viewer()

    print('Done!')