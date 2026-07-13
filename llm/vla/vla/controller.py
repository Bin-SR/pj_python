# -*- coding: utf-8 -*-
# =============================================================================
# VLA 控制器模块
# 提供 Panda 机械臂的运动控制:
#   1. JointInterpolationController —— 关节空间平滑插值控制
#   2. IKSolver —— 基于阻尼最小二乘的逆运动学求解器
#   3. GraspTrajectoryGenerator —— 抓取轨迹生成器
# =============================================================================

import time
import numpy as np
import mujoco

from vla.config import (
    SIM_TIMESTEP, CTRL_DECIMATION,
    N_ARM_JOINTS, JOINT_RANGES,
    GRIPPER_CTRL_RANGE,
    IK_MAX_ITER, IK_TOLERANCE, IK_DAMPING,
    TRAJECTORY_STEPS,
    PREGRASP_HEIGHT_OFFSET, GRASP_HEIGHT_OFFSET, LIFT_HEIGHT,
    ARM_JOINT_NAMES,
)


# ============================================================
# 逆运动学求解器 (阻尼最小二乘法)
# ============================================================

class IKSolver:
    """基于阻尼最小二乘 (Damped Least Squares) 的 IK 求解器。

    使用 MuJoCo 的 Jacobian 计算关节修正量，迭代逼近目标位置。

    重要: 求解过程中会临时修改 data.qpos 来计算前向运动学，
    求解完成后会恢复原始 qpos，避免与仿真线程冲突。
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 qpos_adr: np.ndarray,
                 dof_adr: np.ndarray,
                 body_id: int,
                 max_iter: int = IK_MAX_ITER,
                 tolerance: float = IK_TOLERANCE,
                 damping: float = IK_DAMPING):
        self.model = model
        self.data = data
        self.qpos_adr = np.asarray(qpos_adr, dtype=np.int32)
        self.dof_adr = np.asarray(dof_adr, dtype=np.int32)
        self.body_id = body_id
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.damping = damping

        # 预分配 Jacobian 数组 (3 行平移 + 3 行旋转)
        self._jac_pos = np.zeros((3, model.nv))
        self._jac_rot = np.zeros((3, model.nv))

    def solve(self, target_pos: np.ndarray,
              init_qpos: np.ndarray = None) -> np.ndarray:
        """求解 IK: 返回使末端到达 target_pos 的关节角度。

        Args:
            target_pos: (3,) 世界坐标系下的目标位置
            init_qpos: (7,) 初始关节角度 (None 则用当前 data.qpos 值)
        Returns:
            (7,) 关节角度解
        """
        target = np.asarray(target_pos, dtype=np.float64)

        # ---- 保存原始 qpos (关键: 防止干扰仿真线程) ----
        original_qpos = self.data.qpos[self.qpos_adr].copy()

        # 设置初始关节角度
        if init_qpos is not None:
            self.data.qpos[self.qpos_adr] = np.asarray(init_qpos, dtype=np.float64)

        # ---- IK 迭代 ----
        for iteration in range(self.max_iter):
            # 前向运动学 (只读 xpos, 修改内部缓存)
            mujoco.mj_forward(self.model, self.data)

            # 当前末端位置
            current = self.data.xpos[self.body_id].copy()

            # 位置误差
            error = target - current
            err_norm = np.linalg.norm(error)
            if err_norm < self.tolerance:
                break

            # 计算平移 Jacobian (3 x nv)
            mujoco.mj_jacBody(self.model, self.data,
                              self._jac_pos, self._jac_rot,
                              self.body_id)

            # 只取参与 IK 的 DOF 列 (3 x 7)
            jac_sub = self._jac_pos[:, self.dof_adr]

            # 阻尼最小二乘: delta_q = J^T (J J^T + lambda^2 I)^{-1} error
            jjt = jac_sub @ jac_sub.T
            damped = jjt + (self.damping ** 2) * np.eye(3)
            try:
                delta_q = jac_sub.T @ np.linalg.solve(damped, error)
            except np.linalg.LinAlgError:
                delta_q = jac_sub.T @ np.linalg.lstsq(damped, error, rcond=None)[0]

            # 更新关节角度 (使用 qpos 地址)
            self.data.qpos[self.qpos_adr] += delta_q

            # 裁剪到关节限位
            for i, (lo, hi) in enumerate(JOINT_RANGES):
                idx = self.qpos_adr[i]
                self.data.qpos[idx] = np.clip(self.data.qpos[idx], lo, hi)

        # 保存解
        solution = self.data.qpos[self.qpos_adr].copy()

        # ---- 恢复原始 qpos (关键: 不影响仿真状态) ----
        self.data.qpos[self.qpos_adr] = original_qpos

        return solution


# ============================================================
# 关节插值控制器
# ============================================================

class JointInterpolationController:
    """在关节空间中进行平滑插值控制。

    将目标关节角度分解为若干小步，逐步执行，实现平滑运动。
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 actuator_ids: np.ndarray,
                 qpos_adr: np.ndarray,
                 steps: int = TRAJECTORY_STEPS):
        self.model = model
        self.data = data
        self.actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
        self.qpos_adr = np.asarray(qpos_adr, dtype=np.int32)
        self.steps = steps

    def move_to_joints(self,
                       target_arm_qpos: np.ndarray,
                       gripper_ctrl: float,
                       step_callback=None) -> bool:
        """平滑移动到目标关节位置。"""
        target = np.asarray(target_arm_qpos, dtype=np.float64)
        start = self.data.qpos[self.qpos_adr].copy()
        delta = target - start

        for i in range(self.steps):
            alpha = (i + 1) / self.steps
            interp = start + delta * alpha

            # 设置控制信号
            for j in range(7):
                self.data.ctrl[self.actuator_ids[j]] = interp[j]
            # 夹爪
            gripper_val = np.clip(gripper_ctrl, 0.0, 255.0)
            self.data.ctrl[self.actuator_ids[7]] = gripper_val

            # 等待仿真步
            for _ in range(CTRL_DECIMATION):
                time.sleep(SIM_TIMESTEP)

            if step_callback:
                step_callback(i + 1, self.steps)

        return True

    def get_current_arm_qpos(self) -> np.ndarray:
        """获取当前手臂关节角度。"""
        return self.data.qpos[self.qpos_adr].copy()


# ============================================================
# 抓取轨迹生成器
# ============================================================

class GraspTrajectoryGenerator:
    """生成从当前位置到抓取目标的完整轨迹。"""

    def __init__(self,
                 ik_solver: IKSolver,
                 controller: JointInterpolationController):
        self.ik = ik_solver
        self.ctrl = controller

    def execute_grasp(self,
                      cube_pos: np.ndarray,
                      gripper_open: float = 0.0,
                      gripper_close: float = 255.0) -> bool:
        """执行完整的抓取动作。"""
        cube = np.asarray(cube_pos, dtype=np.float64)

        # ---- 阶段 1: 预抓取 ----
        pregrasp_pos = cube.copy()
        pregrasp_pos[2] += PREGRASP_HEIGHT_OFFSET
        pregrasp_qpos = self.ik.solve(pregrasp_pos)
        self.ctrl.move_to_joints(pregrasp_qpos, gripper_open)
        time.sleep(0.1)

        # ---- 阶段 2: 下降抓取 ----
        grasp_pos = cube.copy()
        grasp_pos[2] += GRASP_HEIGHT_OFFSET + 0.01
        grasp_qpos = self.ik.solve(grasp_pos, init_qpos=pregrasp_qpos)
        self.ctrl.move_to_joints(grasp_qpos, gripper_open)
        self.ctrl.move_to_joints(grasp_qpos, gripper_close)
        time.sleep(0.3)

        # ---- 阶段 3: 抬升 ----
        lift_pos = cube.copy()
        lift_pos[2] += LIFT_HEIGHT
        lift_qpos = self.ik.solve(lift_pos, init_qpos=grasp_qpos)
        self.ctrl.move_to_joints(lift_qpos, gripper_close)

        return True


# ============================================================
# 工厂函数: 从 VLAEnv 创建控制器组件
# ============================================================

def create_controller_from_env(env):
    """从 VLAEnv 实例创建 IK 求解器和插值控制器。"""
    model = env.get_model()
    data = env.get_data()
    qpos_adr = env.get_arm_qpos_adr()
    dof_adr = env.get_arm_dof_adr()
    hand_body_id = env.get_hand_body_id()
    actuator_ids = env._actuator_ids

    ik = IKSolver(model, data, qpos_adr, dof_adr, hand_body_id)
    ctrl = JointInterpolationController(model, data, actuator_ids, qpos_adr)

    return ik, ctrl


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    print("IKSolver / Controller 模块已加载。")
    print("请通过 env.py 中的 VLAEnv 进行实际测试。")
