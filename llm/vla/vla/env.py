# -*- coding: utf-8 -*-
# =============================================================================
# VLA 环境模块 —— 封装 MuJoCo 仿真环境
# 基于 mj_env.py 的原有架构，提供同步的 RL 风格接口
# =============================================================================

import time
import threading
import numpy as np
import cv2
import mujoco
import mujoco.viewer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from vla.config import (
    SCENE_PATH, CAMERA_NAME,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    SIM_TIMESTEP, CTRL_DECIMATION,
    ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES, ACTUATOR_NAMES,
    DEFAULT_ARM_QPOS, DEFAULT_GRIPPER_QPOS,
    CUBE_BODY_NAME, N_ARM_JOINTS,
)


class VLAEnv:
    """VLA 任务的 MuJoCo 仿真环境。

    提供类 Gym 接口:
        - reset()        -> 观测
        - step(action)   -> (观测, 奖励, 完成标志, 信息)
        - get_observation() -> 字典形式的观测
        - close()        -> 清理资源

    内部使用双线程 (仿真 + 可视化)，与 mj_env.py 保持一致。
    """

    def __init__(self, render: bool = True):
        """初始化环境。

        Args:
            render: 是否启动可视化窗口。
        """
        # ---- 加载模型 ----
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_TIMESTEP

        # ---- 获取关节在 qpos 中的地址 ----
        # 注意: mj_name2id 返回 joint index, qpos 地址需通过 jnt_qposadr 获取
        # arm
        self._arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
        self._arm_qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in self._arm_joint_ids], dtype=np.int32)
        # gripper
        self._gripper_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)for name in GRIPPER_JOINT_NAMES]
        self._gripper_qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in self._gripper_joint_ids], dtype=np.int32)

        # ---- 获取执行器索引 ----
        self._actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES], dtype=np.int32)

        # ---- 获取 body 索引 ----
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)
        self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # ---- 获取关节 DOF 地址 (用于 Jacobian) ----
        self._arm_dof_adr = np.array([self.model.jnt_dofadr[jid] for jid in self._arm_joint_ids], dtype=np.int32)

        # ---- 渲染器 ----
        self.renderer = mujoco.Renderer(self.model, IMAGE_HEIGHT, IMAGE_WIDTH)

        # ---- 锁和线程 ----
        self._lock = threading.Lock()
        self._render = render
        self._stop = False

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
            self._viewer_thread.start()

        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

        # ---- 步数计数器 ----
        self._step_count = 0

    # ============================================================
    # 线程循环
    # ============================================================

    def _sim_loop(self):
        """后台仿真线程循环。"""
        while not self._stop:
            step_start = time.perf_counter()
            with self._lock:
                # print(np.max(np.abs(self.data.qpos)))
                # print(np.max(np.abs(self.data.qvel)))
                mujoco.mj_step(self.model, self.data)
            elapsed = time.perf_counter() - step_start
            if elapsed < self.model.opt.timestep:
                time.sleep(self.model.opt.timestep - elapsed)

    def _viewer_loop(self):
        """后台可视化线程循环。"""
        while self.viewer.is_running() and not self._stop:
            with self._lock:
                self.viewer.sync()
            time.sleep(0.02)

    # ============================================================
    # 环境接口
    # ============================================================

    def reset(self, cube_pos: np.ndarray = None) -> dict:
        """重置环境到初始状态。

        Args:
            cube_pos: 可选，重置方块位置 (x, y)。z 固定在桌面高度。
        Returns:
            observation: 包含 image, proprioception 等字段的字典。
        """
        with self._lock:
            # 重置到 home keyframe
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            # 如果需要，重新放置方块
            if cube_pos is not None:
                from vla.config import CUBE_SIZE as CS
                # qpos 布局: arm[7] + gripper[2] + cube_free[7]
                cube_start_idx = N_ARM_JOINTS + 2  # = 9
                self.data.qpos[cube_start_idx + 0] = float(cube_pos[0])
                self.data.qpos[cube_start_idx + 1] = float(cube_pos[1])
                self.data.qpos[cube_start_idx + 2] = float(CS)

            # 前向运动学
            # 在reset中，使用mj_forward
            mujoco.mj_forward(self.model, self.data)
            # ？？？？？？？？？？？？？？？？？？
            # 关于mj_forward和mj_step的区别
            # mj_forward根据当前状态计算机器人现在是什么样,不进行时间积分
            # mj_step, 让时间往前走一步，更新状态，是一次完整的仿真循环
            # ？？？？？？？？？？？？？？？？？？

        self._step_count = 0
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple:
        """执行一步动作。

        Args:
            action: (8,) 数组 [arm_joint_0..6, gripper_ctrl]。
                    手臂部分为关节角度 (弧度)，
                    夹爪部分为执行器值 (0~255)。
        Returns:
            (observation, reward, done, info)
        """
        action = np.asarray(action, dtype=np.float64)

        with self._lock:
            # 设置控制信号 (执行器索引直接对应 ctrl 数组)
            for i in range(7):
                self.data.ctrl[self._actuator_ids[i]] = action[i] # 前七个控制器为arm
            # gripper ctrl
            self.data.ctrl[self._actuator_ids[7]] = np.clip(action[7], 0.0, 255.0) # 第八个为gripper

        # 等待若干仿真步
        for _ in range(CTRL_DECIMATION):
            time.sleep(SIM_TIMESTEP)

        obs = self.get_observation()
        reward = self._compute_reward()
        done = False
        info = {}
        self._step_count += 1

        # 类似强化学习，gym的env.step的返回值
        return obs, reward, done, info

    def get_observation(self) -> dict:
        """获取当前观测。

        Returns:
            dict with:
                - "image":          (H, W, 3) uint8 RGB 图像
                - "arm_qpos":       (7,)  手臂关节角度 (弧度)
                - "gripper_qpos":   (2,)  手指关节位置 (米)
                - "gripper_ctrl":   (1,)  夹爪执行器值
                - "cube_pos":       (3,)  方块世界坐标
                - "hand_pos":       (3,)  手爪世界坐标
        """
        with self._lock:
            # 渲染 RGB
            self.renderer.update_scene(self.data, camera=CAMERA_NAME)
            image = self.renderer.render().copy()

            # 本体感知 (通过 qpos 地址正确读取)
            arm_qpos = self.data.qpos[self._arm_qpos_adr].copy()
            gripper_qpos = self.data.qpos[self._gripper_qpos_adr].copy()
            gripper_ctrl = np.array([self.data.ctrl[self._actuator_ids[7]]])

            # 物体 & 手爪位置
            cube_pos = self.data.body(self._cube_body_id).xpos.copy()
            hand_pos = self.data.body(self._hand_body_id).xpos.copy()

        return {
            "image":        image,         # (H, W, 3) uint8
            "arm_qpos":     arm_qpos,      # (7,) float64
            "gripper_qpos": gripper_qpos,  # (2,) float64
            "gripper_ctrl": gripper_ctrl,  # (1,) float64
            "cube_pos":     cube_pos,      # (3,) float64
            "hand_pos":     hand_pos,      # (3,) float64
        }

    def _compute_reward(self) -> float:
        """计算抓取奖励：手爪越接近方块奖励越高。"""
        cube_pos = self.data.body(self._cube_body_id).xpos.copy()
        hand_pos = self.data.body(self._hand_body_id).xpos.copy()
        distance = np.linalg.norm(cube_pos - hand_pos) # 计算二范数
        return float(-distance) # 相当于自己定义的奖励函数 reward = float(-distance)

    # ============================================================
    # 便利方法
    # ============================================================

    def get_cube_position(self) -> np.ndarray:
        """获取方块的世界坐标。"""
        with self._lock:
            return self.data.body(self._cube_body_id).xpos.copy()

    def get_hand_position(self) -> np.ndarray:
        """获取手爪的世界坐标。"""
        with self._lock:
            return self.data.body(self._hand_body_id).xpos.copy()

    def set_cube_position(self, pos: np.ndarray):
        """设置方块的世界坐标 (x, y, z)。"""
        with self._lock:
            cube_start = N_ARM_JOINTS + 2
            self.data.qpos[cube_start:cube_start + 3] = np.asarray(pos, dtype=np.float64)
            mujoco.mj_forward(self.model, self.data)

    def get_arm_qpos_adr(self) -> np.ndarray:
        """返回手臂关节在 qpos 中的地址 (用于 IK)。"""
        return self._arm_qpos_adr.copy()

    def get_arm_dof_adr(self) -> np.ndarray:
        """返回手臂关节在 DOF 中的地址 (用于 Jacobian 切片)。"""
        return self._arm_dof_adr.copy()

    def get_hand_body_id(self) -> int:
        """返回手爪 body 的 id。"""
        return self._hand_body_id

    def get_model(self) -> mujoco.MjModel:
        """返回 MuJoCo 模型。"""
        return self.model

    def get_data(self) -> mujoco.MjData:
        """返回 MuJoCo 数据。"""
        return self.data

    def get_lock(self) -> threading.Lock:
        """返回线程锁。"""
        return self._lock

    # ============================================================
    # 清理
    # ============================================================

    def close(self):
        """关闭环境和可视化窗口。"""
        self._stop = True
        if self._render:
            self.viewer.close()
        self._sim_thread.join(timeout=2.0)
        if self._render:
            self._viewer_thread.join(timeout=2.0)
        print("[VLAEnv] 环境已关闭。")


# ============================================================
# 快速测试
# ============================================================
TEST = 1
if TEST:
    if __name__ == "__main__":
        env = VLAEnv(render=True)
        mj_model = env.get_model()
        mj_data = env.get_data()

        print(mj_model.nv)
        jacp = np.zeros((3, mj_model.nv))
        print(jacp.shape)
        mat = jacp.T @ jacp  # @表示矩阵乘法， *是逐个元素相乘
        print(mat.shape)

        _arm_joint_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
        arm_qpos_adr = np.array([mj_model.jnt_qposadr[jid] for jid in _arm_joint_ids], dtype=np.int32)

        _actuator_ids = np.array([mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES], dtype=np.int32)
        print(type(_arm_joint_ids))
        print(arm_qpos_adr)
        print(_actuator_ids)

        _cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        cube_pos = mj_data.body(_cube_body_id).xpos.copy()

        _hand_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        current = mj_data.xpos[_hand_body_id].copy()

        print(_hand_body_id, current)
        # try:
        #     obs = env.reset()
        #     print("观测键:", obs.keys())
        #     print("图像尺寸:", obs["image"].shape)
        #     print("方块位置:", obs["cube_pos"])
        #     print("手爪位置:", obs["hand_pos"])
        #     print("手臂关节:", obs["arm_qpos"])

        #     # 显示图像
        #     for _ in range(200):
        #         obs, reward, done, info = env.step(obs["arm_qpos"])
        #         img_bgr = cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR)
        #         cv2.imshow("VLAEnv", img_bgr)
        #         if cv2.waitKey(1) & 0xFF == 27:
        #             break

        # finally:
        #     env.close()
        #     cv2.destroyAllWindows()
