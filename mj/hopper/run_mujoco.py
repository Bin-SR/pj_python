# run_mujoco.py — 在原生 MuJoCo 中运行训练好的 PPO 模型
# ======================================================
# 不使用 Gymnasium，直接用 mujoco.MjModel.from_xml_path() 加载 Hopper，
# 用训练好的 PPO Actor 网络控制关节，在 viewer 中实时渲染。

import sys
import os
import time
import threading
import numpy as np

import torch
import mujoco
import mujoco.viewer

# 将 rl/hopper 目录加入 Python path，以便导入 PPO 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl", "hopper"))
from ppo_networks import ActorNetwork
from ppo_agent import PPOAgent


# ============================================================
# 路径配置
# ============================================================
XML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "hopper", "hopper.xml"
)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "rl", "hopper", "models", "hopper_ppo_best.pth"
)


# ============================================================
# 观测构建：从 MuJoCo qpos/qvel → Gymnasium 风格的 11 维观测
# ============================================================
def build_observation(mj_data):
    """
    从 MuJoCo 的 qpos 和 qvel 构造与训练时一致的 11 维观测。

    Gymnasium Hopper-v5 观测:
      obs[0]  = 躯干高度 z      (qpos[1])
      obs[1]  = 躯干倾角         (qpos[2])
      obs[2]  = 大腿关节角       (qpos[3])
      obs[3]  = 小腿关节角       (qpos[4])
      obs[4]  = 脚部关节角       (qpos[5])
      obs[5]  = 躯干 x 方向速度  (qvel[0])
      obs[6]  = 躯干 z 方向速度  (qvel[1])
      obs[7]  = 躯干角速度       (qvel[2])
      obs[8]  = 大腿关节角速度   (qvel[3])
      obs[9]  = 小腿关节角速度   (qvel[4])
      obs[10] = 脚部关节角速度   (qvel[5])

    参数:
        mj_data: MuJoCo MjData 对象

    返回:
        obs: numpy 数组 shape (11,)
    """
    obs = np.zeros(11, dtype=np.float32)
    obs[0] = mj_data.qpos[1]   # rootz — 躯干高度
    obs[1] = mj_data.qpos[2]   # rooty — 躯干倾角
    obs[2] = mj_data.qpos[3]   # thigh_joint
    obs[3] = mj_data.qpos[4]   # leg_joint
    obs[4] = mj_data.qpos[5]   # foot_joint
    obs[5] = mj_data.qvel[0]   # 躯干 vx
    obs[6] = mj_data.qvel[1]   # 躯干 vz
    obs[7] = mj_data.qvel[2]   # 躯干角速度
    obs[8] = mj_data.qvel[3]   # thigh 角速度
    obs[9] = mj_data.qvel[4]   # leg 角速度
    obs[10] = mj_data.qvel[5]  # foot 角速度
    return obs


# ============================================================
# PPO 控制器：加载 Actor 网络，根据观测输出动作
# ============================================================
class PPOController:
    """封装 PPO Actor，将观测映射为 MuJoCo 控制信号。"""

    def __init__(self, model_path, obs_dim=11, act_dim=3, device="cpu"):
        """
        参数:
            model_path: .pth 模型文件路径
            obs_dim: 观测维度
            act_dim: 动作维度
            device: 计算设备
        """
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载 PPO Agent（其中包含 Actor 网络）
        self.agent = PPOAgent(obs_dim, act_dim, device=device)
        self.agent.load(model_path)

        # 切换到评估模式
        # ????????? 为什么是eval 在actor中是 evaluate函数
        self.agent.actor.eval()
        # ?????????

    def get_action(self, obs, deterministic=True):
        """
        给定观测，返回动作。

        参数:
            obs: numpy 数组 [11]
            deterministic: True 使用均值（推荐），False 加入采样噪声

        返回:
            action: numpy 数组 [3]，范围 [-1, 1]
        """
        action, _, _ = self.agent.select_action(obs, deterministic=deterministic)
        return action


# ============================================================
# MuJoCo 仿真 + PPO 控制循环
# ============================================================
class HopperMujocoRunner:
    """
    加载 Hopper XML，加载 PPO 模型，运行仿真并实时控制。
    """

    def __init__(self, xml_path, model_path, timestep=0.005):
        """
        参数:
            xml_path: Hopper XML 文件路径
            model_path: 训练好的 .pth 模型路径
            timestep: 仿真步长 (s)
        """
        # --- 加载 MuJoCo ---
        print(f"[MuJoCo] 加载模型: {xml_path}")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = timestep

        # --- 加载 PPO ---
        print(f"[PPO] 加载模型: {model_path}")
        self.controller = PPOController(model_path, device="cpu")

        # --- Mujoco查看器 ---
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        # --- 线程同步 ---
        self.locker = threading.Lock()
        self.running = True

        print(f"[信息] 自由度: nq={self.mj_model.nq}, nv={self.mj_model.nv}, nu={self.mj_model.nu}")
        print(f"[信息] 关节: {[mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.mj_model.njnt)]}")
        print(f"[信息] 执行器: {[mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.mj_model.nu)]}")
        print(f"[控制] 按 R 键重置仿真 | 关闭窗口退出")
        print()

    def start(self):
        """启动仿真和渲染线程。"""
        sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)

        sim_thread.start()
        viewer_thread.start()

        # 主线程等待 viewer 关闭
        try:
            while self.viewer.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[退出] 用户中断")
        finally:
            self.running = False
            sim_thread.join(timeout=2.0)
            viewer_thread.join(timeout=2.0)
            print("[退出] 完成")

    def reset(self):
        """重置仿真到初始状态。"""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        print("[重置] 仿真已重置")

    def _simulation_loop(self):
        """仿真线程：步进物理，PPO 输出控制。"""
        step_count = 0

        while self.running and self.viewer.is_running():
            step_start = time.perf_counter()

            self.locker.acquire()

            # 1. 构建观测
            obs = build_observation(self.mj_data)

            # 2. PPO 推理获取动作
            action = self.controller.get_action(obs, deterministic=True)

            # 3. 写入控制信号（3 个电机：大腿、小腿、脚）
            self.mj_data.ctrl[:] = action

            # 4. 仿真步进
            mujoco.mj_step(self.mj_model, self.mj_data)

            step_count += 1

            # 5. 检测摔倒（躯干高度 < 0.7m 或倾角过大），自动重置
            torso_height = self.mj_data.qpos[1]
            torso_angle = abs(self.mj_data.qpos[2])
            if torso_height < 0.7 or torso_angle > 1.5:
                # 摔倒，自动重置
                if step_count > 50:  # 避免刚启动就重置
                    mujoco.mj_resetData(self.mj_model, self.mj_data)
                    step_count = 0

            self.locker.release()

            # 控制仿真频率
            elapsed = time.perf_counter() - step_start
            if elapsed < self.mj_model.opt.timestep:
                time.sleep(self.mj_model.opt.timestep - elapsed)

        print(f"[仿真] 线程退出，共 {step_count} 步")

    def _viewer_loop(self):
        """渲染线程：同步并刷新 viewer，处理键盘输入。"""
        while self.running and self.viewer.is_running():
            self.locker.acquire()
            self.viewer.sync()
            self.locker.release()
            time.sleep(0.02)  # ~50 FPS


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("请先运行 rl/hopper/train.py 训练模型。")
        sys.exit(1)

    if not os.path.exists(XML_PATH):
        print(f"[错误] XML 文件不存在: {XML_PATH}")
        sys.exit(1)

    runner = HopperMujocoRunner(
        xml_path=XML_PATH,
        model_path=MODEL_PATH,
        timestep=0.005,
    )
    runner.start()
