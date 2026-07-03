import os
import sys
import time
import mujoco
import threading
import mujoco.viewer
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl", "hopper"))
from ppo_agent import PPOAgent

# class PPOAgent:
#     def __init__(self):
#         super().__init__()

#         self.agent = ppo_agent(obs_dim=11, act_dim=3, device="cpu")

def build_obs(mj_data):
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

class PPO_controller:
    def __init__(self, model_path, obs_dim=11, act_dim=3, device="cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.PPO_Agent = PPOAgent(obs_dim, act_dim, device)
        self.PPO_Agent.load(model_path)
        self.PPO_Agent.actor.eval()

    def get_action(self, obs, deterministic=True):
        action, _, _ = self.PPO_Agent.select_action(obs, deterministic=deterministic)
        return action



class Mj_sim:
    def __init__(self, xml_path,  model_path, timestep=0.005):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.mj_model.opt.timestep = timestep
        self.locker = threading.Lock()

        self.controller = PPO_controller(model_path)

        viewer_thread = threading.Thread(target=self._PhysicsViewerThread)
        sim_thread = threading.Thread(target=self._SimulationThread)

        viewer_thread.start()
        sim_thread.start()

    def _PhysicsViewerThread(self):
        while self.viewer.is_running():
            self.locker.acquire()
            self.viewer.sync()
            self.locker.release()
            time.sleep(0.02)  # ~50 FPS

    def _SimulationThread(self):
        step_count = 0

        while self.viewer.is_running():
            step_start = time.perf_counter()

            self.locker.acquire()

            obs = build_obs(self.mj_data)
            action = self.controller.get_action(obs, deterministic=True)
            self.mj_data.ctrl[:] = action
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

            elapsed = time.perf_counter() - step_start
            if elapsed < self.mj_model.opt.timestep:
                time.sleep(self.mj_model.opt.timestep - elapsed)

        print(f"[仿真] 线程退出，共 {step_count} 步")

if __name__ == "__main__":
    XML_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "hopper", "hopper.xml")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "rl", "hopper", "models", "hopper_ppo_best.pth")

    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("请先运行 rl/hopper/train.py 训练模型。")
        sys.exit(1)

    if not os.path.exists(XML_PATH):
        print(f"[错误] XML 文件不存在: {XML_PATH}")
        sys.exit(1)

    sim = Mj_sim(XML_PATH, MODEL_PATH)
        