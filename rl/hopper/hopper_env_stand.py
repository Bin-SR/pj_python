# hopper_env_stand.py — MuJoCo Hopper 站立任务环境封装
# ======================================================
# 基于 Gymnasium 的 Hopper-v5，添加自定义站立奖励 Wrapper，
# 使智能体学会稳定站立而非跑步前进。

import gymnasium as gym
import numpy as np


# ============================================================
# 自定义奖励 Wrapper：稳定站立
# ============================================================
class StandRewardWrapper(gym.Wrapper):
    """
    将 Hopper 默认的「前进」奖励替换为「稳定站立」奖励。

    观测空间 (11 维):
        obs[0] = 躯干高度 z (m)        — 站立时约 1.25
        obs[1] = 躯干倾角 (rad)        — 站立时接近 0
        obs[2] = 大腿关节角 (rad)
        obs[3] = 小腿关节角 (rad)
        obs[4] = 脚部关节角 (rad)
        obs[5] = 躯干 x 方向速度 (m/s)  — 站立时接近 0
        obs[6] = 躯干 z 方向速度 (m/s)  — 站立时接近 0
        obs[7] = 躯干角速度 (rad/s)
        obs[8] = 大腿关节角速度
        obs[9] = 小腿关节角速度
        obs[10]= 脚部关节角速度

    奖励设计:
        + 高度奖励:  躯干高度越接近目标高度，奖励越高
        + 姿态奖励:  躯干倾角越小（越竖直），奖励越高
        − 速度惩罚:  水平 / 垂直速度越小越好
        − 控制惩罚:  动作幅度越小越好（省力）
    """

    def __init__(self, env, target_height=1.25):
        """
        参数:
            env: 原始 Gymnasium Hopper 环境
            target_height: 目标站立高度 (m)
        """
        super().__init__(env)
        self.target_height = target_height

        # 奖励权重（可调）
        self.w_height = 10.0     # 高度奖励权重
        self.w_upright = 5.0     # 竖直姿态奖励权重
        self.w_vel_penalty = 1.0 # 速度惩罚权重
        self.w_ctrl_penalty = 0.1# 控制代价权重

    def step(self, action):
        """
        执行一步，返回自定义奖励。

        参数:
            action: 动作向量 [3]

        返回:
            obs, custom_reward, terminated, truncated, info
        """
        # 调用原始环境 step，拿到观测和终止标志
        obs, _, terminated, truncated, info = self.env.step(action)

        # ---- 解析观测 ----
        torso_height = obs[0]           # 躯干高度
        torso_angle  = obs[1]           # 躯干倾角
        torso_vel_x  = obs[5]           # 躯干水平速度
        torso_vel_z  = obs[6]           # 躯干垂直速度

        # ---- 自定义奖励 ----
        # 1. 高度奖励：高斯型，越接近目标高度越接近 1
        height_error = abs(torso_height - self.target_height)
        reward_height = np.exp(-5.0 * height_error)

        # 2. 姿态奖励：倾角越小越好
        reward_upright = np.exp(-3.0 * abs(torso_angle))

        # 3. 速度惩罚：水平 + 垂直速度平方和
        penalty_vel = torso_vel_x ** 2 + torso_vel_z ** 2

        # 4. 控制代价：动作幅度平方和
        penalty_ctrl = np.sum(action ** 2)

        # 组合总奖励
        reward = (
            self.w_height * reward_height
            + self.w_upright * reward_upright
            - self.w_vel_penalty * penalty_vel
            - self.w_ctrl_penalty * penalty_ctrl
        )

        # 将自定义奖励写入 info，便于调试
        info["stand_reward"] = reward
        info["height_error"] = height_error

        return obs, reward, terminated, truncated, info


# ============================================================
# 环境工厂函数
# ============================================================
def make_hopper_stand_env(render_mode=None, max_episode_steps=1000, target_height=1.25):
    """
    创建带站立奖励的 Hopper 环境。

    参数:
        render_mode: 渲染模式 (None / "human")
        max_episode_steps: 单回合最大步数
        target_height: 目标站立高度 (m)，默认 1.25

    返回:
        env: 包装了 StandRewardWrapper 的环境对象
    """
    # 创建原始环境
    raw_env = gym.make(
        "Hopper-v5",
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    # 套上站立奖励 Wrapper
    env = StandRewardWrapper(raw_env, target_height=target_height)
    return env


def get_env_info(env):
    """
    打印环境的基本信息，便于调试。
    """
    # 如果被 Wrapper 包裹，找到最底层的 env
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    print("=" * 50)
    print(f"环境名称: Hopper-v5 (站立任务)")
    print(f"观测空间: {base_env.observation_space}")
    print(f"观测维度: {base_env.observation_space.shape[0]}")
    print(f"动作空间: {base_env.action_space}")
    print(f"动作维度: {base_env.action_space.shape[0]}")
    print(f"动作范围: {base_env.action_space.low} ~ {base_env.action_space.high}")
    print(f"目标站立高度: {getattr(env, 'target_height', 'N/A')} m")
    print("=" * 50)


if __name__ == "__main__":
    # 快速测试：创建环境并运行随机策略
    TEST = 0
    if TEST:
        print("**********hopper_env_stand.py : TEST*********")
        env = make_hopper_stand_env(render_mode="human", max_episode_steps=1000)
        get_env_info(env)

        obs, info = env.reset()
        total_reward = 0.0

        for step in range(500):
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"回合结束，步数: {step + 1}, 总奖励: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0.0

        env.close()
