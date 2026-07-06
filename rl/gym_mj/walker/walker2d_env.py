# hopper_env.py — MuJoCo Hopper 环境封装
# ======================================
# 基于 Gymnasium 的 Hopper-v5，提供统一的 step / reset 接口，
# 方便后续替换或自定义奖励函数。

import gymnasium as gym
import numpy as np


def make_walker_env(render_mode=None, max_episode_steps=1000):
    """
    创建 Hopper 环境。

    参数:
        render_mode: 渲染模式，None 为不渲染，"human" 为可视化
        max_episode_steps: 单回合最大步数

    返回:
        env: Gymnasium 环境对象
    """
    env = gym.make(
        "Walker2d-v5",
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    return env


def get_env_info(env):
    """
    打印环境的基本信息，便于调试。

    参数:
        env: Gymnasium 环境对象
    """
    print("=" * 50)
    print(f"环境名称: Walker2d-v5")
    print(f"观测空间: {env.observation_space}")
    print(f"观测维度: {env.observation_space.shape[0]}")

    print(f"动作空间: {env.action_space}")
    print(f"动作维度: {env.action_space.shape[0]}")
    print(f"动作范围: {env.action_space.low} ~ {env.action_space.high}")
    print("=" * 50)


if __name__ == "__main__":
    # 快速测试：创建环境并运行随机策略
    env = make_walker_env(render_mode="human", max_episode_steps=1000)
    get_env_info(env)

    obs, info = env.reset()
    # print(obs.shape)
    total_reward = 0.0

    for step in range(1000):
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"回合结束，步数: {step + 1}, 总奖励: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
