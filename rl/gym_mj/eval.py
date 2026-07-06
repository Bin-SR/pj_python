# eval.py — 模型评估与可视化
# ============================
# 加载训练好的模型，在 Hopper 环境中运行并渲染结果。

# import sys
# print(sys.path)

import os
import time
import numpy as np
import torch

from hopper.hopper_env import make_hopper_env
from hopper.hopper_env_stand import make_hopper_stand_env

from walker.walker2d_env import make_walker_env, get_env_info

from ppo.ppo_agent import PPOAgent
import config


def evaluate(model_path=None, num_episodes=5, render=True):
    """
    评估训练好的 PPO 模型。

    参数:
        model_path: 模型文件路径，默认加载最佳模型
        num_episodes: 评估的 episode 数量
        render: 是否渲染可视化
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] 使用: {device}")

    # 确定模型路径
    if model_path is None:
        model_path = os.path.join(config.SAVE_DIR, "walker_ppo_epoch700.pth")

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型。")
        return

    # 创建渲染环境
    render_mode = "human" if render else None
    env = make_walker_env(render_mode=render_mode, max_episode_steps=config.MAX_EPISODE_STEPS,)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 加载模型
    agent = PPOAgent(obs_dim, act_dim, device=device)
    agent.load(model_path)

    print(f"\n{'='*60}")
    print(f"开始评估 — {num_episodes} 个 episode")
    print(f"模型: {model_path}")
    print(f"{'='*60}\n")

    all_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False

        while not done:
            # 确定性策略（不采样，直接使用 Actor 输出的均值）
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step_count += 1

            if render:
                time.sleep(0.01)  # 控制渲染速度，便于观察

        all_rewards.append(episode_reward)
        print(f"Episode {ep + 1:2d}: 步数={step_count:4d}, 奖励={episode_reward:10.2f}")

    # 统计
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    print(f"\n{'='*60}")
    print(f"评估结果:")
    print(f"  平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  最高奖励: {max(all_rewards):.2f}")
    print(f"  最低奖励: {min(all_rewards):.2f}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估 PPO Hopper 模型")
    parser.add_argument("--model", type=str, default=None,help="模型文件路径（默认加载 ./models/hopper_ppo_best.pth）")
    parser.add_argument("--episodes", type=int, default=5,help="评估的 episode 数量")
    parser.add_argument("--no-render", action="store_true",help="不渲染可视化")
    args = parser.parse_args()

    evaluate(model_path=args.model,num_episodes=args.episodes,render=not args.no_render)
