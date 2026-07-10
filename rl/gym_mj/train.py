# train.py — PPO 训练主循环
# ==========================
# 在 Hopper-v5 上训练 PPO 智能体，使 hopper 稳定站立并前进。

import os
import time
import numpy as np
import torch

from hopper.hopper_env import make_hopper_env, get_env_info
from hopper.hopper_env_stand import make_hopper_stand_env

from walker.walker2d_env import make_walker_env

from half_cheetah.halfCheetah_env import make_HalfCheetah_env

from ppo.ppo_agent import PPOAgent
import config


def train():
    """主训练函数。"""
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] 使用: {device}")

    # 创建环境
    env = make_HalfCheetah_env(render_mode=config.RENDER_MODE, max_episode_steps=config.MAX_EPISODE_STEPS)
    get_env_info(env)
    
    # 观测维度和动作维度
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 创建模型保存目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 初始化 PPO 智能体， 包括构建actor, critic网络和优化器，初始化buffer
    agent = PPOAgent(obs_dim, act_dim, device=device)

    # 训练统计
    episode_rewards = []      # 每个 episode 的总奖励
    best_avg_reward = -float("inf")
    total_steps = 0

    print(f"\n{'='*60}")
    print(f"开始训练 — 共 {config.NUM_EPOCHS} 轮，每轮 {config.STEPS_PER_EPOCH} 步")
    print(f"{'='*60}\n")

    start_time = time.time()

    obs, _ = env.reset()  # obs就是state
    episode_reward = 0.0  # episode表示一局完整的尝试，
    # 一个epoch是一个训练周期，epoch中有多个episode， 一个episode中有多个step

    for epoch in range(config.NUM_EPOCHS):
        epoch_rewards = []

        # --- 收集经验 (on-policy) ---
        # 每个epoch里面有config.STEPS_PER_EPOCH=2048步
        for step in range(config.STEPS_PER_EPOCH):
            # 选择动作
            # 在select_action函数中，相当于进行了A2C的操作
            # 内部：actor网络初步获取action(有tanh压缩后的), critic网络评估当前状态state获取value
            action, log_prob, value = agent.select_action(obs)

            # 环境步进
            # 和之前学的next_state, reward, terminated, truncated, info = env.step(action)相同
            next_obs, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated

            # 存储经验
            agent.store_transition(obs, action, reward, value, log_prob, done)

            # 累计一次完整的尝试episode所得到的reward
            episode_reward += reward
            obs = next_obs  # state = next_state
            total_steps += 1

            # Episode 结束时重置环境， 当done = True时就表示一次尝试结束，episode + 1
            if done:
                # 把每个epoch中的每一个完整的episode所累积的episode_reward存入epoch_rewards中
                epoch_rewards.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0.0

        # --- PPO 更新 ---
        metrics = agent.update()

        # --- 日志输出 ---
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        episode_rewards.append(avg_reward)

        if (epoch + 1) % config.LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            recent_avg = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(
                f"Epoch {epoch + 1:4d}/{config.NUM_EPOCHS} | "
                f"Avg Reward: {recent_avg:8.2f} | "
                f"Episodes: {len(epoch_rewards):2d} | "
                f"Actor Loss: {metrics['actor_loss']:.4f} | "
                f"Critic Loss: {metrics['critic_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        # --- 保存最佳模型 ---
        if avg_reward > best_avg_reward and len(epoch_rewards) > 0:
            best_avg_reward = avg_reward
            agent.save(os.path.join(config.SAVE_DIR, "HalfCheetah_ppo_best.pth"))

        # --- 定期保存 ---
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            agent.save(
                os.path.join(config.SAVE_DIR, f"HalfCheetah_ppo_epoch{epoch+1}.pth")
            )

    # 训练结束
    total_time = time.time() - start_time
    agent.save(os.path.join(config.SAVE_DIR, "HalfCheetah_ppo_final.pth"))

    print(f"\n{'='*60}")
    print(f"训练完成！总耗时: {total_time:.1f}s")
    print(f"最佳平均奖励: {best_avg_reward:.2f}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    train()
