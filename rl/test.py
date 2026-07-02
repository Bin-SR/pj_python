import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

print(f"观测空间: {env.observation_space}")
print(f"观测维度: {env.observation_space.shape[0]}")

print(f"动作空间: {env.action_space}")
print(f"动作维度: {env.action_space.shape}")