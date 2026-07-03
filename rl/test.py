import gymnasium as gym
# "Hopper-v5" "CartPole-v1"
env = gym.make("Hopper-v5", render_mode="human")
print(env.unwrapped.model)
print(f"观测空间: {env.observation_space}")
print(f"观测维度: {env.observation_space.shape[0]}")

print(f"动作空间: {env.action_space}")
print(f"动作维度: {env.action_space.shape}")