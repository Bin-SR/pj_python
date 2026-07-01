import torch
import time
from pathlib import Path
from net import QNEtwork
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
model = Path(__file__).parent / "cartpole_double_dqn.pth"
net = QNEtwork()
net.load_state_dict(torch.load(model))

net.eval()

state, info = env.reset()
done = False
total_reward = 0

for episode in range(10): 
    state, info = env.reset()
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_value = net(state_tensor)
            action = q_value.argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)
        total_reward += reward
        done = terminated or truncated
        state = next_state

    print(f"episode={episode}, reward={total_reward}")
    done = False
    total_reward = 0

env.close()


