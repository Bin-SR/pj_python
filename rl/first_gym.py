import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

# print("状态数:", env.observation_space.n)
# 状态数16,即16个格子
# print("动作数:", env.action_space.n)
# 动作数4, 即0 1 2 3 分别是左 下 右 上

# 2表示右
# 奖励reward：到达终点为1，否则为0
# terminated表示任务是否结束，如掉坑则terminated=True
# truncated表示超时退出，当达到一定时间没有找到终点时，truncated=True
"""
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(2)

print("next_state", next_state)
print("reward", reward)
print("terminated", terminated)
"""

# 随机策略 不聪明
"""
state, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f'state={state} action={action}  next_state={next_state}  reward={reward} terminated={terminated} \n')

    state = next_state
    if terminated == True:
        break
"""
# 引入Q-table
# 16*4 的矩阵，即16个状态(即16个格子) 每个状态可以有4中action:上下左右
# eg. Q[2, 3]: 代表在状态2时执行动作3的未来总收益
# Q = np.random.rand(env.observation_space.n, env.action_space.n)
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.2
gamma = 0.95 
epoches = 5000
success = 0

for epoch in range(epoches):
    state, info = env.reset()
    done = False

    while not done:
        if np.random.rand() < 0.6:
            # 随机搜索
            print("    **********Random****************    ")
            action = env.action_space.sample()
        else:
            # 取当前Q的第state行中最大的未来收益值的索引，作为action
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncaterd, info = env.step(action)
        print(f'epoch={epoch} state={state} action={action}  next_state={next_state}  reward={reward} terminated={terminated} truncaterd={truncaterd}')
        best_next_q = np.max(Q[next_state])
        print("best_next_q: ", best_next_q)
        print("current_q:   ", Q[state, action])
        # Q-learning的核心公式
        Q[state, action] += alpha *  (reward + gamma * best_next_q - Q[state, action])
        print("eq:          ", reward + gamma * best_next_q - Q[state, action], '\n')

        state = next_state
        done = terminated or truncaterd
        
        if reward == 1:
            success += 1
            # input("Enter")
    print("Q[s, a]: ", Q[state, action])
    print("**************Fall hole********************\n\n")
print(Q)
print(success)

policy = np.argmax(Q, axis=1)
print(policy)

