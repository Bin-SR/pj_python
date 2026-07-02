# config.py — PPO + Hopper 超参数集中管理
# =============================================

# ---------- 环境参数 ----------
ENV_NAME = "Hopper-v5"          # Gymnasium MuJoCo Hopper 环境
RENDER_MODE = None              # 训练时不渲染 ("human" 可开启可视化)
MAX_EPISODE_STEPS = 1000        # 单回合最大步数

# ---------- PPO 超参数 ----------
GAMMA = 0.99                    # 折扣因子
LAMBDA = 0.95                   # GAE λ 参数
CLIP_EPSILON = 0.2              # PPO clip 范围
VALUE_COEF = 0.5                # 价值损失权重
ENTROPY_COEF = 0.01             # 熵正则系数（鼓励探索）
MAX_GRAD_NORM = 0.5             # 梯度裁剪阈值

# ---------- 训练参数 ----------
NUM_EPOCHS = 800               # 总训练回合数
STEPS_PER_EPOCH = 2048          # 每轮收集的步数
BATCH_SIZE = 64                 # 小批量大小
PPO_EPOCHS = 10                 # 每轮 PPO 更新次数
LEARNING_RATE = 3e-4            # Adam 学习率

# ---------- 网络结构 ----------
HIDDEN_SIZE = 64                # 隐藏层神经元数
ACTOR_HIDDEN = [64, 64]         # Actor 隐藏层
CRITIC_HIDDEN = [64, 64]        # Critic 隐藏层

# ---------- 保存 & 日志 ----------
SAVE_DIR = "C:/Disk/Data/New_Project/VScode_project/py/rl/hopper/models"         # 模型保存目录
LOG_INTERVAL = 10               # 每 N 回合打印一次日志
SAVE_INTERVAL = 100             # 每 N 回合保存一次模型
EVAL_INTERVAL = 50              # 每 N 回合评估一次
