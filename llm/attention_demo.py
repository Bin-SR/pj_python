import torch
import torch.nn.functional as F

# 假设三个Token，每个Embedding维度为4
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0]
])

print("输入Embedding：")
print(x)

# 为了简单
# Q = x
# K = x
# V = x

Wq = torch.randn(4,4)

Wk = torch.randn(4,4)

Wv = torch.randn(4,4)

Q = x @ Wq

K = x @ Wk

V = x @ Wv

score = torch.matmul(Q, K.T)

print("\nAttention Score：")
print(score)

weight = F.softmax(score, dim=-1)

print("\nAttention Weight：")
print(weight)

output = torch.matmul(weight, V)

print("\nAttention Output：")
print(output)