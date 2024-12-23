import torch

# 假设两个张量的形状都为 (batch_size, N)
batch_size = 5
N = 10
P = 0.7

# 创建两个二进制张量
tensor1 = torch.bernoulli(torch.full((batch_size, N), P)).long()
tensor2 = torch.bernoulli(torch.full((batch_size, N), P)).long()

# 执行按位抑或操作
result = tensor1 ^ tensor2

print(result)