import torch

length_loss_weight = torch.arange(2048)
length_loss_weight = torch.exp((length_loss_weight / 2048) * 4) - 1


print(length_loss_weight)