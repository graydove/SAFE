import torch

# 加载模型（注意设备映射）
checkpoint = torch.load('checkpoint-best2.pth', map_location=torch.device('cpu'))

# 查看所有层参数名称
for i in checkpoint:
    print(i)
for key in checkpoint['model'].keys():
    print(key)