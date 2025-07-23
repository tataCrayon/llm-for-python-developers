import torch
# 检查 GPU 可用性
print(torch.cuda.is_available())  # 应返回 True
print(torch.cuda.device_count())  # 检查 GPU 数量