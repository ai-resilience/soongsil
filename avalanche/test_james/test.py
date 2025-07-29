import torch
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0))
