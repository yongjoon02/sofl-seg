import torch

if not torch.cuda.is_available():
    print("CUDA unavailable")
else:
    print("device_count:", torch.cuda.device_count())
    cur = torch.cuda.current_device()
    print("current_device:", cur)
    print("current_name:", torch.cuda.get_device_name(cur))
    # 전체 목록
    for i in range(torch.cuda.device_count()):
        print(f"{i}: {torch.cuda.get_device_name(i)}")
