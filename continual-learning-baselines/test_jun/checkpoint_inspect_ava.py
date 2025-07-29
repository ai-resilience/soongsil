import torch

# 체크포인트 파일 경로
ckpt_path = '/home/jun/work/continual-learning-baselines/test_jun/checkpoints_cifar10_resnet/epoch50.pth'

# 체크포인트 로드
ckpt = torch.load(ckpt_path, map_location='cpu')

# 최상위 키 출력 (구조화된 형태로 예쁘게 출력)
print("Checkpoint Structure:\n" + "="*30)
for key in ckpt.keys():
    print(f"▶ {key}")

    if isinstance(ckpt[key], dict):
        for subkey in ckpt[key].keys():
            print(f"  └─ {subkey}")
    else:
        print(f"  └─ Type: {type(ckpt[key])}")

print("="*30 + "\nCheckpoint Inspection Completed.")