import torch

ckpt_path = '/home/jun/work/AccumulativeAttack/checkpoints_base_bn/epoch40.pth'
ckpt = torch.load(ckpt_path)

print("\n📦 Checkpoint keys:")
for key in ckpt:
    print(f"- {key}")

print("\n🔍 Checking each key in detail:\n")

# 1. Epoch
if 'epoch' in ckpt:
    print(f"✅ epoch: {ckpt['epoch']}")

# 2. Error
if 'err_cls' in ckpt:
    print(f"✅ err_cls: {ckpt['err_cls']}")

# 3. Args (argparse.Namespace)
if 'args' in ckpt:
    print("✅ args:")
    for arg_key, arg_val in vars(ckpt['args']).items():
        print(f"  - {arg_key}: {arg_val}")

# 4. Optimizer state
if 'optimizer' in ckpt:
    print("✅ optimizer keys:")
    for opt_key in ckpt['optimizer'].keys():
        print(f"  - {opt_key}")

# 5. Net state_dict
if 'net' in ckpt:
    print("✅ net state_dict keys:")
    for net_key in ckpt['net'].keys():
        print(f"  - {net_key}")
else:
    print("⚠️ 'net' key not found in checkpoint.")
