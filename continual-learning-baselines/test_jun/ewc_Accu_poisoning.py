import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
sys.path.append("/home/jun/work/AccumulativeAttack")
from online_accu_train import craft_tri, craft_accu
from utils.adapt_helpers import adapt_tensor
from utils.test_helpers import test
from utils.model import resnet18
from utils.misc import Logger, my_makedir, setup_seed
from utils.train_helpers import prepare_train_data, prepare_test_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
# 주요 하이퍼파라미터
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--test_batch_size', default=500, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--epsilon', default=0.3, type=float)
parser.add_argument('--num_steps', default=5, type=int)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--threshold', default=0.18, type=float)
parser.add_argument('--poison_scale', default=1.0, type=float)

# 기타 옵션
parser.add_argument('--resume', default='checkpoints_base_bn')
parser.add_argument('--model_name', default='epoch50.pth')
parser.add_argument('--log_name', default='log_poison.txt')
parser.add_argument('--onlinemode', default='train', type=str)
parser.add_argument('--mode', default='eval', type=str)
parser.add_argument('--roundsign', action='store_true')
parser.add_argument('--only_reg', action='store_true')
parser.add_argument('--only_normal', action='store_true')
parser.add_argument('--only_second', action='store_true')
parser.add_argument('--clip_gradnorm', action='store_true')
parser.add_argument('--clipvalue', default=1.0, type=float)
parser.add_argument('--use_advtrigger', action='store_true')
parser.add_argument('--use_online_advtrigger', action='store_true')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--shuffle', action='store_true')

args = parser.parse_args()

# ✅ Setup
my_makedir(args.resume)
setup_seed(args.seed)
sys.stdout = Logger(os.path.join(args.resume, args.log_name), mode='a')
cudnn.benchmark = False
print(args)

# ✅ Model
def gn_helper(planes): return nn.GroupNorm(32, planes)
norm_layer = gn_helper
net = resnet18(num_classes=10, norm_layer=norm_layer).to(device)
net = torch.nn.DataParallel(net)

# ✅ Load checkpoint
print(f"Resuming from {args.resume}/{args.model_name}")
ckpt = torch.load(os.path.join(args.resume, args.model_name))
net.load_state_dict(ckpt['model_state_dict'])
print("Checkpoint loaded. Epoch:", ckpt['epoch'], "Error:", ckpt['err_cls'])

# ✅ Optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

# ✅ Data
trset, trloader = prepare_train_data(args, shuffle=args.shuffle)
teset, teloader = prepare_test_data(args)

# ✅ Main accumulative poisoning attack
def main_accu():
    err = test(teloader, net, verbose=True, print_freq=0)
    print("Initial test error: %.4f" % err)

    dt_val = next(iter(teloader))
    data_val, y_val = dt_val[0].to(device), dt_val[1].to(device)
    dt_tri = next(iter(trloader))
    data_tri, y_tri = dt_tri[0].to(device), dt_tri[1].to(device)

    # ✅ Craft initial trigger
    if args.use_advtrigger:
        data_tri_adv = craft_tri(net, data_tri, data_val, y_tri, y_val)
        data_tri_adv = data_tri_adv.detach()
        if args.poison_scale < 1:
            normal_indices = torch.randperm(len(data_tri))[int(args.poison_scale * len(data_tri)):]
            data_tri_adv[normal_indices] = data_tri[normal_indices]
        data_tri = data_tri_adv.clone()

    for epoch in range(args.epochs):
        dt = next(iter(trloader))
        data_train, y_train = dt[0].to(device), dt[1].to(device)

        data_train_adv = craft_accu(net, data_tri.detach(), data_train, data_val, y_tri, y_train, y_val)
        data_train_adv = data_train_adv.detach()
        if args.poison_scale < 1:
            data_train_adv[normal_indices] = data_train[normal_indices]

        adapt_tensor(net, data_train_adv, y_train, optimizer, criterion,
                     args.niter, args.batch_size, args.onlinemode, args)

        err = test(teloader, net, verbose=True, print_freq=0)
        print(f"Epoch:{epoch} Test error: {err:.4f}")

        if err > args.threshold:
            print("Threshold reached. Stopping.")
            break

        # ✅ Online trigger update
        if args.use_online_advtrigger:
            data_tri_adv = craft_tri(net, data_tri, data_val, y_tri, y_val)
            data_tri_adv = data_tri_adv.detach()
            if args.poison_scale < 1:
                data_tri_adv[normal_indices] = data_tri[normal_indices]
            data_tri = data_tri_adv.clone()

    # Final evaluation
    print("Test error before final tri:", test(teloader, net, verbose=True, print_freq=0))
    adapt_tensor(net, data_tri, y_tri, optimizer, criterion,
                 args.niter, args.batch_size, args.onlinemode, args)
    print("Test error after final tri:", test(teloader, net, verbose=True, print_freq=0))


if __name__ == '__main__':
    main_accu()
