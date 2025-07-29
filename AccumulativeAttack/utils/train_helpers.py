import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data

# 원본 학습과 동일하게 정규화 추가
normalize = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010)
)

# 훈련 데이터 변환 (data augmentation 유지 가능)
tr_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,  # 추가
])

# 테스트 데이터 변환 (augmentation 없음, 정규화 추가)
te_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize,  # 추가
])

def prepare_train_data(args, shuffle=False):
    print('Preparing data...')
    trset = datasets.CIFAR10(
        root=args.dataroot,  # ./data에서 args.dataroot로 수정
        train=True,
        download=True,
        transform=tr_transforms
    )
    trloader = torch.utils.data.DataLoader(
        trset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True
    )
    return trset, trloader

def prepare_test_data(args, shuffle=False):
    teset = datasets.CIFAR10(
        root=args.dataroot,  # ./data에서 args.dataroot로 수정
        train=False,
        download=True,
        transform=te_transforms  # 정규화 포함된 transform 사용
    )
    teloader = torch.utils.data.DataLoader(
        teset,
        batch_size=args.test_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True
    )
    return teset, teloader