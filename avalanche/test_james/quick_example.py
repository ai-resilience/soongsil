import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training import EWC

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model 정의
model = SimpleMLP(num_classes=10)

# SplitMNIST 벤치마크 설정
split_mnist = SplitMNIST(n_experiences=5)

train_stream = split_mnist.train_stream
test_stream = split_mnist.test_stream

print("Benchmark:", SplitMNIST.__name__)
print("Number of Experiences in train_stream:", len(train_stream))
for i, exp in enumerate(train_stream):
    print(f" - Experience {i}: Classes = {exp.classes_in_this_experience}, Task Label = {exp.task_label}")

# Optimizer, 손실 함수
optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
criterion = CrossEntropyLoss()

# EWC 전략 적용
cl_strategy = EWC(
    model, optimizer, criterion,
    ewc_lambda=1000,  # 강력한 EWC penalty로 과거 정보 유지 강화
    mode='separate',
    decay_factor=None,
    train_mb_size=32,
    train_epochs=5,   # epoch 수 증가
    eval_mb_size=32,
    device=device
)


# 학습 및 평가
results = []
for train_exp in train_stream:
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(test_stream))

# 결과 출력
for idx, res in enumerate(results):
    print(f"\n===== Result after Experience {idx} =====")
    print(f"Train Accuracy: {res['Top1_Acc_Epoch/train_phase/train_stream/Task000']:.4f}")
    print(f"Train Loss: {res['Loss_Epoch/train_phase/train_stream/Task000']:.4f}")

    print("\nTest Accuracy per Experience:")
    for exp_num in range(len(test_stream)):
        acc_key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{exp_num}"
        loss_key = f"Loss_Exp/eval_phase/test_stream/Task000/Exp00{exp_num}"
        print(f"  - Experience {exp_num}: Accuracy = {res[acc_key]:.4f}, Loss = {res[loss_key]:.4f}")

    print(f"\nStream Avg. Accuracy: {res['Top1_Acc_Stream/eval_phase/test_stream/Task000']:.4f}")
    print(f"Stream Avg. Loss: {res['Loss_Stream/eval_phase/test_stream/Task000']:.4f}")