import avalanche as avl
import torch
import os
import sys

sys.path.append("/home/jun/work/continual-learning-baselines")
from models.models import MLP  

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics
from experiments.utils import set_seed, create_default_args
from utils.misc import Logger

def ewc_pmnist_full_epoch(override_args=None):
    args = create_default_args({
        'cuda': 0, 'ewc_lambda': 1, 'hidden_size': 512,
        'hidden_layers': 1, 'epochs': 50, 'dropout': 0,
        'ewc_mode': 'separate', 'ewc_decay': None,
        'learning_rate': 0.001, 'train_mb_size': 256,
        'seed': 0, 'save_freq': 10
    }, override_args)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(checkpoint_dir, 'log.txt'), mode='a')
    print(args)

    # Benchmark
    benchmark = avl.benchmarks.PermutedMNIST(1, seed=0)
    experience = benchmark.train_stream[0]
    test_stream = benchmark.test_stream

    # Model & Strategy
    model = MLP(hidden_size=args.hidden_size,
            hidden_layers=args.hidden_layers,
            drop_rate=args.dropout).to(device)

    # DataParallel로 감싸기
    model = torch.nn.DataParallel(model).to(device)

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[avl.logging.InteractiveLogger()]
    )

    cl_strategy = avl.training.EWC(
        model, optimizer, criterion,
        ewc_lambda=args.ewc_lambda, mode=args.ewc_mode, decay_factor=args.ewc_decay,
        train_mb_size=args.train_mb_size, train_epochs=1, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin
    )

    results_summary = {}

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Start training epoch {epoch} =====")
        cl_strategy.train(experience)
        res = cl_strategy.eval(test_stream)

        # get top1 acc
        err_cls = 1.0 - res["Top1_Acc_Stream/eval_phase/test_stream/Task000"]

        # save checkpoint
        if epoch % args.save_freq == 0:
            ckpt = {
                'epoch': epoch,
                'args': args,
                'err_cls': err_cls,
                'optimizer': optimizer.state_dict(),
                'net': model.state_dict()
            }
            ckpt_path = os.path.join(checkpoint_dir, f"epoch{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

        for key, val in res.items():
            results_summary[key] = val

    # Final save
    final_ckpt = {
        'epoch': args.epochs,
        'args': args,
        'err_cls': err_cls,
        'optimizer': optimizer.state_dict(),
        'net': model.state_dict()
    }
    final_ckpt_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(final_ckpt, final_ckpt_path)
    print(f"\nFinal checkpoint saved at {final_ckpt_path}")

    print("\n==== Final Results ====")
    for metric, value in results_summary.items():
        print(f"{metric}: {value:.4f}")

    return results_summary


if __name__ == '__main__':
    ewc_pmnist_full_epoch()
