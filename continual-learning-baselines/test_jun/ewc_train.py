import avalanche as avl
import torch
import os
import sys

sys.path.append("/home/jun/work/continual-learning-baselines")
from models.models import MLP  

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from experiments.utils import set_seed, create_default_args
from utils.misc import Logger

def ewc_pmnist_full_epoch(override_args=None):
    args = create_default_args({'cuda': 0, 'ewc_lambda': 1, 'hidden_size': 512,
                                'hidden_layers': 1, 'epochs': 50, 'dropout': 0,
                                'ewc_mode': 'separate', 'ewc_decay': None,
                                'learning_rate': 0.001, 'train_mb_size': 256,
                                'seed': None, 'save_freq': 10}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(checkpoint_dir, 'log.txt'), mode='a')
    
    # 전체 PermutedMNIST를 하나의 task로 묶어서 가져옴
    benchmark = avl.benchmarks.PermutedMNIST(1)
    experience = benchmark.train_stream[0]
    test_stream = benchmark.test_stream

    model = MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                drop_rate=args.dropout)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = avl.training.EWC(
        model, SGD(model.parameters(), lr=args.learning_rate), criterion,
        ewc_lambda=args.ewc_lambda, mode=args.ewc_mode, decay_factor=args.ewc_decay,
        train_mb_size=args.train_mb_size, train_epochs=1, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    results_summary = {}

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Start training epoch {epoch} =====")
        cl_strategy.train(experience) 
        res = cl_strategy.eval(test_stream)

        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer_state_dict': cl_strategy.optimizer.state_dict(),
                'results': res,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        for key, val in res.items():
            results_summary[key] = val

    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save({
        'net': model.state_dict(),
        'optimizer_state_dict': cl_strategy.optimizer.state_dict(),
        'results': results_summary,
    }, final_checkpoint_path)
    print(f"\nFinal checkpoint saved at {final_checkpoint_path}")

    print("\n==== Final Results ====")
    for metric, value in results_summary.items():
        print(f"{metric}: {value:.4f}")

    return results_summary


if __name__ == '__main__':
    ewc_pmnist_full_epoch()
