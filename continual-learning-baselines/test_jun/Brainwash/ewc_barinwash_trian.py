import sys, os, time
import numpy as np
import pickle as pkl
import torch

# --- [최종 수정] 경로 문제 자동 해결 ---
# 이 스크립트가 어디에 있든 프로젝트 루트를 자동으로 찾아 경로에 추가합니다.
project_root = '/home/jun/work/Brainwash' 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import utils 
from approaches.arguments import get_args
from resnet import ResNet18


tstart = time.time()

def main(args):
    # 0. 초기 설정 (시드 고정, 디바이스 설정)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        print('[CUDA unavailable]'); sys.exit()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. 데이터셋 준비
    if args.experiment == 'split_cifar100':
        from approaches.data_utils import generate_split_cifar100_tasks
        data, taskcla, inputsize, task_order = generate_split_cifar100_tasks(args.tasknum, args.seed)
    else:
        raise NotImplementedError("해당 데이터셋을 지원하지 않습니다: " + args.experiment)
    print('\nInput size =', inputsize, '\nTask info =', taskcla)

    # 2. 학습 방법론(Approach) 선택
    if args.approach == 'ewc':
        from approaches import ewc as approach
    else:
        raise NotImplementedError("해당 접근법을 지원하지 않습니다: " + args.approach)
    
    # 3. 모델 초기화
    print('Inits...')
    # ResNet18 호출 방식을 수정하여 TypeError 방지
    net = ResNet18(args.tasknum, data['ncla']//args.tasknum).to(device)

    # 4. EWC 전략 객체 생성
    appr = approach.Appr(model=net, lamb=args.lamb, sbatch=args.batch_size, lr=args.lr,
                         nepochs=args.nepochs, args=args, clipgrad=args.clip)

    # 정확도와 손실을 기록할 행렬 초기화
    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

    # 5. 순차적 학습 루프
    for t, ncla in taskcla:
        if t >= args.lasttask:
            break

        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        xtrain = data[t]['train']['x']
        ytrain = data[t]['train']['y']
        xvalid = data[t]['test']['x']
        yvalid = data[t]['test']['y']
        
        # Train
        appr.train(t, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
        print('-' * 100)

        # Test
        for u in range(t + 1):
            xtest = data[u]['test']['x']
            ytest = data[u]['test']['y']
            test_loss, test_acc = appr.eval(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss
        
        print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t, :t+1])))
        with np.printoptions(precision=2, suppress=True):
            print("Accuracy Matrix:\n", acc)
    
    # 6. 최종 결과 출력 및 저장
    print('*' * 100)
    print('Accuracies =')
    last_trained_task_idx = args.lasttask - 1
    for i in range(last_trained_task_idx + 1):
        print('\t', end='')
        for j in range(last_trained_task_idx + 1):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    
    # BWT (Backward Transfer) 및 평균 정확도 계산
    avg_acc_after = np.mean(acc[last_trained_task_idx, :args.lasttask])
    diag_acc = np.diag(acc)
    bwt = np.mean((acc[last_trained_task_idx, :last_trained_task_idx] - diag_acc[:last_trained_task_idx])) if last_trained_task_idx > 0 else 0

    print(f'After Avg Acc: {100 * avg_acc_after:.1f}%, After BWT: {100 * bwt:.2f}%')

    # --- [최종 수정] 기존 코드와 동일한 구조로 save_dict 생성 ---
    save_dict = {}
    # getattr(args, 'scenario_name', None)은 args에 scenario_name이 없으면 None을 반환하여 오류를 방지
    save_dict['scenario'] = getattr(args, 'scenario_name', None) 
    save_dict['model_type'] = 'resnet'
    save_dict['dataset'] = args.experiment
    save_dict['class_num'] = data['ncla'] // args.tasknum
    save_dict['bs'] = args.batch_size
    save_dict['lr'] = args.lr
    save_dict['n_epochs'] = args.nepochs
    # model.state_dict()는 파라미터와 함께 등록된 버퍼(fisher 정보)도 모두 저장합니다.
    save_dict['model'] = net.state_dict() 
    save_dict['model_name'] = net.__class__.__name__
    # 기존 파일은 task_num에 lasttask 값을 저장했으므로 동일하게 맞춰줍니다.
    save_dict['task_num'] = args.lasttask 
    save_dict['task_order'] = task_order
    save_dict['seed'] = args.seed
    save_dict['emb_fact'] = 1 # CIFAR100의 경우 1로 가정
    save_dict['im_sz'] = inputsize[1]
    
    # cont_method_args 딕셔너리를 생성하고 'method' 키를 명시적으로 추가
    cont_method_args = vars(args)
    cont_method_args['method'] = args.approach
    save_dict['cont_method_args'] = cont_method_args
    
    save_dict['last_task'] = args.lasttask
    save_dict['acc_mat'] = acc
    save_dict['avg_acc'] = avg_acc_after
    save_dict['bwt'] = bwt
    # KeyError 방지를 위해 추가
    save_dict['optim_name'] = args.optim
    # -----------------------------------------------------------------
    
    # 저장 폴더 및 파일 이름 설정
    save_dir = '/home/jun/work/continual-learning-baselines/test_jun/Brainwash' 
    os.makedirs(save_dir, exist_ok=True)
    
    # 기존 파일 이름 생성 방식과 유사하게 수정
    file_name_parts = [
        args.approach,
        f"lamb_{args.lamb}",
        f"dataset_{args.experiment}",
        f"seed_{args.seed}",
        f"task_num_{args.lasttask}"
    ]
    file_name = "_".join(file_name_parts) + ".pkl"
    
    print('Saving results to: ', os.path.join(save_dir, file_name))
    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pkl.dump(save_dict, f)

if __name__ == '__main__':
    args = get_args()
    if not hasattr(args, 'optim'):
        args.optim = 'sgd'
    main(args)