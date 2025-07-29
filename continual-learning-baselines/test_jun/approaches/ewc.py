import sys, os, time
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader

# 프로젝트 루트의 utils.py를 사용하도록 import 경로를 정리합니다.
# 이 코드가 정상 동작하려면, ewc.py를 호출하는 메인 스크립트 상단에
# 프로젝트 루트를 sys.path에 추가하는 코드가 있어야 합니다.
import utils

class Appr(object):
    """ 
    Elastic Weight Consolidation (EWC) 구현체.
    Brainwash 프로젝트의 모든 스크립트와 호환되도록 수정된 최종 버전입니다.
    """
    def __init__(self, model, nepochs=100, sbatch=256, lr=0.01, clipgrad=100.0, lamb=1.0, args=None, **kwargs):
        self.model = model
        self.model_old = None
        self.fisher = None

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.clipgrad = clipgrad
        self.lamb = lamb
        self.args = args

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"EWC Approach Initialized with lamb: {self.lamb}, optim: {self.args.optim}")
        
    def _get_optimizer(self, lr=None):
        """인자로 받은 옵티마이저 종류에 따라 옵티마이저를 생성합니다."""
        if lr is None:
            lr = self.lr
        
        # args.optim은 main 스크립트에서 get_args()를 통해 전달됩니다.
        if self.args.optim == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif self.args.optim == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optim} is not supported.")

    def register_dummies_into_buffer(self):
        """ Fisher 정보를 저장할 빈 버퍼를 모델에 미리 등록합니다. """
        for n, p in self.model.named_parameters():
            buffer_name = '{}_fisher'.format(n.replace('.', '_'))
            self.model.register_buffer(buffer_name, torch.zeros_like(p))

    def load_from_buffers(self):
        """ 등록된 버퍼에서 Fisher 정보를 self.fisher 딕셔너리로 불러옵니다. """
        self.fisher = {}
        for n, p in self.model.named_parameters():
            buffer_name = '{}_fisher'.format(n.replace('.', '_'))
            self.fisher[n] = getattr(self.model, buffer_name)

    def load_model(self, state_dict):
        """ 
        [수정됨] 모델 상태 로드 함수. 
        Fisher 정보가 없는 state_dict도 받을 수 있도록 수정되었습니다.
        """
        # 1. Fisher 정보 저장을 위한 빈 버퍼를 먼저 모델에 등록합니다.
        self.register_dummies_into_buffer()
        
        # 2. strict=False 옵션으로, fisher 키가 없어도 오류 없이 로드합니다.
        self.model.load_state_dict(state_dict, strict=False)
        
        # 3. 로드된 버퍼에서 실제 Fisher 정보를 다시 가져옵니다.
        self.load_from_buffers()

        # 4. 과거 모델을 복사하고 동결합니다.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # utils.py에 해당 함수가 있어야 함

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data=None, input_size=None, taskcla=None):
        """
        [수정됨] 한 태스크에 대한 전체 학습 사이클.
        main_baselines.py와 main_ewc_train.py 모두와 호환되도록 인자를 받습니다.
        """
        self.model.to(self.device)
        self.optimizer = self._get_optimizer(self.lr)

        # Epoch 루프
        for e in range(self.nepochs):
            self.train_epoch(t, xtrain, ytrain)
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(f'| Task: {t:2d}, Epoch {e+1:3d}/{self.nepochs:3d} | Valid: loss={valid_loss:.3f}, acc={100*valid_acc:5.1f}%')

        # 학습 후 Fisher 정보 계산 및 저장
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)

        print("... Calculating Fisher Information ...")
        fisher_new = self._compute_fisher_matrix(t, xtrain, ytrain)

        if self.fisher is not None:
            for n in self.fisher.keys():
                self.fisher[n] = (self.fisher[n] * t + fisher_new[n]) / (t + 1)
        else:
            self.fisher = fisher_new
        
        # Fisher 정보를 모델의 버퍼에도 저장하여 pkl 파일에 함께 저장되도록 함
        for n, p in self.fisher.items():
            buffer_name = '{}_fisher'.format(n.replace('.', '_'))
            self.model.register_buffer(buffer_name, p.data.clone())
            
        print("... Fisher Information Calculated and Updated ...")
        return

    def train_epoch(self, t, x, y):
        self.model.train()
        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r)

        for i in range(0, len(r), self.sbatch):
            b = r[i:min(i + self.sbatch, len(r))]
            images = x[b].to(self.device)
            targets = y[b].to(self.device)

            outputs = self.model.forward(images)[t]
            loss = self.criterion(t, outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        return

    def eval(self, t, x, y):
        total_loss, total_acc, total_num = 0.0, 0.0, 0
        self.model.eval()
        
        r = np.arange(x.size(0))
        r = torch.LongTensor(r)
        
        with torch.no_grad():
            for i in range(0, len(r), self.sbatch):
                b = r[i:min(i + self.sbatch, len(r))]
                images = x[b].to(self.device)
                targets = y[b].to(self.device)

                outputs = self.model.forward(images)[t]
                loss = self.criterion(t, outputs, targets)
                _, pred = outputs.max(1)
                hits = (pred == targets).float()

                total_loss += loss.item() * len(b)
                total_acc += hits.sum().item()
                total_num += len(b)
                
        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):
        """EWC 페널티가 포함된 손실 함수"""
        loss_reg = 0.0
        # `model.train()` 모드이고, 이전 태스크(t>0)가 있을 때만 페널티 계산
        if self.model.training and t > 0:
            for name, param in self.model.named_parameters():
                if name in self.fisher:
                    # model_old가 None이 아닐 때만 페널티 계산
                    if self.model_old is not None:
                        param_old = self.model_old.get_parameter(name)
                        loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        
        return self.ce(output, targets) + self.lamb * loss_reg
    
    def _compute_fisher_matrix(self, t, x, y):
        """Fisher 정보 행렬의 대각 성분을 계산합니다."""
        fisher = {n: torch.zeros_like(p, requires_grad=False) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        
        dataloader = DataLoader(utils.TensorDataset(x, y), batch_size=self.sbatch, shuffle=True)
        
        for images, targets in dataloader:
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model.forward(images)[t]
            log_likelihood = torch.nn.functional.log_softmax(outputs, dim=1)
            
            sampled_labels = log_likelihood.max(1, keepdim=True)[1].squeeze()
            loss = torch.nn.functional.nll_loss(log_likelihood, sampled_labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                 if param.grad is not None:
                    fisher[name] += param.grad.data.clone().pow(2)

        num_samples = len(y)
        for name in fisher.keys():
            fisher[name] /= num_samples
            
        return fisher