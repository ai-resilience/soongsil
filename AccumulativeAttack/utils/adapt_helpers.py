import torch
import torch.nn as nn

from utils.train_helpers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def adapt_tensor(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		optimizer.zero_grad()
		logit = model(inputs)
		loss = criterion(logit, labels)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def adapt_tensor_reverse(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		optimizer.zero_grad()
		logit = model(inputs)
		loss = - args.poisoned_trigger_step * criterion(logit, labels)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def adapt_tensor_PT(model, poisoned_trigger, optimizer, niter=1, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		optimizer.zero_grad()
		for PT, para in zip(poisoned_trigger, model.parameters()):
			para.grad = PT
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type=2.0)
		optimizer.step()

def test_single(model, image, label):
    model.eval()
    inputs = te_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs.to(device))  # <-- 수정된 부분
        probs = nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)  # 가장 높은 확률과 인덱스를 반환
    correctness = 1 if predicted.item() == label else 0
    return correctness, confidence.item()
