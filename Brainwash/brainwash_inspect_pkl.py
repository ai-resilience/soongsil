import pickle
import torch
import numpy as np
import sys

# 사용자님께서 제공한 파일 경로
file_path = '/home/jun/work/Brainwash/afec_ewc_lamb_500000.0_lambemp_100.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl'

print(f"--- 파일 로딩 시작: {file_path} ---\n")

try:
    # 'rb'는 'read binary' 모드를 의미합니다.
    with open(file_path, 'rb') as f:
        # pickle.load를 사용해 파일에서 파이썬 객체를 불러옵니다.
        data = pickle.load(f)

    print(">>> 파일 로딩 성공! <<<\n")

    # 불러온 데이터의 타입 확인 (아마도 dict 타입일 것입니다)
    print(f"1. 데이터 전체 타입: {type(data)}\n")

    if isinstance(data, dict):
        print("2. 저장된 데이터의 주요 키(Keys):")
        # 딕셔너리의 모든 키를 출력하여 어떤 정보가 저장되었는지 확인합니다.
        print(list(data.keys()))
        print("-" * 50)

        print("3. 각 키(Key)에 대한 상세 정보:\n")
        # 각 키와 값의 타입, 형태 등을 자세히 출력합니다.
        for key, value in data.items():
            print(f"  - 키: '{key}'")
            
            # 값의 타입에 따라 다른 정보를 출력
            if isinstance(value, dict):
                print(f"    - 타입: dict")
                print(f"    - 내부 키 개수: {len(value)}")
                # 딕셔너리 내부의 키들도 보여줍니다 (너무 많으면 일부만)
                print(f"    - 내부 키 (최대 5개): {list(value.keys())[:5]}")
            elif isinstance(value, np.ndarray):
                print(f"    - 타입: numpy.ndarray")
                print(f"    - 형태 (shape): {value.shape}")
                print(f"    - 데이터 타입 (dtype): {value.dtype}")
            elif isinstance(value, torch.Tensor):
                print(f"    - 타입: torch.Tensor")
                print(f"    - 형태 (shape): {value.shape}")
                print(f"    - 데이터 타입 (dtype): {value.dtype}")
            elif isinstance(value, list):
                print(f"    - 타입: list")
                print(f"    - 리스트 길이: {len(value)}")
            else:
                # 그 외 (숫자, 문자열 등)
                print(f"    - 타입: {type(value)}")
                print(f"    - 값: {value}")
            
            print() # 가독성을 위한 줄바꿈

except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")