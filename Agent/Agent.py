import subprocess
from openai import OpenAI
import json
from datetime import datetime
from typing import Union, List
import re
import random
import os

# OpenAI 클라이언트 초기화. API 키를 환경 변수나 다른 보안 방법으로 관리하는 것을 권장합니다.
# 예: client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key="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") # 실제 API 키를 이곳에 입력하거나 환경 변수를 사용하세요.

def run_script_and_get_output(command: str, step_title: str) -> str:
    """
    주어진 셸 명령어를 실행하고, 표준 출력을 문자열로 반환합니다.
    오류가 발생하면 오류 메시지를 반환합니다.
    """
    print(f"--- {step_title} 실행 시작 ---")
    print(f"실행 명령어: {command}")
    
    try:
        # 셸 명령어를 실행하고 출력을 캡처합니다.
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            check=True, encoding='utf-8'
        )
        print("--- 실행 성공 ---")
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        # 명령어 실행 중 오류가 발생한 경우
        print("--- 실행 중 오류 발생 ---")
        error_message = (f"명령어 실행에 실패했습니다 (Exit Code: {e.returncode}).\n"
                         f"--- STDOUT ---\n{e.stdout}\n"
                         f"--- STDERR ---\n{e.stderr}\n")
        return error_message
    except Exception as e:
        # 기타 예외 처리
        print(f"--- 알 수 없는 오류 발생 ---")
        return f"스크립트 실행 중 예외가 발생했습니다: {e}"

def format_matrix_as_table(matrix: List[List[float]], num_tasks: int) -> str:
    """정확도 매트릭스(리스트의 리스트)를 마크다운 테이블 문자열로 변환합니다."""
    if not matrix or not all(isinstance(row, list) for row in matrix):
        return "테이블을 생성할 정확도 데이터가 없습니다."

    header = "| Task   | " + " | ".join([f"Task {i}" for i in range(num_tasks)]) + " |"
    separator = "|--------|" + "--------|" * num_tasks
    
    rows = [header, separator]
    for i, row_data in enumerate(matrix):
        # 각 행의 길이를 num_tasks에 맞게 조정하고 빈 값은 '-'로 채웁니다.
        full_row_data = row_data + ['-'] * (num_tasks - len(row_data))
        row_str = f"| Task {i} | " + " | ".join([f"{val:.2f}%" if isinstance(val, float) else str(val) for val in full_row_data]) + " |"
        rows.append(row_str)
        
    return "\n".join(rows)

def analyze_result_with_gpt(log_output: str, step_title: str) -> Union[dict, None]:
    """
    스크립트 실행 결과를 GPT API로 보내 구조화된 데이터(JSON) 분석을 요청하고,
    결과를 표로 출력하며 파싱된 딕셔너리를 반환합니다.
    """
    if not client or client.api_key == "YOUR_API_KEY":
        print("OpenAI API 키가 설정되지 않아 GPT 분석을 건너뜁니다.")
        return None
        
    print(f"\n--- {step_title} GPT 결과 분석 시작 ---")
    
    prompt = f"""
    당신은 머신러닝 결과 분석 전문가입니다.
    아래는 '{step_title}' 단계 스크립트의 전체 실행 로그입니다.

    --- [스크립트 실행 로그] ---
    {log_output}
    --- [스크립트 실행 로그 끝] ---

    **[분석 및 출력 형식 요청]**
    1.  위 로그에서 'Accuracy Matrix'를 찾으십시오.
    2.  이 매트릭스 데이터를 숫자 값만 추출하여 JSON 형식의 2차원 배열(리스트의 리스트)로 만드십시오. (예: [[60.9, 0.0], [55.1, 60.8]])
    3.  로그에 대한 간단한 요약 분석을 문자열로 만드십시오.
    4.  최종 결과를 반드시 다음 JSON 형식으로 제공해주십시오:
        {{
          "accuracy_matrix": [[...], [...]],
          "summary": "<요약 분석 문자열>"
        }}
    5. 만약 분석할 정확도 데이터가 없다면 "accuracy_matrix" 값은 null로 설정하고, "summary"에 로그의 주요 내용을 요약해주세요.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in machine learning result analysis. You must output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1, max_tokens=2048
        )
        
        gpt_response_str = response.choices[0].message.content
        parsed_json = json.loads(gpt_response_str)

        print("\n--- [GPT 최종 분석 보고서] ---")
        matrix_data = parsed_json.get("accuracy_matrix")
        if matrix_data:
            # 가장 긴 행을 기준으로 전체 태스크 수를 결정
            num_tasks = max(len(row) for row in matrix_data) if matrix_data else 0
            # 데이터를 바탕으로 테이블을 생성하여 출력
            table = format_matrix_as_table(matrix_data, num_tasks)
            print(table)
        
        print("\n[요약 분석]")
        print(parsed_json.get("summary", "요약 없음"))
        print("--------------------------")
        
        return parsed_json

    except json.JSONDecodeError:
        print("GPT가 유효한 JSON을 반환하지 않았습니다. 원본 응답을 출력합니다.")
        print(gpt_response_str)
        return {"error": "JSON Decode Error", "raw_response": gpt_response_str}
    except Exception as e:
        print(f"GPT API 호출 중 오류가 발생했습니다: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # --- 실행할 명령어 단계별 정의 ---
    commands_to_run = {
        "단계 1/5: 초기 모델 학습 (EWC)": "CUDA_VISIBLE_DEVICES=1 python /home/jun/work/soongsil/continual-learning-baselines/test_jun/Brainwash/ewc_barinwash_trian.py --experiment split_cifar100 --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 --batch-size 16 --lamb 500000 --lamb_emp 100 --clip 100. --lr 0.01",
        "단계 2/5: Inverse Attack 수행": "python /home/jun/work/soongsil/Brainwash/main_inv.py --pretrained_model_add=/home/jun/work/continual-learning-baselines/test_jun/Brainwash/ewc_lamb_500000.0_dataset_split_cifar100_seed_0_task_num_9.pkl --task_lst=0,1,2,3,4,5,6,7,8 --num_samples=128 --save_dir=/home/jun/work/continual-learning-baselines/test_jun/Brainwash/model_inv --save_every=1000 --batch_reg --init_acc --n_iters=10000",
        "단계 3/5: Brainwash (Reckless Mode) 적용": "CUDA_VISIBLE_DEVICES=0 python /home/jun/work/Brainwash/main_brainwash.py --extra_desc=reckless_test --pretrained_model_add=/home/jun/work/continual-learning-baselines/test_jun/Brainwash/ewc_lamb_500000.0_dataset_split_cifar100_seed_0_task_num_9.pkl --mode='reckless' --target_task_for_eval=0 --delta=0.3 --seed=0 --eval_every=10 --distill_folder=/home/jun/work/continual-learning-baselines/test_jun/Brainwash/model_inv --init_acc --noise_norm=inf --cont_learner_lr=0.001 --n_epochs=5000 --save_every=100",
        "단계 4/5: Brainwash (Cautious Mode) 적용": "CUDA_VISIBLE_DEVICES=0 python main_brainwash.py --extra_desc=cautious_test --pretrained_model_add=/home/jun/work/soongsil/Agent/noise_ewc_wcur_1.0_cautious_test__delta_0.3_dataset_split_cifar100_target_task_0_attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_5000_seed_0_mode_cautious__w_cur_1.0___min_acc_target_9.pkl --mode='cautious' --target_task_for_eval=0 --delta=0.3 --seed=0 --eval_every=10 --distill_folder=/home/jun/work/continual-learning-baselines/test_jun/Brainwash/model_inv --init_acc --noise_norm=inf --cont_learner_lr=0.001 --n_epochs=5000 --save_every=100 --w_cur=1",
        "단계 5/5: 최종 평가": "CUDA_VISIBLE_DEVICES=0 python /home/jun/work/soongsil/Brainwash/main_baselines.py --experiment split_cifar100 --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 --batch-size 16 --lr 0.01 --clip 100. --lamb 500000 --lamb_emp 100 --checkpoint /home/jun/work/soongsil/Agent/noise_ewc_wcur_1.0_cautious_test__delta_0.3_dataset_split_cifar100_target_task_0_attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_5000_seed_0_mode_cautious__w_cur_1.0___min_acc_target_9.pkl --init_acc --addnoise"
    }

    all_gpt_analyses = {}

    # 정의된 각 명령어를 순차적으로 실행
    for step_title, command in commands_to_run.items():
        print("\n" + "="*60)
        script_output = run_script_and_get_output(command, step_title)
        
        if script_output:
            print(f"\n--- 스크립트 전체 실행 로그 ({step_title}) ---")
            print(script_output)
            print("---------------------------------")
            
            # 실행 결과를 GPT로 분석
            analysis_result = analyze_result_with_gpt(script_output, step_title)
            
            if analysis_result:
                all_gpt_analyses[step_title] = analysis_result

    # 모든 분석 결과를 JSON 파일로 저장
    try:
        filename = "analysis_log.json"
        
        print("\n" + "="*60)
        print(f"GPT 분석 결과를 {filename} 파일로 저장합니다...")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_gpt_analyses, f, ensure_ascii=False, indent=4)
            
        print(f"성공적으로 {filename}에 저장했습니다.")

    except Exception as e:
        print(f"로그 파일 저장 중 오류가 발생했습니다: {e}")

    print("\n" + "="*60)
    print("\n--- 모든 작업 완료 ---")
