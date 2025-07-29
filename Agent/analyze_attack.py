import json
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

client = OpenAI(api_key="")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="attack_logs")

def query_chroma_hint(query_text="accuracy drop after task 9", top_k=3):
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "ChromaDB에서 관련 힌트를 찾지 못했습니다."
        return "\n".join(docs)
    except Exception as e:
        return f"ChromaDB 쿼리 중 오류 발생: {e}"

def load_accuracy_matrices(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    before = data["단계 1/4: 초기 모델 학습"]["accuracy_matrix"]
    after = data["단계 4/4: 최종 평가"]["accuracy_matrix"]
    return before, after

def flatten_accuracy(matrix):
    return [round(row[i], 2) if i < len(row) else 0.0 for i, row in enumerate(matrix)]

def generate_gpt_prompt(before_acc, after_acc, chroma_hint: str):
    prompt = f"""
당신은 머신러닝 보안 전문가입니다.
다음은 모델이 학습된 전후의 태스크별 정확도입니다:

- 공격 전 정확도 (마지막 단계 기준): {before_acc}
- 공격 후 정확도 (마지막 단계 기준): {after_acc}

또한, ChromaDB에서 다음과 같은 참고 힌트를 제공합니다:
---
{chroma_hint}
---

이 정보를 바탕으로 아래 질문에 답하십시오:

1. 어떤 태스크에서 정확도가 비정상적으로 하락했습니까?
2. 이 현상이 Brainwash 공격의 특징에 부합합니까? 아니면 Accumulative Attack의 특징에 부합합니까?

반드시 다음 형식으로 한국어 JSON 형식으로 응답하십시오:
{{
  "attack_type": "<공격 이름 또는 없음>",
  "attacked_tasks": [<정확도 하락 task 번호들>],
  "reason": "<판단 근거 한국어 설명>"
}}
"""
    return prompt

def query_gpt_analysis(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "당신은 머신러닝 모델 공격 유형을 판단하는 보안 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    return json.loads(response.choices[0].message.content)

def main():
    before, after = load_accuracy_matrices("/home/jun/work/Agent/analysis_log_20250723_144953.json")
    before_final = flatten_accuracy(before)
    after_final = flatten_accuracy(after)

    # ChromaDB에서 자동 힌트 추출
    chroma_hint = query_chroma_hint("accuracy drop after task 9")

    prompt = generate_gpt_prompt(before_final, after_final, chroma_hint)
    result = query_gpt_analysis(prompt)

    print("\n🔍 공격 판단 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()