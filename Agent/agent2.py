import json
import chromadb
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# ChromaDB 설정
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="attack_logs")

# OpenAI LLM 설정
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    openai_api_key="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"
)

# 프롬프트 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 머신러닝 보안 전문가입니다."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", """다음은 모델이 학습된 전후의 태스크별 정확도입니다:

- 공격 전 정확도 (마지막 단계 기준): {before_acc}
- 공격 후 정확도 (마지막 단계 기준): {after_acc}

ChromaDB 힌트:
---
{chroma_hint}
---

질문:
{question}

반드시 다음 형식으로 JSON 응답하십시오:
{{
  "attack_type": "<공격 이름 또는 없음>",
  "attacked_tasks": [<정확도 하락 task 번호들>],
  "reason": "<판단 근거 한국어 설명>"
}}""")
])

# 메모리
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 대화체인 구성
conversation = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 정확도 JSON 로드
def load_accuracy_matrices(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    before = data["단계 1/4: 초기 모델 학습"]["accuracy_matrix"]
    after = data["단계 4/4: 최종 평가"]["accuracy_matrix"]
    return before, after

# 정확도 평탄화
def flatten_accuracy(matrix):
    return [round(row[i], 2) if i < len(row) else 0.0 for i, row in enumerate(matrix)]

# Chroma 힌트 검색
def query_chroma_hint(query_text="accuracy drop after task 9", top_k=3):
    try:
        results = collection.query(query_texts=[query_text], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "ChromaDB에서 관련 힌트를 찾지 못했습니다."
        return "\n".join(docs)
    except Exception as e:
        return f"ChromaDB 쿼리 중 오류 발생: {e}"

# 분석 함수
def analyze_attack(before_acc, after_acc, question: str, session_id="default-session"):
    chroma_hint = query_chroma_hint(question)
    inputs = {
        "input": {
            "before_acc": str(before_acc),
            "after_acc": str(after_acc),
            "chroma_hint": chroma_hint,
            "question": question
        }
    }
    config = {"configurable": {"session_id": session_id}}
    return conversation.invoke(inputs, config=config)

# 실행
def main():
    before, after = load_accuracy_matrices("/home/jun/work/soongsil/Agent/analysis_log.json")
    before_flat = flatten_accuracy(before)
    after_flat = flatten_accuracy(after)

    print("\n 1차 분석 결과:")
    output1 = analyze_attack(
        before_flat,
        after_flat,
        "1. 어떤 태스크에서 정확도가 비정상적으로 하락했습니까?\n2. 이 현상이 Brainwash 공격의 특징에 부합합니까? 아니면 Accumulative Attack의 특징에 부합합니까?",
        session_id="session1"
    )
    print(output1.content)

    print("\n 2차 분석 요청:")
    output2 = analyze_attack(
        before_flat,
        after_flat,
        "해당 공격이 다른 태스크에 전파될 가능성은 어느 정도인가요?",
        session_id="session1"
    )
    print(output2.content)

if __name__ == "__main__":
    main()
