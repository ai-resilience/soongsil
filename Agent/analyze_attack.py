import json
import chromadb
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

# OpenAI 클라이언트 설정
openai_client = OpenAI(api_key="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")
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    openai_api_key="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"
)

# ChromaDB 설정
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="attack_logs")

def query_chroma_hint(query_text="brainwash continual learning attack", top_k=3):
    try:
        results = collection.query(query_texts=[query_text], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "(ChromaDB에서 관련 내용을 찾지 못했습니다.)"
        return "\n---\n".join(docs)
    except Exception as e:
        return f"ChromaDB 쿼리 오류: {e}"

def load_accuracy_matrices(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    before = data["단계 1/5: 초기 모델 학습 (EWC)"]["accuracy_matrix"]
    after = data["단계 5/5: 최종 평가"]["accuracy_matrix"]
    return before, after

def flatten_accuracy(matrix):
    return [round(v, 2) for v in matrix[-1]]

# Step 1: 정황 판단용 프롬프트
detect_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 머신러닝 보안 전문가야. 정확도 변화와 출력 로그를 기반으로 BrainWash 공격 정황이 있는지 판단해."),
    ("user", """
[정확도 변화]\n- 공격 전: {before_acc}\n- 공격 후: {after_acc}

[출력 로그]\n{previous_output_log}

BrainWash 공격은 마지막 태스크 이후 과거 태스크들의 정확도가 급격히 하락하는 특징이 있어.
정황이 보이면 'brainwash_sign_detected': true 로 응답해.

JSON 형식으로 응답:
{{
  "brainwash_sign_detected": true or false,
  "reason": "<판단 근거>"
}}
""")
])

def detect_brainwash_sign(before_acc, after_acc, previous_output_log):
    chain = detect_prompt | llm
    return chain.invoke({
        "before_acc": str(before_acc),
        "after_acc": str(after_acc),
        "previous_output_log": previous_output_log
    })

# Step 2: 최종 공격 분석 프롬프트
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 머신러닝 보안 전문가야."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", """
[정확도 변화]\n- 공격 전: {before_acc}\n- 공격 후: {after_acc}

[관련 논문 요약]\n{chroma_hint}

[출력 로그]\n{previous_output_log}

{question}

다음 형식의 JSON으로 답해:
{{
  "attack_type": "<공격 이름 또는 없음>",
  "attacked_tasks": [<정확도 하락 task 번호들>],
  "reason": "<판단 근거>"
}}
""")
])

conversation = RunnableWithMessageHistory(
    final_prompt | llm,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="chat_history"
)

def analyze_attack(before_acc, after_acc, question, previous_output_log="", session_id="attack-session"):
    sign_result_raw = detect_brainwash_sign(before_acc, after_acc, previous_output_log)
    try:
        sign_result = json.loads(sign_result_raw.content.strip().strip('```json').strip('```'))
    except Exception as e:
        print(" GPT 정황 판단 응답 파싱 실패:", e)
        sign_result = {"brainwash_sign_detected": False, "reason": "응답 파싱 실패"}

    print("\n BrainWash 정황 판단 결과:", sign_result)

    if sign_result.get("brainwash_sign_detected"):
        chroma_hint = query_chroma_hint("brainwash attack in continual learning")
    else:
        chroma_hint = "(정황 없음 → ChromaDB 참조 생략)"

    inputs = {
        "before_acc": str(before_acc),
        "after_acc": str(after_acc),
        "chroma_hint": chroma_hint,
        "question": question,
        "previous_output_log": previous_output_log
    }
    config = {"configurable": {"session_id": session_id}}
    return conversation.invoke(inputs, config=config)

def main():
    before, after = load_accuracy_matrices("/home/jun/work/soongsil/Agent/analysis_log.json")
    before_flat = flatten_accuracy(before)
    after_flat = flatten_accuracy(after)

    print(f"\n Accuracy 확인\nBefore: {before_flat}\nAfter: {after_flat}")

    previous_output_log = "초기 정확도는 높았으나 후속 태스크 학습 이후 급격히 하락한 양상이 있음."

    print("\n 공격 분석:")
    q1 = "이 정확도 변화는 어떤 유형의 공격에 해당하나요?"
    result = analyze_attack(before_flat, after_flat, q1, previous_output_log)
    print(result.content)

if __name__ == "__main__":
    main()
