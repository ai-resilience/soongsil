import json
import chromadb
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

openai_client = OpenAI(api_key="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") 
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.01, #창의성 조절 0에 가까울수록 결정적
    openai_api_key="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" 
)

chroma_client = chromadb.HttpClient()

try:

    collection = chroma_client.get_collection(name="papers")
    print(f"[ChromaDB] '{collection.name}' 컬렉션에 성공적으로 연결했습니다.")
except Exception as e:
    print(f"[오류] ChromaDB 컬렉션을 가져오는 데 실패했습니다: {e}")
    print("이전에 poisoningpdf.py 스크립트로 데이터를 성공적으로 저장했는지, ChromaDB 서버가 실행 중인지 확인해주세요.")
    exit()

print("[ChromaDB] 문서 수:", collection.count())

def query_chroma_hint(query_text="Poisoning Attack", top_k=3):
    """ChromaDB에서 관련 정보를 검색하는 함수"""
    try:
        results = collection.query(query_texts=[query_text], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "(ChromaDB에서 관련 내용을 찾지 못했습니다.)"
        return "\n---\n".join(f"- {doc}" for doc in docs)
    except Exception as e:
        return f"ChromaDB 쿼리 오류: {e}"

def load_accuracy_matrices(filename: str):
    """JSON 파일에서 정확도 행렬을 로드하는 함수"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    before = data["단계 1/5: 초기 모델 학습 (EWC)"]["accuracy_matrix"]
    after = data["단계 5/5: 최종 평가"]["accuracy_matrix"]
    return before, after

def flatten_accuracy(matrix):
    """정확도 행렬의 마지막 행을 1차원 리스트로 변환하는 함수"""
    return [round(v, 2) for v in matrix[-1]]

# Step 1: 정황 판단용 프롬프트
detect_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 머신러닝 보안 전문가야. 정확도 변화와 출력 로그를 기반으로 BrainWash 공격 정황이 있는지 판단해."),
    ("user", """
[정확도 변화]
- 공격 전: {before_acc}
- 공격 후: {after_acc}

[출력 로그]
{previous_output_log}

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
    """LLM을 이용해 BrainWash 공격 정황을 1차로 판단하는 함수"""
    chain = detect_prompt | llm
    return chain.invoke({
        "before_acc": str(before_acc),
        "after_acc": str(after_acc),
        "previous_output_log": previous_output_log
    })

# Step 2: 최종 공격 분석 프롬프트
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 머신러닝 보안 전문가야. 제공된 모든 정보를 종합하여 최종 분석 결과를 JSON 형식으로만 제공해."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", """
[정확도 변화]
- 공격 전: {before_acc}
- 공격 후: {after_acc}

[관련 논문 요약]
{chroma_hint}

[출력 로그]
{previous_output_log}

{question}

다음 형식의 JSON으로 답해:
{{
  "attack_type": "<공격 이름 또는 없음>",
  "attacked_tasks": [<정확도 하락 task 번호들>],
  "reason": "<판단 근거>"
}}
""")
])

# 대화 기록을 관리하는 LangChain 객체
conversation = RunnableWithMessageHistory(
    final_prompt | llm,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="question",
    history_messages_key="chat_history"
)

def analyze_attack(before_acc, after_acc, question, previous_output_log="", session_id="attack-session"):
    """공격 분석 파이프라인을 실행하는 함수"""
    # 1. BrainWash 정황 판단
    sign_result_raw = detect_brainwash_sign(before_acc, after_acc, previous_output_log)
    try:
        sign_result = json.loads(sign_result_raw.content.strip().strip('```json').strip('```'))
    except Exception as e:
        print(" GPT 정황 판단 응답 파싱 실패:", e)
        sign_result = {"brainwash_sign_detected": False, "reason": "응답 파싱 실패"}

    print("\n[1단계] BrainWash 정황 판단 결과:", sign_result)

    # 2. 정황에 따라 ChromaDB에서 정보 검색
    if sign_result.get("brainwash_sign_detected"):
        print("\n[2단계] 정황 포착, ChromaDB에서 관련 정보 검색...")
        chroma_hint = query_chroma_hint("brainwash attack in continual learning")
    else:
        print("\n[2단계] 정황 없음, ChromaDB 참조 생략.")
        chroma_hint = "(정황 없음 → ChromaDB 참조 생략)"

    # 3. 최종 분석
    print("\n[3단계] 최종 분석 실행...")
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
    """메인 실행 함수"""
    try:
        before, after = load_accuracy_matrices("/home/jun/work/soongsil/Agent/analysis_log.json")
    except FileNotFoundError:
        print("[오류] analysis_log.json 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return
        
    before_flat = flatten_accuracy(before)
    after_flat = flatten_accuracy(after)

    print(f"\n[초기 데이터] Accuracy 확인\n- Before: {before_flat}\n- After:  {after_flat}")

    previous_output_log = "초기 정확도는 높았으나 후속 태스크 학습 이후 급격히 하락한 양상이 있음."

    q1 = "이 정확도 변화는 어떤 유형의 공격에 해당하나요? 모든 정보를 종합해서 분석해주세요."
    final_result_raw = analyze_attack(before_flat, after_flat, q1, previous_output_log)

    print("\n--- 최종 분석 결과 ---")
    try:
        final_result = json.loads(final_result_raw.content.strip().strip('```json').strip('```'))
        print(f"  - 공격 유형: {final_result.get('attack_type', 'N/A')}")
        print(f"  - 공격 대상 태스크: {final_result.get('attacked_tasks', 'N/A')}")
        print(f"  - 분석 근거: {final_result.get('reason', 'N/A')}")
    except json.JSONDecodeError:
        print("최종 결과가 JSON 형식이 아니어서 파싱에 실패했습니다.")
        print(final_result_raw.content)
    except Exception as e:
        print(f"최종 결과 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
