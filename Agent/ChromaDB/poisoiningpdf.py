import fitz  # PyMuPDF
import chromadb

# PDF에서 텍스트 추출하는 함수
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 텍스트를 일정 크기로 나누는 함수
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Chroma DB 클라이언트 연결
client = chromadb.HttpClient()

# 논문 컬렉션 생성 또는 가져오기
collection = client.get_or_create_collection("papers")

# 논문 1에서 텍스트 추출 및 Chunking
pdf_text_1 = extract_text_from_pdf("/home/jun/work/Agent/ChromaDB/Abbasi_BrainWash_A_Poisoning_Attack_to_Forget_in_Continual_Learning_CVPR_2024_paper.pdf")
text_chunks_1 = chunk_text(pdf_text_1)

# 논문 2에서 텍스트 추출 및 Chunking
pdf_text_2 = extract_text_from_pdf("/home/jun/work/Agent/ChromaDB/NeurIPS-2021-accumulative-poisoning-attacks-on-real-time-data-Paper.pdf")
text_chunks_2 = chunk_text(pdf_text_2)

# 메타데이터 정의 (수정)
metadata_1 = {
    "title": "BrainWash: A Poisoning Attack to Forget in Continual Learning", 
    "authors": "Ali Abbasi; Parsa Nooralinejad; Hamed Pirsiavash; Soheil Kolouri", 
    "year": 2024
}

metadata_2 = {
    "title": "Accumulative Poisoning Attacks on Real-time Data", 
    "authors": "논문2 저자1; 논문2 저자2", 
    "year": 2021
}

# Chroma DB에 논문 1 추가
for idx, chunk in enumerate(text_chunks_1):
    collection.add(
        documents=[chunk],
        metadatas=[metadata_1],
        ids=[f"paper1_chunk{idx}"]
    )

# Chroma DB에 논문 2 추가
for idx, chunk in enumerate(text_chunks_2):
    collection.add(
        documents=[chunk],
        metadatas=[metadata_2],
        ids=[f"paper2_chunk{idx}"]
    )

print("두 개의 논문이 Chroma DB에 성공적으로 추가되었습니다.")
