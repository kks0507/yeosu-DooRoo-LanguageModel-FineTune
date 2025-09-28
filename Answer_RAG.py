# ✨ sqlite3 버전 충돌 해결을 위한 코드 (최상단에 위치해야 함)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ==========================================================

import os
import pandas as pd
from tqdm import tqdm
import argparse
import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer
import textwrap # ✨ 1. textwrap 라이브러리 임포트

# =================================================================================
# 설정 부분
# =================================================================================
INPUT_DIR = "evaluation_datasets"
API_URL = "http://127.0.0.1:8001/generate/"
SYSTEM_PROMPT = "You are a helpful assistant who is an expert on the tourist attractions of Yeosu, South Korea. Please provide kind and accurate answers to the user's questions based on the provided context."

# ✨ 2. RAG 설정
CHROMA_DB_PATH = "./chroma_db"  # ChromaDB 데이터가 저장된 경로
COLLECTION_NAME = "yeosu_tour_db" # 미리 생성해둔 컬렉션 이름
EMBEDDING_MODEL = 'nlpai-lab/KURE-v1' # 한국어 임베딩 모델

# ✨ 3. RAG 모델 및 DB 클라이언트 초기화 (스크립트 실행 시 한 번만 로드)
print("임베딩 모델과 ChromaDB 클라이언트를 로드합니다...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
print("로드 완료.")

def generate_answers_via_api(category):
    """지정된 카테고리의 CSV 파일에 대해 RAG를 적용하여 모델 답변을 생성합니다."""
    print("\n" + "="*50)
    print(f"### 2단계: RAG 기반 API 호출로 답변 생성 ({category}) ###")
    print("="*50)

    input_file = os.path.join(INPUT_DIR, f"evaluation_{category}.csv")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"오류: '{input_file}' 파일을 찾을 수 없습니다.")
        return

    model_answers = []
    for question in tqdm(df['question'], desc=f"RAG 및 API 호출 중 ({category})"):
        try:
            # 1. 검색(Retrieve): 질문을 벡터로 변환하고 ChromaDB에서 관련 문서 검색
            query_embedding = embedding_model.encode(question, convert_to_tensor=False).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3 
            )
            context = "\n\n".join(results['documents'][0])

            # ✨ 2. 증강(Augment): textwrap.dedent를 사용하여 불필요한 공백 제거
            rag_prompt = textwrap.dedent(f"""
                [Context]
                {context}

                [Question]
                {question}

                [Instruction]
                You are a friendly guide for Yeosu tourism. Based on the provided [Context], please answer the [Question] clearly and concisely in Korean.
                - Structure your answer in a bulleted list (using '-') for easy reading.
                - Each bullet point should explain one key place or item.
                - Ensure your response is helpful and directly answers the user's question.
            """).strip() # 맨 앞/뒤의 불필요한 줄바꿈도 제거
            
            # 3. 생성(Generate)
            payload = {
                "query": rag_prompt,
                "system_prompt": SYSTEM_PROMPT 
            }
            
            # ... (이후 코드는 모두 동일) ...
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            
            answer = response.json().get("response", "Error: No response field")
            model_answers.append(answer)

        except requests.exceptions.RequestException as e:
            print(f"\nAPI 호출 오류 발생: {e}")
            model_answers.append(f"API_ERROR: {e}")
        except Exception as e:
            print(f"\nRAG 처리 중 오류 발생: {e}")
            model_answers.append(f"RAG_ERROR: {e}")

    df['model_answer'] = model_answers
    df.to_csv(input_file, index=False, encoding='utf-8-sig')
    print(f"\n'{input_file}' 파일에 모델 답변 업데이트 완료.")

# ... (파일의 맨 아래 부분은 동일) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 파이프라인을 통해 API를 호출하여 모델 답변을 생성합니다.")
    parser.add_argument("--category", type=str, required=True, choices=["fluency", "coherence", "accuracy", "completeness"])
    args = parser.parse_args()
    
    generate_answers_via_api(args.category)