# ✨ sqlite3 버전 충돌 해결을 위한 코드 (최상단에 위치해야 함)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ==========================================================

import pandas as pd
import chromadb
# ✨ 1. ChromaDB의 임베딩 함수 유틸리티를 임포트합니다.
import chromadb.utils.embedding_functions as embedding_functions
from tqdm import tqdm

# =================================================================================
# 설정 부분
# =================================================================================
CSV_FILE_PATH = "yeosu-tour-final.csv"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "yeosu_tour_db"
EMBEDDING_MODEL = 'nlpai-lab/KURE-v1'

# =================================================================================
# 데이터 로드 및 ChromaDB 저장 로직
# =================================================================================
def setup_chromadb_from_csv():
    """Q&A 형식의 CSV 파일을 읽어 RAG에 최적화된 방식으로 ChromaDB에 저장합니다."""
    print("="*50)
    print("### 1단계: CSV 파일 로드 ###")
    print("="*50)
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        # 비어있는(NaN) 셀을 빈 문자열("")로 대체하여 오류를 방지합니다.
        df = df.fillna("")
        print(f"'{CSV_FILE_PATH}' 파일에서 {len(df)}개의 Q&A 데이터를 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: '{CSV_FILE_PATH}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    print("\n" + "="*50)
    print("### 2단계: ChromaDB 클라이언트 및 컬렉션 설정 ###")
    print("="*50)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"기존 컬렉션 '{COLLECTION_NAME}'을 삭제합니다.")
        client.delete_collection(name=COLLECTION_NAME)
    
    # ✨ 2. 우리가 선택한 한국어 모델로 임베딩 함수를 생성합니다.
    korean_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # ✨ 3. 컬렉션을 만들 때, 위에서 만든 임베딩 함수를 지정해줍니다.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=korean_embedding_function
    )
    
    print("\n" + "="*50)
    print("### 3단계: 답변(Answer) 기반 문서 생성 및 저장 ###")
    print("="*50)

    # (이 부분부터는 기존 코드와 완전히 동일합니다)
    documents = []
    metadatas = []
    ids = []

    print("CSV 데이터로 문서를 생성합니다...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="문서/메타데이터 생성 중"):
        answer_text = row['개요'].strip()
        if answer_text:
            documents.append(answer_text)
            metadatas.append({
                'category': row['category'],
                'subcategory': row['subcategory'],
                '명칭(name)': row['명칭(name)'],
                '주소': row['주소']
            })
            ids.append(f"data_{index}")

    print(f"문서 생성이 완료되었습니다. 총 {len(documents)}개의 유효한 문서가 생성되었습니다.")
    
    print("이제 ChromaDB에 데이터를 추가합니다.")
    batch_size = 10
    for i in tqdm(range(0, len(documents), batch_size), desc="DB 저장 중"):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print("\n" + "="*50)
    print(f"🎉 ChromaDB에 {len(documents)}개의 문서 저장이 완료되었습니다!")
    print("="*50)

if __name__ == "__main__":
    setup_chromadb_from_csv()