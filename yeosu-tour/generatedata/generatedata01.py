# 파일명: generate_worker_template.py

import os
import re
import json
import time
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

# ==========================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
#           파일을 복사할 때마다 여기만 수정하세요 (4줄)
# ==========================================================

# 1. 이 작업자가 사용할 API 키를 지정하세요.
GEMINI_API_KEY = "YOUR_GEMINI_API"

# 2. 이 작업자가 읽을 CSV 파일 경로를 지정하세요.
CSV_FILE_PATH  = "split_data/YOUR_FILE_NAME.csv"

# 3. 좌표 변환 후 중간 저장될 파일 경로를 지정하세요. (겹치지 않게)
PROCESSED_CSV_PATH = "split_data/YOUR_PROCESSED_FILE_NAME.csv"

# 4. 이 작업자의 번호(ID)를 지정하세요. (최종 결과 파일명에 사용)
WORKER_ID = 1

# ==========================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==========================================================

# 고정 파라미터
MODEL_NAME      = "gemini-2.5-flash-lite-preview-06-17"
N_QUESTIONS_PER_ITEM = 4      # 각 정보 항목당 생성할 질문 개수 
SLEEP_SEC       = 7.0
MAX_ROWS        = None
SYSTEM_PROMPT   = "You are a helpful assistant who is an expert on the tourist attractions of Yeosu, South Korea. Please provide kind and accurate answers to the user's questions."
OUTPUT_DIR      = "generated_json" # 생성된 JSON 파일이 저장될 디렉터리

# API 클라이언트 초기화
if not GEMINI_API_KEY or "YOUR" in GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY를 설정해주세요.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def preprocess_csv(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        if '위도' in df.columns and '경도' in df.columns:
            df['위도'] = df['위도'].fillna("")
            df['경도'] = df['경도'].fillna("")
            df['좌표'] = df.apply(lambda r: f"({r['위도']}, {r['경도']})" if r['위도'] and r['경도'] else "", axis=1)
            df = df.drop(columns=['위도', '경도'])
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"좌표 통합 완료. 결과가 '{output_path}'에 저장되었습니다.")
        return df
    except FileNotFoundError:
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다.")
        return None

def to_chat_format(question, answer, system_prompt):
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]

def generate_qa_with_gemini(attraction_name, info_type, info_content, n_questions):
    prompt = f"""
    You are a prompt engineering expert creating a training dataset for a helpful AI assistant.
    Your goal is to generate diverse user questions and a single, perfect answer based on the provided information.

    **ROLE & TONE:** The AI assistant should sound like a friendly, local Yeosu tour guide.

    **Information:**
    - Attraction Name: "{attraction_name}"
    - Information Type: "{info_type}"
    - Information Content: "{info_content}"

    **Instructions:**
    1.  **Generate {n_questions} diverse user questions.**
        - Each question **must** include the attraction name, "{attraction_name}".
        - The questions should reflect different, natural ways a real user might ask. For example, include:
            - A direct question (e.g., "What are the hours for...?").
            - A polite inquiry (e.g., "Could you please tell me about...?").
            - A question for planning purposes (e.g., "If I visit..., can I...?").
        - **Do not** create questions that cannot be answered by the "Information Content".

    2.  **Generate ONE high-quality answer.**
        - The answer must be based **strictly** on the "Information Content". **Do not** add any information that is not provided.
        - The answer must be a complete, helpful sentence. 
        - **Crucially**, if the content is a single word like "가능" or "없음", expand it into a full sentence.
            - For example, if Information Content is "가능" for "주차시설", the answer should be something like "네, {attraction_name}에는 주차시설이 마련되어 있어 주차가 가능합니다." and NOT just "네, 가능합니다."

    **Output Format (provide only the JSON object):**
    {{
        "questions": ["...", "...", "..."],
        "answer": "..."
    }}"
    """
    try:
        response = model.generate_content(prompt)
        json_str = re.search(r'```json\n({.*?})\n```', response.text, re.DOTALL)
        return json.loads(json_str.group(1)) if json_str else json.loads(response.text)
    except Exception as e:
        print(f"  [Error] Gemini API 호출 또는 JSON 파싱 실패: {e}")
        return None

def main():
    print(f"[Worker-{WORKER_ID}] 시작.")
    df = preprocess_csv(CSV_FILE_PATH, PROCESSED_CSV_PATH)
    if df is None:
        return

    df = df.fillna("")
    all_generated_data = []
    columns_to_use = [
    '개요', '주소', '좌표',
    '문의 및 안내', '이용시간', '쉬는날', '주차시설',
    '애완동물 동반 가능 여부', '유모차 대여 여부', '개장일',
    '신용카드 가능 여부', '상세정보'
    ]
    total_iterations = sum(1 for _, row in df.iterrows() for col in columns_to_use if col in df.columns and str(row.get(col, "")).strip())

    with tqdm(total=total_iterations, desc=f"Worker-{WORKER_ID} 진행률") as pbar:
        for _, row in df.iterrows():
            name = row.get('명칭(name)')
            if not name: continue
            for col in columns_to_use:
                if col in df.columns:
                    content = str(row.get(col, "")).strip()
                    if content and content != '()':
                        qa_pair = generate_qa_with_gemini(name, col, content, N_QUESTIONS_PER_ITEM)
                        if qa_pair and 'questions' in qa_pair and 'answer' in qa_pair:
                            for q in qa_pair['questions']:
                                all_generated_data.append(to_chat_format(q, qa_pair['answer'], SYSTEM_PROMPT))
                        pbar.update(1)
                        time.sleep(SLEEP_SEC)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 상단에 설정된 WORKER_ID를 사용하여 고유한 파일명을 만듭니다.
    json_output_path = os.path.join(OUTPUT_DIR, f"output_part_{WORKER_ID}.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_generated_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ [Worker-{WORKER_ID}] 작업 완료. 총 {len(all_generated_data)}개 데이터 생성.")
    print(f"결과가 '{json_output_path}' 파일에 저장되었습니다.")

# [수정됨] sys.argv 부분을 제거하고 main()을 직접 호출합니다.
if __name__ == "__main__":
    main()