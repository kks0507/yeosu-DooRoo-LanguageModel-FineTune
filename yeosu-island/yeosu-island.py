# requirements:
# pip install pandas datasets huggingface_hub google-generativeai tqdm scikit-learn

import os
import re
import json
import time
import pandas as pd
import google.generativeai as genai
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# =========================
# 0) 설정 & 환경
# =========================
GEMINI_API_KEY = "YOUR_API_KEY"  # 본인의 Gemini API 키로 교체하세요.
HF_TOKEN       = "YOUR_API_KEY" # 본인의 Hugging Face 쓰기(write) 토큰으로 교체하세요.

# 실행 파라미터
CSV_FILE_PATH   = "YOUR_TRAIN_DATA.csv"
HF_REPO_ID      = "YOUR_HF_USERNAME/YOUR_DATASET_NAME"  # 예: "myusername/myislands-dataset"
MODEL_NAME      = "gemini-2.5-flash-lite-preview-06-17"
N_AUGMENTATIONS = 3
SLEEP_SEC       = 7.0
MAX_ROWS        = None
SYSTEM_PROMPT   = "You are a helpful assistant who is an expert on the islands of Yeosu, South Korea. Please provide kind and accurate answers to the user's questions."

# API 클라이언트 초기화
if not GEMINI_API_KEY or "YOUR" in GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY를 설정해주세요.")
if not HF_TOKEN or "YOUR" in HF_TOKEN:
    raise ValueError("HF_TOKEN을 설정해주세요.")

genai.configure(api_key=GEMINI_API_KEY)
hf_api = HfApi(token=HF_TOKEN)

# =========================
# 1) 데이터 증강 (강화된 프롬프트)
# =========================
def paraphrase_question_gemini(model: genai.GenerativeModel, island: str, question: str, n: int) -> list[str]:
    """
    Gemini를 호출하여 다양한 스타일과 어투의 질문을 n개 생성합니다.
    """
    prompt = f"""
# ROLE
You are a creative copywriter specializing in generating diverse user questions. Your goal is to create a variety of questions that real people would ask, based on a single, standardized query.

# MISSION
Generate {n} new questions based on the "ORIGINAL QUESTION" below.
The generated questions must ask for the exact same information, but should have completely different tones, styles, and phrasing, as if asked by different people.

# CORE INFORMATION (MUST BE INCLUDED IN EVERY QUESTION)
- Island Name: "{island}"

# ORIGINAL QUESTION
"{question}"

# GUIDELINES FOR DIVERSIFICATION (Generate a mix of styles like these)
1.  **Friendly & Casual (친근한 말투):** Like asking a friend. (e.g., "낭도 이름은 무슨 뜻이야?", "혹시 낭도 주소 알아?")
2.  **Direct & Concise (직접적이고 간결한 말투):** Getting straight to the point. (e.g., "낭도 면적?", "낭도 특산물 알려줘.")
3.  **Polite & Formal (정중한 말투):** As if asking in a formal setting. (e.g., "낭도의 이름이 유래된 배경에 대해 설명해주실 수 있나요?")
4.  **Inquisitive & Detailed (조금 더 상세하게 묻는 말투):** Asking with more curiosity. (e.g., "낭도라는 섬은 왜 그런 이름이 붙게 된 건지 궁금해요.")
5.  **Beginner's Question (여행객이나 초심자의 질문):** As if asking for the first time. (e.g., "여수 낭도에 가려면 배 어디서 타요?", "낭도에 가면 꼭 봐야 할 게 뭔가요?")

# CONSTRAINTS
- ALWAYS maintain the original question's core intent. Do not ask for different information.
- ALWAYS end the sentence with a question mark (?).
- DO NOT simply reorder the words from the original question. Create genuinely new sentences.
- Output ONLY a single JSON array containing exactly {n} question strings. Do not include any other text, explanations, or formatting.
- Example output format: ["질문1", "질문2", "질문3"]
"""
    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            raw_text = (getattr(resp, "text", None) or "").strip()
            match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if not match:
                raise ValueError("JSON array not found in the response.")
            paraphrases = json.loads(match.group(0))
            if isinstance(paraphrases, list) and len(paraphrases) >= n:
                return [str(p).strip() for p in paraphrases[:n]]
            else:
                raise ValueError(f"Not enough paraphrases generated ({len(paraphrases)} < {n}).")
        except Exception as e:
            print(f"  - (Warning) Paraphrase generation failed (Attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(SLEEP_SEC * (attempt + 1))
            else:
                print(f"  - (Failed) Returning original question after final attempt: {question}")
                return [question] * n
    return [question] * n

# =========================
# 2) 데이터 포맷팅 및 분할
# =========================
def to_qwen_chat_format(question: str, answer: str, system_prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question).strip()},
            {"role": "assistant", "content": str(answer).strip()}
        ]
    }

# =========================
# 3) 메인 파이프라인
# =========================
def main():
    print(f"[Load] Reading CSV file: {CSV_FILE_PATH}")
    df_orig = pd.read_csv(CSV_FILE_PATH)
    
    df_orig.columns = [col.lower().strip() for col in df_orig.columns]
    required_cols = ['island', 'question', 'answer']
    if not all(col in df_orig.columns for col in required_cols):
        raise KeyError(f"CSV must contain the following columns: {required_cols}")

    df_orig = df_orig.dropna(subset=required_cols).copy()
    for col in required_cols:
        df_orig[col] = df_orig[col].astype(str).str.strip()
    df_orig = df_orig[df_orig["question"] != ""].reset_index(drop=True)

    if MAX_ROWS:
        df_orig = df_orig.head(MAX_ROWS)
        print(f"[Note] Dev Mode: Processing only the first {len(df_orig)} rows.")

    print(f"\n[Split] Splitting {len(df_orig)} original samples into Dev/Test sets...")
    test_rows = [
    to_qwen_chat_format(row['question'], row['answer'], SYSTEM_PROMPT)
    for _, row in df_orig.iterrows()
    ]

    print(f"\n[Augment] Creating Train set by paraphrasing {len(df_orig)} questions ({N_AUGMENTATIONS} each)...")
    model = genai.GenerativeModel(MODEL_NAME)
    train_rows = []
    failed_paraphrases = 0

    for _, row in tqdm(df_orig.iterrows(), total=len(df_orig), desc="Augmenting Train Data"):
        paraphrased_qs = paraphrase_question_gemini(
            model,
            island=row['island'],
            question=row['question'],
            n=N_AUGMENTATIONS
        )
        for pq in paraphrased_qs:
            if pq.strip() == row['question'].strip():
                failed_paraphrases += 1
            train_rows.append(to_qwen_chat_format(pq, row['answer'], SYSTEM_PROMPT))
        time.sleep(SLEEP_SEC)

    print(f"[Result] Generated {len(train_rows)} augmented samples for Train set.")
    print(f"  - Failed/identical paraphrases: {failed_paraphrases} out of {len(df_orig) * N_AUGMENTATIONS}")

    unique_train_rows_str = {json.dumps(d, sort_keys=True) for d in train_rows}
    train_rows = [json.loads(s) for s in unique_train_rows_str]
    print(f"  - Train samples after deduplication: {len(train_rows)}")

    if not train_rows or not test_rows:
        raise RuntimeError("Data creation failed. Cannot upload empty datasets.")

    dsd = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test": Dataset.from_list(test_rows)
    })
    
    print("\n[Hugging Face] Generated Dataset Information:")
    print(dsd)

    print(f"\n[Upload] Pushing dataset to https://huggingface.co/datasets/{HF_REPO_ID}")
    try:
        dsd.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print("✅ All tasks completed successfully!")
    except Exception as e:
        print(f"🚨 Failed to upload to Hugging Face Hub: {e}")

if __name__ == "__main__":
    main()