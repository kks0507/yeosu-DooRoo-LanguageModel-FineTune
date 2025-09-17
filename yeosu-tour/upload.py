import json
import os
import glob
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# --- 설정 ---
HF_TOKEN = "Your_HF_KEY" # 본인의 Hugging Face 쓰기(write) 토큰으로 교체
HF_REPO_ID = "YOUR_HF_REPO" # 본인의 Hugging Face 리포지토리 ID로 교체
JSON_DIR = "generatedata/generated_json" # 2단계에서 생성된 JSON 파일들이 있는 디렉터리
# ---

def consolidate_and_upload():
    """여러 JSON 파일을 통합하여 Hugging Face에 업로드하는 함수"""
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    if not json_files:
        print(f"오류: '{JSON_DIR}' 디렉터리에서 생성된 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(json_files)}개의 JSON 파일을 통합합니다...")
    all_data = []
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
        print(f"  - '{file_path}' 로드 완료 (데이터: {len(data)}개)")

    # 중복 제거 (여러 파일에 걸쳐 중복이 있을 수 있으므로 최종적으로 한 번 더 실행)
    unique_data_str = {json.dumps(d, sort_keys=True) for d in all_data}
    final_data = [json.loads(s) for s in unique_data_str]
    print(f"\n[Result] 총 {len(final_data)}개의 학습 데이터 통합 완료 (중복 제거 후).")

    # 3. Hugging Face에 업로드
    print("\n[Upload] Hugging Face Hub에 데이터셋 업로드를 시작합니다.")
    try:
        train_data, test_data = train_test_split(final_data, test_size=0.2, random_state=42)
        
        # Hugging Face 형식에 맞게 변환
        formatted_train_data = [{"messages": item} for item in train_data]
        formatted_test_data = [{"messages": item} for item in test_data]

        dsd = DatasetDict({
            "train": Dataset.from_list(formatted_train_data),
            "test": Dataset.from_list(formatted_test_data)
        })

        print("\n[Hugging Face] 생성된 데이터셋 정보:")
        print(dsd)

        print(f"\n[Upload] '{HF_REPO_ID}' 리포지토리에 데이터셋을 푸시합니다.")
        dsd.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print("✅ 업로드가 성공적으로 완료되었습니다!")

    except Exception as e:
        print(f"[Error] Hugging Face 업로드 중 오류 발생: {e}")
        print("  - HF_TOKEN에 'write' 권한이 있는지 확인하세요.")
        print(f"  - '{HF_REPO_ID}' 리포지토리가 존재하는지 확인하세요.")

if __name__ == "__main__":
    consolidate_and_upload()