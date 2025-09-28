import os
import pandas as pd
import random
from datasets import load_dataset
from typing import List, Dict

# 허깅페이스 토큰 및 설정
HF_TOKEN = "YOUR_HF_KEY"
HF_USERNAME = "YOUR_HF_NAME"
REPO_NAME = "yeosoo_dataset"

# 출력 디렉토리
OUTPUT_DIR = "/home/kjm/dooroo/evaluation_datasets"

# 평가 카테고리별 파일명
EVALUATION_FILES = [
    "evaluation_accuracy.csv",
    "evaluation_coherence.csv",
    "evaluation_completeness.csv",
    "evaluation_fluency.csv"
]

def load_huggingface_dataset() -> List[Dict]:
    """
    허깅페이스에서 데이터셋을 로드합니다.
    """
    try:
        # 데이터셋 로드 (토큰 사용)
        dataset = load_dataset(
            f"{HF_USERNAME}/{REPO_NAME}",
            token=HF_TOKEN,
            split="train"  # 또는 적절한 split
        )

        print(f"데이터셋 로드 완료: {len(dataset)}개 데이터")

        # 데이터를 딕셔너리 리스트로 변환
        data_list = []
        for item in dataset:
            data_list.append(item)

        return data_list

    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return []

def sample_random_data(data_list: List[Dict], num_samples: int = 5) -> List[Dict]:
    """
    데이터에서 랜덤으로 샘플을 추출합니다.
    """
    if len(data_list) < num_samples:
        print(f"경고: 전체 데이터({len(data_list)})가 요청된 샘플 수({num_samples})보다 적습니다.")
        return data_list

    return random.sample(data_list, num_samples)

def convert_to_csv_format(sampled_data: List[Dict]) -> pd.DataFrame:
    """
    샘플링된 데이터를 CSV 형식으로 변환합니다.
    기존 CSV 파일과 동일한 구조: question, ground_truth_answer, model_answer
    """
    csv_data = []

    for item in sampled_data:
        # 데이터셋의 컬럼명에 따라 매핑 (실제 데이터셋 구조에 맞게 수정 필요)
        row = {
            "question": item.get("question", item.get("input", "")),
            "ground_truth_answer": item.get("ground_truth_answer", item.get("output", item.get("answer", ""))),
            "model_answer": item.get("model_answer", "")  # 필요시 추가 로직
        }
        csv_data.append(row)

    return pd.DataFrame(csv_data)

def save_evaluation_datasets(data_list: List[Dict]):
    """
    평가용 데이터셋 4개를 생성하고 저장합니다.
    """
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4개의 평가 파일 각각에 대해 랜덤 샘플링 및 저장
    for filename in EVALUATION_FILES:
        print(f"\n{filename} 생성 중...")

        # 랜덤으로 5개 샘플 추출
        sampled_data = sample_random_data(data_list, 5)

        # CSV 형식으로 변환
        df = convert_to_csv_format(sampled_data)

        # 파일 저장
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')

        print(f"✅ {filename} 저장 완료 ({len(df)}개 데이터)")
        print(f"   저장 경로: {filepath}")

def main():
    """
    메인 실행 함수
    """
    print("=== 허깅페이스 데이터셋 평가용 CSV 생성 ===")

    # 1. 허깅페이스 데이터셋 로드
    print("\n1. 허깅페이스 데이터셋 로드 중...")
    data_list = load_huggingface_dataset()

    if not data_list:
        print("❌ 데이터셋 로드에 실패했습니다.")
        return

    print(f"✅ 총 {len(data_list)}개의 데이터를 로드했습니다.")

    # 2. 데이터 샘플 확인
    if data_list:
        print("\n2. 데이터 샘플 확인:")
        sample_item = data_list[0]
        print(f"   샘플 데이터 키: {list(sample_item.keys())}")
        print(f"   샘플 데이터: {sample_item}")

    # 3. 평가용 데이터셋 생성
    print("\n3. 평가용 데이터셋 생성 중...")
    save_evaluation_datasets(data_list)

    print("\n🎉 모든 평가용 CSV 파일 생성이 완료되었습니다!")
    print(f"📁 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    # 랜덤 시드 설정 (재현가능한 결과를 위해)
    random.seed(42)

    # 메인 함수 실행
    main()
