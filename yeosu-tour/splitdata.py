import pandas as pd
import numpy as np
import os

# --- 설정 ---
CSV_FILE_PATH = "yeosu_normalize.csv"
OUTPUT_DIR = "split_data"
NUM_SPLITS = 17  # 몇 개의 파일로 나눌지 결정
# ---

def split_csv():
    """CSV 파일을 여러 개의 작은 파일로 분할하는 함수"""
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"오류: 원본 CSV 파일 '{CSV_FILE_PATH}'을(를) 찾을 수 없습니다.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"'{OUTPUT_DIR}' 디렉터리를 생성했습니다.")

    # 데이터를 무작위로 섞어 각 파일에 데이터가 편중되지 않도록 합니다.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 데이터프레임을 NUM_SPLITS 개수만큼 분할
    split_dfs = np.array_split(df, NUM_SPLITS)

    print(f"데이터를 {NUM_SPLITS}개의 파일로 분할합니다...")
    for i, part_df in enumerate(split_dfs):
        output_path = os.path.join(OUTPUT_DIR, f"data_part_{i+1:02d}.csv")
        part_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  -> {output_path} (행: {len(part_df)}개)")

    print("\n✅ 분할이 완료되었습니다.")

if __name__ == "__main__":
    split_csv()