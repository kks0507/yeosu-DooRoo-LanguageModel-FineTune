import os
import json
import pandas as pd
from tqdm import tqdm
import time
import google.generativeai as genai
import argparse # argparse 추가

# =================================================================================
# ✨ 설정 부분
# =================================================================================
GEMINI_API_KEY = "YOUR_GEMINI_KEY"
genai.configure(api_key=GEMINI_API_KEY)

INPUT_DIR = "evaluation_datasets"
OUTPUT_DIR = "evaluation_results" # 결과 파일을 저장할 별도 디렉토리

# =================================================================================
# ✨ 카테고리별 상세 평가 프롬프트
# =================================================================================
PROMPT_TEMPLATES = {
    "fluency": """
    당신은 언어 모델의 '유창성'을 평가하는 AI 심판입니다. 아래 [질문], [정답 예시], [모델 답변]을 읽고, 제시된 [평가 기준]에 따라 각 항목별로 1~5점 척도로 평가해주세요.
    최종적으로 5개 항목 점수의 평균을 계산하여 JSON 형식으로만 반환해야 합니다. 다른 설명은 절대 추가하지 마세요.

    [질문]: {question}
    [정답 예시]: {ground_truth_answer}
    [모델 답변]: {model_answer}

    [유창성 평가 기준]:
    1. 문법적 정확성: 문법적 오류가 거의 없고 자연스러운가?
    2. 어휘의 적절성: 문맥에 맞는 어휘와 표현을 사용했는가?
    3. 문장 구조의 다양성: 반복적이지 않고, 다양한 문장 구조를 활용했는가?
    4. 읽기 용이성: 문장이 매끄럽게 이어져 쉽게 읽히는가?
    5. 자연스러움: 한국어 모국어 화자에게 어색하지 않은가?

    [출력 형식 (JSON)]:
    {{
      "scores": {{
        "문법적 정확성": <1-5점>, "어휘의 적절성": <1-5점>, "문장 구조의 다양성": <1-5점>, "읽기 용이성": <1-5점>, "자연스러움": <1-5점>
      }},
      "average_score": <5개 점수의 평균 점수>,
      "reason": "<종합적인 평가 이유>"
    }}
    """,
    "coherence": """
    당신은 언어 모델의 '일관성'을 평가하는 AI 심판입니다. 아래 [질문], [정답 예시], [모델 답변]을 읽고, 제시된 [평가 기준]에 따라 각 항목별로 1~5점 척도로 평가해주세요.
    최종적으로 5개 항목 점수의 평균을 계산하여 JSON 형식으로만 반환해야 합니다. 다른 설명은 절대 추가하지 마세요.

    [질문]: {question}
    [정답 예시]: {ground_truth_answer}
    [모델 답변]: {model_answer}

    [일관성 평가 기준]:
    1. 논리적 연결성: 답변 내 문장과 문장이 논리적으로 연결되어 있는가?
    2. 주제 유지: 답변이 질문의 주제를 벗어나지 않고 일관되게 유지되는가?
    3. 맥락 적합성: 앞뒤 문맥에 어긋나지 않고 자연스럽게 이어지는가?
    4. 모순 여부: 텍스트 내에 자기모순이나 불일치가 없는가?
    5. 흐름의 매끄러움: 문단 전개가 끊기지 않고 매끄럽게 이어지는가?

    [출력 형식 (JSON)]:
    {{
      "scores": {{
        "논리적 연결성": <1-5점>, "주제 유지": <1-5점>, "맥락 적합성": <1-5점>, "모순 여부": <1-5점>, "흐름의 매끄러움": <1-5점>
      }},
      "average_score": <5개 점수의 평균 점수>,
      "reason": "<종합적인 평가 이유>"
    }}
    """,
    "accuracy": """
    당신은 언어 모델의 '정확성'을 평가하는 AI 심판입니다. 아래 [질문], [정답 예시], [모델 답변]을 읽고, 제시된 [평가 기준]에 따라 각 항목별로 1~5점 척도로 평가해주세요.
    최종적으로 4개 항목 점수의 평균을 계산하여 JSON 형식으로만 반환해야 합니다. 다른 설명은 절대 추가하지 마세요.

    [질문]: {question}
    [정답 예시]: {ground_truth_answer}
    [모델 답변]: {model_answer}

    [정확성 평가 기준]:
    1. 사실적 일치성: 제시된 정보가 [정답 예시] 및 실제 사실과 일치하는가?
    2. 세부 정보 정확성: 날짜, 지명, 인물, 수치 등 구체적 정보가 올바른가?
    3. 오정보/환각 여부: 존재하지 않는 사실이나 왜곡된 정보가 포함되지 않았는가?
    4. 질문 대응 정확성: 사용자의 질문에 맞는 정확한 사실을 제시했는가?

    [출력 형식 (JSON)]:
    {{
      "scores": {{
        "사실적 일치성": <1-5점>, "세부 정보 정확성": <1-5점>, "오정보/환각 여부": <1-5점>, "질문 대응 정확성": <1-5점>
      }},
      "average_score": <4개 점수의 평균 점수>,
      "reason": "<종합적인 평가 이유>"
    }}
    """,
    "completeness": """
    당신은 언어 모델의 '완결성'을 평가하는 AI 심판입니다. 아래 [질문], [정답 예시], [모델 답변]을 읽고, 제시된 [평가 기준]에 따라 각 항목별로 1~5점 척도로 평가해주세요.
    최종적으로 5개 항목 점수의 평균을 계산하여 JSON 형식으로만 반환해야 합니다. 다른 설명은 절대 추가하지 마세요.

    [질문]: {question}
    [정답 예시]: {ground_truth_answer}
    [모델 답변]: {model_answer}

    [완결성 평가 기준]:
    1. 질문 대응 충실성: 사용자의 질문/요구사항을 빠짐없이 충실히 반영했는가?
    2. 핵심 요소 포함 여부: 답변에 필요한 핵심 정보(시간, 장소 등)가 모두 포함되었는가?
    3. 세부 정보 보강성: 단순 개요가 아니라 적절한 세부사항을 충분히 제공했는가?
    4. 균형 잡힌 정보 제공: 질문의 모든 측면을 균형 있게 다루었는가?
    5. 답변의 완결성: 답변이 중간에 끊기거나 불완전하지 않고 완결성을 갖추었는가?

    [출력 형식 (JSON)]:
    {{
      "scores": {{
        "질문 대응 충실성": <1-5점>, "핵심 요소 포함 여부": <1-5점>, "세부 정보 보강성": <1-5점>, "균형 잡힌 정보 제공": <1-5점>, "답변의 완결성": <1-5점>
      }},
      "average_score": <5개 점수의 평균 점수>,
      "reason": "<종합적인 평가 이유>"
    }}
    """
}

def evaluate_category(category):
    """지정된 카테고리에 대해 상세 루브릭을 사용하여 Gemini 평가를 수행합니다."""
    print("="*50)
    print(f"### 3단계: Gemini API로 답변 평가 시작 ({category}) ###")
    print("="*50)

    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    input_file = os.path.join(INPUT_DIR, f"evaluation_{category}.csv")
    output_file = os.path.join(OUTPUT_DIR, f"results_{category}.csv")
    
    try:
        df = pd.read_csv(input_file)
        if df['model_answer'].isnull().any() or (df['model_answer'] == '').any():
            print(f"경고: '{input_file}' 파일에 비어있는 'model_answer'가 있습니다.")
            print("2_generate_answers.py를 먼저 실행했는지 확인해주세요.")
            return
    except FileNotFoundError:
        print(f"오류: '{input_file}' 파일을 찾을 수 없습니다.")
        return
        
    # Gemini 모델 및 프롬프트 선택
    judge_model = genai.GenerativeModel('gemini-1.5-flash')
    prompt_template = PROMPT_TEMPLATES[category]

    evaluation_results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"채점 중 ({category})"):
        prompt = prompt_template.format(
            question=row['question'],
            ground_truth_answer=row['ground_truth_answer'],
            model_answer=row['model_answer']
        )
        
        try:
            response = judge_model.generate_content(prompt)
            json_response_str = response.text.strip().replace("```json", "").replace("```", "")
            eval_data = json.loads(json_response_str)
            
            row_dict = row.to_dict()
            row_dict['evaluation_details'] = json.dumps(eval_data.get('scores', {}), ensure_ascii=False)
            row_dict['average_score'] = eval_data.get('average_score')
            row_dict['evaluation_reason'] = eval_data.get('reason', 'N/A')
            evaluation_results.append(row_dict)

        except Exception as e:
            # (이하 오류 처리 코드는 이전과 동일)
            print(f"\n오류 발생 (행: {index}): {e}.")
            row_dict = row.to_dict()
            row_dict.update({'evaluation_details': None, 'average_score': None, 'evaluation_reason': f"ERROR: {e}"})
            evaluation_results.append(row_dict)
        
        time.sleep(0.5)

    result_df = pd.DataFrame(evaluation_results)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    final_avg_score = result_df['average_score'].mean()
    print(f"\n'{category}' 카테고리 채점이 완료되었습니다.")
    print(f"-> 결과가 '{output_file}' 파일에 저장되었습니다.")
    print(f"-> 최종 평균 점수: {final_avg_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="지정된 카테고리에 대해 Gemini 평가를 수행합니다.")
    parser.add_argument("--category", type=str, required=True, choices=["fluency", "coherence", "accuracy", "completeness"],
                        help="평가할 카테고리를 지정합니다.")
    args = parser.parse_args()
    
    evaluate_category(args.category)