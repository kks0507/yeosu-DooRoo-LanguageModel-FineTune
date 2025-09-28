# Dooroo2025: 여수 관광 특화 언어 모델 및 RAG 시스템

## 1\. 프로젝트 개요

  - **과제 명:** (중소벤처기업부 창업성장기술개발사업) 생성형 AI 기반 관광퀴즈 게임 플랫폼 연구 및 개발
  - **과제 기간:** `25.03.12 ~ 25.07.31`
  - **모델 명:** `Dooroo2025`
  - **핵심 기능:** 여수 관광 정보에 특화된 질의응답 및 콘텐츠 생성
  - **주요 기술:**
      - **Fine-tuning:** `unsloth/Qwen3-4B-Instruct-2507` 모델을 여수 관광/섬 데이터로 파인튜닝
      - **RAG (Retrieval-Augmented Generation):** Vector DB (ChromaDB)를 활용하여 실시간으로 정확한 정보 검색 및 답변 생성
      - **LaaJ (LLM as a Judge):** Gemini 1.5 Flash를 이용한 자동화된 성능 평가 파이프라인 구축

## 2\. 전체 시스템 구성도 (System Architecture)
<img width="1218" height="608" alt="KakaoTalk_20250925_145052708" src="https://github.com/user-attachments/assets/a13374b7-1dc3-4a3f-bb56-bb89522d5fb5" />


## 3\. 데이터 구축 및 가공

원천 데이터를 수집하여 모델 학습 및 RAG 시스템에 사용될 데이터셋과 벡터 DB를 구축합니다.

### 원천 데이터

  - **섬, 관광:** 여수 지역사회연구소, 한국관광공사 API 데이터
  - **식당, 숙소:** 한국관광공사 API 데이터

### 가공 및 결과물

  - **Hugging Face Datasets:** 원천 데이터를 Q\&A 형식으로 가공 및 증강하여 업로드
      - `kingkim/yeosu_island`: [https://huggingface.co/datasets/kingkim/yeosu\_island](https://huggingface.co/datasets/kingkim/yeosu_island)
      - `kingkim/yeosu_tour`: [https://huggingface.co/datasets/kingkim/yeosu\_tour](https://huggingface.co/datasets/kingkim/yeosu_tour)
  - **Vector DB:** RAG 시스템에서 실시간 검색에 사용될 ChromaDB 구축 (`Vectordb.py`)

## 4\. 언어 모델 파인튜닝 (Fine-tuning)

Hugging Face에 업로드된 데이터셋을 기반으로 LoRA(Low-Rank Adaptation) 기법을 사용하여 베이스 모델을 파인튜닝합니다.

### 기술 스택

| 구분 | 내용 |
| :--- | :--- |
| **베이스 모델** | `unsloth/Qwen3-4B-Instruct-2507` |
| **파인튜닝 라이브러리** | Unsloth (LoRA-PEFT) |
| **Trainer** | Hugging Face TRL (SFTTrainer) |
| **결과물 (모델)** | `kingkim/Dooroo2025_v1.0` |

## 5\. RAG 기반 답변 생성 및 평가

파인튜닝된 모델과 RAG를 결합하여 사용자 질문에 대한 최종 답변을 생성하고, LaaJ(LLM as a Judge) 파이프라인을 통해 성능을 자동으로 평가합니다.

### 1\) 하이브리드 답변 생성

  - **동작 방식:** 사용자 질문이 입력되면, RAG 시스템이 Vector DB에서 가장 관련성 높은 정보를 검색하여 컨텍스트를 구성하고, 이를 파인튜닝된 모델에 전달하여 최종 답변을 생성합니다. (`answer.py`)
  - **기대 효과:** 파인튜닝을 통해 얻은 도메인 특화 지식과 RAG를 통해 확보한 사실 기반 정보가 결합되어 답변의 정확성과 신뢰도를 극대화합니다.

### 2\) LaaJ 자동화 평가 (GEVAL)

  - **평가 모델:** `Gemini-1.5-Flash`
  - **평가 지표:** **유창성(Fluency), 일관성(Coherence), 정확성(Accuracy), 완결성(Completeness)** 의 4가지 핵심 지표를 자체 정의된 루브릭(Rubric)에 따라 평가합니다.
  - **프로세스:** 이 과정을 자동화하여 모델 개선에 따른 성능 변화를 지속적으로 추적하고, 각 항목별 점수를 높여나가는 방식으로 모델을 고도화합니다. (`evaluation.py`)
