# 여수 Dooroo 전용 언어모델 학습 프로세스

## 중소벤처기업부 창업성장기술개발사업(디딤돌):

  - **과제 명:** 생성형 AI 기반 관광퀴즈 게임 플랫폼 연구 및 개발.
  - 여수 Dooroo 모델 학습은 크게 3가지 과정으로 구성됩니다.

## 1\. 원천데이터를 학습에 맞게 가공하기

### 원천데이터 구성

원천데이터는 총 4가지 유형으로 이루어져 있습니다.

  * **섬, 관광**: 여수 지역사회연구소와 한국관광공사에서 제공하는 API를 원시데이터로 하여 구조화된 형식으로 만들었습니다.
  * **식당, 숙소**: 한국관광공사에서 제공하는 API를 원시데이터로 하여 구조화된 형식으로 만들었으며, 각 요소 별로 100건의 규모를 가집니다.

| 유형 | 원천데이터 파일 이름 |
| :--- | :--- |
| 섬 | island-normalize.csv |
| 관광 | tour-normalize.csv |
| 식당 | yeosu-foodandhotel.xlsx |
| 숙소 | yeosu-foodandhotel.xlsx |

### 진행 현황

현재 4개 중 핵심 요소인 ‘섬’과 ‘관광’에 대한 작업이 마무리된 상태입니다.

  * '섬' 관련 원천 데이터: 총 757건
  * '관광' 관련 원천 데이터: 총 341건

이 원천 데이터를 기반으로 각 유형별 질문-답변 형식의 1차 가공 및 증강을 진행했습니다. 이후 모델 학습에 최적화된 JSON 형식으로 변환하여 허깅페이스 데이터 셋에 업로드했습니다.

| 유형 | 데이터 수 | 데이터 셋 레포지터리 |
| :--- | :--- | :--- |
| 섬 | 학습: 2270개<br>검증: 757개 | [https://huggingface.co/datasets/kingkim/yeosu\_island](https://huggingface.co/datasets/kingkim/yeosu_island) |
| 관광 | 학습: 5430개<br>검증: 1360개 | [https://huggingface.co/datasets/kingkim/yeosu\_tour](https://huggingface.co/datasets/kingkim/yeosu_tour) |

### 데이터 가공 코드

**여수 섬 (yeosu-island 폴더)**

| 목적 | 파일 이름 |
| :--- | :--- |
| 데이터 증강 및 가공 + 허깅페이스 업로드 | yeosu-island.py |

**여수 관광 (yeosu-tour 폴더)**

| 목적 | 파일 이름 |
| :--- | :--- |
| 데이터 분할 (행 기준) | splitdata.py |
| 데이터 증강 및 가공 | generatedata/generatedata01.py |
| 허깅페이스 업로드 | upload.py |

## 2\. 가공한 학습데이터를 언어모델에 학습시키기

허깅페이스에 업로드된 학습용 데이터 셋을 모델에 학습(파인튜닝)시킵니다.

### 핵심 기술 스택

| 구분 | 내용 |
| :--- | :--- |
| 베이스 모델 | unsloth/Qwen3-4B-Instruct-2507 |
| 파인튜닝 라이브러리 | Unsloth (LoRA-PEFT) |
| Trainer | Hugging Face TRL (SFTTrainer) |

### 프레임워크 버전

| 프레임워크 | 버전 |
| :--- | :--- |
| TRL | 0.22.2 |
| Transformers | 4.56.1 |
| pytorch | 2.5.1+cu21 |
| Datasets | 3.6.0 |
| Tokenizers | 0.22.0 |

### 모델 학습 코드

**여수 섬 (yeosu-island 폴더)**

| 목적 | 파일 이름 |
| :--- | :--- |
| 모델 파인튜닝 + 허깅페이스 업로드 | train.py |

## 3\. RAG 및 자동화된 평가 시스템 도입

단순 파인튜닝을 넘어, 모델의 성능을 객관적으로 측정하고 실시간 정보에 기반한 정확한 답변을 생성하기 위해 RAG(Retrieval-Augmented Generation)와 LaaJ(LLM as a Judge) 평가 시스템을 도입했습니다.

### 1\. 하이브리드 접근: 파인튜닝과 RAG의 결합

학습 성과를 극대화하기 위해, **파인튜닝된 모델**과 **RAG 기법**을 결합한 하이브리드 형태로 시스템을 구성했습니다.

  * **파인튜닝 모델:** 여수 관광 도메인에 특화된 지식과 말투를 학습합니다.
  * **RAG:** 질문이 들어왔을 때 Vector DB에서 가장 관련성 높은 최신 정보를 실시간으로 검색하여 모델에 컨텍스트로 제공합니다.

이러한 접근을 통해 모델이 가진 기본 지식에 더해, 특정 질문에 대한 사실 기반의 정확하고 풍부한 답변 생성이 가능해집니다.

| 목적 | 파일 이름 |
| :--- | :--- |
| RAG 파이프라인 구현 | LAAJ\_answer\_rag.py |
| Vector DB 구축 | LAAJ\_vectordb.py |

### 2\. LaaJ 기반 자동화 평가 시스템 (GEVAL)

학습 결과의 성능을 일관성 있고 객관적으로 평가하기 위해, **GEVAL 형식의 LaaJ(LLM as a Judge)** 평가 파이프라인을 구축했습니다.

자체적으로 정의한 상세 평가 루브릭(Rubric)을 기반으로 **유창성(Fluency), 일관성(Coherence), 정확성(Accuracy), 완결성(Completeness)** 4가지 핵심 지표를 평가합니다. 이 과정을 자동화하여 모델 개선에 따른 성능 변화를 지속적으로 추적하고, 각 항목별 점수를 높여나가는 방식으로 모델을 고도화합니다.

| 목적 | 파일 이름 |
| :--- | :--- |
| 4개 지표 자동 평가 수행 | LAAJ\_eval.py |
