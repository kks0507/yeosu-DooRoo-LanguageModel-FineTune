# [cite\_start]여수 모델 학습 프로세스 정리 [cite: 1]

[cite\_start]여수 Dooroo 모델 학습은 크게 2가지 과정으로 구성됩니다. [cite: 2]

## [cite\_start]1. 원천데이터를 학습에 맞게 가공하기 [cite: 3]

### [cite\_start]원천데이터 구성 [cite: 4]

[cite\_start]원천데이터는 총 4가지 유형으로 이루어져 있으며, 해당 사항은 김의향 이사님과 협의가 완료되었습니다. [cite: 4, 5]

  * [cite\_start]**섬, 관광**: 여수 지역사회연구소와 한국관광공사에서 제공하는 API를 원시데이터로 하여 구조화된 형식으로 만들었습니다. [cite: 6, 10]
  * [cite\_start]**식당, 숙소**: 한국관광공사에서 제공하는 API를 원시데이터로 하여 구조화된 형식으로 만들었으며, 각 요소 별로 100건의 규모를 가집니다. [cite: 7]

| 유형 | 원천데이터 파일 이름 |
| :--- | :--- |
| 섬 | island-normalize.csv |
| 관광 | tour-normalize.csv |
| 식당 | yeosu-foodandhotel.xlsx |
| 숙소 | yeosu-foodandhotel.xlsx |
[cite\_start][cite: 8]

### 진행 현황

[cite\_start]현재 4개 중 핵심 요소인 ‘섬’과 ‘관광’에 대한 작업이 마무리된 상태입니다. [cite: 9]

  * [cite\_start]'섬' 관련 원천 데이터: 총 757건 [cite: 11]
  * [cite\_start]'관광' 관련 원천 데이터: 총 341건 [cite: 12]

[cite\_start]이 원천 데이터를 기반으로 각 유형별 질문-답변 형식의 1차 가공 및 증강을 진행했습니다. [cite: 13] [cite\_start]이후 모델 학습에 최적화된 JSON 형식으로 변환하여 허깅페이스 데이터 셋에 업로드했습니다. [cite: 14]

| 유형 | 데이터 수 | 데이터 셋 레포지터리 |
| :--- | :--- | :--- |
| 섬 | 학습: 2270개\<br\>검증: 757개 | [https://huggingface.co/datasets/kingkim/yeosu\_island](https://huggingface.co/datasets/kingkim/yeosu_island) |
| 관광 | 학습: 5430개\<br\>검증: 1360개 | [https://huggingface.co/datasets/kingkim/yeosu\_tour](https://huggingface.co/datasets/kingkim/yeosu_tour) |
[cite\_start][cite: 15]

### [cite\_start]데이터 가공 코드 [cite: 16]

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
[cite\_start][cite: 17]

## [cite\_start]2. 가공한 학습데이터를 언어모델에 학습시키기 [cite: 18]

[cite\_start]허깅페이스에 업로드된 학습용 데이터 셋을 모델에 학습(파인튜닝)시킵니다. [cite: 19]

### [cite\_start]핵심 기술 스택 [cite: 20]

| 구분 | 내용 |
| :--- | :--- |
| 베이스 모델 | unsloth/Qwen3-4B-Instruct-2507 |
| 파인튜닝 라이브러리 | Unsloth (LoRA-PEFT) |
| Trainer | Hugging Face TRL (SFTTrainer) |
[cite\_start][cite: 21]

### [cite\_start]프레임워크 버전 [cite: 22]

| 프레임워크 | 버전 |
| :--- | :--- |
| TRL | 0.22.2 |
| Transformers | 4.56.1 |
| pytorch | 2.5.1+cu21 |
| Datasets | 3.6.0 |
| Tokenizers | 0.22.0 |
[cite\_start][cite: 23]

### [cite\_start]모델 학습 코드 [cite: 24]

**여수 섬 (yeosu-island 폴더)**
| 목적 | 파일 이름 |
| :--- | :--- |
| 모델 파인튜닝 + 허깅페이스 업로드 | train.py |
[cite\_start][cite: 25]
