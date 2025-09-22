from getpass import getpass
from huggingface_hub import login
import torch
import ast
import json
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import shutil
import os

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
# 'SFTConfig' 대신 'TrainingArguments'를 'transformers'에서 가져옵니다.
from transformers import TrainingArguments, TextStreamer
# 'SFTConfig'는 더 이상 여기서 import하지 않습니다.
from trl import SFTTrainer
# =================================================================================
# 1. 사용자 정보 설정
# ⚠️ 주의: 아래 토큰이 포함된 코드를 외부에 공유하지 마세요.
# =================================================================================
HF_TOKEN = "YOUR_HF_TOKEN"  # Hugging Face 토큰 입력
MODEL_REPO = "kingkim/Dooroo2025"   # 학습 결과(모델)를 업로드할 저장소

# =================================================================================
# 2. 모델 및 토크나이저 불러오기
# =================================================================================
max_seq_length = 2048
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# =================================================================================
# 3. LoRA 설정 (PEFT)
# =================================================================================
model = FastModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.05, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# =================================================================================
# 4. 데이터셋 준비
# =================================================================================
import json
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

# 1) Tokenizer에 Qwen3 모델의 공식 채팅 템플릿 적용
print("Tokenizer에 Qwen3 채팅 템플릿을 적용합니다...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-instruct",
)

# 2) 데이터셋 로드
print("데이터셋을 로드합니다...")
print("데이터셋 'kingkim/yeosu_tour'를 로드합니다...")
dataset_tour = load_dataset("kingkim/yeosu_tour", token=HF_TOKEN)
print("데이터셋 'kingkim/yeosu_island'를 로드합니다...")
dataset_island = load_dataset("kingkim/yeosu_island", token=HF_TOKEN)

# 3) ✨ 'train'과 'test' 스플릿을 각각 합칩니다.
# train 스플릿 합치기
combined_train_dataset = concatenate_datasets(
    [dataset_tour["train"], dataset_island["train"]]
)
# test 스플릿 합치기
combined_eval_dataset = concatenate_datasets(
    [dataset_tour["test"], dataset_island["test"]]
)

# 4) ✨ (매우 중요) 합쳐진 학습 데이터셋을 섞어줍니다.
# 이렇게 해야 모델이 두 주제의 데이터를 골고루 학습할 수 있습니다.
combined_train_dataset = combined_train_dataset.shuffle(seed=42)

# 5) ✨ 합쳐진 데이터셋을 사용하도록 변수 이름을 맞춰줍니다.
from datasets import DatasetDict
dataset = DatasetDict({
    "train": combined_train_dataset,
    "test": combined_eval_dataset
})

# 6) 'messages'를 안전하게 파싱하여 chat template 적용
def formatting_prompts_func(examples):
    texts = []
    for item in examples["messages"]:
        try:
            # 문자열이면 JSON으로 파싱, 이미 list/dict면 그대로 사용
            if isinstance(item, str):
                obj = json.loads(item)
            else:
                obj = item

            # {"messages": [...]} 또는 바로 [...] 형태 모두 지원
            if isinstance(obj, dict) and "messages" in obj:
                msgs = obj["messages"]
            else:
                msgs = obj  # list로 가정

            # 최종 text 생성
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception as e:
            print(f"데이터 파싱 오류: {e} — 빈 문자열로 대체")
            text = ""  # 배치 길이 보존

        texts.append(text)

    return {"text": texts}

# (선택) 샘플 타입 빠르게 점검 — 맵핑 전 1회만
try:
    sample = dataset["train"][0]["messages"]
    print("샘플 타입 확인:", type(sample))
    print("샘플 값(앞부분):", str(sample)[:200])
except Exception as _:
    pass

# 7) 전체 스플릿에 포맷팅 적용
print("데이터셋을 모델 학습 형식에 맞게 변환합니다...")
remove_cols = [c for c in ["messages"] if c in dataset["train"].column_names]
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=remove_cols,   # 존재할 때만 제거
)

# 8) 빈 문자열 제거 (파싱 실패 샘플 정리)
dataset = dataset.filter(lambda x: x["text"] != "")

# 9) 스플릿 준비 (validation 이름 주의)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

print("데이터셋 준비 완료:")
print(f"Train 샘플 (학습용): {len(train_dataset)}")
print(f"Eval 샘플 (검증용): {len(eval_dataset)}")

print("\n포맷팅된 샘플 예시:")
if len(train_dataset) > 0:
    print(train_dataset[0]["text"])
else:
    print("⚠️ train 데이터가 비었습니다. 상단의 '샘플 타입 확인' 로그와 원본 데이터의 'messages' 필드를 점검하세요.")

# =================================================================================
# 5. 모델 학습 (Unsloth 최적화 적용)
# =================================================================================
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import train_on_responses_only

# -- SFTTrainer를 설정합니다.
# 전문가 코드의 TrainingArguments 방식을 사용하되, 사용자님의 평가 전략을 결합합니다.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,    # ✨ train_dataset으로 명확히 지정
    eval_dataset=eval_dataset,      # ✨ eval_dataset 추가
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    # 'SFTConfig'를 'TrainingArguments'로 변경합니다. 내용은 동일합니다.
    args=TrainingArguments(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        num_train_epochs=200,
        learning_rate=4e-6,
        bf16=True,
        fp16=False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./outputs",
        report_to="none",
    ),
)

# -- [Unsloth 최적화] 모델이 사용자의 질문이 아닌 '답변' 부분에만 집중하여 학습하도록 설정합니다.
# 이는 모델이 질문 형식을 모방하는 대신, 답변 생성 능력을 높이는 데 효과적입니다.
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# -- 모델 학습 시작 --
print("\n모델 학습을 시작합니다...")
trainer_stats = trainer.train()
print("모델 파인튜닝이 완료되었습니다.")

# =================================================================================
# 6. 모델 평가 및 저장 (수정된 최종안)
# =================================================================================
# 1. 최종 모델 성능을 명시적으로 평가하고 결과를 저장합니다.
#    - trainer.evaluate()를 호출하면 SFTTrainer 설정 시 전달한 eval_dataset으로 평가가 수행됩니다.
print("\n최종 모델의 성능을 평가합니다...")
eval_results = trainer.evaluate()

print("최종 평가 결과:")
print(eval_results)

# --- 👇 [추가] 평가 결과를 JSON 파일로 저장 ---
import json
with open("evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)
print("\n✅ 평가 결과를 'evaluation_results.json' 파일에 저장했습니다.")

# 3. 학습된 최종 LoRA 어댑터를 명시적으로 저장합니다.
# final_adapter_path = "./final_adapter"
# print(f"\n학습된 LoRA 어댑터를 '{final_adapter_path}' 경로에 저장합니다...")
# trainer.save_model(final_adapter_path)
# print("저장 완료.")

# =================================================================================
# 7. LoRA 어댑터 적용하여 베이스 모델과 결합
# =================================================================================
# Hugging Face Hub에 생성될 레포지토리 정보
hf_repo_id = MODEL_REPO # 변경 필요

# --- 1. 저장된 LoRA 어댑터를 다시 로드합니다 ---
# print(f"'{final_adapter_path}'에서 어댑터를 로드하고 베이스 모델과 결합합니다...")
print("\nLoRA 어댑터를 베이스 모델과 병합합니다...")
# 이 시점의 model 객체는 학습이 완료된 LoRA 모델입니다.
model = model.merge_and_unload()
print("병합 완료.")

# from_pretrained에 LoRA 어댑터 경로를 직접 지정합니다.
# Unsloth는 어댑터 설정 파일을 보고 자동으로 올바른 베이스 모델을 로드한 후,
# LoRA 어댑터를 적용하여 완벽한 PeftModel 객체를 반환합니다.
# model, tokenizer = FastLanguageModel.from_pretrained(
#    model_name = final_adapter_path, 
#    max_seq_length = 2048,
#    dtype = torch.bfloat16,
#    load_in_4bit = False,
#    load_in_8bit = False,
#)

# --- 4. 모델을 Hub에 업로드 ---
# 이제 model 객체는 어댑터가 적용된 상태로 올바르게 인식됩니다.
print(f"모델을 병합하여 '{hf_repo_id}' 레포지토리에 업로드합니다...")
print("이 작업은 시간이 다소 걸릴 수 있습니다.")

# 2. 병합된 일반 모델이므로, 일반 업로드 함수를 사용합니다.
model.push_to_hub(
    "kingkim/Dooroo2025", # 레포지터리 ID
    tokenizer = tokenizer,
    token = HF_TOKEN,
)

print("\n업로드 완료!")
print(f"모델 페이지: https://huggingface.co/{hf_repo_id}")
