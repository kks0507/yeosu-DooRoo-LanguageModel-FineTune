# main.py

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# =================================================================================
# 1) 모델·토크나이저 로드
# =================================================================================
# ⚠️ 이전에 파인튜닝하여 업로드한 모델 ID로 변경해야 합니다.
# 예: "kingkim/Dooroo2025"
model_id = "kingkim/Dooroo2025" # ⬅️ 사용자의 파인튜닝 모델 ID로 변경하세요.
HF_TOKEN = "YOUR_HF_KEY" # ⬅️ 비공개 모델이라면 허깅페이스 토큰을 여기에 넣으세요.

print(f"'{model_id}' 모델 로드를 시작합니다...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=HF_TOKEN, # 비공개 모델 접근용
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN, # 비공개 모델 접근용
)
model.eval()
print("모델 로드 완료.")


# =================================================================================
# 2) 생성 파라미터
# =================================================================================
gen_config = GenerationConfig(
    temperature=0.6,
    top_p=0.95,
    max_new_tokens=1024,
    min_new_tokens=80,
    do_sample=False
)

# =================================================================================
# 3) FastAPI 설정
# =================================================================================
app = FastAPI()

# ✨ [수정 1] 시스템 프롬프트를 받을 수 있도록 Pydantic 모델 변경
class QueryRequest(BaseModel):
    query: str
    system_prompt: str | None = None # system_prompt 필드 추가 (선택 사항)


@app.post("/generate/")
async def generate_response(request: QueryRequest):
    try:
        # ✨ [수정 2] 시스템 프롬프트와 사용자 질문을 채팅 형식으로 구성
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.query})

        # ✨ [수정 3] apply_chat_template을 사용하여 모델이 이해하는 형식으로 변환
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True, # 답변 생성을 위해 <|im_start|>assistant 추가
            return_tensors="pt"
        ).to(model.device)

        # 3‑2) 텍스트 생성
        with torch.no_grad():
            output_ids = model.generate(input_ids=inputs, generation_config=gen_config)

        # 3‑3) 디코딩 (입력 부분은 제외하고 디코딩)
        # 중요: inputs.shape[1]를 사용하여 생성된 부분만 잘라냅니다.
        result_text = tokenizer.decode(
            output_ids[0][inputs.shape[1]:],
            skip_special_tokens=True
        )
        return {"response": result_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================================================================================
# 4) 서버 실행
# =================================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)