import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import os

df_undersampled = pd.read_csv('train_undersampled.csv')
train_df = df_undersampled 


MODEL_NAME = "lighthouse/mdeberta-v3-base-korean-further"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, use_safetensors=True) # 이진 분류



class TextDataset(Dataset):
    # 데이터셋 객체 생성 시 초기 설정을 위한 메서드
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts  # 텍스트 데이터 리스트 (예: ['문장1', '문장2', ...])
        self.labels = labels  # 레이블 데이터 리스트 (예: [0, 1, ...])
        self.tokenizer = tokenizer  # Hugging Face의 토크나이저 객체
        self.max_len = max_len  # 토큰화 후 최대 시퀀스 길이

    # 데이터셋의 총 샘플 수를 반환하는 메서드
    def __len__(self):
        return len(self.texts)

    # 주어진 인덱스(idx)에 해당하는 샘플을 반환하는 메서드
    # DataLoader가 이 메서드를 호출하여 미니배치를 구성합니다.
    def __getitem__(self, idx):
        # 인덱스에 해당하는 텍스트와 레이블을 가져옵니다.
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 토크나이저를 사용하여 텍스트를 모델 입력 형식으로 변환합니다.
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # 문장 시작([CLS])과 끝([SEP])에 특수 토큰을 추가합니다.
            max_length=self.max_len,      # 문장의 최대 길이를 설정합니다. 이보다 길면 잘라냅니다.
            return_token_type_ids=False,  # 토큰 타입 ID는 반환하지 않습니다 (BERT와 같은 모델에서 두 문장을 구분할 때 사용).
            padding='max_length',         # 문장 길이를 max_len에 맞추기 위해 짧은 경우 나머지를 [PAD] 토큰으로 채웁니다.
            truncation=True,              # max_len보다 긴 문장을 잘라냅니다.
            return_attention_mask=True,   # 어텐션 마스크를 생성합니다 ([PAD] 토큰은 0, 나머지는 1).
            return_tensors='pt',          # 결과를 PyTorch 텐서 형태로 반환합니다.
        )

        # 모델에 입력으로 들어갈 딕셔너리를 구성하여 반환합니다.
        return {
            'input_ids': encoding['input_ids'].flatten(),           # 토큰화된 ID 텐서. flatten()으로 1차원으로 만듭니다.
            'attention_mask': encoding['attention_mask'].flatten(), # 어텐션 마스크 텐서. flatten()으로 1차원으로 만듭니다.
            'labels': torch.tensor(label, dtype=torch.long)         # 레이블을 long 타입의 PyTorch 텐서로 변환합니다.
        }

train_dataset = TextDataset(train_df['paragraph_text'].tolist(), train_df['generated'].tolist(), tokenizer)


clean_model_name = MODEL_NAME.replace("/", "_")
output_path = os.path.join("./results", clean_model_name, "skip_eval")

training_args = TrainingArguments(
    seed=42,
    output_dir=output_path,          
    num_train_epochs=3,              
    per_device_train_batch_size=64, 
    learning_rate= 2e-5,                
    warmup_steps=500,                
    weight_decay=0.01,              
    logging_dir='./logs',            
    logging_steps=1000,
    bf16=True,                      
    lr_scheduler_type="cosine_with_restarts",
    save_strategy="epoch", 

)        

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

print("\n--- 학습 완료 ---")
print(f"모델 체크포인트가 '{output_path}' 경로에 저장되었습니다.")