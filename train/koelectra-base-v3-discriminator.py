import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

df_undersampled = pd.read_csv('train_undersampled.csv')
df = df_undersampled


# 훈련/검증 데이터셋으로 분리
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['generated'])

# 2. 토크나이저 및 모델 로드
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, use_safetensors=True) # 이진 분류이므로 num_labels=2

# 3. PyTorch Dataset 생성
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 데이터셋 객체 생성
train_dataset = TextDataset(train_df['paragraph_text'].tolist(), train_df['generated'].tolist(), tokenizer)
val_dataset = TextDataset(val_df['paragraph_text'].tolist(), val_df['generated'].tolist(), tokenizer)


# 4. 성능 평가 지표 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 5. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir='./results/monologg-koelectra-base-v3-discriminator2',          # 모델과 결과물이 저장될 디렉토리
    num_train_epochs=3,              # 총 학습 에폭 수
    per_device_train_batch_size=128,  # 학습 시 GPU당 배치 사이즈
    per_device_eval_batch_size=64,   # 평가 시 GPU당 배치 사이즈
    warmup_steps=500,                # 학습률 스케줄러를 위한 워밍업 스텝 수
    weight_decay=0.01,               # 가중치 감소(Weight decay)
    logging_dir='./logs',            # 로그 저장 디렉토리
    logging_steps=10,
    learning_rate= 2e-5,                
    lr_scheduler_type="cosine_with_restarts",  # Cosine Annealing with warm restarts
    bf16=True,                     # mixed precision training (FP16)
)

# 6. Trainer 객체 생성 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 학습 시작
trainer.train()

# 최종 평가
print("\n--- 최종 평가 결과 ---")
eval_results = trainer.evaluate()
print(eval_results)