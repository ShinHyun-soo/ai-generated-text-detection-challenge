import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from scipy.special import softmax
from scipy.stats import rankdata  # ← 여기로 변경


MODEL_INFO_LIST = [
    {
        "path": './results/monologg-koelectra-base-v3-discriminator2/checkpoint-3777',
        "tokenizer_name": "monologg/koelectra-base-v3-discriminator",
        "max_length":256
    },
    {
       "path": './results/lighthouse_mdeberta-v3-base-kor-further/skip_eval/checkpoint-9444',
       "tokenizer_name": "lighthouse/mdeberta-v3-base-kor-further",
        "max_length":256
    },
    {
       "path": './results/team-lucid_deberta-v3-xlarge-korean/skip_eval/checkpoint-37767',
       "tokenizer_name": "team-lucid/deberta-v3-xlarge-korean",
       "max_length":128
    },
    {
       "path": './results/team-lucid_deberta-v3-base-korean/skip_eval/checkpoint-9444',
       "tokenizer_name": "team-lucid/deberta-v3-base-korean",
      "max_length":256
    },
    {
       "path": './results/team-lucid_deberta-v3-xlarge-korean/skip_eval_192/checkpoint-37767',
       "tokenizer_name": "team-lucid/deberta-v3-xlarge-korean",
       "max_length":192 
    },
   {
       "path": './results/klue_roberta-large/skip_eval/checkpoint-9444',
       "tokenizer_name": "klue/roberta-large",
       "max_length":256 
    },
]

test_df = pd.read_csv("test.csv")
submission_df = pd.read_csv('sample_submission.csv')

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts, self.tokenizer, self.max_len = texts, tokenizer, max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            str(self.texts[idx]), add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten()
        }

all_model_preds = [] 

for model_info in MODEL_INFO_LIST:
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_name"])
    model     = AutoModelForSequenceClassification.from_pretrained(model_info["path"])
    test_ds   = InferenceDataset(
        test_df['paragraph_text'].tolist(),
        tokenizer,
        max_len=model_info["max_length"]
    )
    trainer   = Trainer(model=model, args=TrainingArguments(
        output_dir='./tmp', do_train=False, do_eval=False, per_device_eval_batch_size=1024
    ))
    logits    = trainer.predict(test_ds).predictions  
    probs     = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:,1]
    all_model_preds.append(probs)  



P = np.stack(all_model_preds, axis=0)  
R = np.stack([ rankdata(P[i], method='average') for i in range(P.shape[0]) ], axis=0)


mean_rank = R.mean(axis=0)              

normalized_rank = (mean_rank - 1) / (mean_rank.max() - 1)

submission_df['generated'] = normalized_rank
submission_df.to_csv('8.csv', index=False, float_format='%.10f')
