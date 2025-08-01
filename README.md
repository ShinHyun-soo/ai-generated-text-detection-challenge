
# 2025 SW중심대학 디지털 경진대회 : AI부문 
<img width="1486" height="276" alt="image" src="https://github.com/user-attachments/assets/fc62e092-9578-4547-a17e-c1a9f7d27da5" />


## 생성형 AI(LLM)와 인간 : 텍스트 판별 챌린지 (Private 13th) 🏆
<img width="5512" height="11023" alt="12  한성대_Private_13위_Cube_포스터(확인)-1" src="https://github.com/user-attachments/assets/f2b621f7-4742-4ff4-bc83-9d40af896832" />

### Reproduce
1. Download Data in [Dacon](https://dacon.io/competitions/official/236473/data). Data should be include train.csv, test.csv, sample_submission.csv to run full code in this repository.
2. Create python 3.11 environment and run `pip install -r "requirements.txt"`
3. Run `pre-processing.ipynb`, you can get `train_undersampled.csv` .
4. Run `python train/roberta-large.py && train/deberta-v3-xlarge-krean-192.py && deberta-v3-xlarge-korean.py && deberta-ve-base-korean.py && koelectra-base-v3-discriminaotr.py && mdeberta-v3-base-kor-further.py`
5. Run `inference.py`
