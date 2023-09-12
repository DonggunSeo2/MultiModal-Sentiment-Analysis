import utils
from model import TextBERT
from dataset import TextDataset

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW 
from transformers import BertTokenizer, BertModel

# pickle to list 
mosi_train = pd.read_pickle('data/MOSI/train.pkl')
mosi_dev = pd.read_pickle('data/MOSI/dev.pkl')
mosi_test = pd.read_pickle('data/MOSI/test.pkl')

mosi_train_kor_txt = pd.read_pickle('data/MOSI/train_text_ko.pkl')
mosi_train_lat_txt = pd.read_pickle('data/MOSI/train_text_la.pkl')
mosi_train_jap_txt = pd.read_pickle('data/MOSI/train_text_ja.pkl')
mosi_train_deu_txt = pd.read_pickle('data/MOSI/train_text_de.pkl')


mosei_train = pd.read_pickle('data/MOSEI/train.pkl')
mosei_dev = pd.read_pickle('data/MOSEI/dev.pkl')
mosei_test = pd.read_pickle('data/MOSEI/test.pkl')

# modality division
train_ms_txt = utils.modality_div(mosi_train, 3)
train_ms_aud = utils.modality_div(mosi_train, 2)
train_ms_labels = utils.modality_div(mosi_train, -1)

dev_ms_txt = utils.modality_div(mosi_dev, 3)
dev_ms_aud = utils.modality_div(mosi_dev, 2)
dev_ms_labels = utils.modality_div(mosi_dev, -1)

test_ms_txt = utils.modality_div(mosi_test, 3)
test_ms_aud = utils.modality_div(mosi_test, 2)
test_ms_labels = utils.modality_div(mosi_test, -1)

# 모델과 토크나이저
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_model = BertModel.from_pretrained("bert-base-multilingual-uncased")

# tokenization and classification
train_tok_txt = utils.tokenized_data_subword(train_ms_txt, tokenizer = tokenizer)
train_ms_lab = utils.reg2cls(train_ms_labels)
dev_tok_txt = utils.tokenized_data_subword(dev_ms_txt, tokenizer = tokenizer)
dev_ms_lab = utils.reg2cls(dev_ms_labels)

test_tok_txt = utils.tokenized_data_subword(test_ms_txt, tokenizer = tokenizer)
test_ms_lab = utils.reg2cls(test_ms_labels)

# 각 단어를 하나의 토큰으로 일치시키고, 서브워드 분리가 되는 단어는 UNK 토큰으로 설정
train_eng_txt = [" ".join(sentence) for sentence in train_tok_txt]
dev_joined_txt = [" ".join(sentence) for sentence in dev_tok_txt]
test_joined_txt = [" ".join(sentence) for sentence in test_tok_txt]

tok_kor = [tokenizer.tokenize(sentence) for sentence in mosi_train_kor_txt]
tok_lat = [tokenizer.tokenize(sentence) for sentence in mosi_train_lat_txt]
tok_jap = [tokenizer.tokenize(sentence) for sentence in mosi_train_jap_txt]
tok_deu = [tokenizer.tokenize(sentence) for sentence in mosi_train_deu_txt]

train_kor_txt = [" ".join(sentence) for sentence in tok_kor]
train_lat_txt = [" ".join(sentence) for sentence in tok_lat]
train_jap_txt = [" ".join(sentence) for sentence in tok_jap]
train_deu_txt = [" ".join(sentence) for sentence in tok_deu]

train_joined_txt = train_eng_txt
# train_joined_txt = train_eng_txt + train_kor_txt
# train_joined_txt = train_eng_txt + train_lat_txt
# train_joined_txt = train_eng_txt + train_jap_txt
# train_joined_txt = train_eng_txt + train_deu_txt
# train_joined_txt = train_kor_txt
# train_joined_txt = train_lat_txt
# train_joined_txt = train_jap_txt
# train_joined_txt = train_deu_txt

# 인풋 아이디랑 어텐션 마스크 설정
train_encoded_inputs = tokenizer(train_joined_txt, padding = "longest",return_tensors = "pt")
train_input_ids = train_encoded_inputs['input_ids']
train_attention_mask = train_encoded_inputs['attention_mask']

dev_encoded_inputs = tokenizer(dev_joined_txt, padding = "longest",return_tensors = "pt")
dev_input_ids = dev_encoded_inputs['input_ids']
dev_attention_mask = dev_encoded_inputs['attention_mask']

test_encoded_inputs = tokenizer(test_joined_txt, padding = "longest",return_tensors = "pt")
test_input_ids = test_encoded_inputs['input_ids']
test_attention_mask = test_encoded_inputs['attention_mask']

# 데이터 셋 생성
train_dataset = TextDataset(train_input_ids, train_attention_mask, train_ms_lab)
test_dataset = TextDataset(test_input_ids, test_attention_mask, test_ms_lab)

# DataLoader 사용
train_dataloader = DataLoader(train_dataset, batch_size = 16)
test_dataloader = DataLoader(test_dataset, batch_size = 16)
num_cls = 3

# BERT 모델 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 분류기 및 준비
txt_model = TextBERT(bert_model, num_cls)
optimizer = AdamW(txt_model.parameters(), lr = 1e-6)
criterion = nn.CrossEntropyLoss()

# 학습
txt_model.to(device)

num_epochs = 7


print('-----------train started----------')
for epoch in range(num_epochs):
    txt_model.train()
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).long()

        optimizer.zero_grad()
        logits = txt_model(input_ids, attention_mask = attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item()}")
    
# 모델 평가
txt_model.eval()
total_acc = 0
num_batches = 0

# 예측 라벨과 실제 라벨을 저장할 리스트를 초기화합니다.
all_pred_labels = []
all_true_labels = []
print('-----------test started----------')
# 그래디언트 계산 비활성화
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = txt_model(input_ids, attention_mask=attention_mask)
        acc = utils.accuracy(logits, labels)

        total_acc += acc
        num_batches += 1

        # 예측 라벨과 실제 라벨 저장
        pred_labels = torch.argmax(logits, dim=1)
        all_pred_labels.extend(pred_labels.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())
        
# 저장된 예측 라벨과 실제 라벨을 사용하여 F1 스코어 계산
y_pred = torch.tensor(all_pred_labels, dtype=torch.long)
y_true = torch.tensor(all_true_labels, dtype=torch.long)

# 모델 성능 평가
test_accuracy = total_acc / num_batches
test_f1_score = utils.f1_score(y_true, y_pred, 3)
print(f"Test Accuracy: {test_accuracy:.7f}")
print(f"Test F1 Score: {test_f1_score:.7f}")
# torch.save(txt_model.state_dict(), "multilingual_model/eng_kor_model.pt")
torch.save(txt_model.state_dict(), "multilingual_model/eng_model.pt")