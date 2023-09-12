import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import utils
import numpy as np
import model

# pickle to list 
mosi_train = pd.read_pickle('data/MOSI/train.pkl')
mosi_dev = pd.read_pickle('data/MOSI/dev.pkl')
mosi_test = pd.read_pickle('data/MOSI/test.pkl')

# modality division
train_ms_labels = utils.modality_div(mosi_train, -1)
dev_ms_labels = utils.modality_div(mosi_dev, -1)
test_ms_labels = utils.modality_div(mosi_test, -1)

# tokenization and classification
train_ms_lab = utils.reg2cls(train_ms_labels)
dev_ms_lab = utils.reg2cls(dev_ms_labels)
test_ms_lab = utils.reg2cls(test_ms_labels)

# 데이터 로드
train_data = torch.load('train_embedding_data.pt')
train_labels = torch.tensor(train_ms_lab, dtype = torch.long)
test_data = torch.load('test_embedding_data.pt')
test_labels = torch.tensor(test_ms_lab, dtype = torch.long)

# 데이터셋 생성
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 모델, 손실 함수 및 옵티마이저 초기화
main_model = model.MultimodalModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(main_model.parameters(), lr=3e-5)
    
# 학습
num_epochs = 25
for epoch in range(num_epochs):
    main_model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = main_model(inputs.mean(dim=1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# 평가
main_model.eval()
correct = 0
total = 0
all_predicted = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = main_model(inputs.mean(dim=1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predicted.extend(predicted.cpu().numpy())  
        all_labels.extend(labels.cpu().numpy())  
    

print(f'Accuracy: {100 * correct / total}%')
f1 = utils.calculate_f1_score(all_labels, all_predicted)  # 수정한 부분
print(f'F1 Score: {f1}')  # 수정한 부분