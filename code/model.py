import torch
import torch.nn as nn

class TextBERT(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask = attention_mask)
        last_hidden_state = outputs[0]  # BERT의 마지막 은닉 상태
        cls_output = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
    

class MultimodalModel(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes):
        super().__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(74, hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size + hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, audio_features):
        # BERT 출력 얻기
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs[0][:, 0, :]

        # LSTM 출력 얻기
        lstm_outputs, _ = self.lstm(audio_features)
        lstm_output = lstm_outputs[:, -1, :]

        # BERT 출력과 LSTM 출력을 concatenate
        combined_output = torch.cat([cls_output, lstm_output], dim=1)

        # Concatenated 출력을 선형 계층에 통과시켜 3개의 라벨로 분류
        logits = self.classifier(combined_output)
        return logits