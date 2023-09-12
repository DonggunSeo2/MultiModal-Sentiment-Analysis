import numpy as np
import torch

def modality_div(data, mod):
    div_li = []
    
    if mod == 0 or mod == 1 or mod == 2 or mod == 3:
        for idx in range(len(data)):
            div_li.append(data[idx][0][mod])
            
    elif mod == -1:
        for idx in range(len(data)):
            div_li.append(data[idx][1][0][0])
        
    else:
        raise Exception("IndexError")
    
    return div_li

def tokenized_data(txt_data, tokenizer):
    tok_data = []
    
    for instance in txt_data:
        tokenized_instance = []
        for word in instance:
            tokenized_word = tokenizer.tokenize(word)
            if len(tokenized_word) == 1:  # 단어가 하나의 토큰으로 토크나이징되는 경우
                tokenized_instance.extend(tokenized_word)
            else:  # 단어가 여러 서브워드로 토크나이징되는 경우
                tokenized_instance.append("[UNK]")
        tok_data.append(tokenized_instance)
    return tok_data

def tokenized_data_subword(txt_data, tokenizer):
    tok_data = []
    for instance in txt_data:
        tokenized_instance = []
        for word in instance:
            tokenized_word = tokenizer.tokenize(word)
            tokenized_instance.extend(tokenized_word)
        tok_data.append(tokenized_instance)
    return tok_data
            
def reg2cls(data):
    for i in range(len(data)):
        if data[i] >= 1.0:
            data[i] = 2.0
        elif data[i] < 1.0 and data[i] > -1.0:
            data[i] = 1.0
        else:
            data[i] = 0.0
            
    return data

def padded_data(tok_txt_data, tokenizer):
    # 패딩 적용
    max_length = max([len(instance) for instance in tok_txt_data])
    padded_dataset = []
    for instance in tok_txt_data:
        instance += ['[PAD]'] * (max_length - len(instance))
        padded_dataset.append(instance)

    # 텐서 변환
    input_ids = [tokenizer.convert_tokens_to_ids(instance) for instance in padded_dataset]
    input_tensor = torch.tensor(input_ids)
    return input_tensor

        
def pret_emb(input_tensor, model, batch_size = 32):
    
    for i in range(0, len(input_tensor), batch_size):
        # BERT 모델 통과
        with torch.no_grad():
            outputs = model(input_tensor)
            embeddings = outputs.last_hidden_state[0]
            
        embeddings.append(batch_embeddings)
        
        embeddings_tensor = torch.cat(embeddings, dim=0)
    return embeddings_tensor


def pret_emb_batched(embeddings, tok_txt_data, tokenizer, model, batch_size=8):
    # 입력 데이터를 배치 단위로 분할
    for i in range(0, len(tok_txt_data), batch_size):
        batch = tok_txt_data[i:i + batch_size]

        # 패딩 적용
        max_length = max([len(instance) for instance in batch])
        padded_dataset = []
        for instance in batch:
            instance += ['[PAD]'] * (max_length - len(instance))
            padded_dataset.append(instance)

        # 텐서 변환
        input_ids = [tokenizer.convert_tokens_to_ids(instance) for instance in padded_dataset]
        input_tensor = torch.tensor(input_ids)

        # BERT 모델 통과
        with torch.no_grad():
            outputs = model(input_tensor)
            batch_embeddings = outputs.last_hidden_state[0]

        # 임베딩 벡터를 리스트에 추가
        embeddings.append(batch_embeddings)

    # 리스트를 텐서로 변환하여 반환
    embeddings_tensor = torch.cat(embeddings, dim=0)
    return embeddings_tensor

def accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

def f1_score(y_true, y_pred, num_classes):
    # Confusion matrix를 계산합니다.
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    # Precision과 Recall을 계산합니다.
    true_positive = conf_matrix.diag()
    false_positive = conf_matrix.sum(dim=0) - true_positive
    false_negative = conf_matrix.sum(dim=1) - true_positive

    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)

    # F1 스코어를 계산합니다.
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

    return f1.mean().item()

def calculate_f1_score(true_labels, predicted_labels):
    num_classes = len(set(true_labels))
    class_wise_f1 = []

    for cls in range(num_classes):
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            if true_label == cls and predicted_label == cls:
                true_positive += 1
            elif true_label != cls and predicted_label == cls:
                false_positive += 1
            elif true_label == cls and predicted_label != cls:
                false_negative += 1

        if true_positive + false_positive + false_negative == 0:
            f1 = 0
        else:
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        class_wise_f1.append(f1)

    return sum(class_wise_f1) / num_classes

def word2sen(txt_data):
    sen_li = []

    for i in range(len(txt_data)):
        sen_li.append(' '.join(txt_data[i]))
        
    return sen_li

def trans_sen(txt_data, lan):
    translated_list = []
    translator = Translator()
    count = 0
    
    for text in txt_data:
        translated_text = translator.translate(text, dest = lan).text
        translated_list.append(translated_text)
        count += 1
        print('{} {} translated'.format(count, text))
    
    return translated_list
