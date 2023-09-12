import torch
from transformers import BertTokenizer, BertModel
from model import TextBERT

def classify_text(text, path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    multi_bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
    
    model = TextBERT(multi_bert, 3)
    model.load_state_dict(torch.load(path))
    
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 모델 예측
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
    _, predicted_label_idx = torch.max(torch.abs(outputs), dim=1)
    predicted_label = predicted_label_idx.item()
    # logits = outputs.logits
    # predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label - 1


test_bool = True

while test_bool:
    lang = input("사용 언어를 입력하세요(k : English-Korean model, d : English-Deutsch model): ")
    if lang == 'k':
        text = input('텍스트를 입력하세요 : ')

        result = classify_text(text, 'multilingual_model/eng_kor_model.pt')
        if result == 0:
            print(f'{result}, neural')
        elif result == 1:
            print(f'{result}, positive')
        else:
            print(f'{result}, negative')
        # print(result)
            
    elif lang == 'd':
        text = input('텍스트를 입력하세요 : ')

        result = classify_text(text, 'multilingual_model/eng_deu_model.pt')
        if result == 0:
            print(f'{result}, neural')
        elif result == 1:
            print(f'{result}, positive')
        else:
            print(f'{result}, negative')
        # print(result)
            
    else:
        test_bool = False
    