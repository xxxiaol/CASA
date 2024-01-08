import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

data_name = 'casa_llama2.json'
data = json.load(open(data_name, 'r'))

nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
nli_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
label_mapping = ['contradiction', 'neutral', 'entailment']

device = torch.device("cuda") 
nli_model.to(device)
nli_model.eval()

for idxi, i in enumerate(data):
    if not 'contexts' in i:
        continue
    pred = [0, 0]  # sufficient, insufficient
    for j in range(len(i['contexts'])):
        if len(i['contexts'][j]) == 0:
            continue

        scores = []
        score_max = []
        for k in np.arange(len(i['contexts'][j])):
            cur_context = i['contexts'][j][k]
            if len(cur_context)>0 and cur_context[0] == '"' and cur_context[-1] == '"':
                cur_context = cur_context[1:-1]
            x = nli_tokenizer.encode(cur_context, i['conclusion'], return_tensors='pt', truncation_strategy='only_first').to(device)
            logits = nli_model(x)[0]
            scores.append(logits.softmax(dim=1)[0])
            score_max.append(logits.softmax(dim=1)[0].argmax(axis=0))
        scores = torch.stack(scores, dim=0)
        
        i.update({'entailment': [label_mapping[x] for x in score_max]})
        votes = [0, 0, 0]  # 'contradiction', 'neutral', 'entailment'
        for k in score_max:
            votes[k] += 1
        if votes[0] > votes[2]:
            pred[1] += 1
        elif votes[0] < votes[2]:
            pred[0] += 1
        else:  # compare probability sum
            scores_sum = torch.sum(scores, axis=0)
            if scores_sum[0] > scores_sum[2]:
                pred[1] += 1
            else:
                pred[0] += 1
            
    if pred[1] > 0:
        i.update({"prediction": 'insufficient'})
    else:
        i.update({"prediction": 'sufficient'})
    
with open(data_name, 'w') as f:
    f.write(json.dumps(data, indent=2))