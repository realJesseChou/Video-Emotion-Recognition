# import torch
# from torch.distributions import Categorical
# from torch import Tensor
# from typing import Optional,Union
# import collections
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
# from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
#
#
# model_path = "E:\Re\VER\\audioR\pretrained_models\distilbert-base-multilingual-cased"
# def func():
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = DistilBertForSequenceClassification.from_pretrained(model_path)
#
#     text = "今天天气不错，非常适合踢足球，我很喜欢"
#     inputs = tokenizer(text, return_tensors="pt", padding=True)
#     res = model(inputs["input_ids"], inputs["attention_mask"])
#     pred = torch.argmax(res.logits.detach(), dim=-1).data
#
#
# def compute_prob(x:list):
#     counts = collections.Counter(x)
#     total = len(x)
#
#     probs = []
#     for k, v in counts.items():
#         probs.append(v / total)
#     probs = np.array(list(counts.values())) / total
#     return torch.tensor(probs)
#
# def compute_mi(x:Union[Tensor, list], y:Union[Tensor, list]):
#     if isinstance(x, Tensor):
#         x = x.detach().tolist()
#     if isinstance(y, Tensor):
#         y = y.detach().tolist()
#     x_ten = compute_prob(x)
#     y_ten = compute_prob(y)
#
#     px = Categorical(probs=x_ten)
#     py = Categorical(probs=y_ten)
#     pxy = Categorical(probs=torch.outer(x_ten, y_ten))
#
#     mi = torch.sum(pxy.log_prob(pxy.sample()) - px.log_prob(px.sample()) - py.log_prob(py.sample()))
#     return mi
#
#
# if __name__ == "__main__":
#     x = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
#     y = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
#     z = [0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 1]
#     # # print(compute_mi(compute_prob(x), compute_prob(y)))
#     # func()
#     # print(compute_mi(x, x))
#     print(mutual_info_score(x, y))
#     print(mutual_info_score(x, y))
#     print(mutual_info_score(x, z))
#     print(mutual_info_score(z, y))


import os
import random

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel

model_name_or_path = "xmj2002/hubert-base-ch-speech-emotion-recognition"
duration = 6
sample_rate = 16000

config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
)


def id2class(id):
    if id == 0:
        return "angry"
    elif id == 1:
        return "fear"
    elif id == 2:
        return "happy"
    elif id == 3:
        return "neutral"
    elif id == 4:
        return "sad"
    else:
        return "surprise"


def predict(path, processor, model):
    speech, sr = librosa.load(path=path, sr=sample_rate)
    speech = processor(speech, padding="max_length", truncation=True, max_length=duration * sr,
                       return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        logit = model(speech)
    score = F.softmax(logit, dim=1).detach().cpu().numpy()[0]
    id = torch.argmax(logit).cpu().numpy()
    print(f"file path: {path} \t predict: {id2class(id)} \t score:{score[id]} ")


class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x


processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
model = HubertForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)
model.eval()

file_path = [f"test_data/{path}" for path in os.listdir("test_data")]
path = random.sample(file_path, 1)[0]
predict(path, processor, model)
