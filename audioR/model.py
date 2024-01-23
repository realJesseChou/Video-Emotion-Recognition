import os

import librosa
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel, Wav2Vec2ForCTC, Wav2Vec2Processor, DistilBertForSequenceClassification

here = os.path.abspath(__file__)
father = os.path.dirname
audio_re_model_name = os.path.join(father(here), r"pretrained_models", r"hubert-base-ch-speech-emotion-recognition")
sst_extrac_model_name = os.path.join(father(here), r"pretrained_models", r"wav2vec2-large-xlsr-53-chinese-zh-cn")
text_re_model_name = os.path.join(father(here), r"pretrained_models", r"distilbert-base-multilingual-cased")

sst_emo_map = {
    0: "积极",
    1: "中性",
    2: "消极"
}

duration = 6
sample_rate = 16000

config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=audio_re_model_name,
)


def id2class(id):
    if id == 0:
        return "消极" #"愤怒"
    elif id == 1:
        return "消极" #"害怕"
    elif id == 2:
        return "积极" #"高兴"
    elif id == 3:
        return "中性"
    elif id == 4:
        return "消极" #"悲伤"
    else:
        return "积极"


def predict(path, audio_processor, audio_model, sst_processor, sst_model):
    audio_model = audio_model.to("cuda" if torch.cuda.is_available() else "cpu")
    sst_model = sst_model.to("cuda" if torch.cuda.is_available() else "cpu")

    speech, sr = librosa.load(path=path, sr=sample_rate)
    # 语音情感识别
    audio_speech = audio_processor(speech, padding="max_length", truncation=True, max_length=duration * sr,
                       return_tensors="pt", sampling_rate=sr).input_values
    audio_speech = audio_speech.to("cuda" if torch.cuda.is_available() else "cpu")
    # 字幕提取
    sst_speech = sst_processor(speech, padding="max_length", truncation=True, max_length=13 * sr,
                       return_tensors="pt", sampling_rate=sr).input_values
    sst_speech = sst_speech.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # 情感识别
        audio_logit = audio_model(audio_speech)
        # 字幕提取
        sst_logit = sst_model(sst_speech)
        sst_preds = torch.argmax(sst_logit.logits, dim=-1)
        sens = sst_processor.batch_decode(sst_preds)

    # score = F.softmax(audio_logit, dim=1).detach().cpu().numpy()[0]
    id = torch.argmax(audio_logit).cpu().numpy()
    print(id)
    return id2class(id), sens


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


def audio_predict_and_sst_extrac():
    # 语音情感识别模型
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_re_model_name)
    audio_model = HubertForSpeechClassification.from_pretrained(
        audio_re_model_name,
        config=config,
    )
    audio_model.eval()

    # 字幕提取模型
    sst_pocessor = Wav2Vec2Processor.from_pretrained(sst_extrac_model_name)
    sst_model = Wav2Vec2ForCTC.from_pretrained(sst_extrac_model_name)

    # 文本情感识别模型
    text_tokenizer = AutoTokenizer.from_pretrained(text_re_model_name)
    text_re_model = DistilBertForSequenceClassification.from_pretrained(text_re_model_name)

    file_path = [os.path.join(father(father(here)), r"Repo", r"data", r"audio", path) for path in os.listdir(os.path.join(father(father(here)), r"Repo", r"data", r"audio"))]
    audio_emo_res = []
    sst_emo_res = []

    # 循环处理每个audio文件
    for _, audio_path in enumerate(file_path):
        # 这里返回每个音频情感识别的结果id和对应的文字。
        print(f"正在对{audio_path}进行处理......")
        emo, sentence = predict(audio_path, audio_processor, audio_model, sst_pocessor, sst_model)
        audio_emo_res.append(emo)
        inputs = text_tokenizer(sentence, return_tensors="pt", padding=True)
        r = text_re_model(inputs["input_ids"], inputs["attention_mask"])
        sst_emo_res.append(sst_emo_map[torch.argmax(r.logits.detach(), dim=-1).item()])

    return audio_emo_res, sst_emo_res

if __name__ == "__main__":
    x, y = audio_predict_and_sst_extrac()
    print(len(x), x)
    print(len(y), y)
