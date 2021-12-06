import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi import ViTokenizer
from datasets.preprocess import preprocess_sentence


class Predictor(object):
    def __init__(self, model_path, device, num_cls, preprocess=True):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.preprocess = preprocess
        self.model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=num_cls)
        self.device = torch.device('cuda:{}'.format(device)) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    def predict(self, texts: list):
        if self.preprocess:
            texts = preprocess_sentence(texts)
        else:
            texts = list(map(ViTokenizer.tokenize, texts))
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        input_ids = torch.tensor(encodings['input_ids']).to(self.device)
        attention_mask = torch.tensor(encodings['attention_mask']).to(self.device)
        labels = torch.tensor(np.zeros(len(encodings["input_ids"]), 
                                      dtype=np.int64)).to(self.device)
        outputs = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask,
                             labels=labels)
        results = torch.max(torch.sigmoid(outputs[1]), 1)
        confs = results[0].detach().cpu().numpy()
        lbls = results[1].detach().cpu().numpy()
        return confs, lbls


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=str, nargs="+",
                        default=["POS", "NEG", "NEU"], 
                        help='classes')
    parser.add_argument('--device', default='cpu', 
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, 
                        default='log/VNEX2021--2021-12-06--08-34/ckpt_best.pth', 
                        help='path to checkpoint')
    parser.add_argument('--file', type=str, 
                        default="test.txt", 
                        help='path to file that contains test examples')
    opt = parser.parse_args()
    predictor = Predictor(opt.weights, opt.device, len(opt.cls))
    f = open(opt.file, "r", encoding="utf-8")
    texts = f.read().splitlines()
    confs, lbls = predictor.predict(texts)
    for i in range(len(texts)):
        print(texts[i])
        print("Label: %s, Conf: %.4f" % (opt.cls[lbls[i]], confs[i]))
