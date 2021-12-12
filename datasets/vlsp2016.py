import torch
import os
from random import shuffle
from .preprocess import preprocess_sentence
from .datasets import BaseDataset


VLSP2016_LABELS = {'POS': 0, 'NEG': 1, 'NEU': 2}
VLSP2016_LABELS_MAP = {0: 'POS', 1: 'NEG', 2: 'NEU'}


class VLSP2016Dataset(BaseDataset):
    def __init__(self, cfg, ds_type, preprocess=True):
        super(VLSP2016Dataset).__init__()
        self.data = list()
        self.labels = list()   
        train_ratio = cfg["DATASET"]["TRAIN"]
        val_ratio = cfg["DATASET"]["VAL"]
        sum_ratio = train_ratio + val_ratio
        train_ratio /= sum_ratio
        val_ratio /= sum_ratio
        
        print("Reading data from %s..." % cfg["DATASET"]["PATH"])
        
        if ds_type == "test":
            cur_data = None
            cur_lbl = None
            root_path = os.path.join(cfg["DATASET"]["PATH"], "test")
            with open(os.path.join(root_path, "test_raw_ANS.txt"), "r", 
                      encoding="utf8") as file:
                for i, line in enumerate(file.read().splitlines()):
                    if i&1:
                        cur_lbl = VLSP2016_LABELS[line.strip()]
                    else:
                        cur_data = line.strip("\xa0")
                    if cur_lbl is not None and cur_data is not None:
                        self.data.append(cur_data)
                        self.labels.append(cur_lbl)
                        cur_data = None
                        cur_lbl = None
        else:
            root_path = os.path.join(cfg["DATASET"]["PATH"], "train")
            pos_path = os.path.join(root_path, "POS")
            neg_path = os.path.join(root_path, "NEG")
            neu_path = os.path.join(root_path, "NEU")
            for fn in os.listdir(pos_path):
                if fn[-4:] != ".txt":
                    continue
                data_file = os.path.join(pos_path, fn)
                data = open(data_file, "r", encoding="utf-8").read().splitlines()
                if cfg["DATASET"]["SHUFFLE"]:
                    shuffle(data)
                frac = int(train_ratio*len(data))
                for i, comment in enumerate(data):
                    if ds_type == "train" and i > frac:
                        continue
                    elif ds_type == "val" and i <= frac:
                        continue
                    if comment:
                        comment = " " + comment.strip() + " "
                        self.data.append(comment)
                        self.labels.append(VLSP2016_LABELS["POS"]) 
                    
            for fn in os.listdir(neg_path):
                if fn[-4:] != ".txt":
                    continue
                data_file = os.path.join(neg_path, fn)
                data = open(data_file, "r", encoding="utf-8").read().splitlines()
                if cfg["DATASET"]["SHUFFLE"]:
                    shuffle(data)
                frac = int(train_ratio*len(data))
                for i, comment in enumerate(data):
                    if ds_type == "train" and i > frac:
                        continue
                    elif ds_type == "val" and i <= frac:
                        continue
                    if comment:
                        comment = " " + comment.strip() + " "
                        self.data.append(comment)
                        self.labels.append(VLSP2016_LABELS["NEG"])
                    
            for fn in os.listdir(neu_path):
                if fn[-4:] != ".txt":
                    continue
                data_file = os.path.join(neu_path, fn)
                data = open(data_file, "r", encoding="utf-8").read().splitlines()
                if cfg["DATASET"]["SHUFFLE"]:
                    shuffle(data)
                frac = int(train_ratio*len(data))
                for i, comment in enumerate(data):
                    if ds_type == "train" and i > frac:
                        continue
                    elif ds_type == "val" and i <= frac:
                        continue
                    if comment:
                        comment = " " + comment.strip() + " "
                        self.data.append(comment)
                        self.labels.append(VLSP2016_LABELS["NEU"])
            
        if not self.data:
            raise Exception("There's no data in", cfg["DATASET"]["PATH"])
        
        if preprocess:
            self.data = preprocess_sentence(self.data)
            
        self.describle()
    
    
if __name__ == "__main__":
    pass
