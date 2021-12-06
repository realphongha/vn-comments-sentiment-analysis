import torch
import os
from .preprocess import preprocess_sentence
from .datasets import BaseDataset


VNEX2021_LABELS = {'POS': 0, 'NEG': 1, 'NEU': 2, 'IGNORE': 3}
VNEX2021_LABELS_MAP = {0: 'POS', 1: 'NEG', 2: 'NEU', 3: 'IGNORE'}


class VNEX2021Dataset(BaseDataset):
    def __init__(self, cfg, ds_type, preprocess=True):
        super(VNEX2021Dataset).__init__()
        self.data = list()
        self.labels = list()
        if ds_type == "train":
            root_path = os.path.join(cfg["DATASET"]["PATH"], 
                                     cfg["DATASET"]["TRAIN"])
        elif ds_type == "val":
            root_path = os.path.join(cfg["DATASET"]["PATH"], 
                                     cfg["DATASET"]["VAL"])
        elif ds_type == "test":
            root_path = os.path.join(cfg["DATASET"]["PATH"], 
                                     cfg["DATASET"]["TEST"])
        else:
            raise NotImplementedError("%s is not implemented!" % ds_type)
        data_path = os.path.join(root_path, "data")
        label_path = os.path.join(root_path, "label")
        print("Reading data from %s..." % root_path)
        for fn in os.listdir(data_path):
            if fn[-4:] != ".txt":
                continue
            label_file = os.path.join(label_path, fn)
            if not os.path.exists(label_path):
                continue
            data_file = os.path.join(data_path, fn)
            data = open(data_file, "r", encoding="utf-8").read().splitlines()
            labels = open(label_file, "r").read().splitlines()[0]
            labels = list(map(int, labels.strip().split()))
            for i, comment in enumerate(data):
                comment = " " + comment.strip() + " "
                lbl = labels[i]
                if lbl == VNEX2021_LABELS["IGNORE"]:
                    continue
                self.data.append(comment)
                self.labels.append(lbl) 
            
        if not self.data:
            raise Exception("There's no data in", root_path)
        
        if preprocess:
            self.data = preprocess_sentence(self.data)
            
        self.describle()
    
    
if __name__ == "__main__":
    pass
