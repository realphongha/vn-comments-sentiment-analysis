import sys
import os
import json
import datetime

import pickle

sys.path.append("..")
from datasets import DATASET

from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer


class MLBase(ABC):
    def __init__(self, mode, cfg):
        self.cfg = cfg
        datetime_str = datetime.datetime.now().strftime("--%Y-%m-%d--%H-%M")
        output_path = os.path.join(self.cfg["OUTPUT"], 
                                   self.cfg["MODEL"]["NAME"] + "--" + 
                                   self.cfg["DATASET"]["NAME"] + 
                                   datetime_str)
        self.output_path = output_path
            
        if mode == "train":    
            if not os.path.isdir(self.output_path):
                os.mkdir(self.output_path)
            with open(os.path.join(self.output_path, "configs.txt"), "w") as output_file:
                json.dump(cfg, output_file)

            if cfg["DATASET"]["NAME"].lower() in DATASET:
                dataset, labels, labels_map = DATASET[cfg["DATASET"]["NAME"].lower()]
                self.labels, self.labels_map = labels, labels_map 
            else:
                raise NotImplementedError("%s is not implemented!" % 
                                        cfg["DATASET"]["NAME"])
            
            self.num_cls = len(labels.keys()) if "IGNORE" not in labels \
                else len(labels.keys()) - 1 
           
            self.train_ds = dataset(cfg, "train", cfg["TRAIN"]["PREPROCESS"])
            self.val_ds = dataset(cfg, "val", cfg["TEST"]["PREPROCESS"])
            
            if cfg["MODEL"]["EMBEDDING"].lower() == "tfidf":
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=20000)
                self.X_train = self.vectorizer.fit_transform(self.train_ds.data)
                self.save_vectorizer()
                self.X_val = self.vectorizer.transform(self.val_ds.data)

            else:
                raise NotImplementedError("%s embedding is not implemented!" % 
                                      cfg["MODEL"]["EMBEDDING"])
        elif mode == "test":
            if cfg["DATASET"]["NAME"].lower() in DATASET:
                dataset, labels, labels_map = DATASET[cfg["DATASET"]["NAME"].lower()]
                self.labels, self.labels_map = labels, labels_map 
            else:
                raise NotImplementedError("%s is not implemented!" % 
                                        cfg["DATASET"]["NAME"])
            
            self.num_cls = len(labels.keys()) if "IGNORE" not in labels \
                else len(labels.keys()) - 1 
            self.test_ds = dataset(cfg, "test", cfg["TEST"]["PREPROCESS"])
            
            if cfg["MODEL"]["EMBEDDING"].lower() == "tfidf":
                self.load_vectorizer()
                self.load_model()
                self.X_test = self.vectorizer.transform(self.test_ds.data)

            else:
                raise NotImplementedError("%s embedding is not implemented!" % 
                                      cfg["MODEL"]["EMBEDDING"])
        else: # mode == "predict"
            self.load_vectorizer()
            self.load_model()
            
        # if mode == "train" or mode == "test":
        #     if cfg["DATASET"]["NAME"].lower() in DATASET:
        #         dataset, labels, labels_map = DATASET[cfg["DATASET"]["NAME"].lower()]
        #         self.labels, self.labels_map = labels, labels_map 
        #     else:
        #         raise NotImplementedError("%s is not implemented!" % 
        #                                 cfg["DATASET"]["NAME"])
            
        #     self.num_cls = len(labels.keys()) if "IGNORE" not in labels \
        #         else len(labels.keys()) - 1 
        #     if mode == "train":
        #         self.train_ds = dataset(cfg, "train", cfg["TRAIN"]["PREPROCESS"])
        #         self.val_ds = dataset(cfg, "val", cfg["TEST"]["PREPROCESS"])
        #     else: # mode == "test"
        #         self.test_ds = dataset(cfg, "test", cfg["TEST"]["PREPROCESS"])
            
        #     if cfg["MODEL"]["EMBEDDING"].lower() == "tfidf":
        #         if mode == "train":
        #             self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=20000)
        #             self.X_train = self.vectorizer.fit_transform(self.train_ds.data)
        #             self.save_vectorizer()
        #             self.X_val = self.vectorizer.transform(self.val_ds.data)
        #         elif mode == "test":
        #             self.load_vectorizer()
        #             self.load_model()
        #             self.X_test = self.vectorizer.transform(self.test_ds.data)

        #     else:
        #         raise NotImplementedError("%s embedding is not implemented!" % 
        #                               cfg["MODEL"]["EMBEDDING"])
        # else:
        #     self.load_vectorizer()
        #     self.load_model()
            

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def predict(self):
        pass
    
    def save_vectorizer(self):
        pickle.dump(self.vectorizer, open(os.path.join(self.output_path, "vectorizer.pkl"), "wb"))
        print("Saved vectorizer to %s" % os.path.join(self.output_path, "vectorizer.pkl"))
        
    def load_vectorizer(self):
        f = open(self.cfg["TEST"]["VECTORIZER"], 'rb')
        self.vectorizer = pickle.load(f)
        print("Loaded vectorizer from %s" % self.cfg["TEST"]["VECTORIZER"])

    def save_model(self):
        pickle.dump(self.model, open(os.path.join(self.output_path, "model.pkl"), "wb"))
        print("Saved model to %s" % os.path.join(self.output_path, "model.pkl"))
        
    def load_model(self):
        f = open(self.cfg["TEST"]["WEIGHTS"], 'rb')
        self.model = pickle.load(f)
        print("Loaded model from %s" % self.cfg["TEST"]["WEIGHTS"])
