from datasets.preprocess import preprocess_sentence
from .machine_learning_base import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_v = np.vectorize(sigmoid)


class RF(MLBase, ABC):
    def __init__(self, mode, cfg):
        super().__init__(mode, cfg)        
        if mode == "train":
            rf_cfg = cfg["TRAIN"]["RF"]
            if cfg["TRAIN"]["GRIDSEARCH"]:
                rf_model = RandomForestClassifier()
                hyperparams = {
                    "n_estimators": rf_cfg["N_ES"],
                    "max_depth": rf_cfg["MAX_DEPTH"],
                    "min_samples_split": rf_cfg["MIN_SPLIT"],
                    "min_samples_leaf": rf_cfg["MIN_LEAF"],
                    "max_features": rf_cfg["MAX_FEATURES"]
                }
                self.model = GridSearchCV(rf_model, hyperparams,
                                          n_jobs=cfg["WORKERS"],
                                          cv=rf_cfg["CV"],
                                          verbose=2)
            else:
                self.model = RandomForestClassifier(n_estimators=rf_cfg["N_ES"],
                                                    max_depth=rf_cfg["MAX_DEPTH"],
                                                    min_samples_split=rf_cfg["MIN_SPLIT"],
                                                    min_samples_leaf=rf_cfg["MIN_LEAF"],
                                                    max_features=rf_cfg["MAX_FEATURES"])

    def train(self):
        self.model.fit(self.X_train, self.train_ds.labels)
        if self.cfg["TRAIN"]["GRIDSEARCH"]:
            best_params_str = "Best params: " + str(self.model.best_params_)
            print(best_params_str)
            with open(os.path.join(self.output_path, "best_params.txt"), "w") as f:
                f.write(best_params_str)
            self.model = self.model.best_estimator_
        self.save_model()
        pred_train = self.model.predict(self.X_train)
        pred_val = self.model.predict(self.X_val)
        print("Result on valset:")
        print(accuracy_score(self.val_ds.labels, pred_val))
        matrix = confusion_matrix(self.val_ds.labels, pred_val)
        accs = matrix.diagonal()/matrix.sum(axis=1)
        print(matrix)
        print(accs) 
        print(classification_report(self.val_ds.labels, pred_val))
        print("Result on trainset:")
        print(accuracy_score(self.train_ds.labels, pred_train))
        matrix = confusion_matrix(self.train_ds.labels, pred_train)
        accs = matrix.diagonal()/matrix.sum(axis=1)
        print(matrix)
        print(accs) 
        print(classification_report(self.train_ds.labels, pred_train))

    def test(self):
        pred_test = self.model.predict(self.X_test)
        print("Result on valset:")
        print(accuracy_score(self.test_ds.labels, pred_test))
        matrix = confusion_matrix(self.test_ds.labels, pred_test)
        accs = matrix.diagonal()/matrix.sum(axis=1)
        print(matrix)
        print(accs) 
        print(classification_report(self.test_ds.labels, pred_test))
        
    def predict(self, sentences):
        sentences = preprocess_sentence(sentences)
        data = self.vectorizer.transform(sentences)
        pred = self.model.predict(data)
        return pred, np.full((len(pred),), -1)
