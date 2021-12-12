import argparse
import yaml

from time import time

from machine_learning.random_forest import RF
from machine_learning.svm import SVM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='predict', 
                        help='train, test or predict?')
    parser.add_argument('--cfg', type=str, default='configs/vlsp2016_svm.yaml', 
                        help='path to config file')
    parser.add_argument('--file', type=str, default='test.txt', 
                        help='path to data file for predict')
    opt = parser.parse_args()
    
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
            
    if cfg["MODEL"]["NAME"].lower() == "rf":
        rf_engine = RF(opt.mode, cfg)
        if opt.mode.lower() == "train":
            rf_engine.train()
        elif opt.mode.lower() == "test":
            rf_engine.test()
        elif opt.mode.lower() == "predict":
            f = open(opt.file, "r", encoding="utf8")
            lines = f.read().splitlines()
            begin = time()
            res = rf_engine.predict(lines)
            sps = len(lines)/(time()-begin) # sentences per second
            print(res)
            print("Sentences per second:", sps)
        else:
            raise NotImplementedError("%s mode is not implemented!" % opt.mode)
    elif cfg["MODEL"]["NAME"].lower() == "svm":
        svm_engine = SVM(opt.mode, cfg)
        if opt.mode.lower() == "train":
            svm_engine.train()
        elif opt.mode.lower() == "test":
            svm_engine.test()
        elif opt.mode.lower() == "predict":
            f = open(opt.file, "r", encoding="utf8")
            lines = f.read().splitlines()
            begin = time()
            res = svm_engine.predict(lines)
            sps = len(lines)/(time()-begin) # sentences per second
            print(res)
            print("Sentences per second:", sps)
        else:
            raise NotImplementedError("%s mode is not implemented!" % opt.mode)
    else:
        raise NotImplementedError("%s model is not implemented!" % opt.model)
