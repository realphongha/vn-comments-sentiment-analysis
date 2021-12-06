import argparse
import os

import torch
import yaml
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from datasets import DATASET
        
        
def main(cfg, opt):
    cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
    cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = cfg["CUDNN"]["ENABLED"]
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUS"]
    
    if cfg["DATASET"]["NAME"].lower() in DATASET:
        dataset, labels, labels_map = DATASET[cfg["DATASET"]["NAME"].lower()]
    else:
        raise NotImplementedError("%s is not implemented!" % 
                                  cfg["DATASET"]["NAME"])
    
    num_cls = len(labels.keys()) if "IGNORE" not in labels \
        else len(labels.keys()) - 1 
    
    if cfg["MODEL"]["NAME"].lower() == "phobert":
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base", num_labels=num_cls)
    else:
        raise NotImplementedError("%s is not implemented!" % 
                                  cfg["MODEL"]["NAME"])
        
    
    val_ds = dataset(cfg, "test", cfg["TEST"]["PREPROCESS"])
    val_ds.set_encodings(tokenizer(val_ds.data, 
                         truncation=True, 
                         padding=True))
    val_loader = DataLoader(val_ds, 
                            batch_size=cfg["TEST"]["BATCH_SIZE"], 
                            shuffle=False,
                            num_workers=cfg["WORKERS"])
    
    gpus = tuple(range(len(cfg["GPUS"].strip().split(","))))
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    
    weights = opt.weights if opt.weights else cfg["TEST"]["WEIGHTS"]
    if weights:
        model.module.load_state_dict(torch.load(weights))
        print("Loaded weights from", weights)
    
    model.eval()
    logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []
    class_correct = list(0. for i in range(num_cls))
    class_total = list(0. for i in range(num_cls))
    
    step = 0
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        attention_mask = batch['attention_mask'].cuda(non_blocking=True)
        labels = batch['labels'].cuda(non_blocking=True)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(torch.sigmoid(outputs[1]), 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            logit_pred = outputs[1]
            pred_label = torch.sigmoid(logit_pred)
            logit_pred = logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            labels = labels.to('cpu').numpy()
        tokenized_texts.append(input_ids)
        logit_preds.append(logit_pred)
        true_labels.append(labels)
        pred_labels.append(pred_label)
        step += 1

    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    pred_labels = [np.argmax(pl) for pl in pred_labels]
    true_labels = [item for sublist in true_labels for item in sublist]
    acc = accuracy_score(true_labels, pred_labels)
    print('Accuracy val:', acc, '\n')
    clf_report = classification_report(true_labels, pred_labels)
    print(clf_report)
    for i in range(num_cls):
        acc_log = 'Accuracy of %5s: %2d %%' % (
            labels_map[i], 100 * class_correct[i] / class_total[i])
        print(acc_log)
    avg_acc_log = 'Average accuracy: %2d %%' % (
            100 * sum(class_correct) / sum(class_total))
    print(avg_acc_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='configs/vnex2021_lite.yaml', 
                        help='path to config file')
    parser.add_argument('--weights', 
                        type=str, 
                        default='', 
                        help='path to weights file')

    opt = parser.parse_args()
    with open(opt.config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
        
    main(cfg, opt)
