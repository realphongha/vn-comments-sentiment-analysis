import datetime
import argparse
import os
import json
import math

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from loss.focal_loss import FocalLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AdamW, Adafactor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from datasets import DATASET


# use for focal loss:
class CustomAutoModelForSequenceClassification(AutoModelForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = FocalLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
def main(cfg, output_path):
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
        if cfg["FOCAL_LOSS"]:
            model = CustomAutoModelForSequenceClassification.from_pretrained(
                "vinai/phobert-base", num_labels=num_cls)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                "vinai/phobert-base", num_labels=num_cls)
    else:
        raise NotImplementedError("%s is not implemented!" % 
                                  cfg["MODEL"]["NAME"])
        
    
    train_ds = dataset(cfg, "train", cfg["TRAIN"]["PREPROCESS"])
    train_ds.set_encodings(tokenizer(train_ds.data, 
                           truncation=True, 
                           padding=True))

    train_loader = DataLoader(train_ds, 
                              batch_size=cfg["TRAIN"]["BATCH_SIZE"], 
                              shuffle=cfg["TRAIN"]["SHUFFLE"],
                              num_workers=cfg["WORKERS"])
    val_ds = dataset(cfg, "val", cfg["TEST"]["PREPROCESS"])
    val_ds.set_encodings(tokenizer(val_ds.data, 
                         truncation=True, 
                         padding=True))
    val_loader = DataLoader(val_ds, 
                            batch_size=cfg["TEST"]["BATCH_SIZE"], 
                            shuffle=False,
                            num_workers=cfg["WORKERS"])
    
    # Optimizer
    if cfg["TRAIN"]["OPTIMIZER"].lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=cfg["TRAIN"]["LR"])
    elif cfg["TRAIN"]["OPTIMIZER"] == 'adafactor':
        optimizer = Adafactor(model.parameters(), lr=cfg["TRAIN"]["LR"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg["TRAIN"]["LR"], momentum=0.9)
    
    gpus = tuple(range(len(cfg["GPUS"].strip().split(","))))
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    
    if cfg["TRAIN"]["WEIGHTS"]:
        model.module.load_state_dict(torch.load(cfg["TRAIN"]["WEIGHTS"]))
        print("Loaded weights from", cfg["TRAIN"]["WEIGHTS"])
        
    if cfg["TRAIN"]["FREEZE_BACKBONE"]:
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    if cfg["TRAIN"]["DROPOUT"] and 0 < cfg["TRAIN"]["DROPOUT"] < 1:
        model.module.classifier.dropout = nn.Dropout(cfg["TRAIN"]["DROPOUT"])
        
    # print(model)
    
    iteration = 100 # to be printed
    if cfg["TRAIN"]["EARLY_STOP"]:
        patience = cfg["TRAIN"]["EARLY_STOP"]
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    best_val_loss = math.inf
    ckpt_best = os.path.join(output_path, 'ckpt_best.pth')
    ckpt_latest = os.path.join(output_path, 'ckpt_latest.pth')
    log_path = os.path.join(output_path, 'log.txt')
    
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        print("EPOCH %i:" % epoch)
        log_file = open(log_path, 'a+')
        log_file.write("EPOCH %i:\n" % epoch)
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        # Training:
        model.train()
        step = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            results = torch.max(torch.sigmoid(outputs[1]), 1)
            correct += (results[1] == labels).sum().item()
            total += labels.size(0)
            if len(loss.shape) > 0:
                loss.sum().backward()
            else:
                loss.backward()
            optimizer.step()
            if len(loss.shape) > 0:
                running_loss += loss.sum().item()
                train_losses.append(loss.sum().item())
            else:
                running_loss += loss.item()
                train_losses.append(loss.item())
            if step % iteration == iteration - 1:
                logs = "[EPOCH: {} STEP: {}] loss train: {:.3f}, acc train: {:.3f}" \
                    .format(epoch, step + 1, running_loss / iteration, correct / total)
                # print('\n', logs)
                log_file.write(logs)
                log_file.write("\n")
                running_loss = 0.0
                correct = 0.0
                total = 0.0
            step += 1
        
        # Evaluate:   
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
                loss = outputs[0]
                if len(loss.shape) > 0:
                    valid_losses.append(loss.sum().item())
                else:
                    valid_losses.append(loss.item())
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
        print('Accuracy val of Epoch {}: '.format(epoch), acc, '\n')
        clf_report = classification_report(true_labels, pred_labels)
        print(clf_report)
        log_file.write(clf_report)
        log_file.write("\n")
        clf_report_dict = classification_report(true_labels, pred_labels, 
                                                output_dict=True)
        clf_report_df = pd.DataFrame(clf_report_dict)
        clf_report_df.to_csv(
            os.path.join(output_path, 'val_epoch_{}.csv'.format(epoch)), 
            index=True)
        for i in range(num_cls):
            acc_log = 'Accuracy of %5s: %.4f %%' % (
                labels_map[i], class_correct[i] / class_total[i])
            print(acc_log)
            log_file.write(acc_log + "\n")
        avg_acc_log = 'Average accuracy: %.4f %%' % (
                sum(class_correct) / sum(class_total))
        print(avg_acc_log)
        log_file.write(avg_acc_log + "\n")

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(cfg["TRAIN"]["EPOCH"]))
        print_msg = (f'[{epoch:>{epoch_len}}/{cfg["TRAIN"]["EPOCH"]:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)
        log_file.write(print_msg + "\n")

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        torch.save(model.module.state_dict(), ckpt_latest)
        print("Saved checkpoint to", ckpt_latest)
        log_file.write("Saved checkpoint to %s\n" % ckpt_latest)
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            patience = cfg["TRAIN"]["EARLY_STOP"]
            torch.save(model.module.state_dict(), ckpt_best)
            print("Saved checkpoint to", ckpt_best)
            log_file.write("Saved checkpoint to %s\n" % ckpt_best)
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping!")
                break
        log_file.close()

    # visualize the loss as the network trained
    print("avg_train_losses ", avg_train_losses)
    print("avg_valid_losses ", avg_valid_losses)
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 2)  # consistent scale
    plt.xlim(0, len(avg_train_losses) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, 'loss_plot.png'), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='configs/vnex2021_lite.yaml', 
                        help='path to config file')

    opt = parser.parse_args()
    with open(opt.config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
        
    datetime_str = datetime.datetime.now().strftime("--%Y-%m-%d--%H-%M")
    output_path = os.path.join(cfg["OUTPUT"], 
                               cfg["MODEL"]["NAME"] + "--" +
                               cfg["DATASET"]["NAME"] + 
                               datetime_str)
    os.makedirs(output_path)    
    with open(os.path.join(output_path, "configs.txt"), "w") as output_file:
        json.dump(cfg, output_file)
    
    main(cfg, output_path)
