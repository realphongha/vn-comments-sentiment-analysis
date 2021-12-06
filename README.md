# Sentiment analysis for Vietnamese comments from VNExpress

## Installation 

### Install `transformer`
 -  Python 3.6+, and PyTorch 1.1.0+ (or TensorFlow 2.0+)
 -  Install `transformers`:
	- `git clone https://github.com/huggingface/transformers.git`
	- `cd transformers`
	- `pip3 install --upgrade .`

### Install requirements
```pip install -r requirements.txt```

## How to run:

### Training:
```python train.py --config path/to/config.yaml```

### Testing:
```python evaluate.py --config path/to/config.yaml --weights path/to/weights.pth```

### Predicting:
```python predict.py --weights path/to/weights.pth --file test.txt --device cpu --cls POS NEG NEU```

## Our project based on `PhoBERT`:
```
@inproceedings{phobert,
title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
year      = {2020},
pages     = {1037--1042}
}
```
