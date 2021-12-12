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

### For demo webapp and crawler:
 - Install Chrome, Selenium and compatible chromedriver

## How to run:

### PhoBERT:
#### Training:
```python train.py --config path/to/config.yaml```

#### Testing:
```python evaluate.py --config path/to/config.yaml --weights path/to/weights.pth```

#### Predicting:
```python predict.py --weights path/to/weights.pth --file test.txt --device cpu --cls POS NEG NEU```

### Machine learning algorithms:
#### Training:
```python machine_learning.py --mode train --cfg path/to/config.yaml```

#### Testing:
```python machine_learning.py --mode test --cfg path/to/config.yaml```

#### Predicting:
```python machine_learning.py --mode predict --cfg path/to/config.yaml --file path/to/data/file```

### Webapp:
```cd demo_webapp```
```python webapp.py --weights path/to/weights.pth --device cpu --cls POS NEG NEU```

### Crawler:
```cd crawler```
```python crawler.py```

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
