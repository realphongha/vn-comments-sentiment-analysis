import torch
import numpy as np
import pandas as pd


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def set_encodings(self, encodings):
        self.encodings = encodings
 
    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

    def __len__(self):
        return len(self.data)
    
    def describle(self):
        count_dict = dict()
        sentence_lens = []
        for i in range(len(self.data)):
            sentence = self.data[i]
            sentence_len = len(sentence.split())
            sentence_lens.append(sentence_len)
            label = self.labels[i]
            count_dict[label] = count_dict.get(label, 0) + 1
        print("Total of %i sentence(s)!" % len(self.data))
        print("Number of each label:", count_dict)
        print("Sentence length stats:")
        print(pd.Series(np.array(sentence_lens)).describe())
