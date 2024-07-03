import torch
import numpy as np
import os
import json

from torch.utils.data import Dataset, DataLoader


class NerDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]["text"]
        labels = self.data[item]["labels"]
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data
    
def get_data(data_path, tokenizer, max_seq_len, label2id, train_batch_size, dev_batch_size):
    
    
    with open(os.path.join(data_path, "train.txt"), "r") as fp:
        train_data = fp.read().split("\n")
    train_data = [json.loads(d) for d in train_data]

    with open(os.path.join(data_path, "dev.txt"), "r") as fp:
        dev_data = fp.read().split("\n")
        
    dev_data = [json.loads(d) for d in dev_data]

    train_dataset = NerDataset(train_data, tokenizer, label2id, max_seq_len)
    dev_dataset = NerDataset(dev_data, tokenizer, label2id, max_seq_len)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, num_workers=2)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=dev_batch_size, num_workers=2)
    
    return train_loader, dev_loader