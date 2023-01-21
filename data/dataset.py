import pandas as pd
from sklearn import preprocessing
import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader

class TweetsDataset():
    def __init__(self, train_size=23480):

        self.device = 'cpu'

        file_out = pd.read_csv('data/Tweets.csv')
        file_out = file_out.sample(frac = 1)
        x_train = file_out.iloc[0:(train_size+1), 1].values
        y_train = file_out.iloc[0:(train_size+1), 3].values

        x_test = file_out.iloc[(train_size+1):27480, 1].values
        y_test = file_out.iloc[(train_size+1):27480, 3].values

        label_encoder = preprocessing.LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)


        self.train_iter = list(zip(y_train, x_train))
        self.test_iter = list(zip(y_test, x_test))

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self.yield_tokens(), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: int(x)

    def train_test_split(self):
        return self.train_iter, self.test_iter

    def to_device(self, device):
        self.device = device

    def yield_tokens(self):
        for _, text in self.train_iter:
            yield self.tokenizer(text)

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)

    def make_dataloaders(self, batch_size=10):
        train_dataset = to_map_style_dataset(self.train_iter)
        test_dataset = to_map_style_dataset(self.test_iter)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, collate_fn=self.collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=True, collate_fn=self.collate_batch)

        return train_dataloader, test_dataloader

    

