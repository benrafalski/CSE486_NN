from ast import Try
import random
import sys
import time
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
#import matplotlib.pyplot as plt
import json
from collections import Counter
# from torchtext.vocab import vocab
# from torchtext.data.utils import get_tokenizer
from nltk.stem import WordNetLemmatizer
import jamspell

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string

corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('data/en.bin')
lemmatizer = WordNetLemmatizer()


batch_size = 1
lr = 0.005

class BiLSTM_SentimentAnalysis(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_units, num_layers, dropout) :
        super().__init__()

        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, lstm_units, num_layers=num_layers, bidirectional=False, batch_first=True)
        
        self.fc1 = nn.Linear(lstm_units, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)
        self.relu = nn.ReLU()

        # We use dropout before the final layer to improve with regularization
        self.dropout = nn.Dropout(dropout)

        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        # self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, 3)
        

    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state 
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        embs = self.embedding(x)

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hidden = self.lstm(embs, hidden)

        # Dropout is applied to the output and fed to the FC layer
        #out = self.dropout(out)
        #out = self.fc(out)

        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1]

        # rel = self.relu(out)
        # dense1 = self.fc1(rel)
        # drop = self.dropout(dense1)
        # preds = self.fc2(drop)

        # out = self.dropout(out)
        # out = torch.nn.functional.relu(self.fc1(out))
        # out = torch.nn.functional.softmax(self.fc2(out), dim=1)        
        # return out, hidden

        rel = self.relu(out)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds, hidden
    
    def init_hidden(self):
        # return (torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32))
        return (torch.zeros(self.num_layers, batch_size, self.lstm_units), torch.zeros(self.num_layers, batch_size, self.lstm_units))


def data_preprocessing(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text) # Remove HTML from text
    text = corrector.FixFragment(text) #fixes spelling
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = [word for word in text.split() if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text if len(text)>1]
    text = ' '.join(text)
    return text

def Padding(review_int, seq_len):
    '''
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(review_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)
            
    return features


test_case = "Today is a good day, I am very happy"
test_case = data_preprocessing(test_case)
corpus = [word for word in test_case.split()]
count_words = Counter(corpus)
sorted_words = count_words.most_common()
vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

reviews_int = []
r = [vocab_to_int[word] for word in test_case.split()]
reviews_int.append(r)
features = Padding(reviews_int, 106).tolist()
# bad = 0.0, neutral = 1.0, good = 2.0
labels = [2.0]
data = []
for i in range(len(features)):
    f = features[i]
    l = labels[i]
    data.append((f, l))

# print(data)
test_x = np.array([tweet for tweet, label in data])
test_y = np.array([label for tweet, label in data])

test_x = test_x.astype(int)
# train_y = train_y.astype(int)
test_y = test_y.astype(int)
test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).type(torch.LongTensor))
test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)


model = torch.load("models/rnn_10000.pth")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for batch_idx, batch in enumerate(test_dl):
    input = batch[0]
    target = batch[1]

    # need to save the model.hidden state...
    h0, c0 = model.init_hidden()

    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        out, hidden = model(input, (h0, c0))
        _, preds = torch.max(out, 1)
        preds = preds.to("cpu").tolist()

        print(f'Preds {preds}')
        print(f'Target {target.tolist()}')

        print("Correct" if accuracy_score(preds, target.tolist()) == 1.0 else "Wrong")

        # batch_acc.append(accuracy_score(preds, target.tolist()))