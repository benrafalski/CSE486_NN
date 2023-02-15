from ast import Try
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
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
nltk.download('punkt')



#train_path = "data/Tweets.csv"
#train_df = pd.read_csv(train_path)
## train_df = train_df.sample(frac=1)

#train_df = train_df.drop(columns={"textID", "selected_text"})


#train_df, test_df = train_test_split(train_df, test_size=0.15)

#remove_pos = 7312 - 6590
#remove_neut = 9456 - 6590


#neg_df = train_df[train_df["sentiment"] == "negative"] 

#pos_df = train_df[train_df["sentiment"] == "positive"]
#neut_df = train_df[train_df["sentiment"] == "neutral"]

#pos_drop_indices = np.random.choice(pos_df.index, remove_pos, replace=False)
#neut_drop_indices = np.random.choice(neut_df.index, remove_neut, replace=False)

#pos_undersampled = pos_df.drop(pos_drop_indices)
#neut_undersampled = neut_df.drop(neut_drop_indices)

#balanced_train_df = pd.concat([neg_df, pos_undersampled, neut_undersampled])



## print(balanced_train_df["sentiment"].value_counts())

#train_clean_df, test_clean_df = train_test_split(balanced_train_df, test_size=0.15)

#train_set = list(train_clean_df.to_records(index=False))
#test_set = list(test_clean_df.to_records(index=False))

## print(train_set[:10])

#def remove_links_mentions(tweet):
#    link_re_pattern = "https?:\/\/t.co/[\w]+"
#    mention_re_pattern = "@\w+"
#    tweet = re.sub(link_re_pattern, "", tweet)
#    tweet = re.sub(mention_re_pattern, "", tweet)
#    return tweet.lower()

#train_set = [(label, word_tokenize(remove_links_mentions(tweet))) for tweet, label in train_set]
#test_set = [(label, word_tokenize(remove_links_mentions(tweet))) for tweet, label in test_set]

## print(test_set[:3])


testdata = []
for line in open('data/Software.json', 'r'):
  testdata.append(json.loads(line))

df = pd.DataFrame(testdata)
df = df[["overall", "reviewText"]]

##df.columns = ['overall', 'reviewText']
##print(df.columns)
print(df.head())
df.replace([1.0, 2.0, 3.0], 0, inplace=True)
df.replace(4.0, 1, inplace=True)
df.replace(5.0, 2, inplace=True)
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

noPunct = []
for line in df["reviewText"]:
    noPunct.append(line.translate({ord(c): None for c in ".!,:;'\/?~+_\"[]{}"}))
df["reviewText"] = noPunct

train_set, test_set = train_test_split(df.values.tolist(), test_size=0.15, random_state=64)
train_df, valid_df = train_test_split(df.values.tolist(), test_size=0.15, random_state=64)



index2word = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


for ds in [train_set, test_set]:
    for label, tweet in ds:
        for token in tweet:
            if token not in index2word:
                index2word.append(token)


word2index = {token: idx for idx, token in enumerate(index2word)}

seq_length = 106

def label_map(label):
    if label == "0":
        return 0
    elif label == "1":
        return 1
    else: #positive
        return 2


#tokenizer = get_tokenizer('spacy')
#counter = Counter()
#for (label, line) in train_set:
#    counter.update(tokenizer(line))
#vocabs = vocab(counter, min_freq=10, specials=["<unk>", "<sos>", "<eos>", "<pad>"])
#word2index = vocabs.get_stoi()

def encode_and_pad(tweet, length):
    sos = [word2index["<SOS>"]]
    eos = [word2index["<EOS>"]]
    pad = [word2index["<PAD>"]]

    if len(tweet) < length - 2: # -2 for SOS and EOS
        n_pads = length - 2 - len(tweet)
        #encoded = [word2index[w] for w in tweet]

        encoded = []
        
        for w in tweet:
            try:
                encoded.append(word2index[w])
            
            except KeyError:
                if (w != " "):
                    encoded.append(word2index["<UNK>"])
                    print("word not recognized: " + w)
        sos.extend(encoded)
        return sos +  eos + pad * n_pads 
    else: # tweet is longer than possible; truncating
        #encoded = [word2index[w] for w in tweet]
        encoded = []
        
        for w in tweet:
            try:
                encoded.append(word2index[w])
            
            except KeyError:
                if (w != " "):
                    encoded.append(word2index["<UNK>"])
                    print("word not recognized: " + w)
        truncated = encoded[:length - 2]
        sos.extend(truncated)
        return sos   + eos





train_encoded = [(encode_and_pad(tweet, seq_length), label) for label, tweet in train_set]

test_encoded = [(encode_and_pad(tweet, seq_length), label) for label, tweet in test_set]

batch_size = 128

train_x = np.array([tweet for tweet, label in train_encoded])
train_y = np.array([label for tweet, label in train_encoded])
test_x = np.array([tweet for tweet, label in test_encoded])
test_y = np.array([label for tweet, label in test_encoded])


train_x = train_x.astype(float)
test_x = test_x.astype(float)


train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).type(torch.LongTensor))
test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).type(torch.LongTensor))

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)


class BiLSTM_SentimentAnalysis(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_units, num_layers, dropout) :
        super().__init__()

        self.lstm_units = lstm_units
        self.num_layers = num_layers
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

        rel = self.relu(out)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds, hidden
    
    def init_hidden(self):
        return (torch.zeros(self.num_layers, batch_size, self.lstm_units), torch.zeros(self.num_layers, batch_size, self.lstm_units))

model = BiLSTM_SentimentAnalysis(len(word2index), 64, 32, 32, 2, 0.2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

epochs = 30
losses = []

for e in range(epochs):
    batch_acc = []
    print(f'epoch {e+1}')

    h0, c0 =  model.init_hidden()

    h0 = h0
    c0 = c0

    for batch_idx, batch in enumerate(train_dl):

        if batch_idx % (len(train_dl)//100) == 0:
            print(f'\tstarting batch number {batch_idx} of {len(train_dl)}')

        input = batch[0]
        target = batch[1]

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out, hidden = model(input, (h0, c0))
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(out, 1)
            preds = preds.to("cpu").tolist()
            batch_acc.append(accuracy_score(preds, target.tolist()))

        
    print(f'epoch accracy = {sum(batch_acc)/len(batch_acc)}')



    losses.append(loss.item())

batch_acc = []
for batch_idx, batch in enumerate(test_dl):

    input = batch[0]
    target = batch[1]

    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        out, hidden = model(input, (h0, c0))
        _, preds = torch.max(out, 1)
        preds = preds.to("cpu").tolist()
        batch_acc.append(accuracy_score(preds, target.tolist()))

sum(batch_acc)/len(batch_acc)

print(sum(batch_acc)/len(batch_acc))
