from ast import Try
import sys
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
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string



testdata = []
i = 0
for line in open('data/Software.json', 'r'):
  i+=1
  if i ==10000:
      break
  testdata.append(json.loads(line))

df = pd.DataFrame(testdata)
df = df[["overall", "reviewText"]]

##df.columns = ['overall', 'reviewText']
##print(df.columns)
print(df.head())
df.replace([1.0, 2.0, 3.0], 0, inplace=True)
df.replace(4.0, 1, inplace=True)
df.replace(5.0, 2, inplace=True)
# df.replace([4.0, 5.0, 1, inplace=True)
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))



def label_map(label):
    if label == "0":
        return 0
    elif label == "1":
        return 1
    else: #positive
        return 2

index2word = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
word2index = {token: idx for idx, token in enumerate(index2word)}

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
        # print(sos +  eos + pad * n_pads )
        # sys.exit()
        return pad * n_pads + sos +  eos 
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



def preprocess1():
    noPunct = []
    for line in df["reviewText"]:
        noPunct.append(line.translate({ord(c): None for c in ".!,:;'\/?~+_\"[]{}"}))
    df["reviewText"] = noPunct

    train_set, test_set = train_test_split(df.values.tolist(), test_size=0.15, random_state=64)
    train_df, valid_df = train_test_split(df.values.tolist(), test_size=0.15, random_state=64)



    


    for ds in [train_set, test_set]:
        for label, tweet in ds:
            for token in tweet:
                if token not in index2word:
                    index2word.append(token)


    

    seq_length = 106
    train_encoded = [(encode_and_pad(tweet, seq_length), label) for label, tweet in train_set]

    test_encoded = [(encode_and_pad(tweet, seq_length), label) for label, tweet in test_set]

    return (train_encoded, test_encoded)


def data_preprocessing(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text) # Remove HTML from text
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = [word for word in text.split() if word not in stop_words]
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

def preprocess2():
    df['cleaned_reviews'] = df['reviewText'].apply(data_preprocessing)
    corpus = [word for text in df['cleaned_reviews'] for word in text.split()]
    count_words = Counter(corpus)
    sorted_words = count_words.most_common()


    keys = []
    values = []
    for key, value in sorted_words[:20]:
        keys.append(key)
        values.append(value)

    # plt.plot()
    # plt.bar(keys, values)
    # plt.theme('matrix')
    # # plt.plotsize(100, 15)
    # plt.show()

    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    reviews_int = []
    for text in df['cleaned_reviews']:
        r = [vocab_to_int[word] for word in text.split()]
        reviews_int.append(r)

    print(reviews_int[:1])
    df['Review int'] = reviews_int

    review_len = [len(x) for x in reviews_int]
    df['Review len'] = review_len
    print(df.head())




    features = Padding(reviews_int, 106).tolist()
    labels = []
    for score in df['overall']:
        labels.append(score)

    # print(features[0])
    # print(labels)


    data = []
    for i in range(len(features)):
        f = features[i]
        l = labels[i]
        data.append((f, l))

    # data = zip(features, labels)

    # 
    # print(len(data))
    # sys.exit()

    train_encoded = data[:9000]
    test_encoded = data[9000:]

    return (train_encoded, test_encoded, len(vocab_to_int))




train_encoded, test_encoded, vocab_size = preprocess2()


# sys.exit()

batch_size = 32

train_x = np.array([tweet for tweet, label in train_encoded])
train_y = np.array([label for tweet, label in train_encoded])
test_x = np.array([tweet for tweet, label in test_encoded])
test_y = np.array([label for tweet, label in test_encoded])


train_x = train_x.astype(int)
test_x = test_x.astype(int)


train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).type(torch.LongTensor))
test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).type(torch.LongTensor))

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)

print(len(train_dl))


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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=False, batch_first=True)
        

        # We use dropout before the final layer to improve with regularization
        self.dropout = nn.Dropout(dropout)

        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)
        

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

        out = self.dropout(out)
        out = torch.nn.functional.relu(self.fc1(out))
        out = torch.nn.functional.softmax(self.fc2(out), dim=1)        
        return out, hidden
    
    def init_hidden(self):
        # return (torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32))
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim), torch.zeros(self.num_layers, batch_size, self.hidden_dim))



hidden_dim = 256
embedding_dim = 64
# vocab_size = len(word2index)
vocab_size = vocab_size + 1

model = BiLSTM_SentimentAnalysis(vocab_size, embedding_dim, hidden_dim, 64, 3, 0.2)

# 3e-4
lr = 0.005

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 100
losses = []

for e in range(epochs):
    batch_acc = []
    print(f'epoch {e+1}')

    h0, c0 =  model.init_hidden()

    h0 = h0
    c0 = c0

    for batch_idx, batch in enumerate(train_dl):

        if batch_idx % 100 == 0:
            print(f'\tstarting batch number {batch_idx} of {len(train_dl)}')

        input = batch[0]
        target = batch[1]

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out, hidden = model(input, (h0, c0))
            loss = criterion(out, target)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
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
