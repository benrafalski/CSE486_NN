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


# data_size  test_acc   train_acc   epochs        
# 10000      0.633      0.968       15
# 20000      0.651      0.966       15
# 50000      0.682      0.950       20
# 100000     0.693      0.925       30 
# 250000     0.713      0.885       50 

DATA_SIZE = 250000

def load_json(filename):
    testdata = []
    i = 0
    for line in open(filename, 'r'):
        i+=1
        if i == DATA_SIZE // 4:
            break
        testdata.append(json.loads(line))
    return testdata


# load_json(filename='data/Software.json')

software = load_json(filename='data/Software.json')
electronic = load_json(filename='data/Electronics.json')
appliance = load_json(filename='data/Appliances.json')
video_games = load_json(filename='data/Video_Games.json')

testdata = software + electronic + appliance + video_games

df = pd.DataFrame(testdata)
df = df[["overall", "reviewText"]]
# testdata = []
# i = 0
# for line in open('data/Software.json', 'r'):
#   i+=1
# #   if i ==10000:
# #       break
#   testdata.append(json.loads(line))

# df = pd.DataFrame(testdata)
# df = df[["overall", "reviewText"]]


# testdata2 = []
# i = 0
# for line in open('data/Appliances.json', 'r'):
#   i+=1
# #   if i ==10000:
# #       break
#   testdata.append(json.loads(line))

# df2 = pd.DataFrame(testdata)
# df2 = df2[["overall", "reviewText"]]

##df.columns = ['overall', 'reviewText']
##print(df.columns)
print(df.head())
df.replace([1.0, 2.0, 3.0], 0, inplace=True)
df.replace(4.0, 1, inplace=True)
df.replace(5.0, 2, inplace=True)
# df.replace([4.0, 5.0, 1, inplace=True)
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))


# df2.replace([1.0, 2.0, 3.0], "0", inplace=True)
# df2.replace(4.0, "1", inplace=True)
# df2.replace(5.0, "2", inplace=True)

# df2['reviewText'] = df2['reviewText'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))


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

def preprocess2():
    print("Applying preprocessing...")
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


    #different type of encoding that might work better after we get new dataset
    #pos_counts = Counter()
    #neg_counts = Counter()
    #total_counts = Counter()

    #labels = []
    #for score in df['overall']:
    #    labels.append(score)
    #listCopy = list(df['cleaned_reviews'])
    #for i in range(len(listCopy)):
    #  if (labels[i] == "2"):
    #    for word in listCopy[i].split():
    #      pos_counts[word] += 1
    #      total_counts[word] += 1
    #  elif (labels[i] == "1"):
    #    for word in listCopy[i].split():
    #      pos_counts[word] += 0
    #      neg_counts[word] += 0
    #      total_counts[word] += 1
    #  elif (labels[i] == "0"):
    #    for word in listCopy[i].split():
    #      neg_counts[word] += 1
    #      total_counts[word] += 1
    #pos_neg_ratios = Counter()

    #for word, count in list(total_counts.most_common()):
    #  if (count >= 10):
    #    ratio = pos_counts[word] / float(neg_counts[word] + 1)
    #    pos_neg_ratios[word] = ratio

    #for word, ratio in pos_neg_ratios.most_common():
    #  if (ratio > 1):
    #    pos_neg_ratios[word] = np.log(ratio)
    #  else:
    #    pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

    #new_words = set()
    #for line in listCopy:
    #  for word in line.split():
    #    if (total_counts[word] > 10):
    #         if (word in pos_neg_ratios.keys()):
    #            if ((pos_neg_ratios[word] >= 0.05) or (pos_neg_ratios[word] <= -0.05)):
    #                new_words.add(word)
    #         else:
    #            new_words.add(word)

    #listNew = list(new_words)
    ## print(listNew[0])
    ## #print(df.head())

    #word2index = {}
    #for i, word in enumerate(listNew):
    #  word2index[word] = i

    #ignoring = set()
    #reviews_int = []
    #for text in df['cleaned_reviews']:
    #    textLine = []
    #    for word in text.split():
    #      try:
    #        textLine.append(word2index[word])
    #      except KeyError:
    #        ignoring.add(word)
    #    r = textLine
    #    reviews_int.append(r)





    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    print("Applying vocab to int...")
    reviews_int = []
    for text in df['cleaned_reviews']:
        r = [vocab_to_int[word] for word in text.split()]
        reviews_int.append(r)

    # print(reviews_int[:1])
    df['Review int'] = reviews_int

    review_len = [len(x) for x in reviews_int]
    df['Review len'] = review_len
    # print(df.head())




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

    # df2['cleaned_reviews'] = df2['reviewText'].apply(data_preprocessing)
    # reviews_int2 = []
    # unknowns = []
    # for text in df2['cleaned_reviews']:
      
    #     words = []
        
    #     for word in text.split():
    #       try:
    #         words.append(vocab_to_int[word])
            
    #       except KeyError:
    #         if word not in unknowns:
    #           unknowns.append(word)
            
    #     reviews_int2.append(words)
    # # print(unknowns)
    # df2['Review int'] = reviews_int2

    # review_len2 = [len(x) for x in reviews_int2]
    # df2['Review len'] = review_len2

    # features2 = Padding(reviews_int2, 106).tolist()
    # labels2 = []
    # for score in df2['overall']:
    #     labels2.append(score)

    # data2 = []
    # for i in range(len(features2)):
    #     f2 = features2[i]
    #     l2 = labels2[i]
    #     data2.append((f2, l2)) 


    random.shuffle(data)
        
        
    train_encoded = data[:int(DATA_SIZE*0.90)]
    test_encoded = data[int(DATA_SIZE*0.90):]
    print(len(train_encoded))
    print(len(test_encoded))

    return (train_encoded, test_encoded, len(vocab_to_int))



pre_start = time.time()
train_encoded, test_encoded, vocab_size = preprocess2()
pre_end = time.time()
print(f'Preprocessing took : {pre_end-pre_start} s')

# sys.exit()

batch_size = 32

train_x = np.array([tweet for tweet, label in train_encoded])
train_y = np.array([label for tweet, label in train_encoded])
test_x = np.array([tweet for tweet, label in test_encoded])
test_y = np.array([label for tweet, label in test_encoded])

train_x = train_x.astype(int)
test_x = test_x.astype(int)
train_y = train_y.astype(int)
test_y = test_y.astype(int)


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



hidden_dim = 256
embedding_dim = 32
# vocab_size = len(word2index)
vocab_size = vocab_size + 1

model = BiLSTM_SentimentAnalysis(vocab_size, embedding_dim, hidden_dim, 64, 3, 0.2)
# model = torch.load("models/rnn.pth")

# 3e-4
lr = 0.005

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 50
losses = []

for e in range(epochs):
    batch_acc = []
    print(f'epoch {e+1}')

    h0, c0 =  model.init_hidden()

    h0 = h0
    c0 = c0

    for batch_idx, batch in enumerate(train_dl):

        if batch_idx % 1000 == 0:
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


print('saving model...')
PATH = f"models/rnn_{DATA_SIZE}.pth"

# state = {
#     'epoch': epochs,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict(),
# }
torch.save(model, PATH)
