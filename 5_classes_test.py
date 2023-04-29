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
import json
from collections import Counter
from nltk.stem import WordNetLemmatizer
import jamspell
import pickle

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



# these are the hyper parameters that need to be defined
DATA_SIZE = 100000 # amount of data
batch_size = 32 # how many records for each batch
hidden_dim = 256 # size of the hidden dimension of embedding layer
embedding_dim = 32 # size of embedding dim of embedding layer
lr = 0.005 # learing rate 
epochs = 30 # number of rounds used to train

# function to load json data into a list
def load_json(filename):
    testdata = []
    i = 0
    for line in open(filename, 'r'):
        i+=1
        if i == DATA_SIZE // 4:
            break
        testdata.append(json.loads(line))
    return testdata


# Delena added these new stopwords
more_stop_words = {"would", "get", "game", "product", "software", "also", "got", "thing",
                    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "version", "may", "although", "mine", "must", "neither", "became", 
                   "become", "oh", "whereas", "could", "whether", "either", "story", "program",
                   "else", "first", "next", "say", "lot", "go", "paper", "book", "manual", "im",
                   "youll", "youve", "ill", "youre", "theyre", "hes", "shes", "ive", "new", "old", 
                   "card", "computer", "use", "install", "made", "think", "back", "many", "used"
                  }
# add the new stopwords and remove 'not' and 'no'
stop_words.update(more_stop_words)
stop_words.discard("not")
stop_words.discard("no")


df2 = pd.read_csv("data/Comments.csv")
df2 = df2[['Comment', 'Sentiment']]

df2 = pd.DataFrame(df2)
df2.replace(['negative', 'Negative', 'Negative '], "0", inplace=True)
df2.replace(['Neutral','neutral'],  "1", inplace=True)
df2.replace(['Positive', 'positive'], "2", inplace=True)
df2.rename(columns={'Comment':'reviewText', 'Sentiment':'overall'}, inplace=True)
df2['reviewText'] = df2['reviewText'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
df2 = df2[df2['overall'] != "1"]


# preprocessing function
# does various operations on the comments
def data_preprocessing(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text) # Remove HTML from text
    text = corrector.FixFragment(text) #fixes spelling
    text = ''.join([c for c in text if c not in string.punctuation and c not in string.digits])# Remove punctuation
    text = [word for word in text.split() if word not in stop_words] # remove stopwords
    text = [lemmatizer.lemmatize(word) for word in text if len(text)>1] # lemmatize
    text = ' '.join(text)
    return text

# function to pad each sequence
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

# real preprocessing function currently being used
def preprocess3():
    print("Applying preprocessing...")
    
    df2['cleaned_reviews'] = df2['reviewText'].apply(data_preprocessing)
    df3 = pd.read_pickle("data/preprocessing.pkl")
    corpus = [word for text in df3['cleaned_reviews'] for word in text.split()]
    count_words = Counter(corpus)
    sorted_words = count_words.most_common()
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    reviews_int2 = []
    unknowns = []
    for text in df2['cleaned_reviews']:
      
        words = []
        
        for word in text.split():
            try:
                words.append(vocab_to_int[word])
            
            except KeyError:
                if word not in unknowns:
                    unknowns.append(word)
            
        reviews_int2.append(words)
    # print(unknowns)
    df2['Review int'] = reviews_int2

    review_len2 = [len(x) for x in reviews_int2]
    df2['Review len'] = review_len2

    features2 = Padding(reviews_int2, 106).tolist()
    labels2 = []
    for score in df2['overall']:
        labels2.append(score)

    data2 = []
    for i in range(len(features2)):
        f2 = features2[i]
        l2 = labels2[i]
        data2.append((f2, l2)) 

    test_encoded = data2
    print(len(test_encoded))

    return (test_encoded, len(vocab_to_int))


# preprocess the data here
pre_start = time.time()
test_encoded, vocab_size = preprocess3()
print(f"vocab size = {vocab_size}")
pre_end = time.time()
print(f'Preprocessing took : {pre_end-pre_start} s')

# get the training and testing labels here
test_x = np.array([tweet for tweet, label in test_encoded])
test_y = np.array([label for tweet, label in test_encoded])
test_x = test_x.astype(int)
test_y = test_y.astype(int)

# convert the testing data into a PyTorch Dataset
test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).type(torch.LongTensor))
# now we can convert the Dataset into a PyTorch DataLoader
test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)

# this is the defenition of the model we are using
class BiLSTM_SentimentAnalysis(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_units, num_layers, dropout, num_classes) :
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
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

        # We use dropout before the final layer to improve with regularization
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state 
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        embs = self.embedding(x)

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hidden = self.lstm(embs, hidden)



        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1]


        rel = self.relu(out)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds, hidden
    
    def init_hidden(self):
        # return (torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32))
        return (torch.zeros(self.num_layers, batch_size, self.lstm_units), torch.zeros(self.num_layers, batch_size, self.lstm_units))


vocab_size = vocab_size + 1
# define the model like below if you want to load an existing model
model = torch.load("models/rnn_100000_5_classes.pth")
# optimizer to update the weights and biases
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# this commented out part pretty much just tests using a loaded model and exits
model.eval()

# define a loss function
criterion = nn.CrossEntropyLoss()
losses = []



# after training we test on the testing dataset
# notice there is no backward propogation or updating of the weights
batch_acc = []
for batch_idx, batch in enumerate(test_dl):

    input = batch[0]
    target = batch[1]
    optimizer.zero_grad()
    h0, c0 = model.init_hidden()

    with torch.set_grad_enabled(False):
        out, hidden = model(input, (h0, c0))
        _, preds = torch.max(out, 1)
        preds = preds.to("cpu").tolist()
        for k in range(len(preds)):
          if(preds[k] == 1 or preds[k] == 2):
            preds[k] = 0
          elif(preds[k] == 3):
            preds[k] = 1
          elif(preds[k] == 4):
            preds[k] = 2
        batch_acc.append(accuracy_score(preds, target.tolist()))

sum(batch_acc)/len(batch_acc)
print(sum(batch_acc)/len(batch_acc))


