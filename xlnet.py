import pandas as pd
import json
from transformers import XLNetTokenizer,XLNetForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import AdamW
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler
import torch.nn as nn
import numpy as np


DATA_SIZE = 1000
filename = 'data/Software.json' 

testdata = []
i = 0
for line in open(filename, 'r'):
    i+=1
    if i == DATA_SIZE:
        break
    testdata.append(json.loads(line))

df = pd.DataFrame(testdata)
df = df[["overall", "reviewText"]]


sentences  = []
for sentence in df['reviewText']:
  sentence = sentence+"[SEP] [CLS]"
  sentences.append(sentence)


tokenizer  = XLNetTokenizer.from_pretrained('xlnet-base-cased',do_lower_case=True)
tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]

ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

labels = df['overall'].values

max1 = len(ids[0])
for i in ids:
  if(len(i)>max1):
    max1=len(i)
print(max1)
MAX_LEN = max1

input_ids2 = pad_sequences(ids,maxlen=MAX_LEN,dtype="long",truncating="post",padding="post")

xtrain,xtest,ytrain,ytest = train_test_split(input_ids2,labels,test_size=0.15)


Xtrain = torch.tensor(xtrain)
Ytrain = torch.tensor(ytrain)
Xtest = torch.tensor(xtest)
Ytest = torch.tensor(ytest)

batch_size = 32

train_data = TensorDataset(Xtrain,Ytrain)
test_data = TensorDataset(Xtest,Ytest)
loader = DataLoader(train_data,batch_size=batch_size)
test_loader = DataLoader(test_data,batch_size=batch_size)

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=5)

optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)# We pass model parameters
criterion = nn.CrossEntropyLoss()

def flat_accuracy(preds,labels):  # A function to predict Accuracy
  correct=0
  for i in range(0,len(labels)):
    if(preds[i]==labels[i]):
      correct+=1
  return (correct/len(labels))*100


no_train = 0
epochs = 3
for epoch in range(epochs):
    model.train()
    loss1 = []
    steps = 0
    train_loss = []
    l = []
    b = 0
    for inputs,labels1 in loader :
        b+=1
        print(f'Batch number {b}/{len(loader)}')
        inputs
        labels1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[0],labels1)
        logits = outputs[1]
        #ll=outp(loss)
        [train_loss.append(p.item()) for p in torch.argmax(outputs[0],axis=1).flatten() ]#our predicted 
        [l.append(z.item()) for z in labels1]# real labels
        loss.backward()
        optimizer.step()
        loss1.append(loss.item())
        no_train += inputs.size(0)
        steps += 1
    print("Current Loss is : {} Step is : {} number of Example : {} Accuracy : {}".format(loss.item(),epoch,no_train,flat_accuracy(train_loss,l)))


