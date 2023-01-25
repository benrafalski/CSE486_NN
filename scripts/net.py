import torch.nn as nn
import torch.nn.functional as F
import time
import torch

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_class):
        super(TextClassificationModel, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # # self.flat = nn.Flatten()
        
        # # self.conv1 = nn.Conv2d(embed_dim, embed_dim, 5)
        # # self.conv2 = nn.Conv2d(embed_dim, embed_dim, 5)
        # self.fc1 = nn.Linear(embed_dim, embed_dim)
        # self.fc2 = nn.Linear(embed_dim, embed_dim)
        # self.fc3 = nn.Linear(embed_dim, num_class)
        # self.init_weights()

        self.embedding = nn.Embedding(vocab_size,64)
        self.lstm = nn.LSTM( input_size=64,hidden_size=64,num_layers=3,batch_first=True)
        self.fc = nn.Linear(64, num_class)


    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()


    def forward(self, text, offsets):
        # x = self.embedding(text, offsets)
        # # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1) 
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)


        res = self.embedding(text)
        h_0 = torch.zeros(3,len(text),64)
        c_0 = torch.zeros(3,len(text),64)
        output, (h_n,c_n) = self.lstm(res,(h_0,c_0))
        output = output.reshape(len(text),-1)
        return self.fc(output)

        

    def fit(self, dataloader, epoch, optimizer):
        criterion = torch.nn.CrossEntropyLoss()
    
        self.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = self(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(self, dataloader):
        self.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = self(text, offsets)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count