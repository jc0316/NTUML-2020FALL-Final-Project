import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer ,AdamW, get_linear_schedule_with_warmup, BertweetTokenizer
import tensorflow as tf
import re
import string  

def encode_tweets(tokenizer, tweets, max_len):
    nb_tweets = len(tweets)
    tokens = np.ones((nb_tweets,max_len),dtype='int32')
    masks = np.zeros((nb_tweets,max_len),dtype='int32')
    segs = np.zeros((nb_tweets,max_len),dtype='int32')

    for k in range(nb_tweets):        
        # INPUT_IDS
        tweet = tweets[k]
        enc = tokenizer.encode(tweet)                   
        if len(enc) < max_len-2:
            tokens[k,:len(enc)+2] = [0] + enc + [2]
            masks[k,:len(enc)+2] = 1
        else:
            tokens[k,:max_len] = [0] + enc[:max_len-2] + [2]
            masks[k,:max_len] = 1 
    return tokens,masks,segs

embedding_dict = {}

with open('./glove.6B.100d.txt','r') as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors
        
glove.close()

tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base")



def load_data(path='./train.csv', shuffle=True):
    train_csv = pd.read_csv(path)
    train_labels = train_csv["target"].values.tolist()
    train_datas = list(zip(train_csv["text"], train_csv["keyword"] ,train_labels))
    if shuffle == True:
        random.seed(432)
        random.shuffle(train_datas)

    train_set = train_datas[:len(train_labels)//10*9]
    valid_set = train_datas[len(train_labels)//10*9:]
    return train_set, valid_set
class dataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.labels = []
        self.keywords = []
        temp = []
        for i in range(len(data)):
            temp.append(data[i][0])
            self.keywords.append(embedding_dict.get(data[i][1], np.zeros(100,dtype='float32')))
            self.labels.append(data[i][2])
        self.tokens,self.masks,self.segs = encode_tweets(tokenizer, temp, max_len)
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        token = self.tokens[idx]
        mask = self.masks[idx]
        seg = self.segs[idx]
        keyword = self.keywords[idx]
        label = self.labels[idx]
        return token, mask, seg, keyword, label

MAX_LEN = 50
train_set, valid_set = load_data()
train_dataset = dataset(train_set, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

valid_dataset = dataset(valid_set, tokenizer, MAX_LEN)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# model.py

class NN(nn.Module):
    def __init__(self, MAX_LEN):
        super(NN, self).__init__()
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.layer1 = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 , 100 , bias = True), nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(100 , 1 , bias = True))
    def forward(self, token, mask, seg, keyword):
        x = self.bertweet(token.to(torch.int64), attention_mask=mask.to(torch.int64), token_type_ids=seg.to(torch.int64))
        x = self.layer1(x[0][:,0,:])
        x = x + keyword.to(torch.float32)

        x = self.layer2(x).flatten()

        
        return x

isGPU = torch.cuda.is_available()
print ('PyTorch GPU device is available: {}'.format(isGPU))

MAX_LEN = 50
model = NN(MAX_LEN)
if isGPU is True:
    model.cuda()
opt = AdamW(model.parameters(),lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
#loss_fn = nn.BCELoss()
pos_weight = (torch.ones([1])*2).cuda()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
epochs = 100
best_loss = 21379
for epoch in range(epochs):
    model.train()
    train_loss = []
    train_acc = []
    for idx, (token, mask, seg, keyword, label) in enumerate(train_loader):
        if isGPU == True:
            token = token.cuda()
            mask = mask.cuda()
            seg = seg.cuda()
            label = label.cuda()
            keyword = keyword.cuda()
        opt.zero_grad()
        output = model(token, mask, seg, keyword).to(torch.float32)
        loss = loss_fn(output, label.to(torch.float32))
        #predict = torch.argmax(output, dim=1).flatten()
        loss.backward()
        opt.step()
        output = torch.sigmoid(output)
        predict = np.array([1 if i>0.5 else 0 for i in output])
        acc = np.mean((label.cpu().numpy() == predict))
        train_acc.append(acc)
        train_loss.append(loss.item())
    print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)), end = "   ")
    model.eval()

    with torch.no_grad():
        valid_loss = []
        valid_acc = []
        for idx, (token, mask, seg, keyword, label) in enumerate(valid_loader):
            if isGPU == True:
                token = token.cuda()
                mask = mask.cuda()
                seg = seg.cuda()
                label = label.cuda()
                keyword = keyword.cuda()
            output = model(token, mask, seg, keyword).to(torch.float32)
            
            loss = loss_fn(output, label.to(torch.float32))
            #predict = torch.argmax(output, dim=1).flatten()
            output = torch.sigmoid(output)
            predict = np.array([1 if i>0.5 else 0 for i in output])
            acc = np.mean((label.cpu().numpy() == predict))
            valid_loss.append(loss.item())
            valid_acc.append(acc)
        print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))

    torch.save(model.state_dict(), 'best'+str(epoch)+'.bin')
    scheduler.step(np.mean(valid_loss))