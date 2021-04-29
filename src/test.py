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
import os
MAX_LEN = 50
isGPU = torch.cuda.is_available()


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

def load_test(path='./test.csv', shuffle=True):
    test_csv = pd.read_csv(path)
    test_id = test_csv["id"].values.tolist()
    test_data = list(zip(test_csv["text"], test_csv["keyword"]))
    return test_data, test_id
class dataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        temp = []
        self.keywords = []
        for i in range(len(data)):
            temp.append(data[i][0])
            self.keywords.append(embedding_dict.get(data[i][1], np.zeros(100,dtype='float32')))
        self.tokens,self.masks,self.segs = encode_tweets(tokenizer, temp, max_len)
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        token = self.tokens[idx]
        mask = self.masks[idx]
        seg = self.segs[idx]
        keyword = self.keywords[idx]
        return token, mask, seg, keyword

class NN(nn.Module):
    def __init__(self, MAX_LEN):
        super(NN, self).__init__()
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.layer2 = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 , 100 , bias = True), nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.Linear(100 , 1 , bias = True))
    def forward(self, token, mask, seg, keyword):
        x = self.bertweet(token.to(torch.int64), attention_mask=mask.to(torch.int64), token_type_ids=seg.to(torch.int64))
        x = self.layer2(x[0][:,0,:])
        x = x + keyword.to(torch.float32)

        x = self.layer3(x).flatten()

        
        return x
test_data, test_id = load_test()
test_dataset = dataset(test_data, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = NN(50).cuda()

model.load_state_dict(torch.load('best5.bin'))
model.eval()

predicts = []
with torch.no_grad():
    for idx, (token, mask, seg, keyword) in enumerate(test_loader):
        if isGPU == True:
            token = token.cuda()
            mask = mask.cuda()
            seg = seg.cuda()
            keyword = keyword.cuda()
        output = model(token, mask, seg, keyword).to(torch.float32)
        output = torch.sigmoid(output)
        #predict = torch.argmax(output, dim=1).flatten().cpu().numpy().tolist()
        predict = np.array([1 if i>0.5 else 0 for i in output]).tolist()
        predicts += predict

with open('output.csv', 'w') as f:
    f.write('id,target\n')  
    for i in range(len(predicts)):
        
        f.write('{:d},{:d}\n'.format(test_id[i], predicts[i]))

