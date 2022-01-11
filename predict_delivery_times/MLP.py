import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.embd_ship =  nn.Embedding(27, 10) #2 words in vocab, 5 dimensional embeddings
        self.embd_cat =  nn.Embedding(33, 10)
        self.embd_pack =  nn.Embedding(7, 5)
        self.embd_b2c =  nn.Embedding(2, 5)
        
        self.embd_month =  nn.Embedding(12, 12) #2 words in vocab, 5 dimensional embeddings
        self.embd_seller =  nn.Embedding(101, 20)
        
        self.embd_it_gr =  nn.Embedding(11, 5)
        self.embd_by_gr =  nn.Embedding(11, 5)
        
        self.w1 = nn.Linear(85,256,bias=True)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        self.bn1 = torch.nn.BatchNorm1d(256)
        
        self.w2 = nn.Linear(256,256,bias=True)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        self.bn2 = torch.nn.BatchNorm1d(256)
        
        self.w3 = nn.Linear(256,256,bias=True)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.zeros_(self.w3.bias)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.w4 = nn.Linear(256,256,bias=True)
        nn.init.xavier_uniform_(self.w4.weight)
        nn.init.zeros_(self.w4.bias)
        self.bn4 = torch.nn.BatchNorm1d(256)

        self.w5 = nn.Linear(256,256,bias=True)
        nn.init.xavier_uniform_(self.w5.weight)
        nn.init.zeros_(self.w5.bias)
        self.bn5 = torch.nn.BatchNorm1d(256)

        
        self.w7 = nn.Linear(256,2,bias=True)
        nn.init.xavier_uniform_(self.w7.weight)
        nn.init.zeros_(self.w7.bias)


    def forward(self, x):
        
        x_num = x[:,:13]
        x_one_hot = x[:,13:].int()
        #print(x)
        #x = torch.rand(len(x),18)
        #print(x_one_hot[:,5])


        ship = self.embd_ship(x_one_hot[:,0])
        cat = self.embd_cat(x_one_hot[:,1])
        pack = self.embd_pack(x_one_hot[:,2])
        b2c = self.embd_b2c(x_one_hot[:,3])
        
        month = self.embd_month(x_one_hot[:,4])
        seller = self.embd_seller(x_one_hot[:,5])
        
        #print(x_one_hot[:,6])
        it_gr = self.embd_it_gr(x_one_hot[:,6])
        by_gr = self.embd_by_gr(x_one_hot[:,7])
        
        x = torch.cat((x_num, ship, cat, pack, b2c, month,seller,it_gr,by_gr ), 1)# , seller, it_gr, by_gr), 1)
        
        x = self.bn1(torch.tanh(self.w1(x)))
        
        x = self.bn2(torch.tanh(self.w2(x)))
        x = self.bn3(torch.tanh(self.w3(x)))
        
        x = self.bn4(torch.tanh(self.w4(x)))
        x = self.bn5(torch.tanh(self.w5(x)))
        '''
        x = self.bn6(F.relu(self.w6(x)))
        '''
        x = self.w7(x)
        return x




class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        
        self.embd_ship =  nn.Embedding(27, 10) #2 words in vocab, 5 dimensional embeddings
        self.embd_cat =  nn.Embedding(33, 10)
        self.embd_pack =  nn.Embedding(7, 5)
        self.embd_b2c =  nn.Embedding(2, 5)
        
        self.embd_month =  nn.Embedding(12, 12) #2 words in vocab, 5 dimensional embeddings
        self.embd_seller =  nn.Embedding(101, 20)
        
        self.embd_it_gr =  nn.Embedding(11, 5)
        self.embd_by_gr =  nn.Embedding(11, 5)
        
        self.w1 = nn.Linear(85,256,bias=True)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        self.bn1 = torch.nn.BatchNorm1d(256)
        
        self.w2 = nn.Linear(256,256,bias=True)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        self.bn2 = torch.nn.BatchNorm1d(256)
        
        self.w3 = nn.Linear(256,256,bias=True)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.zeros_(self.w3.bias)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.w4 = nn.Linear(256,256,bias=True)
        nn.init.xavier_uniform_(self.w4.weight)
        nn.init.zeros_(self.w4.bias)
        self.bn4 = torch.nn.BatchNorm1d(256)

        self.w5 = nn.Linear(256,256,bias=True)
        nn.init.xavier_uniform_(self.w5.weight)
        nn.init.zeros_(self.w5.bias)
        self.bn5 = torch.nn.BatchNorm1d(256)

        
        self.w7 = nn.Linear(256,3,bias=True)
        nn.init.xavier_uniform_(self.w7.weight)
        nn.init.zeros_(self.w7.bias)


    def forward(self, x):
        
        x_num = x[:,:13]
        x_one_hot = x[:,13:].int()
        #print(x)
        #x = torch.rand(len(x),18)
        #print(x_one_hot[:,5])


        ship = self.embd_ship(x_one_hot[:,0])
        cat = self.embd_cat(x_one_hot[:,1])
        pack = self.embd_pack(x_one_hot[:,2])
        b2c = self.embd_b2c(x_one_hot[:,3])
        
        month = self.embd_month(x_one_hot[:,4])
        seller = self.embd_seller(x_one_hot[:,5])
        
        #print(x_one_hot[:,6])
        it_gr = self.embd_it_gr(x_one_hot[:,6])
        by_gr = self.embd_by_gr(x_one_hot[:,7])
        
        x = torch.cat((x_num, ship, cat, pack, b2c, month,seller,it_gr,by_gr ), 1)# , seller, it_gr, by_gr), 1)
        
        x = self.bn1(torch.tanh(self.w1(x)))
        
        x = self.bn2(torch.tanh(self.w2(x)))
        x = self.bn3(torch.tanh(self.w3(x)))
        
        x = self.bn4(torch.tanh(self.w4(x)))
        x = self.bn5(torch.tanh(self.w5(x)))
        '''
        x = self.bn6(F.relu(self.w6(x)))
        '''
        x = self.w7(x)
        return x


