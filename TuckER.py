import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.autograd import Variable
from ResNet50 import resnet50
from PIL import Image

class TuckER(torch.nn.Module):
    def __init__(self, args, num_entitiy, num_relation, device):
        super(TuckER, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entitiy, args.embedding_dim)
        self.emb_rel = torch.nn.Embedding(num_relation, args.embedding_dim)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (args.embedding_dim, args.embedding_dim, args.embedding_dim)), 
                                    dtype=torch.float, device=device, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.4)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(args.embedding_dim)
        

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.emb_e(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.emb_rel(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred