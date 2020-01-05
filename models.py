import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

"""
Hierarchical ConvNet
"""
class ConvNetEncoder1(nn.Module):
    def __init__(self, config):
        super(ConvNetEncoder1, self).__init__()

        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        #self.weight = torch.randn((2*self.enc_lstm_dim, self.word_emb_dim, 6), requires_grad = True) 
        #with torch.no_grad():
        #    self.weight[:,:,1] = 0            

        self.convnet1 = nn.Sequential(
                       
            nn.Conv1d(self.word_emb_dim, 2*self.enc_lstm_dim, kernel_size=7,
                      stride=1, padding=1),

            nn.ReLU(inplace=True),
            )

        #with torch.no_grad():
        #self.convnet1[0].weight = nn.Parameter(self.weight)
        #f = open("guru99.txt","w+")
        #f.write(weight)
        #f.close()
        #torch.save(self.weight,'guru.pt')

        self.convnet2 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=7,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

        self.convnet3 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=7,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

        self.convnet4 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=7,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0,1).transpose(1,2).contiguous()
        # batch, nhid, seqlen)

        #print (sent.shape)

        sent = self.convnet1(sent)
        u1 = torch.max(sent,2)[0]

        sent = self.convnet2(sent)
        u2 = torch.max(sent, 2)[0]

        sent = self.convnet3(sent)
        u3 = torch.max(sent, 2)[0]

        sent = self.convnet4(sent)
        u4 = torch.max(sent, 2)[0]

        #print("weight",self.weight)

        emb = torch.cat((u1, u2, u3, u4), 1)

        return emb

class ConvNetEncoder2(nn.Module):
    def __init__(self, config):
        super(ConvNetEncoder2, self).__init__()

        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        #self.weight = torch.randn((2*self.enc_lstm_dim, self.word_emb_dim, 6), requires_grad = True)
        #with torch.no_grad():
        #    self.weight[:,:,2] = 0

        self.convnet1 = nn.Sequential(

            nn.Conv1d(self.word_emb_dim, 2*self.enc_lstm_dim, kernel_size=4,
                      stride=1, padding=1),

            nn.ReLU(inplace=True),
            )

        #with torch.no_grad():
        #self.convnet1[0].weight = nn.Parameter(self.weight)
        #f = open("guru99.txt","w+")
        #f.write(weight)
        #f.close()
        #torch.save(self.weight,'guru.pt')

        self.convnet2 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=4,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

        self.convnet3 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=4,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

        self.convnet4 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=4,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0,1).transpose(1,2).contiguous()
        # batch, nhid, seqlen)

        #print (sent.shape)

        sent = self.convnet1(sent)
        u1 = torch.max(sent,2)[0]

        sent = self.convnet2(sent)
        u2 = torch.max(sent, 2)[0]

        sent = self.convnet3(sent)
        u3 = torch.max(sent, 2)[0]

        sent = self.convnet4(sent)
        u4 = torch.max(sent, 2)[0]

        #print("weight",self.weight)

        emb = torch.cat((u1, u2, u3, u4), 1)

        return emb

class ConvNetEncoder3(nn.Module):
    def __init__(self, config):
        super(ConvNetEncoder3, self).__init__()

        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        #self.weight = torch.randn((2*self.enc_lstm_dim, self.word_emb_dim, 6), requires_grad = True)
        #with torch.no_grad():
        #    self.weight[:,:,3] = 0

        self.convnet1 = nn.Sequential(

            nn.Conv1d(self.word_emb_dim, 2*self.enc_lstm_dim, kernel_size=5,
                      stride=1, padding=1),

            nn.ReLU(inplace=True),
            )

        #with torch.no_grad():
        #self.convnet1[0].weight = nn.Parameter(self.weight)
        #f = open("guru99.txt","w+")
        #f.write(weight)
        #f.close()
        #torch.save(self.weight,'guru.pt')

        self.convnet2 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=5,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

        self.convnet3 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=5,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

        self.convnet4 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=5,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0,1).transpose(1,2).contiguous()
        # batch, nhid, seqlen)

        #print (sent.shape)

        sent = self.convnet1(sent)
        u1 = torch.max(sent,2)[0]

        sent = self.convnet2(sent)
        u2 = torch.max(sent, 2)[0]

        sent = self.convnet3(sent)
        u3 = torch.max(sent, 2)[0]

        sent = self.convnet4(sent)
        u4 = torch.max(sent, 2)[0]

        #print("weight",self.weight)

        emb = torch.cat((u1, u2, u3, u4), 1)

        return emb

class ConvNetEncoder4(nn.Module):
    def __init__(self, config):
        super(ConvNetEncoder4, self).__init__()

        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        #self.weight = torch.randn((2*self.enc_lstm_dim, self.word_emb_dim, 6), requires_grad = True)
        #with torch.no_grad():
        #    self.weight[:,:,3] = 0

        self.convnet1 = nn.Sequential(

            nn.Conv1d(self.word_emb_dim, 80, kernel_size=6,
                     stride=1, padding=1),

            nn.ReLU(inplace=True),
            )

        #with torch.no_grad():
        #    self.convnet1[0].weight = nn.Parameter(self.weight)

        #print(self.convnet1[0].weight.data)
        #f = open("guru99.txt","w+")
        #f.write(weight)
        #f.close()
        #torch.save(self.weight,'guru.pt')

        self.convnet2 = nn.Sequential(
            nn.Conv1d(80, 80, kernel_size=6,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

        self.convnet3 = nn.Sequential(
            nn.Conv1d(80, 80, kernel_size=6,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

        self.convnet4 = nn.Sequential(
            nn.Conv1d(80, 80, kernel_size=6,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0,1).transpose(1,2).contiguous()
        # batch, nhid, seqlen)

        #print (sent.shape)

        sent = self.convnet1(sent)
        u1 = torch.max(sent,2)[0]

        sent = self.convnet2(sent)
        u2 = torch.max(sent, 2)[0]

        sent = self.convnet3(sent)
        u3 = torch.max(sent, 2)[0]

        sent = self.convnet4(sent)
        u4 = torch.max(sent, 2)[0]

        #print("weight",self.weight)

        emb = torch.cat((u1, u2, u3, u4), 1)

        return emb


"""
Main module for Natural Language Inference
"""
class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 2*self.enc_lstm_dim #modify here
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder1","ConvNetEncoder2","ConvNetEncoder3","ConvNetEncoder4", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        '''
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        '''
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(320, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(320, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

    def forward(self, s1):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        output = self.classifier(u)
        
        #modify for the remote homology detection
        #v = self.encoder(s2)
        #features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        #output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb


"""
Main module for Classification
"""
class ClassificationNet(nn.Module):
    def __init__(self, config):
        super(ClassificationNet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 2*self.enc_lstm_dim
        self.inputdim = 4*self.inputdim if self.encoder_type == "ConvNetEncoder" else self.inputdim
        self.inputdim = self.enc_lstm_dim if self.encoder_type =="LSTMEncoder" else self.inputdim
        self.classifier = nn.Sequential(nn.Linear(self.inputdim, 512), nn.Linear(512, self.n_classes))

    def forward(self, s1):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)

        output = self.classifier(u)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb