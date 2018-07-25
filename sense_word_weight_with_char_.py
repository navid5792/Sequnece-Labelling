import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket
hostname = socket.gethostname()
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *
from Vocab import *
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
from sorting import sort_sentence
import numpy as np
from CRF import *
import sys
from copy import deepcopy
import highway

# Configure models
attn_model = 'dot'
embed_size = 100
hidden_size = 300
suffix_dim = 100
prefix_dim = 100
n_layers = 3
dropout = 0.5
batch_size = 10
spelling_dim = 100

# Configure training/optimization
clip = 5
teacher_forcing_ratio = 0.5
learning_rate = 0.015
lr_decay = 0.05
decoder_learning_ratio = 5.0
n_epochs = 1000
epoch = 0
plot_every = 20
print_every = 100
evaluate_every = 100



    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, char_input_size, embed_size, hidden_size, n_class, sense_size, suffix_size, spelling_size, crf, n_layers=1, dropout=0.1, if_highway = False):
        super(EncoderRNN, self).__init__()
        self.char_dim = 30
        self.char_hidden = 300
        self.crf = crf
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_class = self.crf.n_labels
        self.embedding = nn.Embedding(input_size, embed_size)
        self.embedding_sense = nn.Embedding(sense_size, hidden_size)
#        self.embedding_spelling = nn.Embedding(spelling_size, spelling_dim)
        self.lstm_word = nn.LSTM(hidden_size + embed_size + 8 , hidden_size, n_layers, dropout = self.dropout ,  bidirectional=True)
        self.lstm_sense = nn.LSTM( hidden_size, hidden_size, n_layers,dropout = self.dropout, bidirectional=True)
        self.lstm_char = nn.GRU(self.char_dim, self.char_hidden, dropout = self.dropout,  bidirectional = False)
        self.lstm_picked_char = nn.LSTM ( self.char_hidden, hidden_size,  n_layers, dropout = self.dropout, bidirectional=True) # picked char embeddings
        self.lstm_bigram = nn.LSTM ( embed_size, hidden_size,  n_layers, dropout = self.dropout, bidirectional=True)
        
        self.concat = nn.Linear(1*hidden_size + 14 + len(char2index)*0 + 0,hidden_size)
        self.fc = nn.Linear(hidden_size,self.n_class)
        self.W = Variable(torch.rand(1,hidden_size))
        
        
        self.char_embedding = nn.Embedding(char_input_size, self.char_dim)
        self.char_lstm = nn.GRU(self.char_dim, self.char_hidden, dropout = self.dropout,  bidirectional = False)
#        self.suffix_embed = nn.Embedding(suffix_size, suffix_dim)
#        self.prefix_embed = nn.Embedding(prefix_size, prefix_dim)
                
        self.W1 = nn.Parameter(torch.FloatTensor([1]).cuda(),requires_grad =True)
        self.W2 = nn.Parameter(torch.FloatTensor([1]).cuda(),requires_grad =True)
        self.W4 = nn.Parameter(torch.FloatTensor([1]).cuda(),requires_grad =True)
        self.W3 = nn.Parameter(torch.FloatTensor([1]).cuda(),requires_grad =True)
        
        self.u_gru = nn.LSTM(hidden_size, hidden_size, 1, dropout = self.dropout ,  bidirectional=False)
       
        if if_highway:
            print("highway on")
            self.fb2char = highway.hw(1 * self.char_hidden, num_layers=1)
        self.if_highway = if_highway
        self.gate = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.conv = nn.Conv1d(embed_size, embed_size, 2, stride=1) # 300 300 kernel = 2  stride =1 
    
    def load_embeddings(self, tensors, sense_tensors):
        self.embedding.weight = nn.Parameter(tensors)
        self.embedding_sense.weight = nn.Parameter(sense_tensors)
        
    def forward(self, input_seqs,  input_lengths, sense_seqs, y, input_suffix, spelling, last_two, hidden=None, return_logits = False, train = True):
        # Note: we run this all at once (over multiple batches of multiple sequences)
#        print(input_seqs)
#        print(input_seqs,input_lengths,y)
        
#        input_seqs = input_seqs.cuda()
#        sense_seqs = se
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)
        embedded_sense = self.embedding_sense(sense_seqs)
        embedded_sense = self.dropout(embedded_sense)
        
        # Bigrams
        
#        start = torch.FloatTensor([input_lang.word2index['SOS']]*input_seqs.size(1)).view(1,-1)
#        input_seqs_cat = Variable(torch.cat([input_seqs.cpu().float().data,start],0).long().cuda())
#           
#        embedded_bi = self.embedding(input_seqs_cat)
##        embedded_bi = self.dropout(embedded_bi)
#        embedded_bi = self.conv(embedded_bi.transpose(0,1).transpose(1,2)) 13 10 300 -> 10 13 300 -> 10 200 13 after conv 10 12 300
#        embedded_bi = embedded_bi.transpose(1,2).transpose(0,1) # 10 12 300 -> 12 10 300
#
        temp = list(input_seqs.data.cpu().numpy())
        temp = np.transpose(temp)

        
        WWW_list = []
        real_length = []
        MAXLEN = []
        
        for i in range(len(temp)): #100
            WW_list= []
            maxlen = 0
            real_length_temp = []
            for j in range(len(temp[i])): #46
                x = list(input_lang.index2word[temp[i][j]])
                W_list =[]
                real_length_temp.append(len(x))
                for k in range(len(x)):
                    if maxlen < len(x):
                        maxlen = len(x)
                    W_list.append(char2index[x[k]])
                WW_list.append(W_list)
            real_length.append(real_length_temp)
            MAXLEN.append(maxlen)
            WWW_list.append(WW_list)     
        for i in range(len(WWW_list)):#100
            for j in range(len(WWW_list[i])):#46
                
                for k in range(MAXLEN[i]-len(WWW_list[i][j])):
                    WWW_list[i][j].append(char2index['^'])

        temp_list_baire = Variable(torch.zeros(len(WWW_list),len(WWW_list[i]), 1 * self.char_hidden).cuda(), requires_grad = True)

        for i in range(len(WWW_list)):
            a,b = sort_sentence(deepcopy(WWW_list[i]),deepcopy(real_length[i]))
            a = np.transpose(a)
            padded_len = [x[0] for x in b]
            sentence = Variable(torch.LongTensor(a).cuda(), requires_grad = True)
            embedded_c = self.char_embedding(sentence)
            packed_c = torch.nn.utils.rnn.pack_padded_sequence(embedded_c, padded_len)
            char_hid = None
            char_embed, char_hid = self.lstm_char(packed_c, char_hid)
            char_hid = char_hid.squeeze(0)
            temp_list = Variable(torch.zeros(char_hid.size(0),char_hid.size(1)).cuda(), requires_grad = True)
            for k in range(len(b)):
               temp_list[b[k][1]] = char_hid[k] 
            temp_list_baire[i] = temp_list
        temp_list_baire = temp_list_baire.transpose(0,1)
        if self.if_highway:
             temp_list_baire = self.fb2char(temp_list_baire)

        embedded = torch.cat([embedded, temp_list_baire, input_suffix],2)
           
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.lstm_word(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = self.dropout(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        
        packed_sense = torch.nn.utils.rnn.pack_padded_sequence(embedded_sense, input_lengths)
        hidden = None
        outputs_sense, hidden = self.lstm_sense(packed_sense, hidden)
        outputs_sense, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs_sense) # unpack (back to padded)
        outputs_sense = self.dropout(outputs_sense)
        outputs_sense = outputs_sense[:, :, :self.hidden_size] + outputs_sense[:, : ,self.hidden_size:]
        
        packed_char = torch.nn.utils.rnn.pack_padded_sequence(temp_list_baire, input_lengths)
        hidden = None
        outputs_char, hidden = self.lstm_picked_char(packed_char, hidden)
        outputs_char, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs_char) # unpack (back to padded)
        outputs_char = self.dropout(outputs_char)
        outputs_char = outputs_char[:, :, :self.hidden_size] + outputs_char[:, : ,self.hidden_size:]
#
#        packed_bigram = torch.nn.utils.rnn.pack_padded_sequence(embedded_bi, input_lengths)
#        hidden = None
#        outputs_bigram, hidden = self.lstm_bigram(packed_bigram, hidden)
#        outputs_bigram, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs_bigram) # unpack (back to padded)
#        outputs_bigram = self.dropout(outputs_bigram)
#        outputs_bigram = outputs_bigram[:, :, :self.hidden_size] + outputs_bigram[:, : ,self.hidden_size:]

#        g1 = nn.functional.sigmoid(self.gate(torch.cat([outputs,outputs_sense],2)))
#        outputs =  g1 * outputs + (1-g1) * outputs_sense
        
        
#        outputs =  self.W1 * outputs + self.W2 * outputs_sense + self.W3 * outputs_char + self.W4 * outputs_bigram 
        outputs =  self.W1 * outputs + self.W2 * outputs_sense + self.W3 * outputs_char
#        outputs =  self.W1 * outputs + self.W2 * outputs_sense
        
#        g1 = nn.functional.sigmoid(self.gate(outputs))
#        g2 = nn.functional.sigmoid(self.gate(outputs_sense))
#        outputs =  g1 * outputs + g2 * outputs_sense
#        outputs =  outputs +  outputs_sense.
        

#        tmp_outputs = outputs.transpose(0,1).transpose(1,2) s b e -> b s e -> b e s
#        U = Variable(torch.zeros(tmp_outputs.size(0), tmp_outputs.size(2))) # B x S
#        
#        tmp_outputs = tmp_outputs.cpu()
#        
#        for i in range(tmp_outputs.size(0)): # batch
#            U[i] = torch.matmul(self.W, F.tanh(tmp_outputs[i])) #1 e x e s -> 1 s
#        if USE_CUDA:
#           U = U.cuda()   
#        A = F.softmax(U).unsqueeze(1); # bx1xs
#        C = A.bmm(outputs.transpose(0, 1)) # bx1xs .  bxsxe -> bx1xe 
#        C = C.squeeze(1) # bx1xe -> bxe
        
        # Another way to add attention
        
        # bxsxe . bxexs bxsxs attn
        # softmax(attn)
        # bxsxs . bxsxe bxsxe output
        
        # calculate context
        # bxsxe . bxex1 -> bxsx1 -
        # bx1xs . bxsxe -> bx1xe
        
#        outputs = torch.cat([outputs,temp_list_baire],2)
        predict = []
        P = []
        
        for i in range(outputs.size(0)):
#            rnn_out = torch.cat([outputs[i],C],1)   
#            rnn_out = torch.cat([outputs[i], spelling[i]], 1)
#            rnn_out = torch.cat([rnn_out, last_two[i]],1)
#            rnn_out = torch.cat([rnn_out,input_suffix[i]],1)
#            rnn_out = torch.cat([rnn_out,embedded_prefix[i]],1)
#            print(rnn_out)
#            rnn_out = outputs[i]
#            rnn_out = self.fc(rnn_out)
#            rnn_out= self.dropout(rnn_out)
            rnn_out = outputs[i].unsqueeze(0)
            P.append(rnn_out)

        for i in range(1,len(P)):
            P[0] = torch.cat([P[0],P[i]],0)
#        print(P[0])
#        asas
        P = P[0]
#        print(P[0])
#        asas
        hid = None
#        outputs,hid = self.u_gru(P[0],hid)
        
        for i in range(outputs.size(0)):
            output,hid = self.u_gru(P[i].unsqueeze(0),hid)
            output = self.dropout(output)
            output = output.squeeze(0)
#            print(output, spelling[i])
#            aas
            rnn_out = torch.cat([output, spelling[i]], 1)
           
 
            rnn_out = F.tanh(self.concat(rnn_out))
            rnn_out = self.dropout(rnn_out)
            rnn_out = self.fc(rnn_out)
            rnn_out = self.dropout(rnn_out)
            predict.append(rnn_out)
            
        predict[0] = predict[0].unsqueeze(0)
        for i in range(1,outputs.size(0)):
            predict[0] = torch.cat([predict[0],predict[i].unsqueeze(0)],0)
        
        logits = predict[0].transpose(0,1)
        lens = Variable(torch.LongTensor(input_lengths).cuda(), requires_grad = True)
        if train ==False:
            scores, preds = self.crf.viterbi_decode(logits, lens)
            return preds
        
        norm_score = self.crf(logits, lens)
        
        transition_score = self.crf.transition_score(y.transpose(0,1), lens)
        bilstm_score = self._bilstm_score(logits, y.transpose(0,1), lens)
        sequence_score = transition_score + bilstm_score
        loglik = sequence_score - norm_score

        if return_logits:
            return loglik, logits
        else:
            return loglik
    
    def _bilstm_score(self, logits, y, lens):
        y_exp = y.unsqueeze(-1)
        scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score

