#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:51:45 2018

@author: bob
"""

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
import numpy as np
from features import *
import sys
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
lemmatizer = WordNetLemmatizer()
import sys
from correct_suffix_read import *



USE_CUDA = True

PAD_token = 0
SOS_token = 1
EOS_token = 2
MIN_LENGTH = 3
MAX_LENGTH = 5000000

ACC_seq = []

#train_data = "data/POS/train.txt"
#test_data = "data/POS/test.txt"
#dicts = "data/POS/all.txt"
#dicts_sense = "data/POS/all_s.txt"
#train_data_sense = "data/POS/train_s.txt"
#test_data_sense = "data/POS/test_s.txt"

train_data = "data/POS+CONLL/train.txt"
test_data = "data/POS+CONLL/test.txt"
dicts = "data/POS+CONLL/all.txt"
dicts_sense = "data/POS+CONLL/all_s.txt"
train_data_sense = "data/POS+CONLL/train_s.txt"
test_data_sense = "data/POS+CONLL/test_s.txt"


class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD" : 0, "SOS" : 1 , "EOS" : 2}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<p>", 1: "<s>", 2: "<e>"}
        self.n_words = 3 # Count default tokens

        for word in keep_words:
            self.index_word(word)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.strip())
#    s = s.rstrip('.').strip()
    
    return s

def read_langs(lang1, lang2, data_dir, reverse=False):
    print("Reading lines...")
    
    filename = data_dir
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH \
            and len(pair[1]) >= MIN_LENGTH and len(pair[1]) <= MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs



def prepare_data(lang1_name, lang2_name, data_dir, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, data_dir, reverse)


    for pair in pairs: 
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    
#    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    return input_lang, output_lang, pairs

## just words
input_lang0, output_lang0, pairs = prepare_data('train_w', 'train_s', train_data, False)
print("training words: ",len(pairs))
input_lang, output_lang, pairs0 = prepare_data('all_w', 'all_s', dicts, False)
print("all words: ",len(pairs0))
input_lang0, output_lang0, test_pairs = prepare_data('test_w', 'test_s', test_data, False)
print("test words: ",len(test_pairs))

#just sense
input_lang_sense, output_lang_sense, pairs0_sense = prepare_data('all_w_sense', 'all_s_sense', dicts_sense, False)
print("all words_sense: ",len(pairs0))
input_lang0_sense, output_lang0_sense, pairs_sense = prepare_data('test_w_sense', 'test_s_sense', train_data_sense, False)
print("training words sense: ",len(pairs_sense))
input_lang0_sense, output_lang0_sense, test_pairs_sense = prepare_data('test_w_sense', 'test_s_sense', test_data_sense, False)
print("test words sense: ",len(test_pairs_sense))


'''stemming and suffix'''


with open("suffix_.txt","r") as f:
    real_suf = f.readlines()
    real_suf = [x.strip() for x in real_suf]


stemmer = SnowballStemmer("english")
suffix = set()
keys = list(input_lang.word2index.keys())

suffix2index = dict()
for i in range(len(real_suf)):
    suffix2index[real_suf[i]] = len(suffix2index)
suffix2index['<uns>'] = len(suffix2index)

word2suffix = dict()

#suffix_list = ['ly', 'tion', 'able', 'er', 'est', 'ing', 'ed', 'ful', 'ology', 'phobia']
with open("No_suffix.pkl","rb") as f:
    noo_suf = pickle.load(f)
    
mercer_korse = list(no_suf.keys())

for x in keys:
    
    flag = 0
    for i in range(len(real_suf)):

        if real_suf[i] in mercer_korse:
            if x.endswith(real_suf[i]) and (x not in no_suf[real_suf[i]]):
                Binary = list(bin(suffix2index[real_suf[i]])[2:])
                Binary = ['0'] * (8 - len(Binary)) + Binary
                for p in range(len(Binary)):
                    Binary[p] = int(Binary[p])
                word2suffix[x] = Binary
                     
                flag = 1
                break

    
    if flag == 0:
        Binary = list(bin(suffix2index['<uns>'])[2:])
        for p in range(len(Binary)):
            Binary[p] = int(Binary[p])
        word2suffix[x] = Binary
#        word2suffix[x] = [0] * len(suffix2index)
            
        
'''not real suffix'''
#for x in keys:
#    root = stemmer.stem(x)
##    if x.endswith("cation"):
##        print(x, root)
#        
#    if (root in x) and x.index(root) == 0:
#        suffix.add(x[len(root):])
#
#
#suffix = list(suffix)
#suffix2index = dict()
#for i in range(len(suffix)):
#    suffix2index[suffix[i]] = i 
#suffix2index['<uns>'] = len(suffix2index)
#
#word2suffix = dict()
#for x in keys:
#    root = stemmer.stem(x)
##    if x.endswith("cation"):
##        print(x, root)
#        
#    if (root in x) and x.index(root) == 0:
#        word2suffix[x] = suffix2index[x[len(root):]]
#    else:
#        word2suffix[x] = suffix2index['<uns>']

''''''''''''
''' prefix'''
#with open("prefix.txt","r") as f:
#    real_pref = f.readlines()
#    real_pref = [x.strip() for x in real_pref]
#
#
#prefix = set()
#keys = list(input_lang.word2index.keys())
#
#prefix2index = dict()
#for i in range(len(real_pref)):
#    prefix2index[real_pref[i]] = len(prefix2index) 
#prefix2index['<unp>'] = len(prefix2index)
#
#word2prefix = dict()
#
#for x in keys:
#    for r_pref in real_pref:
#        if x.startswith(r_pref):
#            word2prefix[x] = prefix2index[r_pref]
#            break
#        else:
#            word2prefix[x] = prefix2index['<unp>']
#            break
'''prefix'''


chars = set()
for i in range(len(pairs0)):
    chars = set.union(chars,set(list(pairs0[i][0])))

char2index = {}
chars = list(chars)
for i in range(len(chars)):
    char2index[chars[i]] = len(char2index)

char2index['^'] = len(char2index)
MIN_COUNT = 1

# print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
keep_pairs = []

for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True
    
    for word in input_sentence.split(' '):
        if word not in input_lang.word2index:
            keep_input = False
            break

    for word in output_sentence.split(' '):
        if word not in output_lang.word2index:
            keep_output = False
            break

    # Remove if pair doesn't match input and output conditions
    if keep_input and keep_output:
        keep_pairs.append(pair)

print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
pairs = keep_pairs


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] 

# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size, pairs, index):
    
 
    input_seqs = []
    target_seqs = []
    sense_seqs = []
    suffix_seqs = []
    
    arr = np.arange(index * batch_size, min ( (index * batch_size + batch_size), len(pairs)))
    np.random.shuffle(arr)
    if index == (len(pairs) // batch_size + 1):
        arr = np.arange((index-1)*batch_size,len(pairs))
        np.random.shuffle(arr)
#        print((index-1)*batch_size,len(pairs))
        
    for j in range(len(arr)):
        i = arr[j]
        pair = pairs[i]
#        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))
        sense_seqs.append(indexes_from_sentence(input_lang_sense, pair[2]))


    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs, sense_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs, sense_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]
    
    sense_lengths = [len(s) for s in sense_seqs]
    sense_padded = [pad_seq(s, max(sense_lengths)) for s in sense_seqs]
    
  # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    sense_var = Variable(torch.LongTensor(sense_padded)).transpose(0, 1)

    if USE_CUDA:
        torch.cuda.set_device(0)
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        sense_var = sense_var.cuda()

        
    return input_var, input_lengths, target_var, target_lengths, sense_var



