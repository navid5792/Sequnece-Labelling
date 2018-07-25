#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:59:41 2018

@author: bob
"""

from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt

with open("Y's_58.pkl", "rb") as f:
    t, p = pickle.load(f)

with open("dictionary.pkl","rb") as f:
    word2index,index2word,pos2index,index2pos = pickle.load(f)

pred = set()    
for i in range(len(p)):
    pred.add(p[i])
predicted_set = list(pred)

true = set()
for i in range(len(t)):
    true.add(t[i])
true_set = list(true)

true_set_labels = [index2pos[x] for x in true_set]

true =[]
pred =[]


important_pos = ["JJ", "RB", "VBD", "RP", "NN", "VBG", "RBR", "VBN", "JJR", "NNPS", "VB", "RBS", "LS", "PDT", "FW", "SYM", "UH"]
important_index = [pos2index[x] for x in important_pos]

for i in range(len(t)):
        if index2pos[t[i]] in important_pos and index2pos[p[i]] in important_pos:
            true.append(index2pos[t[i]])
            pred.append(index2pos[p[i]])
    
