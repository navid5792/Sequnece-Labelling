#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 01:25:16 2018

@author: bob
"""


import pickle

with open("suffix_stat.pkl","rb")as f:
    suffix_stat = pickle.load(f)
    
top_suffix = {} # ly --> RB, NN
for x in suffix_stat.keys():
    a = []
    for i in range(min(len(suffix_stat[x]), 2)):
        if suffix_stat[x][i][2] > 29:
            a.append(suffix_stat[x][i][0])
    top_suffix[x] = a

all_word_pos = []# gono list
for i in range(len(pairs)):
    word = pairs[i][0].split()
    pos = pairs[i][1].split()
    for j in range(len(word)):
        all_word_pos.append([word[j], pos[j]])
        
no_suf = {}
keys = list(sorted(top_suffix.keys(), key = lambda x : len(x), reverse = True))
for x in keys:
    no_suf[x] = []
    
for i in range(len(all_word_pos)):
    for x in keys: 
        if all_word_pos[i][0].endswith(x):
            if all_word_pos[i][1] not in top_suffix[x]:
                no_suf[x].append(all_word_pos[i][0])
            break
for x in keys:
    no_suf[x] = list(set(no_suf[x]))  
    
with open("No_suffix.pkl","wb") as f:
    pickle.dump(no_suf, f)